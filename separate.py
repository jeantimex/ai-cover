"""
Three-pass source separation cascade using audio-separator.

Stage 1: UVR-MDX-NET-Voc_FT.onnx -> Vocals + Instrumental
Stage 2: UVR_MDXNET_KARA_2.onnx -> Main vocals + Backup vocals
Stage 3: Reverb_HQ_By_FoxJoy.onnx -> Dry main vocal (de-reverbed)

Only the dry main vocal goes to voice conversion. Backup vocals and
instrumental are preserved for the final mix.
"""

import os
import gc
import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import librosa
import soundfile as sf
from audio_separator.separator import Separator


class SeparationResult(NamedTuple):
    """Paths to the separated stems."""
    dry_main_vocal: Path
    backup_vocals: Path
    instrumental: Path
    song_id: str


# Model names as used by audio-separator
MODEL_VOCAL_INSTRUMENTAL = "UVR-MDX-NET-Voc_FT.onnx"
MODEL_MAIN_BACKUP = "UVR_MDXNET_KARA_2.onnx"
MODEL_DEREVERB = "Reverb_HQ_By_FoxJoy.onnx"


def check_model_exists(model_dir: Path, model_name: str) -> bool:
    """Check if a model is already downloaded."""
    model_path = model_dir / model_name
    return model_path.exists()


def get_song_id(audio_path: Path) -> str:
    """Generate a stable ID for caching. Uses blake2b hash of file content."""
    hasher = hashlib.blake2b(digest_size=16)
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_stereo_44100(audio_path: Path, output_dir: Path) -> Path:
    """Convert audio to stereo 44.1kHz if needed. MDX models expect this format."""
    output_path = output_dir / "input_stereo.wav"

    if output_path.exists():
        return output_path

    y, sr = librosa.load(audio_path, sr=44100, mono=False)

    # Ensure stereo
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]

    sf.write(output_path, y.T, 44100)
    return output_path


def null_test_separation(original: Path, stem1: Path, stem2: Path, threshold_db: float = -40) -> bool:
    """
    Verify separation quality by checking that stems sum back to original.
    Residual should be below threshold_db RMS.
    """
    y_orig, sr = librosa.load(original, sr=44100, mono=True)
    y_stem1, _ = librosa.load(stem1, sr=44100, mono=True)
    y_stem2, _ = librosa.load(stem2, sr=44100, mono=True)

    # Align lengths
    min_len = min(len(y_orig), len(y_stem1), len(y_stem2))
    y_orig = y_orig[:min_len]
    y_stem1 = y_stem1[:min_len]
    y_stem2 = y_stem2[:min_len]

    # Compute residual
    residual = y_orig - (y_stem1 + y_stem2)
    rms = np.sqrt(np.mean(residual ** 2))

    if rms > 0:
        rms_db = 20 * np.log10(rms)
    else:
        rms_db = -np.inf

    passed = rms_db < threshold_db
    if not passed:
        print(f"  [QA] Null test failed: residual {rms_db:.1f} dB (threshold {threshold_db} dB)")
    else:
        print(f"  [QA] Null test passed: residual {rms_db:.1f} dB")

    return passed


def separate_audio(
    audio_path: Path,
    output_base_dir: Path,
    model_dir: Path | None = None,
    denoise: bool = True,
    use_mps: bool = True,
) -> SeparationResult:
    """
    Run the three-stage separation cascade.

    Args:
        audio_path: Input audio file
        output_base_dir: Base directory for outputs (song_output/)
        model_dir: Directory to store downloaded models
        denoise: Use sign-inversion denoising (2x slower but cleaner)
        use_mps: Use Metal Performance Shaders on Apple Silicon

    Returns:
        SeparationResult with paths to the three stems
    """
    audio_path = Path(audio_path)
    output_base_dir = Path(output_base_dir)

    if model_dir is None:
        model_dir = Path(__file__).parent / "models"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Generate song ID and create output directory
    song_id = get_song_id(audio_path)
    song_dir = output_base_dir / song_id
    song_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    instrumental_path = song_dir / "instrumental.wav"
    vocals_path = song_dir / "vocals.wav"
    main_vocal_path = song_dir / "main_vocal.wav"
    backup_vocal_path = song_dir / "backup_vocals.wav"
    dry_main_vocal_path = song_dir / "dry_main_vocal.wav"

    # Check cache - if all stems exist, skip separation
    if all(p.exists() for p in [dry_main_vocal_path, backup_vocal_path, instrumental_path]):
        print(f"Using cached stems for {song_id}")
        return SeparationResult(
            dry_main_vocal=dry_main_vocal_path,
            backup_vocals=backup_vocal_path,
            instrumental=instrumental_path,
            song_id=song_id,
        )

    # Ensure input is stereo 44.1kHz
    print("Preparing input audio...")
    stereo_input = ensure_stereo_44100(audio_path, song_dir)

    # Configure separator for MPS/CPU
    if use_mps:
        # audio-separator uses onnxruntime execution providers
        # CoreMLExecutionProvider or CPUExecutionProvider for Mac
        pass  # audio-separator auto-detects

    # Stage 1: Vocals + Instrumental
    if not vocals_path.exists() or not instrumental_path.exists():
        print(f"Stage 1: Separating vocals and instrumental...")
        if check_model_exists(model_dir, MODEL_VOCAL_INSTRUMENTAL):
            print(f"  Model {MODEL_VOCAL_INSTRUMENTAL} already downloaded")
        else:
            print(f"  Downloading model {MODEL_VOCAL_INSTRUMENTAL}...")
        separator = Separator(
            model_file_dir=str(model_dir),
            output_dir=str(song_dir),
            output_format="wav",
            normalization_threshold=0.9,
            output_single_stem=None,
        )
        separator.load_model(MODEL_VOCAL_INSTRUMENTAL)
        outputs = separator.separate(str(stereo_input))

        # audio-separator outputs are named based on stem type
        for out in outputs:
            # Find the actual file - could be in song_dir, cwd, or absolute
            out_path = Path(out)
            candidates = [
                out_path,
                song_dir / out_path.name,
                Path.cwd() / out_path.name,
            ]
            actual_path = None
            for candidate in candidates:
                if candidate.exists():
                    actual_path = candidate
                    break

            if actual_path is None:
                print(f"  Warning: Could not find output file: {out}")
                continue

            name_lower = actual_path.name.lower()
            if "instrumental" in name_lower:
                actual_path.rename(instrumental_path)
            elif "vocal" in name_lower:
                actual_path.rename(vocals_path)

        # Cleanup
        del separator
        gc.collect()

        # QA: null test
        null_test_separation(stereo_input, vocals_path, instrumental_path)

    # Stage 2: Main vocals + Backup vocals
    if not main_vocal_path.exists() or not backup_vocal_path.exists():
        print(f"Stage 2: Separating main and backup vocals...")
        if check_model_exists(model_dir, MODEL_MAIN_BACKUP):
            print(f"  Model {MODEL_MAIN_BACKUP} already downloaded")
        else:
            print(f"  Downloading model {MODEL_MAIN_BACKUP}...")
        separator = Separator(
            model_file_dir=str(model_dir),
            output_dir=str(song_dir),
            output_format="wav",
            normalization_threshold=0.9,
        )
        separator.load_model(MODEL_MAIN_BACKUP)
        outputs = separator.separate(str(vocals_path))

        for out in outputs:
            out_path = Path(out)
            candidates = [out_path, song_dir / out_path.name, Path.cwd() / out_path.name]
            actual_path = next((c for c in candidates if c.exists()), None)

            if actual_path is None:
                print(f"  Warning: Could not find output file: {out}")
                continue

            name_lower = actual_path.name.lower()
            # KARA model outputs "Vocals" (main) and "Instrumental" (backup/karaoke)
            if "instrumental" in name_lower or "karaoke" in name_lower:
                actual_path.rename(backup_vocal_path)
            elif "vocal" in name_lower:
                actual_path.rename(main_vocal_path)

        del separator
        gc.collect()

    # Stage 3: De-reverb main vocal
    if not dry_main_vocal_path.exists():
        print(f"Stage 3: Removing reverb from main vocal...")
        if check_model_exists(model_dir, MODEL_DEREVERB):
            print(f"  Model {MODEL_DEREVERB} already downloaded")
        else:
            print(f"  Downloading model {MODEL_DEREVERB}...")
        separator = Separator(
            model_file_dir=str(model_dir),
            output_dir=str(song_dir),
            output_format="wav",
            normalization_threshold=0.9,
        )
        separator.load_model(MODEL_DEREVERB)
        outputs = separator.separate(str(main_vocal_path))

        for out in outputs:
            out_path = Path(out)
            candidates = [out_path, song_dir / out_path.name, Path.cwd() / out_path.name]
            actual_path = next((c for c in candidates if c.exists()), None)

            if actual_path is None:
                print(f"  Warning: Could not find output file: {out}")
                continue

            name_lower = actual_path.name.lower()
            # Reverb model outputs dry (no reverb) and reverb stems
            if "no reverb" in name_lower or "dry" in name_lower:
                actual_path.rename(dry_main_vocal_path)
            else:
                # Discard the reverb-only stem
                actual_path.unlink()

        del separator
        gc.collect()

    print(f"Separation complete. Stems saved to {song_dir}")

    return SeparationResult(
        dry_main_vocal=dry_main_vocal_path,
        backup_vocals=backup_vocal_path,
        instrumental=instrumental_path,
        song_id=song_id,
    )


if __name__ == "__main__":
    import sys
    from download import acquire_audio

    if len(sys.argv) < 2:
        print("Usage: python separate.py <audio_file_or_youtube_url>")
        sys.exit(1)

    source = sys.argv[1]
    output_base = Path("song_output")

    # Handle YouTube URLs or local files
    audio_path, song_id = acquire_audio(source, output_base / "downloads")
    print(f"Song ID: {song_id}")

    result = separate_audio(
        audio_path=audio_path,
        output_base_dir=output_base,
    )

    print(f"\nResults:")
    print(f"  Dry main vocal: {result.dry_main_vocal}")
    print(f"  Backup vocals:  {result.backup_vocals}")
    print(f"  Instrumental:   {result.instrumental}")
