"""
Seed-VC zero-shot voice conversion (Engine B).

Converts vocals to match a target voice using just a 10-30s reference clip.
No training required - uses a single shared model.
"""

import os
import warnings
import logging
from pathlib import Path

# Suppress WhisperFeatureExtractor sampling_rate warning
warnings.filterwarnings("ignore", message=".*sampling_rate.*")
warnings.filterwarnings("ignore", message=".*WhisperFeatureExtractor.*")

# Suppress transformers warnings (including WhisperFeatureExtractor)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.models.whisper").setLevel(logging.ERROR)

# Also suppress via environment variable (for transformers custom logging)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np
import soundfile as sf

# Set MPS fallback before importing torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Monkey-patch BigVGAN to fix huggingface_hub compatibility
def _patch_bigvgan():
    """Fix BigVGAN._from_pretrained signature for newer huggingface_hub versions."""
    try:
        from seed_vc.modules.bigvgan import bigvgan
        original_from_pretrained = bigvgan.BigVGAN._from_pretrained.__func__

        @classmethod
        def patched_from_pretrained(
            cls,
            *,
            model_id: str,
            revision: str = None,
            cache_dir: str = None,
            force_download: bool = False,
            proxies: dict = None,
            resume_download: bool = False,
            local_files_only: bool = False,
            token = None,
            map_location: str = "cpu",
            strict: bool = False,
            use_cuda_kernel: bool = False,
            **model_kwargs,
        ):
            return original_from_pretrained(
                cls,
                model_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                map_location=map_location,
                strict=strict,
                use_cuda_kernel=use_cuda_kernel,
                **model_kwargs,
            )

        bigvgan.BigVGAN._from_pretrained = patched_from_pretrained
    except Exception:
        pass  # If patching fails, continue anyway

_patch_bigvgan()

# Patch the API to use load_models_v1 (which supports F0) instead of load_models_realtime
def _patch_api_for_f0():
    """Fix api.py to use the correct model loader for F0 support."""
    try:
        import seed_vc.api as api_module
        from seed_vc.inference import load_models as load_models_v1

        # Replace the realtime loader reference with the v1 loader
        api_module.load_models_realtime = load_models_v1
        print("  [Seed-VC] Patched API for F0 support")
    except Exception as e:
        print(f"  [Seed-VC] Warning: Could not patch API for F0: {e}")

_patch_api_for_f0()

# Patch torch.from_numpy to force float32 on MPS (MPS doesn't support float64)
def _patch_torch_for_mps():
    """Force float32 for numpy arrays on MPS since MPS doesn't support float64."""
    import torch
    original_from_numpy = torch.from_numpy

    def patched_from_numpy(ndarray):
        tensor = original_from_numpy(ndarray)
        # Convert float64 to float32 for MPS compatibility
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor

    torch.from_numpy = patched_from_numpy

_patch_torch_for_mps()

from seed_vc import api as seed_vc_api
from seed_vc.Models.audio import AudioData
from download import acquire_audio, trim_audio


def load_audio_as_audiodata(audio_path: Path) -> AudioData:
    """Load an audio file and convert to seed_vc AudioData format."""
    audio, sr = sf.read(audio_path)

    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Convert to int16 (AudioData expects int16 samples)
    audio_int16 = (audio * 32767).astype(np.int16)

    duration = len(audio) / sr
    samples_count = len(audio_int16)

    return AudioData(
        samples=audio_int16.tolist(),
        mel_chunks=None,
        duration=duration,
        samples_count=samples_count,
        sample_rate=sr,
        metadata=None,
    )


def resolve_reference(
    reference: str | Path,
    output_dir: Path,
    separate_vocals: bool = True,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Path:
    """
    Resolve reference audio - can be a local file or YouTube URL.
    Downloads if it's a URL, optionally separates vocals and trims.
    """
    reference_str = str(reference)

    if reference_str.startswith(('http://', 'https://', 'www.')):
        print(f"  Downloading reference audio from URL...")
        ref_path, ref_id = acquire_audio(reference_str, output_dir / "references")

        # Trim if time range specified
        if start_time is not None or end_time is not None:
            trimmed_path = output_dir / "references" / f"{ref_id}_trimmed.wav"
            ref_path = trim_audio(ref_path, trimmed_path, start_time, end_time)
            print(f"  Trimmed reference to {start_time or 0:.1f}s - {end_time or 'end'}s")

        if separate_vocals:
            # Import here to avoid circular import
            from separate import separate_audio

            # Generate a unique ID for trimmed version
            trim_suffix = ""
            if start_time is not None or end_time is not None:
                trim_suffix = f"_{int(start_time or 0)}_{int(end_time or 9999)}"

            # Check if we already have separated vocals for this reference
            ref_vocal_path = output_dir / "references" / f"{ref_id}{trim_suffix}" / "dry_main_vocal.wav"
            if ref_vocal_path.exists():
                print(f"  Using cached separated reference vocal")
                return ref_vocal_path

            print(f"  Separating vocals from reference audio...")
            result = separate_audio(
                audio_path=ref_path,
                output_base_dir=output_dir / "references",
            )
            return result.dry_main_vocal

        return ref_path

    # Local file - still apply trimming if specified
    local_path = Path(reference)
    if start_time is not None or end_time is not None:
        trimmed_path = output_dir / "references" / f"{local_path.stem}_trimmed.wav"
        return trim_audio(local_path, trimmed_path, start_time, end_time)

    return local_path


def convert_voice(
    source_path: Path,
    reference: str | Path,
    output_path: Path,
    separate_reference: bool = True,
    diffusion_steps: int = 30,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.7,
    f0_condition: bool = True,
    auto_f0_adjust: bool = False,
    semi_tone_shift: int = 0,
    fp16: bool = False,
    ref_start_time: float | None = None,
    ref_end_time: float | None = None,
) -> Path:
    """
    Convert source vocals to match the target voice.

    Args:
        source_path: Path to dry main vocal (from separation)
        reference: Path or YouTube URL to 10-30s reference audio of target voice
        output_path: Where to save the converted vocal
        diffusion_steps: Number of diffusion steps (higher = better quality, slower)
        length_adjust: Length adjustment factor (1.0 = same length)
        inference_cfg_rate: Classifier-free guidance rate
        f0_condition: Use F0 conditioning (True for singing to preserve melody)
        auto_f0_adjust: Auto-adjust F0 to match target voice range
        semi_tone_shift: Pitch shift in semitones
        fp16: Use half precision (False for MPS compatibility)

    Returns:
        Path to the converted vocal
    """
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source vocal not found: {source_path}")

    # Resolve reference - can be local file or YouTube URL
    reference_path = resolve_reference(
        reference,
        output_path.parent,
        separate_vocals=separate_reference,
        start_time=ref_start_time,
        end_time=ref_end_time,
    )
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Source: {source_path}")
    print(f"  Reference: {reference_path}")
    print(f"  F0 conditioning: {f0_condition}")
    print(f"  Auto F0 adjust: {auto_f0_adjust}")
    print(f"  Diffusion steps: {diffusion_steps}")
    print(f"  CFG rate: {inference_cfg_rate}")

    # Load audio files as AudioData objects
    print("  Loading audio files...")
    source_data = load_audio_as_audiodata(source_path)
    target_data = load_audio_as_audiodata(reference_path)

    print("  Running Seed-VC inference...")

    # Run inference
    # realtime=False uses the full inference path that supports F0 conditioning
    result = seed_vc_api.inference(
        source=source_data,
        target=target_data,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        semi_tone_shift=semi_tone_shift,
        fp16=fp16,
        realtime=False,  # Required for F0 conditioning to work
    )

    # Result is an AudioData object
    out_sr = result.sample_rate
    out_samples = np.array(result.samples, dtype=np.int16)
    out_audio = out_samples.astype(np.float32) / 32767.0

    # Save output
    sf.write(output_path, out_audio, out_sr)
    print(f"  Saved converted vocal to: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed-VC Voice Conversion")
    parser.add_argument("source", type=Path, help="Source vocal to convert")
    parser.add_argument("reference", help="Reference voice - path or YouTube URL (10-30s)")
    parser.add_argument("output", type=Path, help="Output path")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps")
    parser.add_argument("--pitch", type=int, default=0, help="Semitone shift")
    parser.add_argument("--no-f0", action="store_true", help="Disable F0 conditioning")

    args = parser.parse_args()

    convert_voice(
        source_path=args.source,
        reference=args.reference,
        output_path=args.output,
        diffusion_steps=args.steps,
        semi_tone_shift=args.pitch,
        f0_condition=not args.no_f0,
    )
