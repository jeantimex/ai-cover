"""
Voice Training Tool

Train a custom RVC voice model from multiple audio sources.
Supports YouTube URLs and local audio files.

Usage:
    python train.py --name "taylor" \
        "https://youtube.com/watch?v=..." \
        "https://youtube.com/watch?v=..." \
        song.mp3
"""

import os
import gc
import argparse
import json
from pathlib import Path
from typing import List
import shutil

import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm

from download import acquire_audio
from separate import separate_audio

# Voices directory
VOICES_DIR = Path(__file__).parent / "voices"


def get_voice_dir(name: str) -> Path:
    """Get the directory for a voice model."""
    return VOICES_DIR / name


def list_voices() -> List[str]:
    """List all available trained voices."""
    if not VOICES_DIR.exists():
        return []
    return [d.name for d in VOICES_DIR.iterdir() if d.is_dir() and (d / "config.json").exists()]


def collect_training_data(
    sources: List[str],
    voice_name: str,
    output_dir: Path,
) -> List[Path]:
    """
    Collect and separate vocals from multiple sources.

    Args:
        sources: List of YouTube URLs or local file paths
        voice_name: Name for the voice model
        output_dir: Where to save the collected data

    Returns:
        List of paths to separated vocal files
    """
    vocals_dir = output_dir / "raw_vocals"
    vocals_dir.mkdir(parents=True, exist_ok=True)

    vocal_files = []

    for i, source in enumerate(sources):
        print(f"\n[{i+1}/{len(sources)}] Processing: {source[:50]}...")

        try:
            # Download if URL
            audio_path, source_id = acquire_audio(source, output_dir / "downloads")

            # Separate vocals
            result = separate_audio(
                audio_path=audio_path,
                output_base_dir=output_dir / "separated",
            )

            # Copy the dry main vocal to our collection
            vocal_dest = vocals_dir / f"{source_id}_vocal.wav"
            shutil.copy(result.dry_main_vocal, vocal_dest)
            vocal_files.append(vocal_dest)

            print(f"  Extracted vocal: {vocal_dest.name}")

        except Exception as e:
            print(f"  Error processing {source}: {e}")
            continue

    return vocal_files


def preprocess_vocals(
    vocal_files: List[Path],
    output_dir: Path,
    segment_length: float = 10.0,
    sample_rate: int = 44100,
) -> Path:
    """
    Preprocess vocals for training: normalize, slice into segments.

    Args:
        vocal_files: List of vocal audio files
        output_dir: Where to save processed segments
        segment_length: Length of each segment in seconds
        sample_rate: Target sample rate

    Returns:
        Path to the processed segments directory
    """
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_samples = int(segment_length * sample_rate)
    segment_count = 0
    total_duration = 0.0

    print("\nPreprocessing vocals...")

    for vocal_file in tqdm(vocal_files, desc="Processing files"):
        # Load audio
        audio, sr = torchaudio.load(vocal_file)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)

        audio = audio.squeeze().numpy()
        total_duration += len(audio) / sample_rate

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        # Slice into segments
        for start in range(0, len(audio) - segment_samples, segment_samples // 2):
            segment = audio[start:start + segment_samples]

            # Skip silent segments
            if np.max(np.abs(segment)) < 0.01:
                continue

            segment_path = segments_dir / f"segment_{segment_count:05d}.wav"
            sf.write(segment_path, segment, sample_rate)
            segment_count += 1

    print(f"  Total duration: {total_duration/60:.1f} minutes")
    print(f"  Created {segment_count} segments")

    return segments_dir


def extract_features(
    segments_dir: Path,
    output_dir: Path,
    device: torch.device = None,
) -> Path:
    """
    Extract HuBERT features and F0 for training.

    Args:
        segments_dir: Directory containing audio segments
        output_dir: Where to save features
        device: Torch device

    Returns:
        Path to the features directory
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting features on {device}...")

    # Load HuBERT
    from transformers import HubertModel
    print("  Loading HuBERT...")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert = hubert.to(device)
    hubert.eval()

    # Load F0 extractor
    f0_fn = None
    try:
        from seed_vc.modules.rmvpe import RMVPE
        from seed_vc.hf_utils import load_custom_model_from_hf
        print("  Loading RMVPE...")
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        f0_extractor = RMVPE(model_path, is_half=False, device=device)
        f0_fn = f0_extractor.infer_from_audio
    except Exception as e:
        print(f"  Warning: Could not load RMVPE: {e}")

    # Process each segment
    segment_files = sorted(segments_dir.glob("*.wav"))

    for segment_file in tqdm(segment_files, desc="Extracting features"):
        audio, sr = torchaudio.load(segment_file)
        audio = audio.squeeze()

        # Resample to 16kHz for HuBERT
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)

        # Extract HuBERT features
        with torch.no_grad():
            audio_input = audio_16k.unsqueeze(0).to(device)
            outputs = hubert(audio_input, output_hidden_states=True)
            hubert_features = outputs.hidden_states[9].cpu().numpy()

        # Extract F0
        f0 = None
        if f0_fn is not None:
            f0 = f0_fn(audio_16k.numpy(), thred=0.03)

        # Save features
        feature_name = segment_file.stem
        np.save(features_dir / f"{feature_name}_hubert.npy", hubert_features)
        if f0 is not None:
            np.save(features_dir / f"{feature_name}_f0.npy", f0)

    # Cleanup
    del hubert
    gc.collect()

    return features_dir


def train_model(
    features_dir: Path,
    output_dir: Path,
    voice_name: str,
    epochs: int = 100,
) -> Path:
    """
    Train the RVC model.

    Note: Full training requires the VITS architecture.
    This is a placeholder that will be implemented when
    we port the training code.
    """
    print("\nTraining model...")
    print("  [Note] Full RVC training not yet implemented")
    print("  [Note] Features have been extracted and saved")
    print("  [Note] You can use these with external RVC training tools")

    # Create a placeholder model config
    model_dir = output_dir
    config = {
        "name": voice_name,
        "version": "1.0",
        "features_dir": str(features_dir),
        "status": "features_extracted",
        "note": "Use external RVC training or wait for full implementation",
    }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return model_dir


def train_voice(
    name: str,
    sources: List[str],
    epochs: int = 100,
) -> Path:
    """
    Main training function.

    Args:
        name: Name for the voice model
        sources: List of YouTube URLs or local audio files
        epochs: Number of training epochs

    Returns:
        Path to the trained model directory
    """
    print("=" * 60)
    print(f"Training voice model: {name}")
    print("=" * 60)

    # Create voice directory
    voice_dir = get_voice_dir(name)
    if voice_dir.exists():
        response = input(f"Voice '{name}' already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return None
        shutil.rmtree(voice_dir)

    voice_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect training data
    print("\n[Step 1/4] Collecting training data...")
    vocal_files = collect_training_data(sources, name, voice_dir)

    if not vocal_files:
        print("Error: No vocals extracted. Check your input sources.")
        return None

    print(f"  Collected {len(vocal_files)} vocal tracks")

    # Step 2: Preprocess vocals
    print("\n[Step 2/4] Preprocessing vocals...")
    segments_dir = preprocess_vocals(vocal_files, voice_dir)

    # Step 3: Extract features
    print("\n[Step 3/4] Extracting features...")
    features_dir = extract_features(segments_dir, voice_dir)

    # Step 4: Train model
    print("\n[Step 4/4] Training model...")
    model_dir = train_model(features_dir, voice_dir, name, epochs)

    print("\n" + "=" * 60)
    print(f"Voice '{name}' prepared!")
    print(f"Location: {voice_dir}")
    print("=" * 60)

    return voice_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom voice model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from YouTube videos
  python train.py --name "taylor" \\
      "https://youtube.com/watch?v=..." \\
      "https://youtube.com/watch?v=..."

  # Train from local files
  python train.py --name "my_voice" song1.mp3 song2.wav

  # List available voices
  python train.py --list
        """
    )

    parser.add_argument("sources", nargs="*", help="YouTube URLs or local audio files")
    parser.add_argument("--name", "-n", help="Name for the voice model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--list", "-l", action="store_true", help="List available voices")

    args = parser.parse_args()

    # List voices
    if args.list:
        voices = list_voices()
        if voices:
            print("Available voices:")
            for v in voices:
                print(f"  - {v}")
        else:
            print("No trained voices yet.")
        return

    # Validate inputs
    if not args.sources:
        parser.print_help()
        return

    # Get voice name
    name = args.name
    if not name:
        name = input("Enter a name for this voice: ").strip()
        if not name:
            print("Error: Voice name is required.")
            return

    # Train
    train_voice(name, args.sources, args.epochs)


if __name__ == "__main__":
    main()
