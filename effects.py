"""
Vocal post-processing effects and mixing.

Effects chain (from AICoverGen):
- HighpassFilter (remove rumble)
- Compressor (ratio=4, threshold=-15dB)
- Reverb (room_size=0.15, wet=0.2, dry=0.8, damping=0.7)

Mixing (default gains):
- Main vocal: -4 dB
- Backup vocals: -6 dB
- Instrumental: -7 dB
"""

import numpy as np
from pathlib import Path
from pedalboard import (
    Pedalboard,
    HighpassFilter,
    Compressor,
    Reverb,
    Gain,
)
from pedalboard.io import AudioFile
import soundfile as sf


def apply_vocal_effects(
    input_path: Path,
    output_path: Path,
    highpass_hz: float = 80.0,
    compressor_threshold_db: float = -15.0,
    compressor_ratio: float = 4.0,
    reverb_room_size: float = 0.15,
    reverb_wet: float = 0.2,
    reverb_dry: float = 0.8,
    reverb_damping: float = 0.7,
) -> Path:
    """
    Apply the standard vocal effects chain to a converted vocal.

    This re-adds controlled ambience that the de-reverb stage removed,
    preventing the vocal from sounding sterile and "pasted on".
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_hz),
        Compressor(
            threshold_db=compressor_threshold_db,
            ratio=compressor_ratio,
        ),
        Reverb(
            room_size=reverb_room_size,
            wet_level=reverb_wet,
            dry_level=reverb_dry,
            damping=reverb_damping,
        ),
    ])

    with AudioFile(str(input_path)) as f:
        sample_rate = f.samplerate
        audio = f.read(f.frames)

    effected = board(audio, sample_rate)

    sf.write(output_path, effected.T, sample_rate)

    return output_path


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10 ** (db / 20)


def mix_stems(
    main_vocal_path: Path,
    backup_vocal_path: Path,
    instrumental_path: Path,
    output_path: Path,
    main_vocal_db: float = -4.0,
    backup_vocal_db: float = -6.0,
    instrumental_db: float = -7.0,
    output_format: str = "wav",
) -> Path:
    """
    Mix the three stems with gain staging.

    Default gains sit the converted vocal slightly proud of the mix,
    which flatters it.
    """
    output_path = Path(output_path)

    # Load all stems
    main_vocal, sr = sf.read(main_vocal_path)
    backup_vocal, _ = sf.read(backup_vocal_path)
    instrumental, _ = sf.read(instrumental_path)

    # Ensure all are 2D (stereo)
    if main_vocal.ndim == 1:
        main_vocal = np.column_stack([main_vocal, main_vocal])
    if backup_vocal.ndim == 1:
        backup_vocal = np.column_stack([backup_vocal, backup_vocal])
    if instrumental.ndim == 1:
        instrumental = np.column_stack([instrumental, instrumental])

    # Align lengths to shortest
    min_len = min(len(main_vocal), len(backup_vocal), len(instrumental))
    main_vocal = main_vocal[:min_len]
    backup_vocal = backup_vocal[:min_len]
    instrumental = instrumental[:min_len]

    # Apply gains
    main_vocal = main_vocal * db_to_linear(main_vocal_db)
    backup_vocal = backup_vocal * db_to_linear(backup_vocal_db)
    instrumental = instrumental * db_to_linear(instrumental_db)

    # Mix
    mixed = main_vocal + backup_vocal + instrumental

    # Soft clip to prevent clipping while preserving dynamics
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    # Write output
    if output_format == "mp3":
        # For mp3, write wav first then convert
        wav_path = output_path.with_suffix(".wav")
        sf.write(wav_path, mixed, sr)
        # Use pedalboard for mp3 encoding
        with AudioFile(str(wav_path)) as f:
            audio = f.read(f.frames)
        with AudioFile(str(output_path), "w", sr, num_channels=2) as f:
            f.write(audio)
        wav_path.unlink()
    else:
        sf.write(output_path, mixed, sr)

    return output_path


def get_lufs(audio_path: Path) -> float:
    """
    Calculate integrated loudness in LUFS.
    Simplified implementation - for production use pyloudnorm.
    """
    audio, sr = sf.read(audio_path)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])

    # Simple RMS-based approximation (not true LUFS but close enough for QA)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        return 20 * np.log10(rms) - 0.691  # Approximate K-weighting offset
    return -np.inf


def normalize_to_lufs(
    input_path: Path,
    output_path: Path,
    target_lufs: float = -14.0,
) -> Path:
    """Normalize audio to target integrated loudness."""
    audio, sr = sf.read(input_path)

    current_lufs = get_lufs(input_path)
    gain_db = target_lufs - current_lufs

    audio = audio * db_to_linear(gain_db)

    # Prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    sf.write(output_path, audio, sr)
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python effects.py <input_vocal> <output_vocal>")
        sys.exit(1)

    result = apply_vocal_effects(
        input_path=Path(sys.argv[1]),
        output_path=Path(sys.argv[2]),
    )
    print(f"Effected vocal saved to: {result}")
