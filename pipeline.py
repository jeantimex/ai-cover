"""
AI Cover Pipeline - Main orchestration.

This is a Mac-native port of AICoverGen's song_cover_pipeline.
The flow is a strict linear cascade:

1. Input acquisition (YouTube download or local file)
2. Three-pass source separation
3. Voice conversion (RVC or Seed-VC)
4. Vocal post effects
5. Mixing
6. Cleanup (optional)
"""

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from download import acquire_audio
from separate import separate_audio, SeparationResult
from effects import apply_vocal_effects, mix_stems
from convert_seedvc import convert_voice as seedvc_convert
from convert_rvc import convert_voice as rvc_convert
from train import VOICES_DIR, list_voices, get_voice_dir


@dataclass
class PipelineConfig:
    """Configuration for the cover pipeline."""

    # Input
    source: str  # YouTube URL or local file path

    # Voice conversion
    engine: Literal["rvc", "seedvc"] = "seedvc"

    # RVC settings (Engine A)
    rvc_model_path: Path | None = None
    rvc_index_path: Path | None = None
    pitch_change: int = 0  # Semitones (+12 = up one octave)
    index_rate: float = 0.5
    filter_radius: int = 3
    protect: float = 0.33
    rms_mix_rate: float = 0.25
    f0_method: str = "rmvpe"

    # Seed-VC settings (Engine B)
    reference_audio: str | Path | None = None  # 10-30s voice sample (path or YouTube URL)
    separate_reference: bool = True  # Separate vocals from reference audio
    diffusion_steps: int = 50  # Higher = better quality, slower (30-100)
    auto_f0_adjust: bool = True  # Auto-adjust pitch to match reference voice range
    inference_cfg_rate: float = 0.7  # Classifier-free guidance (0.5-0.9)

    # Effects
    reverb_room_size: float = 0.15
    reverb_wet: float = 0.2
    reverb_dry: float = 0.8
    reverb_damping: float = 0.7

    # Mixing (dB)
    main_vocal_gain: float = -4.0
    backup_vocal_gain: float = -6.0
    instrumental_gain: float = -7.0

    # Output
    output_dir: Path = field(default_factory=lambda: Path("song_output"))
    output_format: Literal["wav", "mp3"] = "wav"
    keep_files: bool = True


@dataclass
class PipelineResult:
    """Result of the cover pipeline."""
    output_path: Path
    song_id: str
    stems: SeparationResult
    converted_vocal_path: Path
    effected_vocal_path: Path


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """
    Run the complete AI cover pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        PipelineResult with paths to all outputs
    """
    print("=" * 60)
    print("AI Cover Pipeline")
    print("=" * 60)

    # Stage 0: Input acquisition
    print("\n[Stage 0] Acquiring input audio...")
    audio_path, song_id = acquire_audio(config.source, config.output_dir / "downloads")
    print(f"  Song ID: {song_id}")

    song_dir = config.output_dir / song_id
    song_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Source separation
    print("\n[Stage 1] Source separation (3-pass cascade)...")
    stems = separate_audio(
        audio_path=audio_path,
        output_base_dir=config.output_dir,
    )
    gc.collect()

    # Stage 2: Voice conversion
    print(f"\n[Stage 2] Voice conversion ({config.engine})...")
    converted_vocal_path = song_dir / "converted_vocal.wav"

    if config.engine == "rvc":
        converted_vocal_path = convert_with_rvc(
            input_path=stems.dry_main_vocal,
            output_path=converted_vocal_path,
            model_path=config.rvc_model_path,
            index_path=config.rvc_index_path,
            pitch_change=config.pitch_change,
            index_rate=config.index_rate,
            filter_radius=config.filter_radius,
            protect=config.protect,
            rms_mix_rate=config.rms_mix_rate,
            f0_method=config.f0_method,
        )
    else:  # seedvc
        converted_vocal_path = convert_with_seedvc(
            input_path=stems.dry_main_vocal,
            output_path=converted_vocal_path,
            reference_audio=config.reference_audio,
            separate_reference=config.separate_reference,
            diffusion_steps=config.diffusion_steps,
            semi_tone_shift=config.pitch_change,
            auto_f0_adjust=config.auto_f0_adjust,
            inference_cfg_rate=config.inference_cfg_rate,
        )

    gc.collect()

    # Stage 3: Vocal effects
    print("\n[Stage 3] Applying vocal effects...")
    effected_vocal_path = song_dir / "effected_vocal.wav"
    apply_vocal_effects(
        input_path=converted_vocal_path,
        output_path=effected_vocal_path,
        reverb_room_size=config.reverb_room_size,
        reverb_wet=config.reverb_wet,
        reverb_dry=config.reverb_dry,
        reverb_damping=config.reverb_damping,
    )

    # Stage 4: Mixing
    print("\n[Stage 4] Mixing final output...")
    output_path = song_dir / f"cover.{config.output_format}"
    mix_stems(
        main_vocal_path=effected_vocal_path,
        backup_vocal_path=stems.backup_vocals,
        instrumental_path=stems.instrumental,
        output_path=output_path,
        main_vocal_db=config.main_vocal_gain,
        backup_vocal_db=config.backup_vocal_gain,
        instrumental_db=config.instrumental_gain,
        output_format=config.output_format,
    )

    # Stage 5: Cleanup
    if not config.keep_files:
        print("\n[Stage 5] Cleaning up intermediates...")
        for path in [converted_vocal_path, effected_vocal_path]:
            if path.exists():
                path.unlink()

    print("\n" + "=" * 60)
    print(f"Done! Output saved to: {output_path}")
    print("=" * 60)

    return PipelineResult(
        output_path=output_path,
        song_id=song_id,
        stems=stems,
        converted_vocal_path=converted_vocal_path,
        effected_vocal_path=effected_vocal_path,
    )


def convert_with_rvc(
    input_path: Path,
    output_path: Path,
    model_path: Path | None,
    index_path: Path | None,
    pitch_change: int,
    index_rate: float,
    filter_radius: int,
    protect: float,
    rms_mix_rate: float,
    f0_method: str,
) -> Path:
    """
    Convert voice using RVC (Engine A).
    Requires a trained .pth model and optional .index file.
    """
    if model_path is None:
        raise ValueError("RVC requires a model file (--rvc-model path/to/model.pth)")

    return rvc_convert(
        source_path=input_path,
        output_path=output_path,
        model_path=model_path,
        index_path=index_path,
        pitch_change=pitch_change,
        index_rate=index_rate,
        filter_radius=filter_radius,
        protect=protect,
        rms_mix_rate=rms_mix_rate,
        f0_method=f0_method,
    )


def convert_with_seedvc(
    input_path: Path,
    output_path: Path,
    reference_audio: str | Path | None,
    separate_reference: bool = True,
    diffusion_steps: int = 50,
    semi_tone_shift: int = 0,
    auto_f0_adjust: bool = True,
    inference_cfg_rate: float = 0.7,
) -> Path:
    """
    Convert voice using Seed-VC (Engine B).
    Zero-shot conversion using a 10-30s reference audio (local file or YouTube URL).
    """
    if reference_audio is None:
        raise ValueError("Seed-VC requires a reference audio (--reference path or URL)")

    return seedvc_convert(
        source_path=input_path,
        reference=reference_audio,
        output_path=output_path,
        separate_reference=separate_reference,
        diffusion_steps=diffusion_steps,
        semi_tone_shift=semi_tone_shift,
        f0_condition=True,  # Preserves pitch/melody for singing
        auto_f0_adjust=auto_f0_adjust,  # Match pitch range to reference
        inference_cfg_rate=inference_cfg_rate,
        fp16=False,  # MPS compatibility
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Cover Pipeline")
    parser.add_argument("source", help="YouTube URL or local audio file")
    parser.add_argument("--engine", choices=["rvc", "seedvc"], default="seedvc")
    parser.add_argument("--reference", help="Reference audio for Seed-VC (path or YouTube URL)")
    parser.add_argument("--voice", "-v", help="Use a trained voice by name (see: python train.py --list)")
    parser.add_argument("--no-separate-ref", action="store_true", help="Don't separate vocals from reference")
    parser.add_argument("--rvc-model", type=Path, help="RVC model path (.pth)")
    parser.add_argument("--rvc-index", type=Path, help="RVC index path (.index)")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch change in semitones")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps (30-100, higher=better)")
    parser.add_argument("--auto-f0", action="store_true", default=True, help="Auto-adjust pitch to reference range")
    parser.add_argument("--no-auto-f0", action="store_false", dest="auto_f0", help="Disable auto F0 adjustment")
    parser.add_argument("--cfg", type=float, default=0.7, help="Classifier-free guidance rate (0.5-0.9)")
    parser.add_argument("--output-dir", type=Path, default=Path("song_output"))
    parser.add_argument("--format", choices=["wav", "mp3"], default="wav")
    parser.add_argument("--no-keep-files", action="store_true")

    args = parser.parse_args()

    # Resolve --voice to reference audio
    reference = args.reference
    if args.voice:
        voice_dir = get_voice_dir(args.voice)
        if not voice_dir.exists():
            available = list_voices()
            print(f"Error: Voice '{args.voice}' not found.")
            if available:
                print(f"Available voices: {', '.join(available)}")
            else:
                print("No trained voices yet. Use: python train.py --name <name> <sources...>")
            exit(1)

        # Look for reference vocal in the voice directory
        raw_vocals = voice_dir / "raw_vocals"
        if raw_vocals.exists():
            vocal_files = list(raw_vocals.glob("*.wav"))
            if vocal_files:
                reference = str(vocal_files[0])  # Use first vocal as reference
                print(f"Using trained voice '{args.voice}': {reference}")

    if not reference and args.engine == "seedvc":
        print("Error: Seed-VC requires --reference or --voice")
        print("  Use: --reference <url_or_path>")
        print("  Or:  --voice <trained_voice_name>")
        exit(1)

    config = PipelineConfig(
        source=args.source,
        engine=args.engine,
        reference_audio=reference,
        separate_reference=not args.no_separate_ref,
        rvc_model_path=args.rvc_model,
        rvc_index_path=args.rvc_index,
        pitch_change=args.pitch,
        diffusion_steps=args.steps,
        auto_f0_adjust=args.auto_f0,
        inference_cfg_rate=args.cfg,
        output_dir=args.output_dir,
        output_format=args.format,
        keep_files=not args.no_keep_files,
    )

    run_pipeline(config)
