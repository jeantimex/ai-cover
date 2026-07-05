"""
Quality assurance utilities for the AI cover pipeline.

- Null tests for separation quality
- F0 correlation for voice conversion
- Loudness measurement (LUFS)
"""

import numpy as np
import librosa
from pathlib import Path


def null_test(
    original: Path,
    stem1: Path,
    stem2: Path,
    threshold_db: float = -40.0,
) -> tuple[bool, float]:
    """
    Verify that two stems sum back to the original.

    Args:
        original: Original audio file
        stem1: First separated stem
        stem2: Second separated stem
        threshold_db: Maximum allowed residual RMS in dB

    Returns:
        Tuple of (passed, residual_db)
    """
    y_orig, sr = librosa.load(original, sr=44100, mono=True)
    y1, _ = librosa.load(stem1, sr=44100, mono=True)
    y2, _ = librosa.load(stem2, sr=44100, mono=True)

    min_len = min(len(y_orig), len(y1), len(y2))
    residual = y_orig[:min_len] - (y1[:min_len] + y2[:min_len])

    rms = np.sqrt(np.mean(residual ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf

    return rms_db < threshold_db, rms_db


def f0_correlation(
    source_vocal: Path,
    converted_vocal: Path,
    hop_length: int = 512,
) -> float:
    """
    Compute F0 correlation between source and converted vocal.

    A high correlation (>0.9) indicates the melody survived conversion.

    Returns:
        Pearson correlation coefficient on voiced frames
    """
    y_src, sr = librosa.load(source_vocal, sr=22050)
    y_conv, _ = librosa.load(converted_vocal, sr=22050)

    # Extract F0 using pyin
    f0_src, voiced_src, _ = librosa.pyin(
        y_src, fmin=50, fmax=800, sr=sr, hop_length=hop_length
    )
    f0_conv, voiced_conv, _ = librosa.pyin(
        y_conv, fmin=50, fmax=800, sr=sr, hop_length=hop_length
    )

    # Align lengths
    min_len = min(len(f0_src), len(f0_conv))
    f0_src = f0_src[:min_len]
    f0_conv = f0_conv[:min_len]
    voiced_src = voiced_src[:min_len]
    voiced_conv = voiced_conv[:min_len]

    # Only compare voiced frames
    voiced_both = voiced_src & voiced_conv
    if np.sum(voiced_both) < 10:
        return 0.0

    f0_src_voiced = f0_src[voiced_both]
    f0_conv_voiced = f0_conv[voiced_both]

    # Pearson correlation
    correlation = np.corrcoef(f0_src_voiced, f0_conv_voiced)[0, 1]

    return float(correlation) if not np.isnan(correlation) else 0.0


def duration_drift(
    source: Path,
    converted: Path,
) -> float:
    """
    Measure duration drift between source and converted audio.

    Returns:
        Drift in milliseconds (positive = converted is longer)
    """
    y_src, sr_src = librosa.load(source, sr=None)
    y_conv, sr_conv = librosa.load(converted, sr=None)

    dur_src = len(y_src) / sr_src
    dur_conv = len(y_conv) / sr_conv

    drift_ms = (dur_conv - dur_src) * 1000
    return drift_ms


def measure_loudness(audio_path: Path) -> dict:
    """
    Measure audio loudness metrics.

    Returns:
        Dict with integrated LUFS (approx), RMS dB, and true peak dBTP
    """
    y, sr = librosa.load(audio_path, sr=None, mono=False)

    if y.ndim == 1:
        y = np.stack([y, y])

    # RMS
    rms = np.sqrt(np.mean(y ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf

    # True peak
    true_peak = np.max(np.abs(y))
    true_peak_dbtp = 20 * np.log10(true_peak) if true_peak > 0 else -np.inf

    # Approximate LUFS (simplified - not true ITU-R BS.1770)
    lufs_approx = rms_db - 0.691

    return {
        "lufs_approx": lufs_approx,
        "rms_db": rms_db,
        "true_peak_dbtp": true_peak_dbtp,
    }


def validate_conversion(
    source_vocal: Path,
    converted_vocal: Path,
    f0_threshold: float = 0.9,
    drift_threshold_ms: float = 50.0,
) -> dict:
    """
    Run all conversion quality checks.

    Returns:
        Dict with test results and pass/fail status
    """
    f0_corr = f0_correlation(source_vocal, converted_vocal)
    drift = duration_drift(source_vocal, converted_vocal)

    return {
        "f0_correlation": f0_corr,
        "f0_passed": f0_corr > f0_threshold,
        "duration_drift_ms": drift,
        "drift_passed": abs(drift) < drift_threshold_ms,
        "all_passed": f0_corr > f0_threshold and abs(drift) < drift_threshold_ms,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python qa.py <source_vocal> <converted_vocal>")
        sys.exit(1)

    results = validate_conversion(Path(sys.argv[1]), Path(sys.argv[2]))

    print("Conversion QA Results:")
    print(f"  F0 correlation: {results['f0_correlation']:.3f} ({'PASS' if results['f0_passed'] else 'FAIL'})")
    print(f"  Duration drift: {results['duration_drift_ms']:.1f} ms ({'PASS' if results['drift_passed'] else 'FAIL'})")
    print(f"  Overall: {'PASS' if results['all_passed'] else 'FAIL'}")
