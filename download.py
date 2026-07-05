"""
Audio input acquisition - YouTube download or local file handling.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple

import yt_dlp
import soundfile as sf
import numpy as np


def parse_time(time_str: str) -> float:
    """
    Parse time string to seconds.

    Formats:
        "1:30" -> 90.0 (1 minute 30 seconds)
        "90" -> 90.0 (90 seconds)
        "0:45" -> 45.0 (45 seconds)
        "2:30.5" -> 150.5 (2 minutes 30.5 seconds)
    """
    if time_str is None:
        return None

    time_str = str(time_str).strip()

    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    else:
        return float(time_str)


def trim_audio(
    input_path: Path,
    output_path: Path,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Path:
    """
    Trim audio file to specified time range.

    Args:
        input_path: Input audio file
        output_path: Output trimmed audio file
        start_time: Start time in seconds (None = beginning)
        end_time: End time in seconds (None = end)

    Returns:
        Path to trimmed audio
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if start_time is None and end_time is None:
        return input_path

    # Load audio
    audio, sr = sf.read(input_path)

    # Calculate sample indices
    start_sample = int(start_time * sr) if start_time else 0
    end_sample = int(end_time * sr) if end_time else len(audio)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    # Trim
    trimmed = audio[start_sample:end_sample]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, trimmed, sr)

    duration = (end_sample - start_sample) / sr
    print(f"  Trimmed to {duration:.1f}s ({start_time or 0:.1f}s - {end_time or duration:.1f}s)")

    return output_path


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_file_hash(file_path: Path) -> str:
    """Generate blake2b hash of file for stable ID."""
    hasher = hashlib.blake2b(digest_size=16)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_youtube(
    url: str,
    output_dir: Path,
) -> tuple[Path, str]:
    """
    Download audio from YouTube URL.

    Returns:
        Tuple of (audio_path, song_id)
        song_id is the YouTube video ID
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    output_path = output_dir / f"{video_id}.mp3"

    # Check cache
    if output_path.exists():
        print(f"Using cached download: {output_path}")
        return output_path, video_id

    print(f"Downloading: {url}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / f"{video_id}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path, video_id


def acquire_audio(
    source: str,
    output_dir: Path,
) -> tuple[Path, str]:
    """
    Acquire audio from YouTube URL or local file.

    Returns:
        Tuple of (audio_path, song_id)
    """
    output_dir = Path(output_dir)

    # Check if it's a URL
    if source.startswith(('http://', 'https://', 'www.')):
        return download_youtube(source, output_dir)

    # Local file
    local_path = Path(source)
    if not local_path.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    song_id = get_file_hash(local_path)
    return local_path, song_id


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python download.py <youtube_url_or_file>")
        sys.exit(1)

    path, song_id = acquire_audio(sys.argv[1], Path("downloads"))
    print(f"Audio: {path}")
    print(f"Song ID: {song_id}")
