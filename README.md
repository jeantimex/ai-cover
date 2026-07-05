# AI Cover Mac

[中文文档](README_CN.md)

A Mac-native AI song cover pipeline that converts vocals to sound like a different singer. Built for Apple Silicon with MPS acceleration.

## Features

- **YouTube Support** — Download songs and reference voices directly from YouTube
- **3-Stage Vocal Separation** — Isolates clean vocals from any song using UVR MDX-Net models
- **Zero-Shot Voice Conversion** — Convert vocals using just a 10-30s reference clip (no training needed)
- **F0 Conditioning** — Preserves the original melody and pitch
- **Professional Effects** — Automatic reverb, compression, and mixing

## How It Works

```
Input Song → Separate Vocals → Convert Voice → Apply Effects → Mix → Output
                  ↓
         Instrumental (kept)
         Backup Vocals (kept)
         Main Vocal → Seed-VC → Converted Vocal
```

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3) or Intel
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/my-singer.git
cd my-singer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

First run will download required models (~2GB total).

## Quick Start

### Basic Usage

```bash
# Convert a YouTube song using a YouTube reference voice
python pipeline.py "https://youtube.com/watch?v=SONG_ID" \
    --reference "https://youtube.com/watch?v=VOICE_ID"
```

### Using Local Files

```bash
# Local song with local reference
python pipeline.py /path/to/song.mp3 --reference /path/to/voice.wav

# YouTube song with local reference
python pipeline.py "https://youtube.com/watch?v=..." --reference /path/to/voice.wav
```

### Output

Results are saved to `song_output/<song_id>/`:
- `cover.wav` — Final mixed cover
- `converted_vocal.wav` — Converted vocal only
- `instrumental.wav` — Isolated instrumental
- `dry_main_vocal.wav` — Original isolated vocal

## Pitch Adjustment Guide

When converting between voices with different ranges, use the `--pitch` parameter to shift semitones.

### Cross-Gender Conversion

| Conversion | Command |
|------------|---------|
| Female → Male | `--pitch -12` (down one octave) |
| Male → Female | `--pitch 12` (up one octave) |

**Example: Female song with male voice**
```bash
python pipeline.py "https://youtube.com/watch?v=FEMALE_SONG" \
    --reference "https://youtube.com/watch?v=MALE_VOICE" \
    --pitch -12
```

**Example: Male song with female voice**
```bash
python pipeline.py "https://youtube.com/watch?v=MALE_SONG" \
    --reference "https://youtube.com/watch?v=FEMALE_VOICE" \
    --pitch 12
```

### Same-Gender Conversion

For voices with different ranges within the same gender:

| Situation | Command |
|-----------|---------|
| Source slightly higher than reference | `--pitch -2` to `--pitch -3` |
| Source much higher than reference | `--pitch -4` to `--pitch -5` |
| Source slightly lower than reference | `--pitch 2` to `--pitch 3` |
| Source much lower than reference | `--pitch 4` to `--pitch 5` |

**Example: High soprano to low alto**
```bash
python pipeline.py "soprano_song.mp3" \
    --reference "alto_voice.wav" \
    --pitch -4
```

### Female Voice Ranges

| Voice Type | Range | Typical Adjustment |
|------------|-------|-------------------|
| Soprano | High (C4-C6) | Reference point |
| Mezzo-soprano | Medium (A3-A5) | ±2-3 semitones from soprano |
| Alto/Contralto | Low (F3-F5) | ±4-5 semitones from soprano |

### Auto F0 vs Manual Control

| Setting | Use When |
|---------|----------|
| `--auto-f0` (default) | Cross-gender conversion or very different voice ranges |
| `--no-auto-f0` | Same-gender, similar range, or you want exact original melody |

**Note:** `--auto-f0` shifts the source pitch to match the reference voice's range. This is helpful for cross-gender conversion but may alter the melody. If the tune sounds different but quality is good, try `--no-auto-f0`.

### Tips

- **Melody sounds different?** — Try `--no-auto-f0` to preserve exact original tune
- **Add manual `--pitch` if needed** — When auto-adjust isn't enough
- **Adjust in small steps** — Try ±2 semitones at a time
- **Listen for artifacts** — Too much shift can cause distortion

## Parameter Reference

### Required

| Parameter | Description |
|-----------|-------------|
| `source` | YouTube URL or path to song |
| `--reference` | YouTube URL or path to reference voice (10-30s) |

### Voice Conversion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pitch` | 0 | Pitch shift in semitones (-12 to +12) |
| `--steps` | 50 | Diffusion steps (30-100, higher = better quality) |
| `--cfg` | 0.7 | Classifier-free guidance rate (0.5-0.9) |
| `--auto-f0` | enabled | Auto-adjust pitch to match reference range |
| `--no-auto-f0` | — | Disable auto pitch adjustment |

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | `song_output` | Output directory |
| `--format` | `wav` | Output format (`wav` or `mp3`) |
| `--no-keep-files` | — | Delete intermediate files |

### Advanced

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--engine` | `seedvc` | Conversion engine (`seedvc` or `rvc`) |
| `--no-separate-ref` | — | Don't separate vocals from reference |

## Examples

### High-Quality Conversion

For best results, increase diffusion steps:

```bash
python pipeline.py "song.mp3" \
    --reference "voice.wav" \
    --steps 100
```

### Stronger Voice Similarity

Increase CFG rate to match the reference voice more closely:

```bash
python pipeline.py "song.mp3" \
    --reference "voice.wav" \
    --cfg 0.85
```

### Female Pop Song → Male Rock Voice

```bash
python pipeline.py "https://youtube.com/watch?v=FEMALE_POP" \
    --reference "https://youtube.com/watch?v=MALE_ROCK" \
    --pitch -12 \
    --steps 75
```

### Male R&B Song → Female Voice

```bash
python pipeline.py "https://youtube.com/watch?v=MALE_RNB" \
    --reference "https://youtube.com/watch?v=FEMALE_VOICE" \
    --pitch 12 \
    --cfg 0.8
```

### Same Singer, Different Song

When the reference and source are similar voices, minimal adjustment needed:

```bash
python pipeline.py "new_song.mp3" \
    --reference "same_artist_clip.wav"
```

## Reference Audio Tips

The quality of the reference audio significantly affects results:

1. **Length**: 15-25 seconds of singing works best
2. **Quality**: Clean, isolated vocals (no background music)
3. **Style**: Similar tempo/style to source song helps
4. **Range**: Include both high and low notes if possible

**Good reference sources:**
- A cappella performances
- Isolated vocal stems
- Clear live performances
- YouTube videos with clean vocals (auto-separated)

## Troubleshooting

### Voice sounds robotic/glitchy
- Increase `--steps` to 75 or 100
- Try a cleaner reference audio

### Pitch sounds wrong
- Add `--pitch` adjustment (see guide above)
- Try `--no-auto-f0` and manually set pitch

### Voice doesn't match reference
- Increase `--cfg` to 0.8 or 0.85
- Use a longer/cleaner reference clip
- Ensure reference has similar vocal style

### Out of memory
- Close other applications
- Process shorter songs
- Reduce `--steps`

### Model download fails
- Check internet connection
- Set `HF_TOKEN` environment variable for faster downloads
- Retry — downloads are cached

## Voice Training

Train custom voice models from multiple audio sources for better similarity.

### Train a Voice

```bash
# From YouTube videos (3-5 songs recommended)
python train.py --name "taylor" \
    "https://youtube.com/watch?v=SONG1" \
    "https://youtube.com/watch?v=SONG2" \
    "https://youtube.com/watch?v=SONG3"

# From local files
python train.py --name "my_voice" song1.mp3 song2.wav vocals/*.mp3

# Mixed sources
python train.py --name "artist" \
    "https://youtube.com/watch?v=..." \
    local_song.mp3
```

### Use a Trained Voice

```bash
# Use by name instead of --reference
python pipeline.py "https://youtube.com/watch?v=SONG" --voice taylor

# List available voices
python train.py --list
```

### Training Tips

- **3-5 songs** from the same singer works best (~15-20 min total)
- Include variety: different tempos, pitches, styles
- Only use songs from the **same singer** — don't mix voices
- Cleaner source audio = better results

## Project Structure

```
my-singer/
├── pipeline.py        # Main entry point
├── train.py           # Voice training tool
├── download.py        # YouTube/local audio acquisition
├── separate.py        # 3-stage vocal separation
├── convert_seedvc.py  # Seed-VC voice conversion
├── convert_rvc.py     # RVC voice conversion (WIP)
├── effects.py         # Audio effects and mixing
├── qa.py              # Quality assurance utilities
├── models/            # Downloaded model checkpoints
├── voices/            # Trained voice models
└── song_output/       # Output directory
```

## License

This project is for personal and educational use. Please respect copyright when using copyrighted songs.

## Acknowledgments

- [Seed-VC](https://github.com/BytedanceSpeech/seed-vc) — Zero-shot voice conversion
- [audio-separator](https://github.com/karaokenerds/python-audio-separator) — Vocal separation
- [UVR MDX-Net](https://github.com/kuielab/mdx-net) — Separation models
