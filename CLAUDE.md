# AI Cover Mac

Mac-native AI song cover pipeline - a port of AICoverGen for Apple Silicon.

## Project Structure

```
my-singer/
├── pipeline.py        # Main orchestration
├── download.py        # YouTube/local audio acquisition
├── separate.py        # 3-stage source separation (audio-separator)
├── effects.py         # Pedalboard effects chain + mixing
├── qa.py              # Quality assurance tests
├── models/            # Downloaded model checkpoints
└── song_output/       # Output stems and covers
```

## Pipeline Stages

1. **Input** - Download from YouTube or use local file
2. **Separation** - 3-pass cascade: vocals/instrumental → main/backup → dry vocal
3. **Conversion** - RVC (trained models) or Seed-VC (zero-shot)
4. **Effects** - Highpass → Compressor → Reverb
5. **Mix** - Gain-staged combination of all stems

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run with local file
python pipeline.py /path/to/song.mp3 --reference /path/to/voice.wav

# Run with YouTube URL
python pipeline.py "https://youtube.com/watch?v=..." --reference /path/to/voice.wav
```

## Development Status

- [x] M1: Separation module (audio-separator)
- [x] M2: Seed-VC engine (zero-shot voice conversion with F0)
- [x] M3: Effects & mixing
- [ ] M4: RVC engine (blocked - fairseq dependency issues, VITS model needs porting)
- [ ] M5: Polish (CLI, golden-set testing)

## Current Recommendation

Use **Seed-VC** (the default engine) for voice conversion. It provides:
- Zero-shot conversion (no training needed)
- F0 conditioning (preserves melody)
- Good quality for most use cases

RVC would provide better timbre matching with trained models, but requires
porting ~500 lines of VITS architecture code to work on Mac without fairseq.

## Key Dependencies

- `audio-separator` - UVR MDX-Net models for stem separation
- `pedalboard` - Spotify's audio effects library
- `torch` with MPS support for Apple Silicon
- `yt-dlp` for YouTube downloads

## MPS Notes

- Run in fp32 (`is_half=False`) - MPS fp16 can cause artifacts
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for ops that fall back to CPU
- Models are loaded/released stage-by-stage to fit in 16GB unified memory
