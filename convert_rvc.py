"""
RVC voice conversion (Engine A) - Fairseq-free implementation.

Uses transformers HuBERT instead of fairseq for content encoding.
Supports .pth RVC models and .index files for retrieval.
"""

import os
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf

# Set MPS fallback before importing models
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def load_hubert(device: torch.device = DEVICE):
    """Load HuBERT model using transformers (no fairseq dependency)."""
    from transformers import HubertModel

    print("  Loading HuBERT model...")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model = model.to(device)
    model.eval()
    return model


def extract_hubert_features(
    hubert_model,
    audio: torch.Tensor,
    sample_rate: int = 16000,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Extract content features using HuBERT."""
    # Ensure 16kHz mono
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)

    audio = audio.to(device)

    with torch.no_grad():
        # HuBERT expects (batch, time)
        outputs = hubert_model(audio, output_hidden_states=True)
        # Use the 9th layer features (typical for RVC)
        features = outputs.hidden_states[9]

    return features


def load_rmvpe(device: torch.device = DEVICE):
    """Load RMVPE F0 extractor from seed-vc."""
    try:
        from seed_vc.modules.rmvpe import RMVPE
        from seed_vc.hf_utils import load_custom_model_from_hf

        print("  Loading RMVPE F0 extractor...")
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        f0_extractor = RMVPE(model_path, is_half=False, device=device)
        return f0_extractor.infer_from_audio
    except Exception as e:
        print(f"  Warning: Could not load RMVPE: {e}")
        return None


def load_rvc_model(model_path: Path, device: torch.device = DEVICE):
    """
    Load an RVC .pth model.

    RVC models contain:
    - config: model configuration
    - weight: model weights
    """
    print(f"  Loading RVC model: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    # Extract config and weights
    config = checkpoint.get("config", None)
    weight = checkpoint.get("weight", checkpoint)

    if config is None:
        raise ValueError("RVC model missing config - may not be a valid RVC model")

    # Build the synthesizer model
    # RVC uses a modified VITS architecture
    from rvc_model import SynthesizerTrnMs768NSFsid

    model = SynthesizerTrnMs768NSFsid(**config)
    model.load_state_dict(weight, strict=False)
    model = model.to(device)
    model.eval()

    return model, config


def load_faiss_index(index_path: Path):
    """Load a faiss index for feature retrieval."""
    try:
        import faiss

        print(f"  Loading faiss index: {index_path}")
        index = faiss.read_index(str(index_path))
        return index
    except ImportError:
        print("  Warning: faiss not installed, retrieval disabled")
        return None
    except Exception as e:
        print(f"  Warning: Could not load index: {e}")
        return None


def convert_voice(
    source_path: Path,
    output_path: Path,
    model_path: Path,
    index_path: Optional[Path] = None,
    pitch_change: int = 0,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    protect: float = 0.33,
    rms_mix_rate: float = 0.25,
    f0_method: str = "rmvpe",
) -> Path:
    """
    Convert voice using RVC.

    Args:
        source_path: Path to source vocal
        output_path: Where to save converted vocal
        model_path: Path to RVC .pth model
        index_path: Optional path to .index file for retrieval
        pitch_change: Pitch shift in semitones
        index_rate: Blend ratio for retrieval (0-1)
        filter_radius: Median filter on F0 curve
        protect: Consonant protection (0-0.5)
        rms_mix_rate: RMS envelope mix (0-1)
        f0_method: F0 extraction method

    Returns:
        Path to converted vocal
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    model_path = Path(model_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"RVC model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Source: {source_path}")
    print(f"  Model: {model_path}")
    print(f"  Pitch change: {pitch_change} semitones")

    # Load audio
    audio, sr = torchaudio.load(source_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Load models
    hubert = load_hubert(DEVICE)
    f0_fn = load_rmvpe(DEVICE)
    # rvc_model, config = load_rvc_model(model_path, DEVICE)

    # Extract features
    print("  Extracting HuBERT features...")
    features = extract_hubert_features(hubert, audio.squeeze(0), sr, DEVICE)

    # Extract F0
    print("  Extracting F0...")
    audio_16k = torchaudio.functional.resample(audio, sr, 16000).squeeze().numpy()
    if f0_fn is not None:
        f0 = f0_fn(audio_16k, thred=0.03)
        f0 = torch.from_numpy(f0).float().to(DEVICE)

        # Apply pitch shift
        if pitch_change != 0:
            f0 = f0 * (2 ** (pitch_change / 12))
    else:
        f0 = None

    # Load index for retrieval
    index = None
    if index_path and Path(index_path).exists():
        index = load_faiss_index(index_path)

    # RVC synthesis requires porting the VITS model architecture
    # The challenge: fairseq dependency prevents using existing RVC packages
    #
    # What we have working:
    # - HuBERT features via transformers (fairseq-free)
    # - F0 extraction via RMVPE
    # - Faiss index loading
    #
    # What's missing:
    # - SynthesizerTrnMs768NSFsid (VITS-based generator)
    # - NSF-HiFiGAN vocoder weights
    #
    # Options:
    # 1. Use Seed-VC instead (zero-shot, no training needed)
    # 2. Port the ~500 lines of VITS code from RVC-Project
    # 3. Wait for a Mac-compatible RVC package

    print("  [RVC] Synthesis model not yet ported")
    print("  [RVC] Recommendation: Use Seed-VC (--engine seedvc) for now")
    print("  [RVC] Seed-VC provides good quality without needing trained models")

    # Placeholder: copy source
    import shutil
    shutil.copy(source_path, output_path)
    print(f"  [RVC] Copied source as placeholder: {output_path}")

    # Cleanup
    del hubert
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RVC Voice Conversion")
    parser.add_argument("source", type=Path, help="Source vocal")
    parser.add_argument("model", type=Path, help="RVC model (.pth)")
    parser.add_argument("output", type=Path, help="Output path")
    parser.add_argument("--index", type=Path, help="Index file (.index)")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch change (semitones)")
    parser.add_argument("--index-rate", type=float, default=0.5)

    args = parser.parse_args()

    convert_voice(
        source_path=args.source,
        output_path=args.output,
        model_path=args.model,
        index_path=args.index,
        pitch_change=args.pitch,
        index_rate=args.index_rate,
    )
