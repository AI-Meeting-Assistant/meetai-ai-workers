"""Lazy SpeechBrain ECAPA speaker embedding (mono float32 PCM)."""

from __future__ import annotations

import math
import numpy as np

TARGET_SR = 16000
_encoder = None


def _resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.size / float(src_sr)
    n_out = max(1, int(math.ceil(duration * dst_sr)))
    x_old = np.linspace(0.0, 1.0, num=samples.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.float32)


def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    _encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return _encoder


def warmUpSpeakerEmbedder() -> None:
    """Load ECAPA weights once (optional worker startup)."""
    _get_encoder()


def _encoder_torch_device(enc) -> "torch.device":
    import torch

    dev = getattr(enc, "device", None)
    if dev is not None:
        return dev if isinstance(dev, torch.device) else torch.device(dev)
    for p in enc.parameters():
        return p.device
    return torch.device("cpu")


def embedPcmMono(
    pcm_mono_f32: np.ndarray,
    *,
    sample_rate: int,
) -> np.ndarray | None:
    """
    Returns L2-normalized embedding as float32 1D array, or None on failure.
    """
    if pcm_mono_f32.size == 0:
        return None
    import torch

    wav = _resample_linear(pcm_mono_f32.astype(np.float32, copy=False), sample_rate, TARGET_SR)
    if wav.size < int(0.05 * TARGET_SR):
        return None
    enc = _get_encoder()
    dev = _encoder_torch_device(enc)
    t = torch.from_numpy(wav).float().unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = enc.encode_batch(t)
    v = emb.detach().squeeze().flatten().cpu().numpy().astype(np.float64, copy=False)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return (v / n).astype(np.float32, copy=False)
