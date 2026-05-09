"""Decode upload / raw audio bytes → mono float32 PCM at ``TARGET_SAMPLE_RATE``."""

from __future__ import annotations

import math
from io import BytesIO
from typing import Tuple

import numpy as np


def pcmMonoF32FromWavBytes(data: bytes, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Read WAV container bytes; convert to mono float32 in [-1, 1] at ``target_sr``.
    Requires ``soundfile``.
    """
    import soundfile as sf

    buf = BytesIO(data)
    samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    if sr != target_sr:
        samples = _resampleLinear(samples, sr, target_sr)
        sr = target_sr
    samples = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    return samples, sr


def pcmMonoF32FromNumpy(samples: np.ndarray, src_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    samples = samples.astype(np.float32, copy=False)
    if src_sr != target_sr:
        samples = _resampleLinear(samples, src_sr, target_sr)
        src_sr = target_sr
    return np.clip(samples, -1.0, 1.0), src_sr


def _resampleLinear(samples: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling (no scipy dependency). Good enough for speech paths."""
    if src_sr == target_sr or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.size / float(src_sr)
    n_out = max(1, int(math.ceil(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, num=samples.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.float32)


def pcmToBytesF32(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def bytesToPcmF32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).copy()
