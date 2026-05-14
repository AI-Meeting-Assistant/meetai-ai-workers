"""Decode upload / raw audio bytes → mono float32 PCM at ``TARGET_SAMPLE_RATE``.

Supports WAV, WebM/Opus, OGG, MP4, and any container FFmpeg can demux.
``soundfile`` (libsndfile) is tried first for speed; for containers it cannot
handle (e.g. WebM from the browser MediaRecorder), we fall back to ``av``
(PyAV — pip-installable FFmpeg bindings, no system binary required).
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Optional, Tuple

import numpy as np

from utils.webm_init_cache import prepareWebmChunk


# ---------------------------------------------------------------------------
# Public decoder — accepts WAV *and* WebM/Opus/OGG/MP4/etc.
# ---------------------------------------------------------------------------

def pcmMonoF32FromWebmBytes(
    data: bytes,
    target_sr: int,
    meeting_id: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Decode WebM audio bytes to mono float32 PCM at ``target_sr``.

    Pass ``meeting_id`` when decoding browser MediaRecorder chunks so that
    WebM continuation fragments (chunks 2, 3, … which lack the EBML header)
    are automatically patched with the cached init segment.
    """
    if meeting_id is not None:
        data = prepareWebmChunk(meeting_id, data)
    return _decodeViaAv(data, target_sr)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _decodeViaAv(data: bytes, target_sr: int) -> Tuple[np.ndarray, int]:
    """Decode any FFmpeg-supported container via PyAV (``pip install av``)."""
    import av  # PyAV — bundles FFmpeg libs, no system binary needed

    chunks: list[np.ndarray] = []
    with av.open(BytesIO(data)) as container:
        resampler = av.AudioResampler(
            format="fltp",       # float32 planar
            layout="mono",
            rate=target_sr,
        )
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                # frame.to_ndarray() returns shape (channels, samples) for planar
                arr = resampled.to_ndarray()  # shape: (1, N) for mono fltp
                chunks.append(arr[0].astype(np.float32))

        # Flush resampler
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            chunks.append(arr[0].astype(np.float32))

    if not chunks:
        return np.zeros(0, dtype=np.float32), target_sr

    samples = np.concatenate(chunks)
    samples = np.clip(samples, -1.0, 1.0)
    return samples, target_sr


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
