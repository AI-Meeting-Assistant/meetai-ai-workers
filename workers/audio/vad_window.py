"""Lightweight **window-local** VAD metrics from PCM (no pyannote per-chunk by default)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Absolute minimum RMS to be considered speech.
# Frames below this are always silence regardless of the relative quantile threshold,
# preventing background noise / silence from being misclassified as speech.
_ABS_SPEECH_RMS_FLOOR = 0.01


@dataclass(frozen=True)
class WindowVadMetrics:
    speech_ms: float
    silence_ms: float
    speech_ratio_percent: float

    def toApi(self) -> tuple[float, float, float]:
        return self.speech_ms, self.silence_ms, self.speech_ratio_percent


def computeEnergyVadMetrics(
    pcm_mono_f32: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: float = 20.0,
    speech_energy_quantile: float = 0.35,
) -> WindowVadMetrics | None:
    """RMS per frame; frames above both the quantile threshold and the absolute
    RMS floor are counted as speech."""
    if pcm_mono_f32.size == 0 or sample_rate <= 0:
        return None
    frame_sz = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = (pcm_mono_f32.size + frame_sz - 1) // frame_sz
    rms_vals: list[float] = []
    for i in range(n_frames):
        chunk = pcm_mono_f32[i * frame_sz : (i + 1) * frame_sz]
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        rms_vals.append(rms)
    if not rms_vals:
        return None
    thresh = float(np.quantile(np.array(rms_vals), speech_energy_quantile))
    speech_frames = sum(1 for r in rms_vals if r >= thresh and r >= _ABS_SPEECH_RMS_FLOOR)
    silence_frames = len(rms_vals) - speech_frames
    frame_dur_ms = frame_ms
    speech_ms = speech_frames * frame_dur_ms
    silence_ms = silence_frames * frame_dur_ms
    total_ms = speech_ms + silence_ms
    ratio = (speech_ms / total_ms * 100.0) if total_ms > 0 else 0.0
    return WindowVadMetrics(speech_ms=speech_ms, silence_ms=silence_ms, speech_ratio_percent=ratio)
