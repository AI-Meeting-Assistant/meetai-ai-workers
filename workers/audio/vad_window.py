"""Lightweight **window-local** VAD metrics from PCM (no pyannote per-chunk by default)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Absolute minimum RMS to be considered speech.
_ABS_SPEECH_RMS_FLOOR = 0.01


@dataclass(frozen=True)
class WindowVadMetrics:
    speech_ms: float
    silence_ms: float
    speech_ratio_percent: float

    def toApi(self) -> tuple[float, float, float]:
        return self.speech_ms, self.silence_ms, self.speech_ratio_percent


def speech_rms_threshold_for_frames(
    rms_vals: list[float] | np.ndarray,
    *,
    abs_rms: float,
) -> float:
    """
    Fixed absolute RMS gate (not quantile — quantile pinned ~65% speech on tab audio).
    """
    _ = rms_vals
    return max(abs_rms, _ABS_SPEECH_RMS_FLOOR)


def computeWindowVadMetrics(
    pcm_mono_f32: np.ndarray,
    *,
    sample_rate: int,
    speech_rms_threshold: float,
    frame_ms: float = 20.0,
    merge_gap_ms: float = 350.0,
) -> WindowVadMetrics | None:
    """
    Speech / silence for one ingest window from merged VAD segment lengths
    (same rules as speaker diarization segments).
    """
    if pcm_mono_f32.size == 0 or sample_rate <= 0:
        return None
    from workers.audio.vad_segments import (
        mergeAdjacentSpeechSegmentRanges,
        vadSpeechSegmentSampleRanges,
    )

    window_ms = float(pcm_mono_f32.shape[0]) * 1000.0 / float(sample_rate)
    gap_samples = max(1, int(sample_rate * merge_gap_ms / 1000.0))
    segs = vadSpeechSegmentSampleRanges(
        pcm_mono_f32,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        speech_rms_threshold=speech_rms_threshold,
    )
    segs = mergeAdjacentSpeechSegmentRanges(segs, max_gap_samples=gap_samples)
    speech_ms = 0.0
    for sa, sb in segs:
        speech_ms += (float(sb - sa) / float(sample_rate)) * 1000.0
    speech_ms = min(speech_ms, window_ms)
    silence_ms = max(0.0, window_ms - speech_ms)
    ratio = (speech_ms / window_ms * 100.0) if window_ms > 0 else 0.0
    return WindowVadMetrics(
        speech_ms=speech_ms,
        silence_ms=silence_ms,
        speech_ratio_percent=ratio,
    )


def computeEnergyVadMetrics(
    pcm_mono_f32: np.ndarray,
    *,
    sample_rate: int,
    speech_energy_quantile: float = 0.35,
    speech_rms_threshold: float | None = None,
    frame_ms: float = 20.0,
) -> WindowVadMetrics | None:
    """Backward-compatible entry; prefers ``speech_rms_threshold`` when set."""
    _ = speech_energy_quantile
    if speech_rms_threshold is None:
        speech_rms_threshold = 0.022
    return computeWindowVadMetrics(
        pcm_mono_f32,
        sample_rate=sample_rate,
        speech_rms_threshold=speech_rms_threshold,
        frame_ms=frame_ms,
    )
