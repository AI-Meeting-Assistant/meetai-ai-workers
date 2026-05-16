"""Frame-level energy VAD → contiguous speech sample ranges (mono float32 PCM)."""

from __future__ import annotations

import numpy as np

from workers.audio.vad_window import speech_rms_threshold_for_frames

# Mirrors workers.audio.vad_window thresholds so behavior stays consistent.
_ABS_SPEECH_RMS_FLOOR = 0.01


def vadSpeechSegmentSampleRanges(
    pcm_mono_f32: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: float = 20.0,
    speech_rms_threshold: float = 0.022,
    speech_energy_quantile: float | None = None,
) -> list[tuple[int, int]]:
    """
    Returns inclusive-exclusive sample intervals ``[(start_sample, end_sample), ...]``
    classified as speech, merged contiguously.

    ``speech_energy_quantile`` is deprecated (quantile VAD pinned ~65% speech); ignored.
    """
    _ = speech_energy_quantile
    if pcm_mono_f32.size == 0 or sample_rate <= 0:
        return []
    pcm = pcm_mono_f32.astype(np.float32, copy=False)
    frame_sz = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = (pcm.size + frame_sz - 1) // frame_sz
    rms_vals: list[float] = []
    for i in range(n_frames):
        chunk = pcm[i * frame_sz : (i + 1) * frame_sz]
        rms_vals.append(float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64))))
    if not rms_vals:
        return []
    thresh = speech_rms_threshold_for_frames(rms_vals, abs_rms=speech_rms_threshold)
    is_speech = [r >= thresh and r >= _ABS_SPEECH_RMS_FLOOR for r in rms_vals]

    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(is_speech):
        if not is_speech[i]:
            i += 1
            continue
        j = i
        while j < len(is_speech) and is_speech[j]:
            j += 1
        start_s = i * frame_sz
        end_s = min(pcm.size, j * frame_sz)
        if end_s > start_s:
            ranges.append((start_s, end_s))
        i = j
    return ranges


def mergeAdjacentSpeechSegmentRanges(
    ranges: list[tuple[int, int]],
    *,
    max_gap_samples: int,
) -> list[tuple[int, int]]:
    """Join speech ranges separated by brief pauses (same utterance, one embedding)."""
    if not ranges or max_gap_samples < 0:
        return list(ranges)
    merged: list[tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_samples:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged
