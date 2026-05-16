"""Match Whisper words to speaker-labeled time intervals → transcript lines + talk-time breakdown."""

from __future__ import annotations

from typing import Iterable


def _overlap_ms(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def clipIntervalMs(
    t0_ms: float,
    t1_ms: float,
    win_lo_ms: float,
    win_hi_ms: float,
) -> tuple[float, float] | None:
    lo = max(t0_ms, win_lo_ms)
    hi = min(t1_ms, win_hi_ms)
    if hi <= lo + 1e-6:
        return None
    return (lo, hi)


def mergeOverlappingLabeledClips(
    clips: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Union overlapping/touching intervals per speaker (avoids double-counting talk ms)."""
    if not clips:
        return []
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for lo, hi, speaker in clips:
        if hi <= lo:
            continue
        by_speaker.setdefault(speaker, []).append((float(lo), float(hi)))
    merged: list[tuple[float, float, str]] = []
    for speaker, intervals in by_speaker.items():
        intervals.sort(key=lambda x: x[0])
        cur_lo, cur_hi = intervals[0]
        for lo, hi in intervals[1:]:
            if lo <= cur_hi + 1e-6:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append((cur_lo, cur_hi, speaker))
                cur_lo, cur_hi = lo, hi
        merged.append((cur_lo, cur_hi, speaker))
    return merged


def aggregateSpeakerTalkMs(
    labeled_clips_ms: Iterable[tuple[float, float, str]],
) -> dict[str, float]:
    clips = mergeOverlappingLabeledClips(list(labeled_clips_ms))
    out: dict[str, float] = {}
    for t0, t1, speaker in clips:
        dur = float(t1 - t0)
        if dur <= 0:
            continue
        out[speaker] = out.get(speaker, 0.0) + dur
    return out


def capSpeakerTalkMsToWindow(
    speaker_talk_ms: dict[str, float],
    window_duration_ms: float,
) -> dict[str, float]:
    """Scale talk-time down if union intervals exceed the ingest window."""
    if window_duration_ms <= 0 or not speaker_talk_ms:
        return speaker_talk_ms
    total = sum(speaker_talk_ms.values())
    if total <= window_duration_ms + 1e-6:
        return speaker_talk_ms
    scale = window_duration_ms / total
    return {k: round(v * scale, 2) for k, v in speaker_talk_ms.items()}


def windowSpeakerRatioPercent(speaker_talk_ms: dict[str, float]) -> dict[str, float] | None:
    total = sum(speaker_talk_ms.values())
    if total <= 0:
        return None
    return {k: round(100.0 * v / total, 2) for k, v in sorted(speaker_talk_ms.items())}


def dominantSpeaker(speaker_talk_ms: dict[str, float]) -> str | None:
    if not speaker_talk_ms:
        return None
    return max(speaker_talk_ms.keys(), key=lambda k: speaker_talk_ms[k])


def assignWordsToSpeakers(
    words: list[tuple[int, int, str]],
    labeled_intervals: list[tuple[float, float, str]],
    *,
    fallback_speaker: str,
    window_lo_ms: float,
    window_hi_ms: float,
) -> list[tuple[str, str]]:
    """
    Each word is ``(start_ms, end_ms, token)``.
    Keeps tokens fully inside `[window_lo_ms, window_hi_ms]` (exclusive end ok).
    Returns list of ``(speaker, text_piece)``.
    """
    spans: list[tuple[str, str]] = []
    for w0, w1, token in words:
        if not token.strip():
            continue
        if w1 <= window_lo_ms or w0 >= window_hi_ms:
            continue
        best_lab = fallback_speaker
        best_ov = 0.0
        wf0, wf1 = float(w0), float(w1)
        for i0, i1, lab in labeled_intervals:
            ov = _overlap_ms(wf0, wf1, i0, i1)
            if ov > best_ov:
                best_ov = ov
                best_lab = lab
        spans.append((best_lab, token.strip()))
    return spans


def mergeConsecutiveSpeakerText(spans: list[tuple[str, str]]) -> list[dict[str, str]]:
    if not spans:
        return []
    lines: list[dict[str, str]] = []
    cur_sp, cur_txt = spans[0]
    buf = cur_txt
    for sp, tx in spans[1:]:
        if sp == cur_sp:
            buf += " " + tx
        else:
            lines.append({"speaker": cur_sp, "text": buf})
            cur_sp, buf = sp, tx
    lines.append({"speaker": cur_sp, "text": buf})
    return lines
