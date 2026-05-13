"""Rolling audio context per meeting for ASR (PYTHON_WORKERS_IMPLEMENTATION.md §1.2–§1.3)."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np


def _samplesToMs(n: int, sample_rate: int) -> int:
    return int(n * 1000 / sample_rate)


class RollingAudioContextBuffer:
    """
    Concatenate recent PCM chunks; trim to ``context_ms`` of audio (~max_samples).

    Client must send **contiguous** ``offset_ms`` windows so the earliest retained
    sample maps to ``first_offset_ms``.
    """

    def __init__(self, *, sample_rate: int, context_ms: int) -> None:
        self.sample_rate = sample_rate
        self.context_ms = context_ms
        self.max_samples = max(sample_rate // 10, int(sample_rate * (context_ms / 1000.0)))
        self._chunks: Deque[Tuple[int, np.ndarray]] = deque()

    def appendChunk(self, chunk_offset_start_ms: int, pcm_mono: np.ndarray) -> None:
        if pcm_mono.size == 0:
            return
        self._chunks.append((chunk_offset_start_ms, pcm_mono.astype(np.float32, copy=False)))
        self._enforceMaxLength()

    @property
    def totalSamples(self) -> int:
        return sum(c[1].size for c in self._chunks)

    def _enforceMaxLength(self) -> None:
        overflow = self.totalSamples - self.max_samples
        if overflow <= 0:
            return
        while overflow > 0 and self._chunks:
            head_off, head = self._chunks[0]
            if head.size <= overflow:
                overflow -= head.size
                self._chunks.popleft()
                continue
            drop = overflow
            self._chunks[0] = (head_off + _samplesToMs(drop, self.sample_rate), head[drop:])
            return

    def buildConcatenated(self) -> Tuple[np.ndarray, int]:
        """
        Concatenated float32 mono PCM and **meeting-relative ms** at sample index 0
        (**start** time of retained audio window).
        """
        if not self._chunks:
            return np.array([], dtype=np.float32), 0
        first_ms, first_arr = self._chunks[0]
        concat = np.concatenate([a for _, a in self._chunks], dtype=np.float32)
        return concat, first_ms

    def getSnapshots(self) -> List[Tuple[int, np.ndarray]]:
        return list(self._chunks)


_MEETING_BUFFERS: dict[str, RollingAudioContextBuffer] = {}


def getBuffer(
    meeting_id: str,
    *,
    sample_rate: int,
    context_ms: int,
) -> RollingAudioContextBuffer:
    buf = _MEETING_BUFFERS.get(meeting_id)
    if buf is None or buf.sample_rate != sample_rate or buf.context_ms != context_ms:
        buf = RollingAudioContextBuffer(sample_rate=sample_rate, context_ms=context_ms)
        _MEETING_BUFFERS[meeting_id] = buf
    return buf


def clearMeetingBuffers(meeting_id: str | None = None) -> None:
    if meeting_id is None:
        _MEETING_BUFFERS.clear()
        from workers.audio.pipeline import clear_live_webm_accum_all

        clear_live_webm_accum_all()
        return
    _MEETING_BUFFERS.pop(meeting_id, None)
    from workers.audio.pipeline import clear_live_webm_accum

    clear_live_webm_accum(meeting_id)
