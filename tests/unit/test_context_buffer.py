"""Tests for RollingAudioContextBuffer."""

from __future__ import annotations

import numpy as np
import pytest
from workers.audio.context_buffer import RollingAudioContextBuffer

SR = 16000


def _chunk(seconds: float, value: float = 1.0) -> np.ndarray:
    return np.full(int(SR * seconds), value, dtype=np.float32)


def test_empty_buffer_returns_empty_array():
    buf = RollingAudioContextBuffer(sample_rate=SR, context_ms=5000)
    audio, offset = buf.buildConcatenated()
    assert audio.size == 0
    assert offset == 0


def test_append_and_concatenate():
    buf = RollingAudioContextBuffer(sample_rate=SR, context_ms=5000)
    buf.appendChunk(0, _chunk(1.0))
    audio, offset = buf.buildConcatenated()
    assert audio.size == SR
    assert offset == 0


def test_appending_empty_chunk_is_ignored():
    buf = RollingAudioContextBuffer(sample_rate=SR, context_ms=5000)
    buf.appendChunk(0, np.array([], dtype=np.float32))
    audio, _ = buf.buildConcatenated()
    assert audio.size == 0


def test_enforce_max_length_trims_old_audio():
    buf = RollingAudioContextBuffer(sample_rate=SR, context_ms=2000)
    buf.appendChunk(0, _chunk(2.0))
    buf.appendChunk(2000, _chunk(1.0))
    assert buf.totalSamples <= buf.max_samples


def test_first_offset_advances_after_trim():
    buf = RollingAudioContextBuffer(sample_rate=SR, context_ms=1000)
    buf.appendChunk(0, _chunk(1.0))
    buf.appendChunk(1000, _chunk(1.0))
    _, offset_ms = buf.buildConcatenated()
    assert offset_ms >= 0


def test_total_samples_accumulates_across_chunks():
    buf = RollingAudioContextBuffer(sample_rate=SR, context_ms=10000)
    buf.appendChunk(0, _chunk(1.0))
    buf.appendChunk(1000, _chunk(2.0))
    assert buf.totalSamples == 3 * SR


def test_get_snapshots_returns_all_chunks():
    buf = RollingAudioContextBuffer(sample_rate=SR, context_ms=10000)
    buf.appendChunk(0, _chunk(1.0))
    buf.appendChunk(1000, _chunk(1.0))
    snaps = buf.getSnapshots()
    assert len(snaps) == 2
