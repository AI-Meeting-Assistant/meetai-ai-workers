# @trace NFR-PERF-01 — VAD window CPU path (micro-benchmark)

from __future__ import annotations

import time

import numpy as np
import pytest

from workers.audio.vad_window import computeWindowVadMetrics

WARMUP_ITERATIONS = 5
BENCHMARK_ITERATIONS = 50
THRESHOLD_SECONDS = 2.0
SAMPLE_RATE = 16000
WINDOW_SECONDS = 6.0


@pytest.fixture
def pcm_6s() -> np.ndarray:
    samples = int(SAMPLE_RATE * WINDOW_SECONDS)
    t = np.linspace(0, WINDOW_SECONDS, samples, endpoint=False)
    return (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.mark.perf
def test_vad_window_6s_pcm_within_threshold(pcm_6s: np.ndarray) -> None:
    for _ in range(WARMUP_ITERATIONS):
        computeWindowVadMetrics(
            pcm_6s,
            sample_rate=SAMPLE_RATE,
            speech_rms_threshold=0.022,
        )

    t0 = time.perf_counter()
    for _ in range(BENCHMARK_ITERATIONS):
        metrics = computeWindowVadMetrics(
            pcm_6s,
            sample_rate=SAMPLE_RATE,
            speech_rms_threshold=0.022,
        )
        assert metrics is not None
    elapsed = time.perf_counter() - t0

    assert elapsed < THRESHOLD_SECONDS
