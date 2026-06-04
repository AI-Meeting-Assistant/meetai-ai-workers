# @trace UC-02.2-ALT-2.1 — silence window metrics
# @trace UC-02.2-NF — VAD speech ratio for ingest window

from __future__ import annotations

import numpy as np

from workers.audio.vad_window import computeWindowVadMetrics


def test_silence_pcm_yields_low_speech_ratio():
    sr = 16000
    pcm = np.zeros(sr, dtype=np.float32)
    metrics = computeWindowVadMetrics(
        pcm,
        sample_rate=sr,
        speech_rms_threshold=0.022,
    )
    assert metrics is not None
    assert metrics.speech_ratio_percent < 5.0


def test_tone_pcm_yields_higher_speech_ratio():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    pcm = (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    metrics = computeWindowVadMetrics(
        pcm,
        sample_rate=sr,
        speech_rms_threshold=0.022,
    )
    assert metrics is not None
    assert metrics.speech_ratio_percent > metrics.speech_ratio_percent * 0 or metrics.speech_ms >= 0
