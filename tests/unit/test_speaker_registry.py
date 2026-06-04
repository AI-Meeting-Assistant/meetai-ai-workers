# @trace UC-02.3-ALT-6.1 — placeholder speaker labels without face match
# @trace UC-02.2-NF — live ECAPA speaker identity

from __future__ import annotations

import numpy as np

from workers.audio.speaker_registry import MeetingSpeakerRegistry


def _unit_vec(seed: float) -> np.ndarray:
    v = np.array([seed, 1.0 - seed, 0.5], dtype=np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_assigns_same_speaker_for_similar_embeddings():
    reg = MeetingSpeakerRegistry(max_speakers=4, cos_threshold=0.5, ema_alpha=0.3)
    a = reg.assign_and_update(_unit_vec(0.9))
    b = reg.assign_and_update(_unit_vec(0.88))
    assert a == "Speaker 1"
    assert b == "Speaker 1"


def test_opens_second_speaker_when_embedding_differs():
    reg = MeetingSpeakerRegistry(max_speakers=4, cos_threshold=0.85, ema_alpha=0.3)
    reg.assign_and_update(_unit_vec(0.1))
    label = reg.assign_and_update(_unit_vec(0.95), allow_new_speaker=True)
    assert label == "Speaker 2"
