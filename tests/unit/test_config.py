# @trace NFR-MAINT-01 — config defaults align chunk stride with text analysis cadence

from __future__ import annotations

import os

import pytest

from config import Settings, reload_settings


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "MEDIA_CHUNK_DURATION_MS",
        "TEXT_ANALYSIS_INTERVAL_MS",
        "TEXT_TRANSCRIPT_RING_BUFFER_SLOTS",
    ):
        monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


def test_default_chunk_duration_is_6000ms():
    settings = Settings.load()
    assert settings.media_chunk_duration_ms == 6000


def test_ring_buffer_slots_derived_as_five_for_30s_at_6s_chunks():
    settings = Settings.load()
    assert settings.text_analysis_interval_ms == 30000
    assert settings.text_transcript_ring_buffer_slots == 5


def test_explicit_ring_buffer_slots_override_derivation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEXT_TRANSCRIPT_RING_BUFFER_SLOTS", "3")
    settings = Settings.load()
    assert settings.text_transcript_ring_buffer_slots == 3


def test_ring_slots_recomputed_when_intervals_change(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_CHUNK_DURATION_MS", "10000")
    monkeypatch.setenv("TEXT_ANALYSIS_INTERVAL_MS", "30000")
    settings = Settings.load()
    assert settings.text_transcript_ring_buffer_slots == 3
