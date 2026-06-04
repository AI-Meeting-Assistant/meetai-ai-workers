# @trace UC-02.4-NF — transcript buffer flush triggers analysis
# @trace UC-02.4-ALT-4.1 — skip when meeting meta missing

from __future__ import annotations

import pytest

from workers.text import pipeline as text_pipeline


@pytest.fixture(autouse=True)
def clear_text_state():
    text_pipeline._meta.clear()
    text_pipeline._buffers.clear()
    yield
    text_pipeline._meta.clear()
    text_pipeline._buffers.clear()


@pytest.mark.asyncio
async def test_buffer_does_not_flush_before_threshold(monkeypatch):
    monkeypatch.setattr(
        "workers.text.pipeline.get_settings",
        lambda: type("S", (), {"text_transcript_ring_buffer_slots": 3})(),
    )
    text_pipeline.updateMeta("m1", "Title", "Agenda")
    await text_pipeline.handleTranscript("m1", 0, "one")
    await text_pipeline.handleTranscript("m1", 6000, "two")
    assert "m1" in text_pipeline._buffers
    assert len(text_pipeline._buffers["m1"]) == 2


@pytest.mark.asyncio
async def test_skips_analysis_without_meta(monkeypatch):
    monkeypatch.setattr(
        "workers.text.pipeline.get_settings",
        lambda: type("S", (), {"text_transcript_ring_buffer_slots": 1})(),
    )
    called = False

    async def fake_analyze(**_kwargs):
        nonlocal called
        called = True
        return {"context_fit": 0.9, "on_topic": True, "reason": None}

    monkeypatch.setattr("workers.text.pipeline.analyzeAdherence", fake_analyze)
    await text_pipeline.handleTranscript("m1", 0, "only chunk")
    assert called is False


@pytest.mark.asyncio
async def test_flushes_and_publishes_on_threshold(monkeypatch):
    monkeypatch.setattr(
        "workers.text.pipeline.get_settings",
        lambda: type(
            "S",
            (),
            {
                "text_transcript_ring_buffer_slots": 2,
                "ollama_url": "http://localhost:11434",
                "ollama_model": "test",
            },
        )(),
    )
    text_pipeline.updateMeta("m1", "Standup", "Goals")

    async def fake_analyze(**_kwargs):
        return {"context_fit": 0.65, "on_topic": True, "reason": None}

    published: list[dict] = []

    class FakeRedis:
        raw = object()

    monkeypatch.setattr("workers.text.pipeline.analyzeAdherence", fake_analyze)
    monkeypatch.setattr(
        "workers.text.pipeline.publish_json",
        lambda _r, _ch, payload: published.append(payload),
    )
    monkeypatch.setattr(
        "workers.text.pipeline.get_redis_client",
        lambda: FakeRedis(),
    )

    await text_pipeline.handleTranscript("m1", 0, "a")
    await text_pipeline.handleTranscript("m1", 6000, "b")

    assert "m1" not in text_pipeline._buffers
    assert len(published) == 1
    assert published[0]["contextFit"] == 0.65
    assert published[0]["chunksAnalysed"] == 2
