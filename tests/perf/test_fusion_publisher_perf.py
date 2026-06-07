# @trace NFR-PERF-01 — Redis publish serialization (micro-benchmark)
# @trace NFR-MAINT-01 — fusion publisher hot path

from __future__ import annotations

import json
import time

import pytest

from core.fusion_publisher import channel_audio, publish_json

WARMUP_ITERATIONS = 100
BENCHMARK_ITERATIONS = 5000
THRESHOLD_SECONDS = 1.0


class FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.messages.append((channel, message))
        return 1


def realistic_audio_payload(meeting_id: str, offset_ms: int) -> dict:
    return {
        "meetingId": meeting_id,
        "offsetMs": offset_ms,
        "transcript": "Speaker 1: Toplantı gündem maddesine geçelim.",
        "transcriptLines": [
            {"speaker": "Speaker 1", "text": "Toplantı gündem maddesine geçelim."},
        ],
        "speakerTalkMs": {"Speaker 1": 4200},
        "speakerTalkRatioPercent": {"Speaker 1": 78.5},
        "vadSpeechMs": 4200.0,
        "vadSilenceMs": 1800.0,
        "vadSpeechRatioPercent": 70.0,
        "speakerLabelsWindow": None,
    }


@pytest.mark.perf
def test_publish_json_audio_payload_within_threshold() -> None:
    redis = FakeRedis()
    meeting_id = "meet-perf-publish"
    channel = channel_audio(meeting_id)

    for i in range(WARMUP_ITERATIONS):
        publish_json(redis, channel, realistic_audio_payload(meeting_id, i * 6000))

    redis.messages.clear()

    t0 = time.perf_counter()
    for i in range(BENCHMARK_ITERATIONS):
        publish_json(redis, channel, realistic_audio_payload(meeting_id, i * 6000))
    elapsed = time.perf_counter() - t0

    assert len(redis.messages) == BENCHMARK_ITERATIONS
    _, raw = redis.messages[-1]
    data = json.loads(raw)
    assert data["meetingId"] == meeting_id
    assert elapsed < THRESHOLD_SECONDS
