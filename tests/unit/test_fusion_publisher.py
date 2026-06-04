# @trace NFR-MAINT-01 — Redis channel contract
# @trace SDD-DG3 — stateless offset in payloads

from __future__ import annotations

import json

from core.fusion_publisher import (
    channel_audio,
    channel_text,
    channel_vision,
    publish_json,
)


class FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.messages.append((channel, message))
        return 1


def test_channel_names():
    mid = "abc-123"
    assert channel_audio(mid) == "meeting:abc-123:audio"
    assert channel_vision(mid) == "meeting:abc-123:vision"
    assert channel_text(mid) == "meeting:abc-123:text"


def test_publish_json_includes_offset():
    r = FakeRedis()
    publish_json(
        r,
        channel_audio("m1"),
        {"meetingId": "m1", "offsetMs": 6000, "transcript": "hi"},
    )
    assert len(r.messages) == 1
    channel, raw = r.messages[0]
    assert channel == "meeting:m1:audio"
    data = json.loads(raw)
    assert data["offsetMs"] == 6000
    assert data["meetingId"] == "m1"
