# @trace UC-02.2-NF — audio Redis payload shape
# @trace NFR-USA-03 — transcript may be null on degradation

from __future__ import annotations

from workers.audio.schemas import AudioChunkPayload, payloadToRedisDict


def test_payload_to_redis_dict_camel_case():
    p = AudioChunkPayload(
        meeting_id="m1",
        offset_ms=0,
        transcript=None,
        vad_speech_ratio_percent=12.5,
    )
    d = payloadToRedisDict(p)
    assert d["meetingId"] == "m1"
    assert d["offsetMs"] == 0
    assert d["transcript"] is None
    assert d["vadSpeechRatioPercent"] == 12.5
