"""Live vision ingest → Redis Pub/Sub (stub; PYTHON_WORKERS_IMPLEMENTATION.md §5)."""

from __future__ import annotations

from core.fusion_publisher import channel_vision, publish_json
from infrastructure.redis_client import get_redis_client
from workers.video.schemas import VisionChunkPayload, visionPayloadToRedisDict


def processLiveVisionChunk(
    meeting_id: str,
    offset_ms: int,
    video_bytes: bytes,
) -> None:
    """
    Process one window of video bytes. Stub ignores ``video_bytes`` and publishes minimal vision JSON.
    """
    _ = video_bytes  # opaque until real vision pipeline
    payload = VisionChunkPayload(meeting_id=meeting_id, offset_ms=offset_ms, focus_score=None)
    r = get_redis_client()
    publish_json(r, channel_vision(meeting_id), visionPayloadToRedisDict(payload))
