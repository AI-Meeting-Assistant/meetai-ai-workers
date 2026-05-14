"""Live vision ingest → Redis Pub/Sub (PYTHON_WORKERS_IMPLEMENTATION.md §5)."""

from __future__ import annotations

from typing import List

from utils.logger import get_logger

from core.fusion_publisher import channel_vision, publish_json
from infrastructure.redis_client import get_redis_client
from utils.webm_init_cache import prepareWebmChunk
from workers.video.aggregator import PersonAggregate, aggregateChunk
from workers.video.face_mesh import FaceFeatures, analyzeFaces
from workers.video.io_video import framesFromVideoBytes
from workers.video.schemas import PersonVisionResult, VisionChunkPayload, visionPayloadToRedisDict

log = get_logger(__name__)


def processLiveVisionChunk(
    meeting_id: str,
    offset_ms: int,
    video_bytes: bytes,
) -> VisionChunkPayload:
    """
    Process one ingest window of video bytes.

    Decodes frames, runs MediaPipe Face Mesh on each, aggregates per-person
    focus/speaking scores, then publishes to ``meeting:{id}:vision``.
    Returns the payload (useful for testing without Redis).
    """
    log.debug(
        "Processing video chunk",
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        size_bytes=len(video_bytes),
    )

    payload = _buildPayload(meeting_id, offset_ms, video_bytes)

    try:
        r = get_redis_client()
        publish_json(r.raw, channel_vision(meeting_id), visionPayloadToRedisDict(payload))
    except Exception:
        log.error(
            "Failed to publish vision payload to Redis",
            meeting_id=meeting_id,
            offset_ms=offset_ms,
            exc_info=True,
        )

    return payload


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _buildPayload(
    meeting_id: str,
    offset_ms: int,
    video_bytes: bytes,
) -> VisionChunkPayload:
    """Decode video, run analysis, aggregate — returns VisionChunkPayload."""
    if not video_bytes:
        return _stubPayload(meeting_id, offset_ms, reason="empty_bytes")

    frames = _decodeFrames(prepareWebmChunk(meeting_id, video_bytes))
    if not frames:
        log.warning(
            "No video frames decoded",
            meeting_id=meeting_id,
            offset_ms=offset_ms,
        )
        return _stubPayload(meeting_id, offset_ms, reason="no_frames")

    h, w = frames[0].shape[:2]
    log.debug(
        "Frames decoded",
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        n_frames=len(frames),
        frame_size=f"{w}x{h}",
    )

    frame_results = _analyzeFrames(frames)
    faces_per_frame = [len(f) for f in frame_results]
    log.debug(
        "Face detection done",
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        frames_with_faces=sum(1 for f in faces_per_frame if f > 0),
        max_faces_in_frame=max(faces_per_frame) if faces_per_frame else 0,
    )

    aggregates = aggregateChunk(frame_results)

    if not aggregates:
        return _stubPayload(meeting_id, offset_ms, reason="no_faces")

    persons = [_toPersonResult(a) for a in aggregates]
    mean_focus = float(sum(p.focus_score or 0.0 for p in persons) / len(persons))

    log.info(
        "Vision analysis complete",
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        n_frames=len(frames),
        n_persons=len(persons),
        mean_focus=round(mean_focus, 3),
    )

    return VisionChunkPayload(
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        focus_score=mean_focus,
        persons=persons,
    )


def _decodeFrames(video_bytes: bytes):
    try:
        return framesFromVideoBytes(video_bytes)
    except Exception:
        log.error("Frame decode error", exc_info=True)
        return []


_FRAME_STRIDE = 3       # process every Nth frame — 90 → 30 frames per 6s chunk
_MAX_FRAME_WIDTH = 640  # resize before MediaPipe — 1620→640 is ~6x faster per frame


def _analyzeFrames(frames) -> List[List[FaceFeatures]]:
    import cv2
    results: List[List[FaceFeatures]] = []
    for i, frame in enumerate(frames):
        if i % _FRAME_STRIDE != 0:
            continue
        try:
            # Downscale: MediaPipe accuracy is fine at 640px width
            h, w = frame.shape[:2]
            if w > _MAX_FRAME_WIDTH:
                scale = _MAX_FRAME_WIDTH / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            results.append(analyzeFaces(frame))
        except Exception:
            results.append([])
    return results


def _toPersonResult(a: PersonAggregate) -> PersonVisionResult:
    return PersonVisionResult(
        person_id=a.person_id,
        focus_score=round(a.focus_score, 4),
        speaking_ratio=round(a.speaking_ratio, 4),
        frame_count=a.frame_count,
    )


def _stubPayload(
    meeting_id: str,
    offset_ms: int,
    reason: str | None = None,
) -> VisionChunkPayload:
    extra = {"degradedReason": reason} if reason else None
    return VisionChunkPayload(
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        focus_score=None,
        persons=[],
        payload=extra,
    )
