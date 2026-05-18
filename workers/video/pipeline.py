"""Live vision ingest → Redis Pub/Sub (PYTHON_WORKERS_IMPLEMENTATION.md §5)."""

from __future__ import annotations

from typing import List

import numpy as np

from utils.logger import get_logger

from core.fusion_publisher import channel_vision, publish_json
from infrastructure.redis_client import get_redis_client
from workers.video.aggregator import PersonAggregate, aggregateChunk
from workers.video.face_mesh import FaceFeatures, analyzeFaces
from workers.video.schemas import PersonVisionResult, VisionChunkPayload, visionPayloadToRedisDict
from workers.video.tracker import assignStableIds

log = get_logger(__name__)


def processLiveVisionChunk(
    meeting_id: str,
    offset_ms: int,
    jpeg_frames: list[bytes],
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
        frame_count=len(jpeg_frames),
    )

    payload = _buildPayload(meeting_id, offset_ms, jpeg_frames)

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
    jpeg_frames: list[bytes],
) -> VisionChunkPayload:
    """Decode JPEG frames, run analysis, aggregate — returns VisionChunkPayload."""
    if not jpeg_frames:
        return _stubPayload(meeting_id, offset_ms, reason="empty_frames")

    frames = _decodeJpegFrames(jpeg_frames)
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

    frame_results = _analyzeFrames(frames, meeting_id)
    frame_results = _applyTracking(frame_results, meeting_id)
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

    # Frontend pre-filters frames to face-only crops, so a real face should
    # appear in at least half the chunk frames. Chunks below 50% indicate the
    # face was genuinely absent most of the window.
    total_frames = len(frame_results)
    frames_with_faces = sum(1 for f in frame_results if f)
    if total_frames > 0 and frames_with_faces / total_frames < 0.25:
        log.debug(
            "Face coverage too low — skipping chunk",
            meeting_id=meeting_id,
            offset_ms=offset_ms,
            frames_with_faces=frames_with_faces,
            total_frames=total_frames,
            coverage=round(frames_with_faces / total_frames, 3),
        )
        return _stubPayload(meeting_id, offset_ms, reason="low_face_coverage")

    persons = [_toPersonResult(a) for a in aggregates]
    total_frames = sum(p.frame_count or 1 for p in persons)
    mean_focus = float(sum((p.focus_score or 0.0) * (p.frame_count or 1) for p in persons) / total_frames)

    all_faces = [f for frame in frame_results for f in frame]
    if all_faces:
        mean_yaw   = round(sum(f.yaw   for f in all_faces) / len(all_faces), 1)
        mean_pitch = round(sum(f.pitch for f in all_faces) / len(all_faces), 1)
        mean_gaze  = round(sum((f.gaze_x**2 + f.gaze_y**2)**0.5 for f in all_faces) / len(all_faces), 3)
    else:
        mean_yaw = mean_pitch = mean_gaze = 0.0

    log.info(
        "Vision analysis complete",
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        n_frames=len(frames),
        n_persons=len(persons),
        mean_focus=round(mean_focus, 3),
        mean_yaw=mean_yaw,
        mean_pitch=mean_pitch,
        mean_gaze=mean_gaze,
    )

    return VisionChunkPayload(
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        focus_score=mean_focus,
        persons=persons,
    )


def _decodeJpegFrames(jpeg_frames: list[bytes]):
    import cv2 as _cv2
    import numpy as _np
    frames = []
    for data in jpeg_frames:
        arr = _np.frombuffer(data, _np.uint8)
        frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
        else:
            log.warning("JPEG frame decode failed, skipping")
    return frames


_FRAME_STRIDE = 1        # process every frame
_MAX_FRAME_WIDTH = 1106  # downscale cap — matches the frontend's 1280px target downscaled
# Reject faces smaller than 3% of frame area.
# Full-screen JPEG frames at ~1106×720 make faces small relative to the frame,
# so 3% (not 15%) is the right threshold. 15% was for the old 400×400 face-crop pipeline.
_MIN_FACE_AREA_RATIO = 0.03


def _analyzeFrames(frames, meeting_id: str) -> List[List[FaceFeatures]]:
    import cv2
    results: List[List[FaceFeatures]] = []
    for i, frame in enumerate(frames):
        if i % _FRAME_STRIDE != 0:
            continue
        try:
            h, w = frame.shape[:2]
            if w > _MAX_FRAME_WIDTH:
                scale = _MAX_FRAME_WIDTH / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            faces = analyzeFaces(frame)
            frame_area = h * w
            faces = [
                f for f in faces
                if (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) >= _MIN_FACE_AREA_RATIO * frame_area
            ]
            # Sort by left x-edge so the leftmost face is always slot 0 within this frame.
            faces.sort(key=lambda f: f.bbox[0])
            results.append(faces)
        except Exception:
            log.warning("Frame analysis error", meeting_id=meeting_id, frame_idx=i, exc_info=True)
            results.append([])
    return results


def _applyTracking(
    frame_results: List[List[FaceFeatures]],
    meeting_id: str,
) -> List[List[FaceFeatures]]:
    """
    Replace per-frame detection-order person_idx with stable cross-chunk IDs.

    For each positional slot (0, 1, ...) seen across the chunk's frames, compute
    a representative embedding (median → re-normalised) and match it against the
    meeting's embedding gallery via cosine similarity. Returns a new frame_results
    list with person_idx values replaced by the tracker's stable IDs.
    """
    import dataclasses
    from collections import defaultdict

    slot_embeddings: dict[int, list] = defaultdict(list)
    slot_bboxes:     dict[int, list] = defaultdict(list)

    for frame_faces in frame_results:
        for face in frame_faces:
            slot_embeddings[face.person_idx].append(face.embedding)
            slot_bboxes[face.person_idx].append(face.bbox)

    if not slot_embeddings:
        return frame_results

    slots = sorted(slot_embeddings.keys())
    rep_embeddings = []
    rep_bboxes = []
    for s in slots:
        stack = np.stack(slot_embeddings[s])           # (N, 16)
        med = np.median(stack, axis=0)
        norm = np.linalg.norm(med)
        rep_emb = med / norm if norm > 1e-6 else med
        rep_embeddings.append(rep_emb)
        # Pick the actual frame embedding closest to the median as representative bbox
        best = int(np.argmax(stack @ rep_emb))
        rep_bboxes.append(slot_bboxes[s][best])

    stable_ids = assignStableIds(meeting_id, rep_embeddings, rep_bboxes)
    slot_to_stable = {slot: sid for slot, sid in zip(slots, stable_ids)}

    return [
        [dataclasses.replace(face, person_idx=slot_to_stable[face.person_idx])
         for face in frame_faces]
        for frame_faces in frame_results
    ]


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
