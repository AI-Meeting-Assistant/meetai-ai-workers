# Video Worker

Processes screen-share video chunks from active meetings and publishes per-person attention metrics to Redis.

## What it does

Each ingest window (`POST /ingest`, default 2 s) delivers a WebM video chunk captured by the frontend via `getDisplayMedia` (screen share of e.g. Google Meet). The video worker:

1. Decodes the WebM bytes into BGR frames using PyAV (FFmpeg-backed, no system binary required).
2. Samples every 3rd frame and downscales to max 640 px width before analysis.
3. Runs **MediaPipe Face Mesh** on each sampled frame to detect up to 5 faces simultaneously.
4. For each detected face extracts:
   - **Gaze vector** — iris centre offset relative to eye corners (left + right eye averaged).
   - **Head pose** — yaw / pitch / roll in degrees via `cv2.solvePnP` with a 6-point 3-D face model.
   - **MAR** (Mouth Aspect Ratio) — vertical / horizontal lip distance ratio; > 0.06 indicates speaking.
   - **Bounding box** — tight rect around all 478 landmarks + 20 px padding.
5. Assigns **stable person IDs** across chunks using centroid-based tracking (nearest-centroid matching per meeting).
6. Aggregates per-person metrics over all sampled frames in the chunk into `focus_score` and `speaking_ratio`.
7. Publishes a `VisionChunkPayload` JSON to Redis channel `meeting:{meetingId}:vision`.

## Model

**MediaPipe Face Mesh** (`mediapipe==0.10.14`)
- `max_num_faces=5`, `refine_landmarks=True` (478 landmarks including 10 iris points)
- `min_detection_confidence=0.3` — lowered from default to handle small compressed video-grid thumbnails
- Runs on CPU; singleton loaded once at worker startup via `warmUpLiveVideoModels()`

## Focus score formula

```
gaze_offset   = mean(sqrt(gaze_x² + gaze_y²)) per frame
gaze_score    = clip(1 - max(0, gaze_offset - 0.15) / 0.85, 0, 1)

yaw_penalty   = mean(clip(max(0, |yaw|   - 5°) / 40°, 0, 1))
pitch_penalty = mean(clip(max(0, |pitch| - 5°) / 30°, 0, 1))
pose_score    = clip(1 - (yaw_penalty + pitch_penalty) / 2, 0, 1)

focus_score   = 0.5 × gaze_score + 0.5 × pose_score
```

Dead zones (0.15 gaze offset, ±5° pose) absorb natural iris offset and solvePnP noise so a person looking directly at the camera scores ~0.95+.

## Files

| File | Purpose |
|---|---|
| `worker_main.py` | Process entry point. Receives IPC messages from gateway supervisor via `multiprocessing.Pipe`, calls `processLiveVisionChunk`, handles shutdown. |
| `pipeline.py` | Main orchestrator. `processLiveVisionChunk(meeting_id, offset_ms, video_bytes)` decodes frames, runs analysis, aggregates results, publishes to Redis. |
| `io_video.py` | WebM/MP4/MOV → list of BGR numpy frames via PyAV. `framesFromVideoBytes(data, max_frames)`. |
| `face_mesh.py` | MediaPipe Face Mesh wrapper. `analyzeFaces(frame_bgr)` returns a list of `FaceFeatures` (gaze, yaw, pitch, roll, MAR, bbox) for all faces in one frame. |
| `tracker.py` | Cross-chunk stable person ID assignment. `assignStableIds(meeting_id, bboxes)` matches new bboxes to previous-chunk centroids by nearest Euclidean distance (threshold 120 px). New faces get a fresh incremental ID. `evict(meeting_id)` cleans up on meeting end. |
| `aggregator.py` | `aggregateChunk(frame_results)` collapses N frames of `FaceFeatures` into one `PersonAggregate` per person: `focus_score`, `speaking_ratio`, `frame_count`. |
| `schemas.py` | Pydantic models for the Redis payload: `VisionChunkPayload` (top-level) and `PersonVisionResult` (per-person). `visionPayloadToRedisDict` serialises for publish. |

## Redis output

Channel: `meeting:{meetingId}:vision`

```json
{
  "meetingId": "...",
  "offsetMs": 4000,
  "focusScore": 0.87,
  "persons": [
    { "personId": 0, "focusScore": 0.92, "speakingRatio": 0.14, "frameCount": 30 },
    { "personId": 1, "focusScore": 0.81, "speakingRatio": 0.00, "frameCount": 30 }
  ],
  "payload": null
}
```

`focusScore` at the top level is the mean across all detected persons (backward-compat scalar). `persons` array carries the full per-person breakdown.

If no faces are detected or decoding fails, a stub payload is published with `focusScore: null`, `persons: []`, and `payload: { "degradedReason": "no_faces" | "no_frames" | "empty_bytes" }`.

## Deferred

**Emotion detection (HSEmotion ONNX):** valence/arousal via `enet_b0_8_best_afew.onnx` — out of scope for current phase. Implementation plan is documented in `CLAUDE.md`.
