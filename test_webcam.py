"""Webcam test: capture frames, run full vision pipeline, print Redis payload format."""

import json
import time

import cv2
import numpy as np

from workers.video.aggregator import aggregateChunk
from workers.video.face_mesh import analyzeFaces
from workers.video.schemas import PersonVisionResult, VisionChunkPayload, visionPayloadToRedisDict
from workers.video.tracker import assignStableIds, evict

MEETING_ID = "webcam-test"
CHUNK_DURATION_S = 6   # simulate 6-second chunks
FPS = 15               # capture rate
FRAMES_PER_CHUNK = CHUNK_DURATION_S * FPS
STRIDE = 3             # analyze every Nth frame (matches pipeline)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam — macOS camera permission required.")
    exit(1)

cap.set(cv2.CAP_PROP_FPS, FPS)
print(f"Webcam opened. Results will be printed every {CHUNK_DURATION_S}s. Press Ctrl+C to exit.\n")

offset_ms = 0
chunk_idx = 0

try:
    while True:
        chunk_idx += 1
        frames = []
        t0 = time.time()

        for _ in range(FRAMES_PER_CHUNK):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        # Analyze sampled frames
        frame_results = []
        for i, frame in enumerate(frames):
            if i % STRIDE != 0:
                continue
            try:
                faces = analyzeFaces(frame)
                if faces:
                    stable_ids = assignStableIds(MEETING_ID, [f.bbox for f in faces])
                    for face, sid in zip(faces, stable_ids):
                        face.person_idx = sid
                frame_results.append(faces)
            except Exception as e:
                frame_results.append([])

        aggregates = aggregateChunk(frame_results)

        # Build payload identical to Redis format
        if aggregates:
            persons = [
                PersonVisionResult(
                    person_id=a.person_id,
                    focus_score=round(a.focus_score, 4),
                    speaking_ratio=round(a.speaking_ratio, 4),
                    frame_count=a.frame_count,
                )
                for a in aggregates
            ]
            mean_focus = round(sum(p.focus_score or 0 for p in persons) / len(persons), 4)
            payload = VisionChunkPayload(
                meeting_id=MEETING_ID,
                offset_ms=offset_ms,
                focus_score=mean_focus,
                persons=persons,
            )
        else:
            payload = VisionChunkPayload(
                meeting_id=MEETING_ID,
                offset_ms=offset_ms,
                focus_score=None,
                persons=[],
                payload={"degradedReason": "no_faces"},
            )

        d = visionPayloadToRedisDict(payload)
        frames_with_faces = sum(1 for f in frame_results if f)

        # Pose debug: mean yaw/pitch across frames with faces
        all_faces = [f for frame in frame_results for f in frame]
        if all_faces:
            mean_yaw   = round(sum(f.yaw   for f in all_faces) / len(all_faces), 1)
            mean_pitch = round(sum(f.pitch for f in all_faces) / len(all_faces), 1)
            mean_gaze  = round(sum((f.gaze_x**2 + f.gaze_y**2)**0.5 for f in all_faces) / len(all_faces), 3)
            pose_str = f"yaw={mean_yaw}° pitch={mean_pitch}° gaze_offset={mean_gaze}"
        else:
            pose_str = "no faces"

        # Tracker debug: unique IDs seen this chunk and how many frames each appeared
        id_counts: dict[int, int] = {}
        for frame in frame_results:
            for face in frame:
                id_counts[face.person_idx] = id_counts.get(face.person_idx, 0) + 1
        tracker_str = ", ".join(f"id{pid}={cnt}f" for pid, cnt in sorted(id_counts.items()))

        print(f"── chunk {chunk_idx} | offset={offset_ms}ms | "
              f"sampled={len(frame_results)} with_faces={frames_with_faces} | {pose_str}")
        print(f"   tracker: [{tracker_str or 'no faces'}]")
        print(json.dumps(d, indent=2, ensure_ascii=False))
        print()

        offset_ms += CHUNK_DURATION_S * 1000

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()
    evict(MEETING_ID)
