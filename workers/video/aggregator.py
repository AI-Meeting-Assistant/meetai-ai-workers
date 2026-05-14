"""Aggregate per-frame face features into per-person chunk-level scores.

Accepts a list of frame results (each frame is a list of FaceFeatures) and
returns one ``PersonAggregate`` per detected person index.

Person identity = detection order within a frame (index 0, 1, 2…).
This is a simple positional assignment — adequate for short 2-second windows
where face order is stable enough. A tracker-based approach is reserved for
a future phase.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from workers.video.face_mesh import FaceFeatures

# MAR threshold above which a frame is counted as "speaking"
_MAR_SPEAKING_THRESHOLD = 0.06

# Smoothing window for speaking_ratio (frames)
_SPEAKING_SMOOTH_WINDOW = 8

# Weights for focus_score: gaze vs head pose
_GAZE_WEIGHT = 0.5
_POSE_WEIGHT = 0.5

# Gaze: natural iris offset when looking at screen (~0.15 due to camera/screen misalignment).
# Offsets within the dead zone score 1.0; penalty starts only beyond it.
_GAZE_DEAD_ZONE = 0.15

# Head pose: wider tolerances account for solvePnP error from approximate intrinsics.
# Penalty starts only beyond these angles.
_YAW_TOLERANCE_DEG   = 45.0
_PITCH_TOLERANCE_DEG = 35.0


@dataclass
class PersonAggregate:
    person_id: int
    focus_score: float          # 0-1; higher = more attentive
    speaking_ratio: float       # 0-1; fraction of frames where MAR > threshold
    frame_count: int            # frames this person appeared in


def aggregateChunk(
    frame_results: List[List[FaceFeatures]],
) -> List[PersonAggregate]:
    """
    ``frame_results``: one list of ``FaceFeatures`` per frame, in temporal order.
    Returns one ``PersonAggregate`` per person index seen in the chunk.
    """
    if not frame_results:
        return []

    # Collect per-person timeseries
    gaze_xs:  Dict[int, List[float]] = defaultdict(list)
    gaze_ys:  Dict[int, List[float]] = defaultdict(list)
    yaws:     Dict[int, List[float]] = defaultdict(list)
    pitches:  Dict[int, List[float]] = defaultdict(list)
    mars:     Dict[int, List[float]] = defaultdict(list)

    for frame_faces in frame_results:
        for face in frame_faces:
            pid = face.person_idx
            gaze_xs[pid].append(face.gaze_x)
            gaze_ys[pid].append(face.gaze_y)
            yaws[pid].append(face.yaw)
            pitches[pid].append(face.pitch)
            mars[pid].append(face.mar)

    aggregates: List[PersonAggregate] = []
    for pid in sorted(gaze_xs.keys()):
        gx = np.array(gaze_xs[pid], dtype=np.float32)
        gy = np.array(gaze_ys[pid], dtype=np.float32)
        yaw_arr  = np.array(yaws[pid],    dtype=np.float32)
        pitch_arr = np.array(pitches[pid], dtype=np.float32)
        mar_arr  = np.array(mars[pid],    dtype=np.float32)

        focus = _focusScore(gx, gy, yaw_arr, pitch_arr)
        speaking = _speakingRatio(mar_arr)

        aggregates.append(PersonAggregate(
            person_id=pid,
            focus_score=float(focus),
            speaking_ratio=float(speaking),
            frame_count=len(gx),
        ))

    return aggregates


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _focusScore(
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
) -> float:
    """
    Weighted combination of gaze and head-pose scores, both in [0, 1].

    Gaze score: dead zone of _GAZE_DEAD_ZONE absorbs natural iris offset when
                looking at screen. Penalty kicks in only beyond the dead zone
                and ramps to 0 at offset 1.0.

    Pose score: dead zone of ±5° absorbs solvePnP noise. Linear ramp to 0
                beyond ±_YAW_TOLERANCE_DEG / ±_PITCH_TOLERANCE_DEG.
    """
    gaze_offset = float(np.mean(np.sqrt(gaze_x ** 2 + gaze_y ** 2)))
    effective_gaze = max(0.0, gaze_offset - _GAZE_DEAD_ZONE)
    gaze_score = float(np.clip(1.0 - effective_gaze / (1.0 - _GAZE_DEAD_ZONE), 0.0, 1.0))

    _POSE_DEAD_ZONE = 5.0
    yaw_excess   = np.maximum(0.0, np.abs(yaw)   - _POSE_DEAD_ZONE)
    pitch_excess = np.maximum(0.0, np.abs(pitch) - _POSE_DEAD_ZONE)
    yaw_penalty   = np.mean(np.clip(yaw_excess   / (_YAW_TOLERANCE_DEG   - _POSE_DEAD_ZONE), 0.0, 1.0))
    pitch_penalty = np.mean(np.clip(pitch_excess / (_PITCH_TOLERANCE_DEG - _POSE_DEAD_ZONE), 0.0, 1.0))
    pose_score = float(np.clip(1.0 - (yaw_penalty + pitch_penalty) / 2.0, 0.0, 1.0))

    return float(np.clip(
        _GAZE_WEIGHT * gaze_score + _POSE_WEIGHT * pose_score,
        0.0, 1.0,
    ))


def _speakingRatio(mar: np.ndarray) -> float:
    """
    Fraction of frames where MAR > threshold, smoothed with a rolling window.

    The 8-frame window smooths jitter from landmark noise.
    """
    if mar.size == 0:
        return 0.0
    speaking_flags = (mar > _MAR_SPEAKING_THRESHOLD).astype(np.float32)

    # Rolling mean over _SPEAKING_SMOOTH_WINDOW (uniform convolution)
    k = min(_SPEAKING_SMOOTH_WINDOW, len(speaking_flags))
    kernel = np.ones(k, dtype=np.float32) / k
    smoothed = np.convolve(speaking_flags, kernel, mode="same")

    return float(np.mean(smoothed))
