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

# Gaze is iris-relative-to-eye — doesn't change with head turns, only with eye movement.
# So pose is the primary attention signal; gaze is a minor supplement.
_GAZE_WEIGHT = 0.2
_POSE_WEIGHT = 0.8

# Gaze dead zone: iris naturally offset ~0.1 due to camera/screen geometry.
_GAZE_DEAD_ZONE = 0.2

_YAW_TOLERANCE_DEG   = 28.0
_PITCH_TOLERANCE_DEG = 30.0  # widened from 22° — iris compensation handles the rest


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

    Pose score: NaN frames (solvePnP failed) are excluded from pose averaging.
                Yaw: dead zone ±5°, tolerance ±28°.
                Pitch: iris-compensated within a screen-viewing window only.
                  Compensation fires when pitch is in (PITCH_NATURAL_OFFSET, PITCH_COMP_MAX),
                  i.e. head is tilted downward but not so far it's clearly the desk.
                  Outside that window (looking up OR head nearly at desk) the full pitch
                  penalty applies.  gaze_y < -0.08 (iris above eye centre) reduces the
                  pitch penalty by up to 85%.
                Natural offset shifted to 10° to meet most monitor setups halfway.
    """
    gaze_offset = float(np.mean(np.sqrt(gaze_x ** 2 + gaze_y ** 2)))
    effective_gaze = max(0.0, gaze_offset - _GAZE_DEAD_ZONE)
    gaze_score = float(np.clip(1.0 - effective_gaze / (1.0 - _GAZE_DEAD_ZONE), 0.0, 1.0))

    _PITCH_NATURAL_OFFSET = 10.0  # shifted from 0° — typical screen-viewing angle
    _YAW_DEAD_ZONE   = 5.0
    _PITCH_DEAD_ZONE = 15.0

    # --- NaN filtering: exclude frames where solvePnP failed ---
    valid_pose = ~(np.isnan(yaw) | np.isnan(pitch))
    if not np.any(valid_pose):
        pose_score = 0.5  # no pose data — treat as neutral, not perfect
        return float(np.clip(_GAZE_WEIGHT * gaze_score + _POSE_WEIGHT * pose_score, 0.0, 1.0))

    yaw_v   = yaw[valid_pose]
    pitch_v = pitch[valid_pose]
    gaze_y_v = gaze_y[valid_pose]

    # --- Yaw penalty (unchanged) ---
    yaw_excess = np.maximum(0.0, np.abs(yaw_v) - _YAW_DEAD_ZONE)
    yaw_penalty = np.mean(np.clip(yaw_excess / (_YAW_TOLERANCE_DEG - _YAW_DEAD_ZONE), 0.0, 1.0))

    # --- Iris-compensated pitch penalty ---
    # Compensation is gated to the plausible screen-viewing pitch window:
    #   pitch > PITCH_NATURAL_OFFSET  → head is genuinely tilted down toward screen
    #   pitch < PITCH_COMP_MAX        → not so far down it's clearly the desk/lap
    # Outside this window (negative pitch = looking up, or extreme positive = desk)
    # gaze_y upward is not evidence of screen-viewing, so no compensation.
    _IRIS_COMP_THRESHOLD = 0.08
    _PITCH_COMP_MAX = 45.0
    in_screen_range = (pitch_v > _PITCH_NATURAL_OFFSET) & (pitch_v < _PITCH_COMP_MAX)
    compensation = np.where(
        in_screen_range,
        np.clip(-gaze_y_v / _IRIS_COMP_THRESHOLD, 0.0, 1.0),
        0.0,
    )
    pitch_excess = np.maximum(0.0, np.abs(pitch_v - _PITCH_NATURAL_OFFSET) - _PITCH_DEAD_ZONE)
    raw_pitch_penalty = np.clip(pitch_excess / (_PITCH_TOLERANCE_DEG - _PITCH_DEAD_ZONE), 0.0, 1.0)
    pitch_penalty = np.mean(raw_pitch_penalty * (1.0 - 0.85 * compensation))

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
