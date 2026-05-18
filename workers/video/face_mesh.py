"""Per-face feature extraction using MediaPipe Face Mesh.

For each face detected in a frame returns:
  - gaze_x, gaze_y  : normalised iris offset (-1 … +1); 0,0 = looking at camera
  - yaw, pitch, roll : head pose in degrees from solvePnP
  - mar              : Mouth Aspect Ratio (0 = closed, >0.06 ≈ speaking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Lazy singleton — MediaPipe is loaded once per process.
# ---------------------------------------------------------------------------

_face_mesh = None


def _getFaceMesh():
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp  # type: ignore
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,   # 400x400 face crops are clean — per-frame detection
                                      # is reliable and avoids tracker poisoning from blank frames
            max_num_faces=5,
            refine_landmarks=True,    # adds 10 iris landmarks (468-477)
            min_detection_confidence=0.5,
        )
    return _face_mesh




# ---------------------------------------------------------------------------
# 3-D reference face model used by solvePnP (metric units, frontal neutral)
# ---------------------------------------------------------------------------

# Canonical 3-D points in mm for a generic adult face (OpenCV convention: x right, y down, z into face)
_MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),    # nose tip          → landmark 1
    (0.0,  -63.6,  -12.5),   # chin              → landmark 152
    (-43.3,  32.7,  -26.0),  # left eye corner   → landmark 33
    (43.3,   32.7,  -26.0),  # right eye corner  → landmark 263
    (-28.9, -28.9,  -24.1),  # left mouth corner → landmark 61
    (28.9,  -28.9,  -24.1),  # right mouth corner→ landmark 291
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
_POSE_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

# Landmark embedding indices — 8 stable points (eye corners, nose tip, mouth corners)
# Normalised by face bbox → 16-d unit vector used for cross-chunk identity matching.
_EMBEDDING_LANDMARKS = [33, 133, 362, 263, 1, 61, 291, 199]

# Iris landmark indices (refine_landmarks=True required)
_LEFT_IRIS_CENTER  = 468   # iris centre — left eye
_RIGHT_IRIS_CENTER = 473   # iris centre — right eye
_LEFT_EYE_LEFT    = 33
_LEFT_EYE_RIGHT   = 133
_RIGHT_EYE_LEFT   = 362
_RIGHT_EYE_RIGHT  = 263

# MAR landmarks: upper-inner lip (13), lower-inner lip (14),
#                left corner (78), right corner (308)
_MAR_IDX = (13, 14, 78, 308)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FaceFeatures:
    person_idx: int          # order of detection in this frame (0-based); reassigned to stable ID by tracker
    gaze_x: float            # horizontal iris offset; + = looking right
    gaze_y: float            # vertical iris offset;   + = looking down
    yaw: float               # degrees; + = face turned right
    pitch: float             # degrees; + = face tilted down
    roll: float              # degrees; + = face tilted clockwise
    mar: float               # mouth aspect ratio
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) with padding
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))  # 16-d unit vector


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyzeFaces(frame_bgr: np.ndarray) -> List[FaceFeatures]:
    """
    Run MediaPipe Face Mesh on one BGR frame and return features for each face.
    Returns an empty list if no faces are detected or an error occurs.
    """
    try:
        return _analyzeFaces(frame_bgr)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _analyzeFaces(frame_bgr: np.ndarray) -> List[FaceFeatures]:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = _getFaceMesh().process(rgb)

    if not result.multi_face_landmarks:
        return []

    # Camera intrinsics approximation (principal point = image centre, focal = width)
    focal = w
    cam_matrix = np.array([
        [focal, 0,     w / 2],
        [0,     focal, h / 2],
        [0,     0,     1   ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    features: List[FaceFeatures] = []
    for idx, lm_set in enumerate(result.multi_face_landmarks):
        lms = lm_set.landmark  # list of NormalizedLandmark

        bbox = _computeBbox(lms, w, h, padding=20)
        gaze_x, gaze_y = _computeGaze(lms, w, h)
        yaw, pitch, roll = _computeHeadPose(lms, w, h, cam_matrix, dist_coeffs)
        mar = _computeMAR(lms, w, h)
        embedding = _computeEmbedding(lms, w, h, bbox)

        features.append(FaceFeatures(
            person_idx=idx,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            mar=mar,
            bbox=bbox,
            embedding=embedding,
        ))
    return features


def _lmPx(lm, w: int, h: int) -> Tuple[float, float]:
    """Denormalise a single landmark to pixel coordinates."""
    return lm.x * w, lm.y * h


def _computeBbox(
    lms,
    w: int,
    h: int,
    padding: int = 20,
) -> Tuple[int, int, int, int]:
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    return x1, y1, x2, y2


def _computeGaze(lms, w: int, h: int) -> Tuple[float, float]:
    """
    Iris-based gaze estimate.

    The iris centre (landmark 468 / 473) is compared to the midpoint of the
    eye corners.  The offset is normalised by half the inter-corner width so
    the result is approximately in [-1, +1].
    """
    # Left eye
    lx_l, ly_l = _lmPx(lms[_LEFT_EYE_LEFT],  w, h)
    lx_r, ly_r = _lmPx(lms[_LEFT_EYE_RIGHT], w, h)
    ix_l, iy_l = _lmPx(lms[_LEFT_IRIS_CENTER], w, h)

    eye_w_l = abs(lx_r - lx_l) + 1e-6
    eye_cx_l = (lx_l + lx_r) / 2
    eye_cy_l = (ly_l + ly_r) / 2

    gaze_x_l = (ix_l - eye_cx_l) / (eye_w_l / 2)
    gaze_y_l = (iy_l - eye_cy_l) / (eye_w_l / 2)

    # Right eye
    rx_l, ry_l = _lmPx(lms[_RIGHT_EYE_LEFT],  w, h)
    rx_r, ry_r = _lmPx(lms[_RIGHT_EYE_RIGHT], w, h)
    ix_r, iy_r = _lmPx(lms[_RIGHT_IRIS_CENTER], w, h)

    eye_w_r = abs(rx_r - rx_l) + 1e-6
    eye_cx_r = (rx_l + rx_r) / 2
    eye_cy_r = (ry_l + ry_r) / 2

    gaze_x_r = (ix_r - eye_cx_r) / (eye_w_r / 2)
    gaze_y_r = (iy_r - eye_cy_r) / (eye_w_r / 2)

    return (gaze_x_l + gaze_x_r) / 2, (gaze_y_l + gaze_y_r) / 2


def _computeHeadPose(
    lms,
    w: int,
    h: int,
    cam_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Head pose via solvePnP.

    Maps 6 stable 2-D landmarks to a generic 3-D face model and recovers the
    rotation vector.  ``cv2.Rodrigues`` converts the rotation vector to a
    matrix from which yaw / pitch / roll are extracted using Euler ZYX decomp.
    """
    image_points = np.array(
        [(_lmPx(lms[i], w, h)) for i in _POSE_LANDMARK_IDX],
        dtype=np.float64,
    )
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS_3D,
        image_points,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return float('nan'), float('nan'), float('nan')

    # Rodrigues → rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # solvePnP has a flip ambiguity: rmat[2,2] < 0 means the face normal points
    # away from the camera (back-of-head solution). Flip 180° around x-axis to correct.
    if rmat[2, 2] < 0:
        flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        rmat = flip @ rmat

    # Euler angles from ZYX decomposition
    pitch_rad = np.arctan2(-rmat[2, 0], np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    cos_pitch = np.cos(pitch_rad)
    if abs(cos_pitch) < 1e-6:
        yaw_rad, roll_rad = 0.0, np.arctan2(-rmat[0, 1], rmat[1, 1])
    else:
        yaw_rad  = np.arctan2(rmat[1, 0] / cos_pitch, rmat[0, 0] / cos_pitch)
        roll_rad = np.arctan2(rmat[2, 1] / cos_pitch, rmat[2, 2] / cos_pitch)

    yaw_deg   = float(np.degrees(yaw_rad))
    pitch_deg = float(np.degrees(pitch_rad))
    roll_deg  = float(np.degrees(roll_rad))

    # |yaw| > 90° is a solvePnP back-of-head flip artifact (common when the
    # video is horizontally mirrored, e.g. Google Meet self-view). Discard.
    if abs(yaw_deg) > 90:
        return float('nan'), float('nan'), float('nan')

    return yaw_deg, pitch_deg, roll_deg


def _computeEmbedding(lms, w: int, h: int, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    16-d unit-normalised landmark embedding for cross-chunk identity matching.

    Takes the 8 stable landmarks in _EMBEDDING_LANDMARKS, normalises their
    pixel positions by the face bbox dimensions, and returns a unit vector.
    Cosine similarity between two embeddings of the same person is typically > 0.95.
    """
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pts: list[float] = []
    for i in _EMBEDDING_LANDMARKS:
        pts.append((lms[i].x * w - x1) / bw)
        pts.append((lms[i].y * h - y1) / bh)
    v = np.array(pts, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def _computeMAR(lms, w: int, h: int) -> float:
    """
    Mouth Aspect Ratio.

        MAR = |upper_inner - lower_inner| / |left_corner - right_corner|

    Values above ~0.06 indicate the mouth is open (speaking).
    """
    u_idx, l_idx, lc_idx, rc_idx = _MAR_IDX

    ux, uy = _lmPx(lms[u_idx],  w, h)
    lx, ly = _lmPx(lms[l_idx],  w, h)
    lcx, lcy = _lmPx(lms[lc_idx], w, h)
    rcx, rcy = _lmPx(lms[rc_idx], w, h)

    vertical   = np.hypot(ux - lx, uy - ly)
    horizontal = np.hypot(lcx - rcx, lcy - rcy) + 1e-6
    return float(vertical / horizontal)
