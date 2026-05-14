"""Cross-chunk stable person ID assignment via centroid matching.

Each worker process keeps a per-meeting MeetingTracker. On every chunk,
new face bboxes are matched to the previous chunk's centroids by nearest
Euclidean distance. Faces that match an existing centroid keep its stable_id;
unmatched faces get a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Maximum pixel distance to consider two centroids the same person.
# At 640px width a grid cell is ~150-200px wide; 120px covers same-cell drift.
_MAX_MATCH_DISTANCE = 120.0

# process-local tracker registry
_trackers: Dict[str, "MeetingTracker"] = {}


@dataclass
class MeetingTracker:
    centroids: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # stable_id → (cx, cy)
    next_id: int = 0


def getTracker(meeting_id: str) -> MeetingTracker:
    if meeting_id not in _trackers:
        _trackers[meeting_id] = MeetingTracker()
    return _trackers[meeting_id]


def evict(meeting_id: str) -> None:
    """Remove tracker state when a meeting ends."""
    _trackers.pop(meeting_id, None)


def assignStableIds(
    meeting_id: str,
    bboxes: List[Tuple[int, int, int, int]],
) -> List[int]:
    """
    Match ``bboxes`` (x1,y1,x2,y2) to stored centroids and return one
    stable_id per bbox, in the same order.

    Algorithm:
      1. Compute centroid for each incoming bbox.
      2. Greedily match each centroid to the nearest stored centroid within
         _MAX_MATCH_DISTANCE (closest-first to avoid double assignment).
      3. Unmatched → new stable_id.
      4. Update stored centroids with this chunk's positions.
    """
    tracker = getTracker(meeting_id)
    new_centroids = [_centroid(b) for b in bboxes]

    if not tracker.centroids:
        # First chunk for this meeting — assign sequential IDs
        ids = list(range(len(bboxes)))
        tracker.next_id = len(bboxes)
        tracker.centroids = {i: c for i, c in zip(ids, new_centroids)}
        return ids

    stored_ids = list(tracker.centroids.keys())
    stored_pts = np.array([tracker.centroids[i] for i in stored_ids], dtype=np.float32)
    new_pts = np.array(new_centroids, dtype=np.float32)

    # Pairwise distances: shape (n_new, n_stored)
    diffs = new_pts[:, None, :] - stored_pts[None, :, :]   # (N, M, 2)
    dists = np.sqrt((diffs ** 2).sum(axis=2))               # (N, M)

    assigned_ids: List[int] = [-1] * len(bboxes)
    used_stored: set[int] = set()
    used_new: set[int] = set()

    # Sort all (new_i, stored_j) pairs by distance, match greedily
    pairs = sorted(
        ((dists[ni, si], ni, si) for ni in range(len(bboxes)) for si in range(len(stored_ids))),
        key=lambda x: x[0],
    )
    for dist, ni, si in pairs:
        if dist > _MAX_MATCH_DISTANCE:
            break
        if ni in used_new or si in used_stored:
            continue
        assigned_ids[ni] = stored_ids[si]
        used_new.add(ni)
        used_stored.add(si)

    # Unmatched new faces → fresh IDs
    for ni in range(len(bboxes)):
        if assigned_ids[ni] == -1:
            assigned_ids[ni] = tracker.next_id
            tracker.next_id += 1

    # Update stored centroids with current positions
    tracker.centroids = {assigned_ids[ni]: new_centroids[ni] for ni in range(len(bboxes))}

    return assigned_ids


def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
