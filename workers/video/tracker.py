"""Cross-chunk stable person ID assignment via centroid matching with ghost memory.

Each worker process keeps a per-meeting MeetingTracker. On every chunk,
new face bboxes are matched to stored centroids by nearest Euclidean distance.
Faces that match keep their stable_id; unmatched faces get a new one.

Ghost memory: when a face disappears (e.g. head turn causing missed detection),
its centroid is retained for _GHOST_CHUNKS chunks before being discarded.
This prevents a briefly-lost face from getting a new ID when it reappears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import time

import numpy as np

# Maximum pixel distance to consider two centroids the same person.
_MAX_MATCH_DISTANCE = 200.0

# How long (seconds) to keep a face centroid alive after it disappears.
# Handles temporary camera-off, head turns, or brief occlusions.
_GHOST_SECONDS = 30.0

# process-local tracker registry
_trackers: Dict[str, "MeetingTracker"] = {}


@dataclass
class _TrackedFace:
    centroid: Tuple[float, float]
    last_seen_ts: float = field(default_factory=time.monotonic)


@dataclass
class MeetingTracker:
    faces: Dict[int, _TrackedFace] = field(default_factory=dict)  # stable_id → face
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
      2. Greedily match to nearest stored centroid within _MAX_MATCH_DISTANCE.
      3. Unmatched → new stable_id.
      4. Update centroids for matched faces; ghost faces not seen this chunk
         are kept for _GHOST_CHUNKS more chunks before eviction.
    """
    tracker = getTracker(meeting_id)
    now = time.monotonic()
    new_centroids = [_centroid(b) for b in bboxes]

    if not tracker.faces:
        ids = []
        for c in new_centroids:
            new_id = tracker.next_id
            tracker.next_id += 1
            tracker.faces[new_id] = _TrackedFace(centroid=c, last_seen_ts=now)
            ids.append(new_id)
        return ids

    stored_ids = list(tracker.faces.keys())
    stored_pts = np.array([tracker.faces[i].centroid for i in stored_ids], dtype=np.float32)

    assigned_ids: List[int] = [-1] * len(bboxes)

    if new_centroids:
        new_pts = np.array(new_centroids, dtype=np.float32)
        diffs = new_pts[:, None, :] - stored_pts[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))

        used_stored: set[int] = set()
        used_new: set[int] = set()

        pairs = sorted(
            ((dists[ni, si], ni, si) for ni in range(len(bboxes)) for si in range(len(stored_ids))),
            key=lambda x: x[0],
        )
        for dist, ni, si in pairs:
            if dist > _MAX_MATCH_DISTANCE:
                break
            if ni in used_new or si in used_stored:
                continue
            sid = stored_ids[si]
            assigned_ids[ni] = sid
            tracker.faces[sid].centroid = new_centroids[ni]
            tracker.faces[sid].last_seen_ts = now
            used_new.add(ni)
            used_stored.add(si)

    # Unmatched new faces → fresh IDs
    for ni in range(len(bboxes)):
        if assigned_ids[ni] == -1:
            new_id = tracker.next_id
            tracker.next_id += 1
            tracker.faces[new_id] = _TrackedFace(centroid=new_centroids[ni], last_seen_ts=now)
            assigned_ids[ni] = new_id

    # Evict ghost faces not seen within _GHOST_SECONDS
    stale = [sid for sid, f in tracker.faces.items() if now - f.last_seen_ts > _GHOST_SECONDS]
    for sid in stale:
        del tracker.faces[sid]

    return assigned_ids


def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
