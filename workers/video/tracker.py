"""Cross-chunk stable person ID assignment via landmark embedding matching.

Each worker process keeps a per-meeting MeetingTracker. On every chunk, one
representative embedding per detected person (median across the chunk's frames)
is matched to the gallery using cosine similarity. Matched faces keep their
stable_id; unmatched faces get a new one. The gallery embedding is updated with
an exponential moving average so it adapts to gradual lighting/pose drift.

Ghost memory: when a face disappears (e.g. camera off, brief occlusion), its
embedding is retained for _GHOST_SECONDS before eviction. This lets a face
resume its original ID when it reappears within that window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Cosine similarity threshold — above this score two embeddings are the same person.
_SIM_THRESHOLD = 0.85

# How long (seconds) to keep a gallery entry alive after the face disappears.
_GHOST_SECONDS = 30.0

# process-local tracker registry
_trackers: Dict[str, "MeetingTracker"] = {}


@dataclass
class _TrackedFace:
    embedding: np.ndarray        # 16-d unit-normalised landmark vector
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


def _cosSim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity — both vectors must be unit-normalised."""
    return float(np.dot(a, b))


def assignStableIds(
    meeting_id: str,
    embeddings: List[np.ndarray],
    bboxes: List[Tuple[int, int, int, int]],  # kept for future logging/fallback
) -> List[int]:
    """
    Match ``embeddings`` to the gallery and return one stable_id per embedding,
    in the same order.

    Algorithm:
      1. Build cosine similarity matrix (incoming × stored).
      2. Greedily assign highest-similarity pairs above _SIM_THRESHOLD.
      3. Unmatched incoming → new stable_id registered in gallery.
      4. Matched gallery entries: EMA-update embedding, refresh timestamp.
      5. Evict ghost entries older than _GHOST_SECONDS.
    """
    tracker = getTracker(meeting_id)
    now = time.monotonic()

    if not tracker.faces:
        ids = []
        for emb in embeddings:
            new_id = tracker.next_id
            tracker.next_id += 1
            tracker.faces[new_id] = _TrackedFace(embedding=emb.copy(), last_seen_ts=now)
            ids.append(new_id)
        return ids

    stored_ids = list(tracker.faces.keys())
    assigned_ids: List[int] = [-1] * len(embeddings)
    used_stored: set[int] = set()
    used_new: set[int] = set()

    if embeddings and stored_ids:
        # Similarity matrix shape: (n_new, n_stored)
        sims = np.array(
            [[_cosSim(e, tracker.faces[sid].embedding) for sid in stored_ids]
             for e in embeddings],
            dtype=np.float32,
        )

        # Greedy assignment — take the highest-similarity pair first
        n_new, n_stored = sims.shape
        pairs = sorted(
            ((sims[ni, si], ni, si)
             for ni in range(n_new) for si in range(n_stored)),
            key=lambda x: -x[0],
        )

        for sim, ni, si in pairs:
            if sim < _SIM_THRESHOLD:
                break
            if ni in used_new or si in used_stored:
                continue
            sid = stored_ids[si]
            assigned_ids[ni] = sid

            # EMA update: keeps gallery embedding fresh for pose/lighting drift
            updated = 0.9 * tracker.faces[sid].embedding + 0.1 * embeddings[ni]
            norm = np.linalg.norm(updated)
            tracker.faces[sid].embedding = updated / norm if norm > 1e-6 else updated
            tracker.faces[sid].last_seen_ts = now

            used_new.add(ni)
            used_stored.add(si)

    # Unmatched incoming faces → fresh stable IDs
    for ni in range(len(embeddings)):
        if assigned_ids[ni] == -1:
            new_id = tracker.next_id
            tracker.next_id += 1
            tracker.faces[new_id] = _TrackedFace(embedding=embeddings[ni].copy(), last_seen_ts=now)
            assigned_ids[ni] = new_id

    # Evict ghost faces not seen within _GHOST_SECONDS
    stale = [sid for sid, f in tracker.faces.items() if now - f.last_seen_ts > _GHOST_SECONDS]
    for sid in stale:
        del tracker.faces[sid]

    return assigned_ids
