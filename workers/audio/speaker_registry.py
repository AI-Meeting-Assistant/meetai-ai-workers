"""Per-meeting canonical speaker centroids (cosine + EMA), max N speakers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-8


@dataclass
class MeetingSpeakerRegistry:
    max_speakers: int
    cos_threshold: float
    ema_alpha: float
    centroids: list[np.ndarray] = field(default_factory=list)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < EPS:
            return v
        return (v / n).astype(np.float32, copy=False)

    def _best_match(self, e: np.ndarray) -> tuple[int, float]:
        sims = [float(np.dot(e, c.astype(np.float64))) for c in self.centroids]
        best_i = max(range(len(sims)), key=lambda idx: sims[idx])
        return best_i, sims[best_i]

    def _update_centroid(self, idx: int, e: np.ndarray) -> str:
        old = self.centroids[idx].astype(np.float64, copy=False)
        new_c = self.ema_alpha * e + (1.0 - self.ema_alpha) * old
        self.centroids[idx] = self._normalize(new_c)
        return f"Speaker {idx + 1}"

    def assign_and_update(self, emb: np.ndarray, *, allow_new_speaker: bool = True) -> str:
        """
        Map embedding to ``Speaker k`` and update / create centroids.

        When ``allow_new_speaker`` is False, always attach to the best-matching
        existing centroid (never open Speaker N+1) — used for short VAD blobs.
        """
        e = self._normalize(emb.astype(np.float64, copy=False))
        if not self.centroids:
            self.centroids.append(e.astype(np.float32, copy=False))
            return "Speaker 1"

        best_i, best_sim = self._best_match(e)
        if best_sim >= self.cos_threshold or not allow_new_speaker:
            return self._update_centroid(best_i, e)

        if len(self.centroids) < self.max_speakers:
            self.centroids.append(e.astype(np.float32, copy=False))
            return f"Speaker {len(self.centroids)}"

        return self._update_centroid(best_i, e)

    def merge_redundant_centroids(self, merge_cos_threshold: float = 0.72) -> None:
        """Collapse near-duplicate centroids so one timbre stays one Speaker id."""
        if len(self.centroids) < 2:
            return
        changed = True
        while changed and len(self.centroids) >= 2:
            changed = False
            n = len(self.centroids)
            for i in range(n):
                for j in range(i + 1, n):
                    if j >= len(self.centroids):
                        break
                    ci = self.centroids[i].astype(np.float64, copy=False)
                    cj = self.centroids[j].astype(np.float64, copy=False)
                    if float(np.dot(ci, cj)) < merge_cos_threshold:
                        continue
                    merged = self._normalize(0.5 * ci + 0.5 * cj)
                    self.centroids[i] = merged
                    self.centroids.pop(j)
                    changed = True
                    break
                if changed:
                    break


_registries: dict[str, MeetingSpeakerRegistry] = {}


def getOrCreateRegistry(
    meeting_id: str,
    *,
    max_speakers: int,
    cos_threshold: float,
    ema_alpha: float,
) -> MeetingSpeakerRegistry:
    reg = _registries.get(meeting_id)
    if reg is None:
        reg = MeetingSpeakerRegistry(
            max_speakers=max_speakers,
            cos_threshold=cos_threshold,
            ema_alpha=ema_alpha,
        )
        _registries[meeting_id] = reg
    return reg


def clearSpeakerRegistry(meeting_id: str) -> None:
    _registries.pop(meeting_id, None)


def clearAllSpeakerRegistries() -> None:
    _registries.clear()
