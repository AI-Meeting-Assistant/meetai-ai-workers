"""Per-meeting WebM init-segment cache.

Browser MediaRecorder with timeslice emits only the *first* chunk with the
EBML + Segment header (init segment). All subsequent chunks are continuation
fragments that PyAV cannot open independently.

Fix: cache the first header-bearing chunk per meeting; prepend it to every
subsequent fragment before passing bytes to av.open().

Each worker process keeps its own in-process dict — no cross-process sharing
needed because audio and video are decoded in separate worker processes.
"""

from __future__ import annotations

# WebM / Matroska EBML magic bytes (first 4 bytes of every valid init segment)
_EBML_MAGIC = b'\x1a\x45\xdf\xa3'

# process-local cache: meeting_id → init segment bytes
_cache: dict[str, bytes] = {}


def prepareWebmChunk(meeting_id: str, data: bytes) -> bytes:
    """
    Return bytes suitable for ``av.open()``.

    - If ``data`` starts with EBML magic: cache it and return as-is.
    - Otherwise: prepend the cached init segment so PyAV can parse the fragment.
    - If no init segment cached yet (edge case): return ``data`` unchanged
      (caller will likely get a decode error, which is handled upstream).
    """
    if data[:4] == _EBML_MAGIC:
        _cache[meeting_id] = data
        return data

    init = _cache.get(meeting_id)
    if init:
        return init + data

    return data  # no init yet — upstream try/except handles failure


def evict(meeting_id: str) -> None:
    """Remove cached init segment when a meeting ends."""
    _cache.pop(meeting_id, None)
