"""Warm-up hooks for vision worker models."""

from utils.logger import get_logger

log = get_logger(__name__)


def warmUpLiveVideoModels() -> None:
    """Load MediaPipe Face Mesh singleton so the first ingest frame isn't slow."""
    try:
        from workers.video.face_mesh import _getFaceMesh
        _getFaceMesh()
        log.info("MediaPipe Face Mesh warmed up")
    except Exception as exc:
        log.error("MediaPipe warm-up failed", error=str(exc))
        raise
