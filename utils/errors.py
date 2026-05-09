"""Structured errors for workers (optional expansion)."""


class AudioWorkerError(Exception):
    """Recoverable pipeline error — caller may publish graceful-null payload."""

    pass
