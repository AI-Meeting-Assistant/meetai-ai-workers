"""Audio worker: live ingest pipeline, batch analysis, and schemas."""

from __future__ import annotations

__all__ = ["processLiveChunk", "AudioChunkPayload", "TextHandoffMessage"]


def __getattr__(name: str):  # PEP 562 — avoid importing torch/whisper at import time
    if name == "processLiveChunk":
        from workers.audio.pipeline import processLiveChunk as mod

        return mod
    if name == "AudioChunkPayload":
        from workers.audio.schemas import AudioChunkPayload as mod

        return mod
    if name == "TextHandoffMessage":
        from workers.audio.schemas import TextHandoffMessage as mod

        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
