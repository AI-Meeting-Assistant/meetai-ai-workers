"""Vision Redis payload shapes (PYTHON_WORKERS_IMPLEMENTATION.md §5.3)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VisionChunkPayload(BaseModel):
    """JSON published to ``meeting:{meetingId}:vision``."""

    model_config = ConfigDict(populate_by_name=True)

    meeting_id: str = Field(alias="meetingId")
    offset_ms: int = Field(alias="offsetMs")
    focus_score: float | None = Field(None, alias="focusScore")
    payload: dict[str, Any] | None = None


def visionPayloadToRedisDict(p: VisionChunkPayload) -> dict[str, Any]:
    return p.model_dump(mode="json", by_alias=True, exclude_none=False)
