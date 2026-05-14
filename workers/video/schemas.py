"""Vision Redis payload shapes (PYTHON_WORKERS_IMPLEMENTATION.md §5.3)."""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field


class PersonVisionResult(BaseModel):
    """Per-person aggregated results for one video chunk."""

    model_config = ConfigDict(populate_by_name=True)

    person_id: int = Field(alias="personId")
    focus_score: float | None = Field(None, alias="focusScore")
    speaking_ratio: float | None = Field(None, alias="speakingRatio")
    frame_count: int | None = Field(None, alias="frameCount")


class VisionChunkPayload(BaseModel):
    """JSON published to ``meeting:{meetingId}:vision``."""

    model_config = ConfigDict(populate_by_name=True)

    meeting_id: str = Field(alias="meetingId")
    offset_ms: int = Field(alias="offsetMs")
    # Scalar focus kept for backward compat; mean of per-person scores when available.
    focus_score: float | None = Field(None, alias="focusScore")
    persons: List[PersonVisionResult] = Field(default_factory=list)
    payload: dict[str, Any] | None = None


def visionPayloadToRedisDict(p: VisionChunkPayload) -> dict[str, Any]:
    return p.model_dump(mode="json", by_alias=True, exclude_none=False)
