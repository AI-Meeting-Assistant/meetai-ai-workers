"""Wire shapes for Redis :audio payloads and audio→text handoff (see PYTHON_WORKERS_IMPLEMENTATION.md §5.3)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AudioChunkPayload(BaseModel):
    """JSON published to ``meeting:{meetingId}:audio``."""

    model_config = ConfigDict(populate_by_name=True)

    meeting_id: str = Field(alias="meetingId")
    offset_ms: int = Field(alias="offsetMs")
    transcript: str | None = None
    vad_speech_ms: float | None = Field(None, alias="vadSpeechMs")
    vad_silence_ms: float | None = Field(None, alias="vadSilenceMs")
    vad_speech_ratio_percent: float | None = Field(None, alias="vadSpeechRatioPercent")
    speaker_labels_window: list[dict[str, Any]] | None = Field(None, alias="speakerLabelsWindow")
    focus_score: float | None = Field(None, alias="focusScore")
    payload: dict[str, Any] | None = None


class TextHandoffMessage(BaseModel):
    """One-way Pipe/Queue message into ``workers/text`` (§3.2)."""

    model_config = ConfigDict(populate_by_name=True)

    meeting_id: str = Field(alias="meetingId")
    offset_ms: int = Field(alias="offsetMs")
    transcript: str | None = None


class AudioWorkerInboundChunk(BaseModel):
    """IPC / internal shape for ingest (not Redis)."""

    model_config = ConfigDict(populate_by_name=True)

    meeting_id: str = Field(alias="meetingId")
    offset_ms: int = Field(alias="offsetMs")
    audio_pcm_mono_f32: bytes
    sample_rate: int = Field(alias="sampleRate")


def payloadToRedisDict(p: AudioChunkPayload) -> dict[str, Any]:
    return p.model_dump(mode="json", by_alias=True, exclude_none=False)
