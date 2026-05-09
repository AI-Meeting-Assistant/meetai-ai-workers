"""Live ingest path: WAV bytes → Redis ``meeting:{id}:audio`` + optional text handoff Pipe."""

from __future__ import annotations

from typing import Any

from config import Settings, get_settings
from infrastructure.redis_client import RedisClient, get_redis_client

from workers.audio.asr import transcribeWindowText
from workers.audio.context_buffer import getBuffer
from workers.audio.diarization import streamingWindowStub
from workers.audio.io_audio import pcmMonoF32FromWavBytes
from workers.audio.schemas import (
    AudioChunkPayload,
    TextHandoffMessage,
    payloadToRedisDict,
)
from workers.audio.vad_window import computeEnergyVadMetrics

try:
    from core.fusion_publisher import channel_audio, publish_json
except ImportError:
    channel_audio = None  # type: ignore[assignment]
    publish_json = None  # type: ignore[assignment]


def _stubPayload(
    *,
    meeting_id: str,
    offset_ms: int,
    reason: str | None = None,
) -> AudioChunkPayload:
    extra: dict[str, Any] = {}
    if reason:
        extra["degradedReason"] = reason
    return AudioChunkPayload(
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        transcript=None,
        vad_speech_ms=None,
        vad_silence_ms=None,
        vad_speech_ratio_percent=None,
        speaker_labels_window=None,
        focus_score=None,
        payload=extra or None,
    )


def processLiveChunk(
    *,
    meeting_id: str,
    offset_ms: int,
    audio_wav_bytes: bytes,
    settings: Settings | None = None,
    redis_client: RedisClient | None = None,
    text_pipe_send_end: Any | None = None,
) -> AudioChunkPayload:
    """
    Process one ingest window (**WAV container bytes** recommended).

    Publishes JSON to Redis and optionally emits :class:`TextHandoffMessage`
    through ``text_pipe_send_end`` (writable ``multiprocessing.Connection``, one-way Pipe).
    """
    settings = settings or get_settings()
    r = redis_client or get_redis_client()

    def publishPayload(p: AudioChunkPayload) -> None:
        if channel_audio is None or publish_json is None:
            return
        publish_json(r.raw, channel_audio(meeting_id), payloadToRedisDict(p))

    def sendHandoff(transcript_val: str | None) -> None:
        if text_pipe_send_end is None:
            return
        hm = TextHandoffMessage(meeting_id=meeting_id, offset_ms=offset_ms, transcript=transcript_val)
        text_pipe_send_end.send(hm.model_dump(mode="json", by_alias=True))

    try:
        pcm, sr = pcmMonoF32FromWavBytes(audio_wav_bytes, settings.target_sample_rate)
    except Exception:
        p = _stubPayload(meeting_id=meeting_id, offset_ms=offset_ms, reason="decode_error")
        publishPayload(p)
        sendHandoff(None)
        return p

    window_end_ms = offset_ms + settings.media_chunk_duration_ms

    vad_metrics = None
    if settings.run_live_vad_energy:
        vad_metrics = computeEnergyVadMetrics(
            pcm,
            sample_rate=sr,
            speech_energy_quantile=settings.vad_energy_rms_quantile_for_speech,
        )

    buf = getBuffer(
        meeting_id,
        sample_rate=sr,
        context_ms=settings.asr_context_window_ms,
    )
    buf.appendChunk(offset_ms, pcm)
    ctx_pcm, abs_context_start_ms = buf.buildConcatenated()

    transcript_val: str | None = None
    if settings.run_live_asr:
        try:
            transcript_val = transcribeWindowText(
                context_pcm=ctx_pcm,
                sample_rate=sr,
                abs_context_start_ms=abs_context_start_ms,
                window_start_ms=offset_ms,
                window_end_ms=window_end_ms,
                model_size=settings.whisper_model_size,
                language=settings.whisper_language,
            )
            transcript_val = transcript_val or None
        except Exception:
            transcript_val = None

    labels = None
    if settings.run_live_diarization_stub:
        labels = streamingWindowStub(
            meeting_id, offset_ms=offset_ms, duration_ms=settings.media_chunk_duration_ms
        )

    payload = AudioChunkPayload(
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        transcript=transcript_val,
        vad_speech_ms=float(vad_metrics.speech_ms) if vad_metrics else None,
        vad_silence_ms=float(vad_metrics.silence_ms) if vad_metrics else None,
        vad_speech_ratio_percent=float(vad_metrics.speech_ratio_percent) if vad_metrics else None,
        speaker_labels_window=labels,
        focus_score=None,
        payload=None,
    )
    publishPayload(payload)
    sendHandoff(transcript_val)
    return payload
