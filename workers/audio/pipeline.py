"""Live ingest path: WAV bytes → Redis ``meeting:{id}:audio`` + optional text handoff Pipe."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from config import Settings, get_settings
from infrastructure.redis_client import RedisClient, get_redis_client

from workers.audio.asr import transcribeWindowText
from workers.audio.context_buffer import clearMeetingBuffers, getBuffer
from workers.audio.io_audio import pcmMonoF32FromWebmBytes
from workers.audio.schemas import (
    AudioChunkPayload,
    TextHandoffMessage,
    payloadToRedisDict,
)
from workers.audio.vad_window import computeEnergyVadMetrics
from utils.logger import get_logger

log = get_logger(__name__)

# Live MediaRecorder chunks are often non-standalone WebM; concatenate per meeting then decode.
_MAX_LIVE_WEBM_ACCUM_MEETINGS = 200
_live_webm_accum_by_meeting: OrderedDict[str, bytearray] = OrderedDict()


def clear_live_webm_accum(meeting_id: str) -> None:
    """Drop concatenated WebM bytes for one meeting (e.g. stream ended)."""
    _live_webm_accum_by_meeting.pop(meeting_id, None)


def clear_live_webm_accum_all() -> None:
    _live_webm_accum_by_meeting.clear()


def _touch_live_webm_accum_lru(meeting_id: str) -> None:
    _live_webm_accum_by_meeting.move_to_end(meeting_id)
    while len(_live_webm_accum_by_meeting) > _MAX_LIVE_WEBM_ACCUM_MEETINGS:
        _live_webm_accum_by_meeting.popitem(last=False)


def _append_and_decode_live_webm_pcm(
    meeting_id: str,
    offset_ms: int,
    audio_webm_bytes: bytes,
    *,
    window_duration_ms: int,
    target_sr: int,
) -> tuple[Any, int]:
    """
    Append this ingest blob to the meeting WebM buffer, decode the full buffer once,
    return mono float32 PCM for ``[offset_ms, offset_ms + window_duration_ms)`` only.
    """
    if offset_ms == 0:
        buf = bytearray()
        _live_webm_accum_by_meeting[meeting_id] = buf
    else:
        buf = _live_webm_accum_by_meeting.get(meeting_id)
        if buf is None:
            buf = bytearray()
            _live_webm_accum_by_meeting[meeting_id] = buf

    before_len = len(buf)
    buf.extend(audio_webm_bytes)
    _touch_live_webm_accum_lru(meeting_id)

    try:
        full_pcm, sr = pcmMonoF32FromWebmBytes(bytes(buf), target_sr)
    except Exception:
        del buf[before_len:]
        raise

    start_s = int(offset_ms * sr / 1000.0)
    end_s = int((offset_ms + window_duration_ms) * sr / 1000.0)
    n = int(full_pcm.shape[0])
    start_s = max(0, min(start_s, n))
    end_s = max(start_s, min(end_s, n))
    window_pcm = full_pcm[start_s:end_s].astype(np.float32, copy=False)
    return window_pcm, sr


try:
    from core.fusion_publisher import channel_audio, publish_json
except ImportError:
    channel_audio = None  # type: ignore[assignment]
    publish_json = None  # type: ignore[assignment]


def _decode_failure_degraded_reason(exc: BaseException) -> str:
    """Map decode-time exceptions to wire ``degradedReason`` (Redis :audio payload)."""
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", "") == "av":
        return "pyav_missing"
    if type(exc).__module__ == "av.error" and type(exc).__name__ == "InvalidDataError":
        return "decode_invalid_data"
    return "decode_error"


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
    audio_webm_bytes: bytes,
    settings: Settings | None = None,
    redis_client: RedisClient | None = None,
    text_pipe_send_end: Any | None = None,
) -> AudioChunkPayload:
    """
    Process one ingest window (**WebM container bytes** recommended).

    Live WebM is concatenated per ``meeting_id`` and decoded as one stream; PCM for
    this window is sliced by ``offset_ms`` + ``media_chunk_duration_ms``. A new
    stream at ``offset_ms == 0`` clears prior WebM bytes and rolling ASR context.

    Publishes JSON to Redis and optionally emits :class:`TextHandoffMessage`
    through ``text_pipe_send_end`` (writable ``multiprocessing.Connection``, one-way Pipe).
    """
    settings = settings or get_settings()
    r = redis_client or get_redis_client()

    log.debug("Processing audio chunk", meeting_id=meeting_id, offset_ms=offset_ms, size_bytes=len(audio_webm_bytes))

    if offset_ms == 0:
        clearMeetingBuffers(meeting_id)

    def publishPayload(p: AudioChunkPayload) -> None:
        if channel_audio is None or publish_json is None:
            log.warning("Redis pub/sub not imported, skipping publish", meeting_id=meeting_id)
            return
        chan = channel_audio(meeting_id)
        log.info("Publishing to Redis", channel=chan, meeting_id=meeting_id, offset_ms=offset_ms, has_transcript=bool(p.transcript))
        publish_json(r.raw, chan, payloadToRedisDict(p))

    def sendHandoff(transcript_val: str | None) -> None:
        if text_pipe_send_end is None:
            return
        hm = TextHandoffMessage(meeting_id=meeting_id, offset_ms=offset_ms, transcript=transcript_val)
        text_pipe_send_end.send(hm.model_dump(mode="json", by_alias=True))

    try:
        pcm, sr = _append_and_decode_live_webm_pcm(
            meeting_id,
            offset_ms,
            audio_webm_bytes,
            window_duration_ms=settings.media_chunk_duration_ms,
            target_sr=settings.target_sample_rate,
        )
    except Exception as e:
        degraded = _decode_failure_degraded_reason(e)
        log.error("Failed to decode audio bytes", meeting_id=meeting_id, offset_ms=offset_ms, exc_info=True)
        p = _stubPayload(meeting_id=meeting_id, offset_ms=offset_ms, reason=degraded)
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
            log.error("Whisper transcription failed", meeting_id=meeting_id, offset_ms=offset_ms, exc_info=True)
            transcript_val = None

    labels = None
    if settings.run_live_diarization_stub:
        from workers.audio.diarization import streamingWindowStub  # lazy: avoids pyannote at module load
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
