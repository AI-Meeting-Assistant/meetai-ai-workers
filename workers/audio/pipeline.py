"""Live ingest path: WAV bytes → Redis ``meeting:{id}:audio`` + optional text handoff Pipe."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import Settings, get_settings
from infrastructure.redis_client import RedisClient, get_redis_client

from workers.audio.asr import transcribeWindowText
from workers.audio.context_buffer import clearMeetingBuffers, getBuffer
from workers.audio.io_audio import pcmMonoF32FromWebmBytes
from workers.audio.webm_ebml_clusters import (
    consume_complete_segment_children,
    find_segment_scan_window,
)
from workers.audio.webm_ebml_prefix import webm_init_prefix_bytes
from workers.audio.schemas import (
    AudioChunkPayload,
    TextHandoffMessage,
    payloadToRedisDict,
)
from workers.audio.vad_window import computeEnergyVadMetrics
from utils.logger import get_logger

log = get_logger(__name__)

# Live MediaRecorder: concatenate WebM per meeting; decode new Cluster elements only (incremental PCM).
_MAX_LIVE_WEBM_ACCUM_MEETINGS = 200
_live_webm_state_by_meeting: OrderedDict[str, "_LiveWebmMeetingState"] = OrderedDict()


@dataclass
class _LiveWebmMeetingState:
    buf: bytearray = field(default_factory=bytearray)
    init_prefix: bytes | None = None
    incremental_mode: bool = False
    next_parse_pos: int = 0
    pcm_chunks: list[np.ndarray] = field(default_factory=list)
    cum_lens: list[int] = field(default_factory=lambda: [0])
    sample_rate: int = 0


def clear_live_webm_accum(meeting_id: str) -> None:
    """Drop live WebM / incremental decode state for one meeting (e.g. stream ended)."""
    _live_webm_state_by_meeting.pop(meeting_id, None)


def clear_live_webm_accum_all() -> None:
    _live_webm_state_by_meeting.clear()


def _touch_live_webm_accum_lru(meeting_id: str) -> None:
    _live_webm_state_by_meeting.move_to_end(meeting_id)
    while len(_live_webm_state_by_meeting) > _MAX_LIVE_WEBM_ACCUM_MEETINGS:
        _live_webm_state_by_meeting.popitem(last=False)


def _append_pcm_chunk(st: _LiveWebmMeetingState, arr: np.ndarray) -> None:
    st.pcm_chunks.append(arr.astype(np.float32, copy=False))
    st.cum_lens.append(st.cum_lens[-1] + int(arr.shape[0]))


def _pcm_slice_from_chunks(st: _LiveWebmMeetingState, start_s: int, end_s: int) -> np.ndarray:
    if start_s >= end_s or not st.pcm_chunks:
        return np.zeros(0, dtype=np.float32)
    cl = st.cum_lens
    n = cl[-1]
    start_s = max(0, min(start_s, n))
    end_s = max(start_s, min(end_s, n))
    if start_s >= end_s:
        return np.zeros(0, dtype=np.float32)
    i = bisect.bisect_right(cl, start_s) - 1
    j = bisect.bisect_right(cl, end_s) - 1
    parts: list[np.ndarray] = []
    for k in range(i, j + 1):
        cs, ce = cl[k], cl[k + 1]
        a = start_s - cs
        b = end_s - cs
        a = max(0, a)
        b = min(ce - cs, b)
        if b > a:
            parts.append(st.pcm_chunks[k][a:b])
    if not parts:
        return np.zeros(0, dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def _append_and_decode_live_webm_pcm(
    meeting_id: str,
    offset_ms: int,
    audio_webm_bytes: bytes,
    *,
    window_duration_ms: int,
    target_sr: int,
) -> tuple[Any, int]:
    """
    Append WebM bytes, decode **new complete Cluster** elements only (after init prefix is known),
    and return mono float32 PCM for ``[offset_ms, offset_ms + window_duration_ms)``.
    Falls back to full-container decode until the first Segment-level Cluster is visible.
    """
    if offset_ms == 0:
        st = _LiveWebmMeetingState()
        _live_webm_state_by_meeting[meeting_id] = st
    else:
        st = _live_webm_state_by_meeting.get(meeting_id)
        if st is None:
            st = _LiveWebmMeetingState()
            _live_webm_state_by_meeting[meeting_id] = st

    buf = st.buf
    before_len = len(buf)
    snap_next = st.next_parse_pos
    snap_inc = st.incremental_mode
    snap_init = st.init_prefix
    snap_chunk_n = len(st.pcm_chunks)
    snap_cum = list(st.cum_lens)

    buf.extend(audio_webm_bytes)
    _touch_live_webm_accum_lru(meeting_id)

    def _rollback() -> None:
        del buf[before_len:]
        st.next_parse_pos = snap_next
        st.incremental_mode = snap_inc
        st.init_prefix = snap_init
        while len(st.pcm_chunks) > snap_chunk_n:
            st.pcm_chunks.pop()
        st.cum_lens = list(snap_cum)

    try:
        raw = bytes(buf)
        prefix = webm_init_prefix_bytes(raw)
        seg = find_segment_scan_window(raw)

        def _slice_full(full_pcm: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
            n = int(full_pcm.shape[0])
            ss = int(offset_ms * sr / 1000.0)
            es = int((offset_ms + window_duration_ms) * sr / 1000.0)
            ss = max(0, min(ss, n))
            es = max(ss, min(es, n))
            return full_pcm[ss:es].astype(np.float32, copy=False), sr

        if st.incremental_mode and st.init_prefix is not None:
            if seg is None:
                full_pcm, sr = pcmMonoF32FromWebmBytes(raw, target_sr)
                if not st.sample_rate:
                    st.sample_rate = sr
                window, sr2 = _slice_full(full_pcm, sr)
                return window, sr2

            seg_body, seg_lim = seg
            new_spans, next_p = consume_complete_segment_children(
                raw,
                seg_body=seg_body,
                seg_lim=seg_lim,
                parse_from=st.next_parse_pos,
            )
            for cs, ce in new_spans:
                blob = st.init_prefix + raw[cs:ce]
                pcm_c, sr = pcmMonoF32FromWebmBytes(blob, target_sr)
                if st.sample_rate and sr != st.sample_rate:
                    log.warning(
                        "Sample rate changed mid-meeting; keeping first rate",
                        meeting_id=meeting_id,
                        first_sr=st.sample_rate,
                        new_sr=sr,
                    )
                    sr = st.sample_rate
                if not st.sample_rate:
                    st.sample_rate = sr
                _append_pcm_chunk(st, pcm_c)
            st.next_parse_pos = next_p
            sr_use = st.sample_rate or target_sr
            ss = int(offset_ms * sr_use / 1000.0)
            es = int((offset_ms + window_duration_ms) * sr_use / 1000.0)
            window = _pcm_slice_from_chunks(st, ss, es)
            if window.size == 0 and len(raw) > 0:
                full_pcm, sr = pcmMonoF32FromWebmBytes(raw, target_sr)
                if not st.sample_rate:
                    st.sample_rate = sr
                return _slice_full(full_pcm, sr)[0], sr
            return window, sr_use

        if not st.incremental_mode and prefix and seg:
            if st.init_prefix is None:
                st.init_prefix = prefix
                st.next_parse_pos = len(prefix)
                st.pcm_chunks.clear()
                st.cum_lens = [0]
            seg_body, seg_lim = seg
            new_spans, next_p = consume_complete_segment_children(
                raw,
                seg_body=seg_body,
                seg_lim=seg_lim,
                parse_from=st.next_parse_pos,
            )
            for cs, ce in new_spans:
                blob = st.init_prefix + raw[cs:ce]
                pcm_c, sr = pcmMonoF32FromWebmBytes(blob, target_sr)
                if not st.sample_rate:
                    st.sample_rate = sr
                _append_pcm_chunk(st, pcm_c)
            st.next_parse_pos = next_p
            if st.cum_lens[-1] > 0:
                st.incremental_mode = True
                sr_use = st.sample_rate
                ss = int(offset_ms * sr_use / 1000.0)
                es = int((offset_ms + window_duration_ms) * sr_use / 1000.0)
                return _pcm_slice_from_chunks(st, ss, es), sr_use

        full_pcm, sr = pcmMonoF32FromWebmBytes(raw, target_sr)
        if not st.sample_rate:
            st.sample_rate = sr
        window, sr_out = _slice_full(full_pcm, sr)
        return window, sr_out

    except Exception:
        _rollback()
        raise


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

    Live WebM is concatenated per ``meeting_id``. After the first Segment-level Cluster
    is visible, **new Cluster elements only** are decoded (incremental); PCM for this
    window is sliced from a running chunk list. Until then, the full buffer is decoded
    once per chunk (bootstrap). ``offset_ms == 0`` clears prior WebM state and rolling
    ASR context.

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
        pcm, sr = pcmMonoF32FromWebmBytes(audio_webm_bytes, settings.target_sample_rate, meeting_id=meeting_id)
    except Exception:
        log.error("Failed to decode audio bytes", meeting_id=meeting_id, offset_ms=offset_ms, exc_info=True)
        p = _stubPayload(meeting_id=meeting_id, offset_ms=offset_ms, reason=degraded)
        publishPayload(p)
        sendHandoff(None)
        return p

    # Derive window duration from actual decoded PCM length, not from config.
    chunk_duration_ms = int(len(pcm) / sr * 1000) if sr > 0 else settings.media_chunk_duration_ms
    window_end_ms = offset_ms + chunk_duration_ms

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
            meeting_id, offset_ms=offset_ms, duration_ms=chunk_duration_ms
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
