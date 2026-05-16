"""Live ingest path: WAV bytes → Redis ``meeting:{id}:audio`` + optional text handoff Pipe."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import Settings, get_settings
from infrastructure.redis_client import RedisClient, get_redis_client

from workers.audio.asr import filterFillerWords, joinWordsToText, transcribeWindowTextAndWords
from workers.audio.context_buffer import clearMeetingBuffers
from workers.audio.io_audio import pcmMonoF32FromWebmBytes
from workers.audio.webm_ebml_clusters import (
    consume_complete_segment_children,
    find_segment_scan_window,
)
from workers.audio.webm_ebml_prefix import webm_init_prefix_bytes
from workers.audio.schemas import (
    AudioChunkPayload,
    TextHandoffMessage,
    TranscriptLine,
    payloadToRedisDict,
)
from workers.audio.speaker_embedder import embedPcmMono
from workers.audio.speaker_registry import getOrCreateRegistry
from workers.audio.speaker_transcript_align import (
    aggregateSpeakerTalkMs,
    assignWordsToSpeakers,
    capSpeakerTalkMsToWindow,
    clipIntervalMs,
    dominantSpeaker,
    mergeConsecutiveSpeakerText,
    windowSpeakerRatioPercent,
)
from workers.audio.vad_segments import (
    mergeAdjacentSpeechSegmentRanges,
    vadSpeechSegmentSampleRanges,
)
from workers.audio.vad_window import computeWindowVadMetrics
from utils.logger import get_logger

log = get_logger(__name__)

# Live MediaRecorder: concatenate WebM per meeting; decode new Cluster elements only (incremental PCM).
_MAX_LIVE_WEBM_ACCUM_MEETINGS = 200
_live_webm_state_by_meeting: OrderedDict[str, "_LiveWebmMeetingState"] = OrderedDict()

# Last ``speaker_context_prefix_ms`` of decoded **window** PCM for speaker VAD/embedding continuity.
_analysis_prefix_tail_by_meeting: OrderedDict[str, np.ndarray] = OrderedDict()
_MAX_PREFIX_TAIL_MEETINGS = 200

# Avoid re-publishing Whisper words from overlapping ASR context across windows.
_last_published_word_end_ms: OrderedDict[str, float] = OrderedDict()
_MAX_WORD_DEDUP_MEETINGS = 200


def _slice_pcm_to_ingest_window(
    pcm: np.ndarray,
    *,
    sample_rate: int,
    stride_ms: int,
) -> tuple[np.ndarray, int]:
    """Keep the trailing ``stride_ms`` of decoded PCM (matches this ingest offset's audio)."""
    if pcm.size == 0 or sample_rate <= 0 or stride_ms <= 0:
        return pcm, 0
    max_samples = int(sample_rate * stride_ms / 1000.0)
    decoded_ms = int(pcm.shape[0] * 1000 / sample_rate)
    if pcm.shape[0] > max_samples:
        pcm = pcm[-max_samples:].astype(np.float32, copy=False)
    used_ms = int(pcm.shape[0] * 1000 / sample_rate)
    return pcm, decoded_ms if decoded_ms > used_ms else 0


def _filter_words_not_yet_published(
    meeting_id: str,
    words: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    if not words:
        return []
    last_end = _last_published_word_end_ms.get(meeting_id, -1.0)
    fresh: list[tuple[int, int, str]] = []
    for t0, t1, token in words:
        if float(t1) <= last_end + 30.0:
            continue
        fresh.append((t0, t1, token))
    if fresh:
        _last_published_word_end_ms[meeting_id] = max(float(t1) for _, t1, _ in fresh)
        _last_published_word_end_ms.move_to_end(meeting_id)
        while len(_last_published_word_end_ms) > _MAX_WORD_DEDUP_MEETINGS:
            _last_published_word_end_ms.popitem(last=False)
    return fresh


def clearSpeakerAnalysisArtifacts(meeting_id: str | None) -> None:
    """Reset speaker centroids + cross-chunk tail for one meeting (or all)."""
    from utils.webm_init_cache import evict as evict_webm_init

    from workers.audio.speaker_registry import clearAllSpeakerRegistries, clearSpeakerRegistry

    if meeting_id is None:
        _analysis_prefix_tail_by_meeting.clear()
        _last_published_word_end_ms.clear()
        clearAllSpeakerRegistries()
        return
    _analysis_prefix_tail_by_meeting.pop(meeting_id, None)
    _last_published_word_end_ms.pop(meeting_id, None)
    evict_webm_init(meeting_id)
    clearSpeakerRegistry(meeting_id)


def _touch_prefix_tail_lru(meeting_id: str) -> None:
    _analysis_prefix_tail_by_meeting.move_to_end(meeting_id)
    while len(_analysis_prefix_tail_by_meeting) > _MAX_PREFIX_TAIL_MEETINGS:
        _analysis_prefix_tail_by_meeting.popitem(last=False)


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
        transcript_lines=None,
        vad_speech_ms=None,
        vad_silence_ms=None,
        vad_speech_ratio_percent=None,
        speaker_talk_ms=None,
        speaker_talk_ratio_percent=None,
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
        p = _stubPayload(meeting_id=meeting_id, offset_ms=offset_ms, reason="decode_failed")
        publishPayload(p)
        sendHandoff(None)
        return p

    stride_ms = settings.media_chunk_duration_ms
    pcm, decoded_ms_extra = _slice_pcm_to_ingest_window(pcm, sample_rate=sr, stride_ms=stride_ms)
    if decoded_ms_extra > stride_ms:
        log.debug(
            "Using trailing PCM for ingest window",
            meeting_id=meeting_id,
            offset_ms=offset_ms,
            decoded_ms=decoded_ms_extra,
            stride_ms=stride_ms,
        )
    chunk_duration_ms = int(len(pcm) / sr * 1000) if sr > 0 else stride_ms
    window_end_ms = offset_ms + chunk_duration_ms

    vad_metrics = None
    if settings.run_live_vad_energy:
        vad_metrics = computeWindowVadMetrics(
            pcm,
            sample_rate=sr,
            speech_rms_threshold=settings.vad_speech_rms_threshold,
        )

    # Skip ASR if VAD ran and speech ratio is below threshold — prevents Whisper hallucinations on silence.
    _skip_asr = (
        vad_metrics is not None
        and vad_metrics.speech_ratio_percent < settings.vad_speech_ratio_min_asr
    )

    whisper_words: list[tuple[int, int, str]] = []
    transcript_val: str | None = None
    if settings.run_live_asr and not _skip_asr:
        try:
            text_raw, whisper_words = transcribeWindowTextAndWords(
                context_pcm=pcm,
                sample_rate=sr,
                abs_context_start_ms=offset_ms,
                window_start_ms=offset_ms,
                window_end_ms=window_end_ms,
                model_size=settings.whisper_model_size,
                language=settings.whisper_language,
            )
            whisper_words = filterFillerWords(list(whisper_words) if whisper_words else [])
            whisper_words = _filter_words_not_yet_published(meeting_id, whisper_words)
            transcript_val = joinWordsToText(whisper_words) or None
            if not transcript_val and text_raw:
                transcript_val = (text_raw or "").strip() or None
        except Exception:
            log.error("Whisper transcription failed", meeting_id=meeting_id, offset_ms=offset_ms, exc_info=True)
            transcript_val = None
            whisper_words = []
    elif _skip_asr:
        log.debug(
            "ASR skipped: speech ratio below threshold",
            meeting_id=meeting_id,
            offset_ms=offset_ms,
            speech_ratio=round(vad_metrics.speech_ratio_percent, 1),
            threshold=settings.vad_speech_ratio_min_asr,
        )

    transcript_lines_models: list[TranscriptLine] | None = None
    speaker_talk_ms_out: dict[str, float] | None = None
    speaker_ratio_pct_out: dict[str, float] | None = None

    if settings.run_live_speaker_id and sr > 0:
        try:
            # Talk-time + VAD segments: this ingest window PCM only (prefix used only for embed hint).
            segs = vadSpeechSegmentSampleRanges(
                pcm,
                sample_rate=sr,
                speech_rms_threshold=settings.vad_speech_rms_threshold,
            )
            gap_merge_samples = max(1, int(sr * 0.35))  # bridge short pauses within one utterance
            segs = mergeAdjacentSpeechSegmentRanges(segs, max_gap_samples=gap_merge_samples)

            prefix_tail = _analysis_prefix_tail_by_meeting.get(meeting_id)

            reg = getOrCreateRegistry(
                meeting_id,
                max_speakers=settings.speaker_max_canonical,
                cos_threshold=settings.speaker_match_cos_threshold,
                ema_alpha=settings.speaker_centroid_ema_alpha,
            )
            reg.merge_redundant_centroids(merge_cos_threshold=0.72)

            min_seg_samples = max(1, int(sr * settings.speaker_min_segment_ms / 1000.0))
            min_new_samples = max(min_seg_samples, int(sr * settings.speaker_new_identity_min_ms / 1000.0))

            speaker_talk_clips: list[tuple[float, float, str]] = []
            labeled_intervals: list[tuple[float, float, str]] = []
            win_lo = float(offset_ms)
            win_hi = float(window_end_ms)
            for sa, sb in segs:
                seg_pcm = pcm[sa:sb]
                if seg_pcm.size == 0:
                    continue
                seg_len = int(sb - sa)
                t0_abs = win_lo + (float(sa) / float(sr)) * 1000.0
                t1_abs = win_lo + (float(sb) / float(sr)) * 1000.0
                partial = aggregateSpeakerTalkMs(speaker_talk_clips)
                embed_pcm = seg_pcm
                if sa == 0 and prefix_tail is not None and prefix_tail.size > 0:
                    embed_pcm = np.concatenate(
                        [prefix_tail.astype(np.float32, copy=False), seg_pcm],
                        dtype=np.float32,
                    )
                emb = embedPcmMono(embed_pcm, sample_rate=sr)
                if emb is not None:
                    if seg_len < min_seg_samples:
                        speaker_label = reg.assign_and_update(emb, allow_new_speaker=False)
                    else:
                        allow_new = seg_len >= min_new_samples
                        speaker_label = reg.assign_and_update(emb, allow_new_speaker=allow_new)
                else:
                    speaker_label = dominantSpeaker(partial) or "Speaker 1"
                clipped = clipIntervalMs(t0_abs, t1_abs, win_lo, win_hi)
                if clipped is not None:
                    lo, hi = clipped
                    speaker_talk_clips.append((lo, hi, speaker_label))
                    labeled_intervals.append((lo, hi, speaker_label))

            reg.merge_redundant_centroids(merge_cos_threshold=0.72)

            speaker_talk_ms_out = aggregateSpeakerTalkMs(speaker_talk_clips)
            if speaker_talk_ms_out:
                speaker_talk_ms_out = capSpeakerTalkMsToWindow(
                    speaker_talk_ms_out,
                    float(chunk_duration_ms),
                )
            if not speaker_talk_ms_out:
                speaker_talk_ms_out = None
                speaker_ratio_pct_out = None
            else:
                speaker_ratio_pct_out = windowSpeakerRatioPercent(speaker_talk_ms_out)

            fb = dominantSpeaker(speaker_talk_ms_out) if speaker_talk_ms_out else "Speaker 1"
            fb = fb or "Speaker 1"

            if transcript_val and settings.run_live_asr and not _skip_asr:
                if whisper_words:
                    spans = assignWordsToSpeakers(
                        list(whisper_words),
                        labeled_intervals,
                        fallback_speaker=fb,
                        window_lo_ms=float(offset_ms),
                        window_hi_ms=float(window_end_ms),
                    )
                    merged = mergeConsecutiveSpeakerText(spans)
                    transcript_lines_models = [
                        TranscriptLine(speaker=m["speaker"], text=m["text"]) for m in merged
                    ]
                    transcript_val = "\n".join(f"{m.speaker}: {m.text}" for m in transcript_lines_models)
                else:
                    stripped = transcript_val.strip()
                    transcript_lines_models = [TranscriptLine(speaker=fb, text=stripped)]
                    transcript_val = f"{fb}: {stripped}"

        except Exception:
            log.error("Speaker ID / attribution failed", meeting_id=meeting_id, offset_ms=offset_ms, exc_info=True)
            transcript_lines_models = None

    labels = None
    if settings.run_live_diarization_stub and not settings.run_live_speaker_id:
        from workers.audio.diarization import streamingWindowStub  # lazy: avoids pyannote at module load

        labels = streamingWindowStub(meeting_id, offset_ms=offset_ms, duration_ms=chunk_duration_ms)

    payload = AudioChunkPayload(
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        transcript=transcript_val,
        transcript_lines=transcript_lines_models,
        vad_speech_ms=float(vad_metrics.speech_ms) if vad_metrics else None,
        vad_silence_ms=float(vad_metrics.silence_ms) if vad_metrics else None,
        vad_speech_ratio_percent=float(vad_metrics.speech_ratio_percent) if vad_metrics else None,
        speaker_talk_ms=speaker_talk_ms_out,
        speaker_talk_ratio_percent=speaker_ratio_pct_out,
        speaker_labels_window=labels,
        focus_score=None,
        payload=None,
    )
    publishPayload(payload)
    sendHandoff(transcript_val)

    if settings.run_live_speaker_id and sr > 0 and settings.speaker_context_prefix_ms > 0 and pcm.size > 0:
        ms = settings.speaker_context_prefix_ms
        max_tail_samples = max(1, min(int(sr * ms / 1000), int(pcm.shape[0])))
        _analysis_prefix_tail_by_meeting[meeting_id] = np.array(
            pcm[-max_tail_samples:], copy=True, dtype=np.float32
        )
        _touch_prefix_tail_lru(meeting_id)

    return payload
