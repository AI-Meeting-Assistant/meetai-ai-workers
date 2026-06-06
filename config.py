"""Load tunable defaults from environment (see PYTHON_WORKERS_IMPLEMENTATION.md §12)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _optional_int_env(key: str) -> int | None:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw not in ("0", "false", "False")


@dataclass(frozen=True)
class Settings:
    media_chunk_duration_ms: int
    asr_context_window_ms: int
    text_analysis_interval_ms: int
    text_transcript_ring_buffer_slots: int
    ollama_url: str
    ollama_model: str
    redis_url: str
    target_sample_rate: int
    whisper_model_size: str
    whisper_language: str
    hf_token: str | None
    """Hugging Face token for gated pyannote models (batch/offline)."""
    run_live_vad_energy: bool
    run_live_asr: bool
    run_live_diarization_stub: bool
    vad_energy_rms_quantile_for_speech: float
    """Deprecated for live path (quantile pinned ~65% speech); kept for env compat."""
    vad_speech_rms_threshold: float
    vad_speech_ratio_min_asr: float

    #: ECAPA embedding + canonical Speaker 1…N assignment (live)
    run_live_speaker_id: bool
    speaker_max_canonical: int
    speaker_match_cos_threshold: float
    speaker_centroid_ema_alpha: float
    speaker_min_segment_ms: float
    """Minimum VAD speech duration before opening a *new* canonical speaker."""
    speaker_new_identity_min_ms: float
    speaker_context_prefix_ms: int

    upload_dir: str
    max_upload_size_mb: int
    recorded_keep_files: bool

    #: Post-process LLM adherence scores (see workers/text/adherence.py)
    adherence_on_topic_fit_threshold: float

    @staticmethod
    def load() -> "Settings":
        d = _int_env("MEDIA_CHUNK_DURATION_MS", 6000)
        text_iv = _int_env("TEXT_ANALYSIS_INTERVAL_MS", 30000)
        slots = _optional_int_env("TEXT_TRANSCRIPT_RING_BUFFER_SLOTS")
        if slots is None:
            slots = max(1, math.ceil(text_iv / d))
        else:
            slots = max(1, slots)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        return Settings(
            media_chunk_duration_ms=d,
            asr_context_window_ms=_int_env("ASR_CONTEXT_WINDOW_MS", 10000),
            text_analysis_interval_ms=text_iv,
            text_transcript_ring_buffer_slots=slots,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            redis_url=redis_url,
            target_sample_rate=_int_env("TARGET_SAMPLE_RATE", 16000),
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "tr"),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
            run_live_vad_energy=os.getenv("RUN_LIVE_VAD_ENERGY", "1") not in ("0", "false", "False"),
            run_live_asr=os.getenv("RUN_LIVE_ASR", "1") not in ("0", "false", "False"),
            run_live_diarization_stub=os.getenv("RUN_LIVE_DIARIZATION_STUB", "1")
            not in ("0", "false", "False"),
            vad_energy_rms_quantile_for_speech=_float_env("VAD_ENERGY_RMS_QUANTILE", 0.35),
            vad_speech_rms_threshold=_float_env("VAD_SPEECH_RMS_THRESHOLD", 0.022),
            vad_speech_ratio_min_asr=_float_env("VAD_SPEECH_RATIO_MIN_ASR", 10.0),
            run_live_speaker_id=os.getenv("RUN_LIVE_SPEAKER_ID", "1") not in ("0", "false", "False"),
            speaker_max_canonical=max(2, min(8, _int_env("SPEAKER_MAX_CANONICAL", 8))),
            speaker_match_cos_threshold=_float_env("SPEAKER_MATCH_COS_THRESHOLD", 0.25),
            speaker_centroid_ema_alpha=_float_env("SPEAKER_CENTROID_EMA_ALPHA", 0.35),
            speaker_min_segment_ms=_float_env("SPEAKER_MIN_SEGMENT_MS", 400.0),
            speaker_new_identity_min_ms=_float_env("SPEAKER_NEW_IDENTITY_MIN_MS", 700.0),
            speaker_context_prefix_ms=_int_env("SPEAKER_CONTEXT_PREFIX_MS", 320),
            upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
            max_upload_size_mb=_int_env("MAX_UPLOAD_SIZE_MB", 500),
            recorded_keep_files=os.getenv("RECORDED_KEEP_FILES", "0") in ("1", "true", "True"),
            adherence_on_topic_fit_threshold=_float_env("ADHERENCE_ON_TOPIC_FIT_THRESHOLD", 0.4),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings.load()
    return _settings
