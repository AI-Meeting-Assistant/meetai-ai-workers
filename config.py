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


@dataclass(frozen=True)
class Settings:
    media_chunk_duration_ms: int
    asr_context_window_ms: int
    text_analysis_interval_ms: int
    text_transcript_ring_buffer_slots: int
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
    vad_speech_ratio_min_for_asr: float

    @staticmethod
    def load() -> "Settings":
        d = _int_env("MEDIA_CHUNK_DURATION_MS", 2000)
        text_iv = _int_env("TEXT_ANALYSIS_INTERVAL_MS", 30000)
        slots = _optional_int_env("TEXT_TRANSCRIPT_RING_BUFFER_SLOTS")
        if slots is None:
            slots = max(1, math.ceil(text_iv / d))
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        return Settings(
            media_chunk_duration_ms=d,
            asr_context_window_ms=_int_env("ASR_CONTEXT_WINDOW_MS", 10000),
            text_analysis_interval_ms=text_iv,
            text_transcript_ring_buffer_slots=slots,
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
            vad_speech_ratio_min_for_asr=_float_env("VAD_SPEECH_RATIO_MIN_ASR", 10.0),
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
