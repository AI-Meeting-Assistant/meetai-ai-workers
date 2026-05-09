"""Warm-up loaders for models used by the audio worker."""

from __future__ import annotations

from config import get_settings
from workers.audio.asr import warmupWhisper


def warmUpLiveAudioModels() -> None:
    """Pre-load Whisper singleton when ``RUN_LIVE_ASR`` is enabled (§8)."""
    s = get_settings()
    if s.run_live_asr:
        warmupWhisper(model_size=s.whisper_model_size, language=s.whisper_language)
