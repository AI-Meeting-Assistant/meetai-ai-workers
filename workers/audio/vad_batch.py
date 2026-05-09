"""Pyannote VAD pipeline for **file / batch** analysis (offline)."""

from __future__ import annotations

import os

from pyannote.audio import Pipeline


class VoiceActivityDetector:
    def __init__(self, use_auth_token: str | None = None):
        if use_auth_token is None:
            use_auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if use_auth_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", use_auth_token)
            os.environ.setdefault("HF_TOKEN", use_auth_token)
        try:
            self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
        except TypeError:
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection", token=use_auth_token
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection", use_auth_token=use_auth_token
                )

    def runVoiceActivityDetection(self, audio_file: str) -> dict:
        vad = self.pipeline(audio_file)

        speech_time = 0.0
        silence_time = 0.0
        last_end = 0.0

        for segment in vad.get_timeline().support():
            speech_time += segment.duration
            silence_time += max(0, segment.start - last_end)
            last_end = segment.end

        total_time = speech_time + silence_time
        speech_ratio = (speech_time / total_time) * 100 if total_time > 0 else 0.0
        silence_ratio = (silence_time / total_time) * 100 if total_time > 0 else 0.0

        return {
            "speech_time": speech_time,
            "silence_time": silence_time,
            "total_time": total_time,
            "speech_ratio": speech_ratio,
            "silence_ratio": silence_ratio,
        }
