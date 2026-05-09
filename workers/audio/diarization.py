"""
Speaker diarization: **batch** path uses pyannote file pipeline; **streaming** path is stub.

Heavy diarization should not run every ``D`` ms; use batch_runner or extend with a periodic policy later.
"""

from __future__ import annotations

from typing import Any

from collections import defaultdict

import torch

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    PyannotePipeline = None  # type: ignore[misc, assignment]


class SpeakerDiarizer:
    """File-based speaker diarization (offline / batch_runner)."""

    def __init__(self, device: str | None = None, hf_token: str | None = None):
        import os

        if PyannotePipeline is None:
            raise RuntimeError("pyannote.audio is not installed")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)

        try:
            self.pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization")
        except TypeError:
            try:
                self.pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization", token=hf_token
                )
            except TypeError:
                self.pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization", use_auth_token=hf_token
                )
        self.pipeline.to(torch.device(device))

    def runSpeakerDiarization(self, audio_file: str) -> dict[str, Any]:
        diarization = self.pipeline(audio_file)
        speaker_times: defaultdict[str, float] = defaultdict(float)
        speaker_segments: list[dict[str, Any]] = []

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_times[speaker] += segment.duration
            speaker_segments.append(
                {"start": segment.start, "end": segment.end, "speaker": speaker}
            )

        total_time = sum(speaker_times.values())
        return {
            "speaker_times": dict(speaker_times),
            "speaker_segments": speaker_segments,
            "total_speech_time": total_time,
        }


def streamingWindowStub(
    meeting_id: str,
    offset_ms: int,
    duration_ms: int,
) -> list[dict[str, Any]]:
    """
    Phase-3 streaming placeholder until a buffered diarization policy exists.

    Returns a minimal label list for telemetry / fusion contract tests only.
    """
    _ = (meeting_id, duration_ms)
    return [{"speaker": "UNKNOWN", "offsetMs": offset_ms, "confidence": None}]
