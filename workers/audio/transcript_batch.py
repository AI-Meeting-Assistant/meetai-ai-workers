"""File-based Whisper + speaker alignment helpers for **batch_runner**."""

from __future__ import annotations

from typing import Any

import torch
import whisper


class Transcriber:
    """Whisper on a WAV path (offline). Separate from singleton live ASR in ``asr.py``."""

    def __init__(
        self, model_size: str = "medium", device: str | None = None, language: str = "tr"
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)
        self.language = language

    def runTranscription(self, audio_file: str) -> dict[str, Any]:
        return self.model.transcribe(
            audio_file, language=self.language, word_timestamps=True, fp16=False
        )

    def alignWithSpeakers(self, transcription: dict, speaker_segments: list) -> list:
        results: list[dict[str, Any]] = []

        for segment in speaker_segments:
            start = segment["start"]
            end = segment["end"]
            speaker = segment["speaker"]

            words: list[str] = []
            for seg in transcription["segments"]:
                for word in seg.get("words", []):
                    if start <= word["start"] <= end:
                        words.append(word["word"])

            if words:
                results.append(
                    {"speaker": speaker, "start": start, "end": end, "text": " ".join(words)}
                )

        return results
