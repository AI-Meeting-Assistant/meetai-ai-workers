"""OpenAI Whisper — singleton model, transcript clipped to ingest window."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def _resampleLinearTo16k(samples: np.ndarray, src_sr: int) -> np.ndarray:
    import whisper as whisper_pkg

    target = whisper_pkg.audio.SAMPLE_RATE
    if src_sr == target or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.size / float(src_sr)
    n_out = max(1, int(math.ceil(duration * target)))
    x_old = np.linspace(0.0, 1.0, num=samples.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.float32)


_engine_singleton: "_WhisperEngine | None" = None


class _WhisperEngine:
    def __init__(self, *, model_size: str, language: str) -> None:
        import whisper
        import torch

        # Whisper uses float64 ops incompatible with MPS; CUDA only, else CPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = whisper.load_model(model_size, device=device)
        self._language = language

    def transcribeContext(
        self,
        audio_f32: np.ndarray,
        *,
        sample_rate: int,
    ) -> dict:
        import whisper

        if audio_f32.size == 0:
            return {"text": "", "segments": []}
        # Whisper expects 16 kHz float32 mono
        if sample_rate != whisper.audio.SAMPLE_RATE:
            audio_f32 = _resampleLinearTo16k(audio_f32.astype(np.float32), sample_rate)
            sample_rate = whisper.audio.SAMPLE_RATE
        audio_f32 = whisper.pad_or_trim(audio_f32)
        return self._model.transcribe(
            audio_f32,
            language=self._language,
            word_timestamps=True,
            fp16=False,
        )


def getWhisperEngine(*, model_size: str, language: str) -> _WhisperEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = _WhisperEngine(model_size=model_size, language=language)
    return _engine_singleton


def warmupWhisper(*, model_size: str, language: str) -> None:
    """Load model once (health check / startup)."""
    getWhisperEngine(model_size=model_size, language=language)


def transcribeWindowText(
    *,
    context_pcm: np.ndarray,
    sample_rate: int,
    abs_context_start_ms: int,
    window_start_ms: int,
    window_end_ms: int,
    model_size: str,
    language: str,
) -> str:
    """
    Run Whisper on ``context_pcm``; keep tokens/segments overlapping
    ``[window_start_ms, window_end_ms)`` in **meeting** coordinates.
    """
    if context_pcm.size == 0:
        return ""
    eng = getWhisperEngine(model_size=model_size, language=language)
    result = eng.transcribeContext(context_pcm, sample_rate=sample_rate)
    parts: List[str] = []
    ws, we = window_start_ms, window_end_ms
    for seg in result.get("segments", []):
        t0_ms = abs_context_start_ms + int(float(seg["start"]) * 1000)
        t1_ms = abs_context_start_ms + int(float(seg["end"]) * 1000)
        if t1_ms <= ws or t0_ms >= we:
            continue
        text_piece = seg.get("text", "").strip()
        if text_piece:
            parts.append(text_piece)
    if not parts and result.get("text"):
        return str(result["text"]).strip()
    return " ".join(parts).strip()


def collectWordSpansMeetingMs(
    *,
    context_pcm: np.ndarray,
    sample_rate: int,
    abs_context_start_ms: int,
    window_start_ms: int,
    window_end_ms: int,
    model_size: str,
    language: str,
) -> List[Tuple[int, int, str]]:
    """Debug helper: words with meeting-ms (start,end,text)."""
    if context_pcm.size == 0:
        return []
    eng = getWhisperEngine(model_size=model_size, language=language)
    result = eng.transcribeContext(context_pcm, sample_rate=sample_rate)
    out: List[Tuple[int, int, str]] = []
    ws, we = window_start_ms, window_end_ms
    for seg in result.get("segments", []):
        for w in seg.get("words") or []:
            t0_ms = abs_context_start_ms + int(float(w["start"]) * 1000)
            t1_ms = abs_context_start_ms + int(float(w["end"]) * 1000)
            if t1_ms <= ws or t0_ms >= we:
                continue
            wt = str(w.get("word", "")).strip()
            if wt:
                out.append((t0_ms, t1_ms, wt))
    return out
