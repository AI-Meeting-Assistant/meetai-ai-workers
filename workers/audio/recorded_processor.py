"""Orchestrate recorded-meeting upload: decode → batch pipeline → LLM → Redis publish."""

from __future__ import annotations

import asyncio
import shutil
import wave
from pathlib import Path
from typing import Any

import numpy as np

from workers.audio import _torch_compat  # noqa: F401 — before batch_runner / pyannote

from config import get_settings
from core.fusion_publisher import (
    channel_recorded_complete,
    channel_recorded_error,
    publish_json,
)
from infrastructure.redis_client import get_redis_client
from utils.logger import get_logger
from workers.audio.batch_runner import runBatchPipeline
from workers.audio.io_audio import pcmMonoF32FromWebmBytes
from workers.text.pipeline import analyzeFullTranscriptAdherence, summarizeFullTranscript

log = get_logger(__name__)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 samples as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


def convert_to_wav(source_path: Path, wav_path: Path, target_sr: int) -> int:
    """Decode any supported container to mono 16 kHz WAV; return duration in ms."""
    data = source_path.read_bytes()
    samples, sr = pcmMonoF32FromWebmBytes(data, target_sr)
    _write_wav(wav_path, samples, sr)
    duration_ms = int(len(samples) / sr * 1000) if sr > 0 else 0
    return duration_ms


def _build_final_payload(
    meeting_id: str,
    analysis: Any,
    duration_ms: int,
    adherence: dict,
    ai_summary: str,
) -> dict[str, Any]:
    return {
        "meetingId": meeting_id,
        "offsetMs": 0,
        "aiSummary": ai_summary or None,
        "audio": {
            "vadSpeechMs": analysis.vad_speech_ms,
            "vadSilenceMs": analysis.vad_silence_ms,
            "vadSpeechRatioPercent": analysis.vad_speech_ratio_percent,
            "transcript": analysis.transcript or None,
            "speakerLabelsWindow": analysis.speaker_labels_window or None,
            "speakerTalkMs": analysis.speaker_talk_ms or None,
            "speakerTalkRatioPercent": analysis.speaker_talk_ratio_percent or None,
        },
        "video": None,
        "recorded": {
            "durationMs": duration_ms or analysis.duration_ms,
            "transcriptLines": analysis.transcript_lines,
            "speakers": analysis.speakers,
            "adherence": adherence,
        },
    }


def _cleanup_upload_dir(upload_dir: Path, keep: bool) -> None:
    if keep:
        return
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)


def _publish_error(meeting_id: str, reason: str) -> None:
    try:
        r = get_redis_client()
        publish_json(
            r.raw,
            channel_recorded_error(meeting_id),
            {"meetingId": meeting_id, "reason": reason},
        )
    except Exception:
        log.error("Failed to publish recorded-error", meeting_id=meeting_id, exc_info=True)


def process_recorded_meeting_sync(
    meeting_id: str,
    source_path: str,
    title: str,
    agenda: str,
) -> None:
    """Synchronous entry for BackgroundTasks / thread pool."""
    asyncio.run(process_recorded_meeting(meeting_id, source_path, title, agenda))


async def process_recorded_meeting(
    meeting_id: str,
    source_path: str,
    title: str,
    agenda: str,
) -> None:
    settings = get_settings()
    upload_root = Path(settings.upload_dir) / meeting_id
    wav_path = upload_root / "audio.wav"
    source = Path(source_path)

    try:
        log.info("Converting recorded upload to WAV", meeting_id=meeting_id, source=str(source))
        duration_ms = convert_to_wav(source, wav_path, settings.target_sample_rate)

        log.info("Running batch pipeline", meeting_id=meeting_id)
        analysis = runBatchPipeline(
            str(wav_path),
            run_vad=True,
            run_diarization=True,
            run_transcription=True,
            run_refinement=False,
        )
        if duration_ms > 0:
            analysis.duration_ms = duration_ms

        transcript = analysis.transcript or ""
        adherence: dict = {"score": None, "onTopic": None, "reason": None}
        ai_summary = ""

        if transcript.strip():
            log.info("Running full-transcript LLM analysis", meeting_id=meeting_id)
            adherence, ai_summary = await asyncio.gather(
                analyzeFullTranscriptAdherence(transcript, title, agenda),
                summarizeFullTranscript(transcript, title, agenda),
            )

        payload = _build_final_payload(meeting_id, analysis, duration_ms, adherence, ai_summary)

        r = get_redis_client()
        publish_json(r.raw, channel_recorded_complete(meeting_id), payload)
        log.info("Recorded meeting processing complete", meeting_id=meeting_id)

        _cleanup_upload_dir(upload_root, settings.recorded_keep_files)
    except Exception as exc:
        log.error("Recorded meeting processing failed", meeting_id=meeting_id, exc_info=True)
        _publish_error(meeting_id, str(exc))
        if not settings.recorded_keep_files:
            _cleanup_upload_dir(upload_root, False)
