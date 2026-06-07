"""
Offline / batch WAV analysis (original ``AudioWorker/app.py`` flow).

Separate from live ``processLiveChunk`` (PYTHON_WORKERS_IMPLEMENTATION §2 / plan).

Environment toggles::
    BATCH_RUN_VAD=1  BATCH_RUN_DIARIZATION=0  BATCH_RUN_TRANSCRIPTION=0
    BATCH_RUN_REFINEMENT=0  BATCH_AUDIO_FILE=audio/meeting_clean.wav
"""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("SB_DISABLE_K2", "1")

from . import _torch_compat  # noqa: F401, E402 — before torch / pyannote checkpoint loads

import torch  # noqa: E402


@dataclass
class RecordedAnalysisResult:
    """Structured output from ``runBatchPipeline`` for recorded meetings."""

    duration_ms: int = 0
    vad_speech_ms: float = 0.0
    vad_silence_ms: float = 0.0
    vad_speech_ratio_percent: float = 0.0
    transcript: str = ""
    transcript_lines: list[dict[str, Any]] = field(default_factory=list)
    speaker_talk_ms: dict[str, float] = field(default_factory=dict)
    speaker_talk_ratio_percent: dict[str, float] = field(default_factory=dict)
    speaker_labels_window: list[dict[str, Any]] = field(default_factory=list)
    speakers: list[dict[str, Any]] = field(default_factory=list)
    vad_result: dict[str, Any] | None = None
    diarization_result: dict[str, Any] | None = None
    transcription: dict[str, Any] | None = None
    aligned_segments: list[dict[str, Any]] = field(default_factory=list)


def _build_speaker_stats(
    speaker_times: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    total = sum(speaker_times.values())
    talk_ms = {k: float(v) * 1000.0 for k, v in speaker_times.items()}
    ratio_percent: dict[str, float] = {}
    speakers: list[dict[str, Any]] = []
    for spk, dur_sec in speaker_times.items():
        ms = dur_sec * 1000.0
        pct = (dur_sec / total * 100.0) if total > 0 else 0.0
        ratio_percent[spk] = pct
        speakers.append({"label": spk, "talkMs": ms, "ratioPercent": pct})
    labels_window = [{"speaker": spk} for spk in speaker_times]
    return talk_ms, ratio_percent, speakers, labels_window


def runBatchPipeline(
    audio_file: str,
    *,
    run_vad: bool = True,
    run_diarization: bool = True,
    run_transcription: bool = True,
    run_refinement: bool = False,
    whisper_size: str | None = None,
    whisper_lang: str | None = None,
) -> RecordedAnalysisResult:
    """
    Run VAD, optional diarization, and Whisper transcription on a WAV file path.
    Returns structured metrics for recorded-meeting publishing.
    """
    from config import get_settings

    settings = get_settings()
    whisper_size = whisper_size or os.getenv("BATCH_WHISPER_MODEL_SIZE") or settings.whisper_model_size
    whisper_lang = whisper_lang or os.getenv("BATCH_WHISPER_LANGUAGE") or settings.whisper_language

    result = RecordedAnalysisResult()
    diarization_result = None
    transcriber = None

    if run_vad:
        from workers.audio.vad_batch import VoiceActivityDetector

        vad = VoiceActivityDetector(use_auth_token=settings.hf_token)
        vad_result = vad.runVoiceActivityDetection(audio_file)
        result.vad_result = vad_result
        result.vad_speech_ms = vad_result["speech_time"] * 1000.0
        result.vad_silence_ms = vad_result["silence_time"] * 1000.0
        result.vad_speech_ratio_percent = vad_result["speech_ratio"]
        result.duration_ms = int(vad_result.get("total_time", 0) * 1000)

    if run_diarization:
        from workers.audio.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(hf_token=settings.hf_token)
        diarization_result = diarizer.runSpeakerDiarization(audio_file)
        result.diarization_result = diarization_result
        talk_ms, ratio_pct, speakers, labels = _build_speaker_stats(
            diarization_result["speaker_times"]
        )
        result.speaker_talk_ms = talk_ms
        result.speaker_talk_ratio_percent = ratio_pct
        result.speakers = speakers
        result.speaker_labels_window = labels

    if not run_transcription:
        return result

    from workers.audio.transcript_batch import Transcriber

    transcriber = Transcriber(model_size=whisper_size, language=whisper_lang)
    transcription = transcriber.runTranscription(audio_file)
    result.transcription = transcription
    result.transcript = (transcription.get("text") or "").strip()

    if run_diarization and diarization_result:
        aligned_segments = transcriber.alignWithSpeakers(
            transcription, diarization_result["speaker_segments"]
        )
        result.aligned_segments = aligned_segments
        result.transcript_lines = [
            {
                "speaker": seg["speaker"],
                "startMs": int(seg["start"] * 1000),
                "endMs": int(seg["end"] * 1000),
                "text": seg["text"],
            }
            for seg in aligned_segments
        ]

        if run_refinement:
            from workers.audio.transcript_refiner import TranscriptRefiner

            refiner = TranscriptRefiner()
            refined_segments = refiner.refineTranscript(aligned_segments)
            result.transcript_lines = [
                {
                    "speaker": seg["speaker"],
                    "startMs": int(seg["start"] * 1000),
                    "endMs": int(seg["end"] * 1000),
                    "text": seg.get("text_clean", seg["text"]),
                }
                for seg in refined_segments
            ]
            result.transcript = "\n".join(
                f"[{s['speaker']}] {s['text']}" for s in result.transcript_lines
            )

    if transcriber is not None:
        del transcriber
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def runBatchRunner() -> None:
    """CLI entry: read env flags and write outputs to disk."""
    audio_file = os.getenv("BATCH_AUDIO_FILE", "audio/meeting_clean.wav")
    run_vad = os.getenv("BATCH_RUN_VAD", "1") not in ("0", "false", "False")
    run_diarization = os.getenv("BATCH_RUN_DIARIZATION", "0") in ("1", "true", "True")
    run_transcription = os.getenv("BATCH_RUN_TRANSCRIPTION", "0") in ("1", "true", "True")
    run_refinement = os.getenv("BATCH_RUN_REFINEMENT", "0") in ("1", "true", "True")

    result = runBatchPipeline(
        audio_file,
        run_vad=run_vad,
        run_diarization=run_diarization,
        run_transcription=run_transcription,
        run_refinement=run_refinement,
    )

    if result.vad_result:
        vr = result.vad_result
        print("\nVAD ANALYSIS")
        print(f"Speech duration : {vr['speech_time']/60:.2f} min")
        print(f"Silence duration : {vr['silence_time']/60:.2f} min")
        print(f"Speech ratio : {vr['speech_ratio']:.2f}%")

    if result.diarization_result:
        print("\nSPEAKER ANALYSIS")
        for spk, dur in sorted(
            result.diarization_result["speaker_times"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"{spk}: {dur/60:.2f} min")

    if not run_transcription:
        print("Transcription disabled, process complete.")
        return

    outputs = Path(os.getenv("BATCH_OUTPUT_DIR", "outputs"))
    outputs.mkdir(parents=True, exist_ok=True)

    if result.aligned_segments:
        out_path = outputs / "transcript_by_speaker.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for seg in result.aligned_segments:
                f.write(
                    f"[{seg['speaker']} | {seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n"
                )
        print("Speaker-aligned transcript ready →", out_path)
    elif result.transcript:
        raw_path = outputs / "transcript_raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(result.transcript)
        print("Raw transcript ready →", raw_path)


if __name__ == "__main__":
    runBatchRunner()
