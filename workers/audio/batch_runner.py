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
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("SB_DISABLE_K2", "1")

import torch  # noqa: E402


def runBatchRunner() -> None:
    audio_file = os.getenv("BATCH_AUDIO_FILE", "audio/meeting_clean.wav")
    run_vad = os.getenv("BATCH_RUN_VAD", "1") not in ("0", "false", "False")
    run_diarization = os.getenv("BATCH_RUN_DIARIZATION", "0") in ("1", "true", "True")
    run_transcription = os.getenv("BATCH_RUN_TRANSCRIPTION", "0") in ("1", "true", "True")
    run_refinement = os.getenv("BATCH_RUN_REFINEMENT", "0") in ("1", "true", "True")

    diarization_result = None
    transcriber = None

    if run_vad:
        print("VAD başlatılıyor...")
        from workers.audio.vad_batch import VoiceActivityDetector

        vad = VoiceActivityDetector()
        vad_result = vad.runVoiceActivityDetection(audio_file)

        print("\nVAD ANALİZİ")
        print(f"Konuşma süresi : {vad_result['speech_time']/60:.2f} dk")
        print(f"Sessizlik süresi : {vad_result['silence_time']/60:.2f} dk")
        print(f"Konuşma oranı : %{vad_result['speech_ratio']:.2f}")

    if run_diarization:
        print("Diarization başlatılıyor...")
        from workers.audio.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer()
        diarization_result = diarizer.runSpeakerDiarization(audio_file)

        print("\nKONUŞMACI ANALİZİ")
        for spk, dur in sorted(
            diarization_result["speaker_times"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"{spk}: {dur/60:.2f} dk")

    if not run_transcription:
        print("Transcription kapalı, işlem tamamlandi.")
        return

    from workers.audio.transcript_batch import Transcriber

    whisper_size = os.getenv("BATCH_WHISPER_MODEL_SIZE") or os.getenv("WHISPER_MODEL_SIZE", "medium")
    whisper_lang = os.getenv("BATCH_WHISPER_LANGUAGE") or os.getenv("WHISPER_LANGUAGE", "tr")
    transcriber = Transcriber(model_size=whisper_size, language=whisper_lang)

    print("\nWhisper transcribe başlıyor...")
    transcription = transcriber.runTranscription(audio_file)

    outputs = Path(os.getenv("BATCH_OUTPUT_DIR", "outputs"))
    outputs.mkdir(parents=True, exist_ok=True)

    if run_diarization and diarization_result:
        print("Konuşmacı ataması yapılıyor...")
        aligned_segments = transcriber.alignWithSpeakers(
            transcription, diarization_result["speaker_segments"]
        )

        out_path = outputs / "transcript_by_speaker.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for seg in aligned_segments:
                f.write(
                    f"[{seg['speaker']} | {seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n"
                )

        print("Speaker-aligned transkript hazır →", out_path)
        if run_refinement:
            print("Speaker-aware metin düzeltiliyor (LLaMA)...")
            from workers.audio.transcript_refiner import TranscriptRefiner

            refiner = TranscriptRefiner()
            refined_segments = refiner.refineTranscript(aligned_segments)

            fp = outputs / "transcript_final_by_speaker.txt"
            with open(fp, "w", encoding="utf-8") as f:
                for seg in refined_segments:
                    f.write(
                        f"[{seg['speaker']} | {seg['start']:.2f}-{seg['end']:.2f}] "
                        f"{seg['text_clean']}\n"
                    )

            print("Final konuşmacılı transkript hazır →", fp)

    else:
        raw_path = outputs / "transcript_raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(transcription["text"])
        print("Raw transkript hazır →", raw_path)

        if run_refinement:
            print("Metin düzeltiliyor (konuşmacısız, LLaMA)...")
            from workers.audio.transcript_refiner import TranscriptRefiner

            refiner = TranscriptRefiner()
            cleaned_text = refiner.refineSegment(current_text=str(transcription["text"]))

            fp = outputs / "transcript_final.txt"
            with open(fp, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print("Final transkript hazır →", fp)

    del transcriber
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Whisper GPU belleği temizlendi")


if __name__ == "__main__":
    runBatchRunner()
