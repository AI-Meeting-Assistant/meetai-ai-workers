import os

os.environ.setdefault("SB_DISABLE_K2", "1")

from modules.vad import VoiceActivityDetector
from modules.speaker_diarization import SpeakerDiarizer
from modules.transcribe import Transcriber
from modules.transcript_refiner import TranscriptRefiner

import torch
import gc


AUDIO_FILE = "audio/meeting_clean.wav"

RUN_VAD = True
RUN_DIARIZATION = False
RUN_TRANSCRIPTION = False
RUN_REFINEMENT = False


def main():
    diarization_result = None
    transcriber = None

    # -------- VAD --------
    if RUN_VAD:
        print("VAD baslatiliyor...")
        vad = VoiceActivityDetector()
        vad_result = vad.run(AUDIO_FILE)

        print("\nVAD ANALIZI")
        print(f"Konusma suresi : {vad_result['speech_time']/60:.2f} dk")
        print(f"Sessizlik suresi : {vad_result['silence_time']/60:.2f} dk")
        print(f"Konusma orani : %{vad_result['speech_ratio']:.2f}")

    # -------- DIARIZATION --------
    if RUN_DIARIZATION:
        print("Diarization baslatiliyor...")
        diarizer = SpeakerDiarizer()
        diarization_result = diarizer.run(AUDIO_FILE)

        print("\nKONUSMACI ANALIZI")
        for spk, dur in sorted(
            diarization_result["speaker_times"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"{spk}: {dur/60:.2f} dk")

    # -------- TRANSCRIPTION --------
    if not RUN_TRANSCRIPTION:
        print("Transcription kapali, islem tamamlandi.")
        return

    transcriber = Transcriber()

    print("\nWhisper transcribe basliyor...")
    transcription = transcriber.transcribe(AUDIO_FILE)

    # ---- SENARYO 1: diarization VAR ----
    if RUN_DIARIZATION and diarization_result:
        print("Konusmaci atamasi yapiliyor...")

        aligned_segments = transcriber.align_with_speakers(
            transcription,
            diarization_result["speaker_segments"]
        )

        with open("outputs/transcript_by_speaker.txt", "w", encoding="utf-8") as f:
            for seg in aligned_segments:
                f.write(
                    f"[{seg['speaker']} | "
                    f"{seg['start']:.2f}-{seg['end']:.2f}] "
                    f"{seg['text']}\n"
                )

        print("Speaker-aligned transkript hazir")
        if RUN_REFINEMENT:
            print("Speaker-aware metin duzeltiliyor (LLaMA)...")
            refiner = TranscriptRefiner()
            refined_segments = refiner.refine_transcript(aligned_segments)

            with open("outputs/transcript_final_by_speaker.txt", "w", encoding="utf-8") as f:
                for seg in refined_segments:
                    f.write(
                        f"[{seg['speaker']} | "
                        f"{seg['start']:.2f}-{seg['end']:.2f}] "
                        f"{seg['text_clean']}\n"
                    )

            print("Final konusmacili transkript hazir")

    # ---- SENARYO 2: diarization YOK ----
    else:
        with open("outputs/transcript_raw.txt", "w", encoding="utf-8") as f:
            f.write(transcription["text"])
        print("Raw transkript hazir")

        if RUN_REFINEMENT:
            print("Metin duzeltiliyor (konusmacisiz, LLaMA)...")
            refiner = TranscriptRefiner()
            cleaned_text = refiner.refine_segment(
                current_text=transcription["text"]
            )

            with open("outputs/transcript_final.txt", "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print("Final transkript hazir")

    del transcriber
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Whisper GPU bellegi temizlendi")


if __name__ == "__main__":
    main()
