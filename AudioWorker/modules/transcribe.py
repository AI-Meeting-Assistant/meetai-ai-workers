import whisper
import torch


class Transcriber:
    def __init__(self, model_size="medium", device=None, language="tr"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)
        self.language = language

    def transcribe(self, audio_file: str) -> dict:
        """
        Whisper ile kelime zaman damgalı transkript üretir
        """
        return self.model.transcribe(
            audio_file,
            language=self.language,
            word_timestamps=True
        )

    def align_with_speakers(self, transcription: dict, speaker_segments: list) -> list:
        """
        Transkripti diarization segmentlerine göre konuşmacılara böler
        """
        results = []

        for segment in speaker_segments:
            start = segment["start"]
            end = segment["end"]
            speaker = segment["speaker"]

            words = []
            for seg in transcription["segments"]:
                for word in seg.get("words", []):
                    if start <= word["start"] <= end:
                        words.append(word["word"])

            if words:
                results.append({
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": " ".join(words)
                })

        return results