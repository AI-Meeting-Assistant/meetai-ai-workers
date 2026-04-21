from pyannote.audio import Pipeline
from collections import defaultdict
import torch
import os


class SpeakerDiarizer:
    def __init__(self, device=None, hf_token=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)

        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization"
            )
        except TypeError:
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    token=hf_token
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=hf_token
                )
        self.pipeline.to(torch.device(device))

    def run(self, audio_file: str) -> dict:
        """
        Konuşmacı bazlı süre ve segment bilgilerini döner
        """
        diarization = self.pipeline(audio_file)

        speaker_times = defaultdict(float)
        speaker_segments = []

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_times[speaker] += segment.duration
            speaker_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })

        total_time = sum(speaker_times.values())

        return {
            "speaker_times": dict(speaker_times),
            "speaker_segments": speaker_segments,
            "total_speech_time": total_time
        }