"""LLaMA-powered transcript polishing — **offline / batch only** (never per ingest chunk)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TranscriptRefiner:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 256,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens
        model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        )

        self.model.config.use_cache = False

    def _buildPrompt(self, previous_text: str, current_text: str) -> str:
        return f"""
You are a professional Turkish meeting transcription editor.

TASK:
- Fix transcription errors
- Correct Turkish grammar
- Fix technical terms
- DO NOT summarize
- DO NOT add new information
- DO NOT remove meaning
- Preserve names and company names
- Output ONLY the corrected sentence

Context (previous speech):
"{previous_text}"

Current transcription to fix:
"{current_text}"

Corrected transcription:
""".strip()

    @torch.no_grad()
    def refineSegment(self, current_text: str, previous_text: str = "") -> str:
        prompt = self._buildPrompt(previous_text, current_text)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            do_sample=False,
        )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return decoded.split("Corrected transcription:")[-1].strip()

    def refineTranscript(self, speaker_segments: list) -> list:
        refined: list[dict[str, object]] = []
        previous_text = ""

        for seg in speaker_segments:
            cleaned_text = self.refineSegment(current_text=str(seg["text"]), previous_text=previous_text)

            refined.append(
                {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text_raw": seg["text"],
                    "text_clean": cleaned_text,
                }
            )

            previous_text = cleaned_text

        return refined
