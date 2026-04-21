from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class TranscriptRefiner:
    def __init__(
        self,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device=None,
        max_new_tokens=256
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
            low_cpu_mem_usage=True
        )

        # Inference icin cache'i kapatmak bellek pikini azaltabilir.
        self.model.config.use_cache = False

    def _build_prompt(self, previous_text, current_text):
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
    def refine_segment(self, current_text, previous_text=""):
        prompt = self._build_prompt(previous_text, current_text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            do_sample=False
        )

        decoded = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return decoded.split("Corrected transcription:")[-1].strip()

    def refine_transcript(self, speaker_segments: list) -> list:
        refined = []
        previous_text = ""

        for seg in speaker_segments:
            cleaned_text = self.refine_segment(
                current_text=seg["text"],
                previous_text=previous_text
            )

            refined.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text_raw": seg["text"],
                "text_clean": cleaned_text
            })

            previous_text = cleaned_text

        return refined
