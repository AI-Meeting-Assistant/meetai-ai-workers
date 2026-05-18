"""Ollama-based meeting adherence analyser.

Calls a local LLM (default: qwen2.5:3b) to evaluate whether the meeting
transcript is on-topic relative to the meeting title and agenda.
"""

from __future__ import annotations

import json

import httpx

from utils.logger import get_logger

log = get_logger(__name__)


async def analyzeAdherence(
    title: str,
    agenda: str,
    transcript: str,
    ollama_url: str,
    model: str,
) -> dict:
    """
    Call the local Ollama LLM and return a parsed adherence result.

    Returns a dict with keys: adherence_score, on_topic, reason.
    On any failure returns the same keys with None / error values.
    """
    prompt = f"""You are analyzing a meeting transcript for topic adherence.

Meeting title: {title}
Meeting agenda: {agenda}

Transcript:
{transcript}

Respond ONLY with a JSON object, no explanation:
{{
  "adherence_score": <float 0.0 to 1.0>,
  "on_topic": <boolean>,
  "reason": "<one sentence if off-topic, else null>"
}}"""

    raw = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
        raw = response.json()["response"]
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        log.warning("Ollama response JSON parse error", raw=raw[:300])
        return {"adherence_score": None, "on_topic": None, "reason": "parse_error"}
    except Exception:
        log.error("Ollama call failed", exc_info=True)
        return {"adherence_score": None, "on_topic": None, "reason": "llm_error"}
