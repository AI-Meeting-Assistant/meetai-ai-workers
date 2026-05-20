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
    prompt = f"""You are a meeting analysis assistant. Your job is to evaluate whether a meeting transcript is on-topic relative to the meeting's title and agenda.

IMPORTANT GUIDELINES:
- Be lenient. Meetings naturally include small talk, clarifications, tangents, and side discussions that are still part of the meeting context. These should NOT be flagged as off-topic.
- Only flag as off-topic if the conversation has clearly and substantially drifted away from the meeting's purpose — not just a brief tangent.
- If the agenda is vague or absent, use the meeting title as the primary reference.
- A context_fit score above 0.5 means the meeting is generally on track.

Meeting title: {title}
Meeting agenda: {agenda or '(no agenda provided — use the title as reference)'}

Transcript excerpt:
{transcript}

Respond ONLY with a JSON object, no explanation:
{{
  "context_fit": <float 0.0 to 1.0, where 1.0 means perfectly on topic>,
  "on_topic": <boolean, true unless conversation has clearly and substantially drifted>,
  "reason": "<one concise sentence explaining why it is off-topic, or null if on-topic>"
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
        return {"context_fit": None, "on_topic": None, "reason": "parse_error"}
    except Exception:
        log.error("Ollama call failed", exc_info=True)
        return {"context_fit": None, "on_topic": None, "reason": "llm_error"}


async def summarizeTranscript(
    title: str,
    agenda: str,
    transcript: str,
    ollama_url: str,
    model: str,
) -> str:
    """Generate a one-shot meeting summary from the full transcript."""
    prompt = f"""You are summarizing a completed meeting.

Meeting title: {title}
Meeting agenda: {agenda or '(none provided)'}

Full transcript:
{transcript}

Write a concise summary in Turkish (unless the transcript is clearly English).
Include: main topics discussed, key decisions, and action items if any.
Use short paragraphs or bullet points. Do not invent facts not present in the transcript."""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
        return (response.json().get("response") or "").strip()
    except Exception:
        log.error("Ollama summary call failed", exc_info=True)
        return ""
