"""Ollama-based meeting adherence analyser.

Calls a local LLM (default: qwen2.5:3b) to evaluate whether the meeting
transcript is on-topic relative to the meeting title and agenda.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)

_ADHERENCE_PROMPT = """You are a meeting analysis assistant. Score how well a transcript excerpt fits the meeting title and agenda.

SCORING RUBRIC (context_fit, 0.0 to 1.0):
- 0.85–1.0: Directly about the agenda subject (main theme, named entities, fans, history, current success).
- 0.70–0.84: Same broad domain as the agenda (e.g. same sport, stadium culture, supporter behavior, comparisons, personal anecdotes tied to the topic). Brief tangents still in-domain count here.
- 0.50–0.69: Weak link only — do NOT treat as a meeting deviation.
- Below 0.50: Clear drift — unrelated sport, unrelated industry, personal small talk with no link to the agenda subject, or a different topic altogether.

IMPORTANT:
- Meetings include natural tangents, comparisons, stories, and side remarks. If they still relate to the agenda's domain, score 0.70+.
- Only set on_topic to false when context_fit would be below 0.50 AND the conversation has clearly left the agenda's subject area.
- If on_topic is true, set reason to null.
- If the agenda mentions personal anecdotes, history, slogans, or fan culture, those are ON-topic when discussing the same subject.

Meeting title: {title}
Meeting agenda: {agenda}

Transcript excerpt:
{transcript}

Respond ONLY with a JSON object:
{{
  "context_fit": <float 0.0 to 1.0>,
  "on_topic": <boolean>,
  "reason": "<one sentence if clearly off-topic, else null>"
}}"""


def normalize_adherence_result(parsed: dict[str, Any]) -> dict[str, Any]:
    """Align context_fit and on_topic; prefer numeric score over LLM boolean."""
    settings = get_settings()
    raw_fit = parsed.get("context_fit")
    if raw_fit is None:
        return {
            "context_fit": None,
            "on_topic": None,
            "reason": parsed.get("reason"),
        }

    try:
        fit = max(0.0, min(1.0, float(raw_fit)))
    except (TypeError, ValueError):
        return {"context_fit": None, "on_topic": None, "reason": "invalid_score"}

    threshold = settings.adherence_on_topic_fit_threshold
    on_topic = fit >= threshold

    if settings.adherence_llm_override_low_boolean and fit >= 0.65:
        on_topic = True

    reason = parsed.get("reason")
    if on_topic:
        reason = None
    elif not reason:
        reason = "Conversation appears off-topic relative to the meeting agenda."

    return {
        "context_fit": round(fit, 4),
        "on_topic": on_topic,
        "reason": reason,
    }


async def analyzeAdherence(
    title: str,
    agenda: str,
    transcript: str,
    ollama_url: str,
    model: str,
) -> dict:
    """
    Call the local Ollama LLM and return a parsed adherence result.

    Returns a dict with keys: context_fit, on_topic, reason.
    On any failure returns the same keys with None / error values.
    """
    agenda_text = agenda or "(no agenda provided — infer the subject domain from the title)"
    prompt = _ADHERENCE_PROMPT.format(
        title=title,
        agenda=agenda_text,
        transcript=transcript,
    )

    raw = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
        raw = response.json()["response"]
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean)
        return normalize_adherence_result(parsed)
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

First detect the primary language of the transcript, then write the summary in that same language.
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
