"""Ollama-based meeting adherence analyser.

Calls a local LLM (default: qwen2.5:3b) to evaluate whether the meeting
transcript is on-topic relative to the meeting title and agenda.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)

ALLOWED_CONTEXT_FITS: tuple[float, ...] = (0.20, 0.40, 0.65, 0.90)

_ADHERENCE_PROMPT = """You are a meeting adherence scorer. Judge how well a transcript excerpt aligns with the meeting title and agenda using general concepts only — never rely on example topics from this instruction.

HOW TO EVALUATE (follow in order):
1. Read the agenda and title. Extract the general subject, intent, and key concepts (not literal word lists).
2. Read the transcript. Identify its main theme and what is being discussed.
3. Compare semantic overlap: same subject, adjacent/related subject, weak possible link, or no shared ground.
4. Check word overlap: distinctive terms from the agenda (including proper names). Even one or two matches in context can support a higher tier when the discussion plausibly concerns the same subject.
5. Pick exactly ONE score tier below.

SCORE TIERS — context_fit must be exactly one of: 0.90, 0.65, 0.40, 0.20

0.90 — Strong alignment
The transcript clearly concerns the same topic or intent as the agenda/title. Direct discussion of the agenda subject counts here. Distinctive agenda terms appearing in the transcript (even briefly) with supporting context also justify 0.90. Do not hesitate to use 0.90 when the link is clear.

0.65 — Related / in-scope tangent
A nearby or overlapping topic: same general domain, natural meeting tangent, comparison, background, or prerequisite that a reasonable participant would still treat as in scope. Use 0.65 when the connection is plausible even if not the exact agenda wording. Prefer 0.65 over 0.40 when unsure between those two.

0.40 — Uncertain relevance
There might be a thematic bridge to the agenda, but evidence is thin or ambiguous. Use when you cannot confidently assign 0.65 or 0.90 yet see a possible link. When torn between 0.40 and a higher tier, choose the higher tier (generous scoring).

0.20 — No meaningful alignment
No shared subject with the agenda/title. Casual small talk (weather, weekend, unrelated personal chat) with no professional link to the agenda. A completely different subject. OR the agenda is empty, placeholder, random characters, or meaningless text — then always use 0.20 regardless of transcript content. Do not infer a substitute agenda from the title when the agenda field is empty or nonsense.

GENEROUS SCORING:
Meetings include tangents, stories, and side remarks. If they still relate to the agenda's domain, score 0.65 or 0.90. When choosing between adjacent tiers, prefer the higher tier unless the lower tier is clearly correct.

Meeting title: {title}
Meeting agenda: {agenda}

Transcript excerpt:
{transcript}

Respond ONLY with valid JSON, no markdown:
{{
  "context_fit": <exactly 0.90 or 0.65 or 0.40 or 0.20>,
  "reason": "<one short sentence only when context_fit is 0.20, otherwise null>"
}}"""


def agenda_unusable(agenda: str) -> bool:
    """True when agenda is empty (deterministic shortcut before LLM)."""
    return not (agenda or "").strip()


def snap_context_fit(raw: Any) -> float | None:
    """Map a raw score to the nearest allowed tier."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0.0 or value > 1.0:
        return None
    return min(ALLOWED_CONTEXT_FITS, key=lambda tier: abs(tier - value))


def normalize_adherence_result(parsed: dict[str, Any]) -> dict[str, Any]:
    """Snap context_fit to tier values; derive on_topic from score only."""
    settings = get_settings()
    raw_fit = parsed.get("context_fit")
    if raw_fit is None:
        return {
            "context_fit": None,
            "on_topic": None,
            "reason": parsed.get("reason"),
        }

    fit = snap_context_fit(raw_fit)
    if fit is None:
        return {"context_fit": None, "on_topic": None, "reason": "invalid_score"}

    threshold = settings.adherence_on_topic_fit_threshold
    on_topic = fit >= threshold

    reason = parsed.get("reason")
    if fit != 0.20:
        reason = None
    elif not reason:
        reason = "Conversation appears off-topic relative to the meeting agenda."

    return {
        "context_fit": fit,
        "on_topic": on_topic,
        "reason": reason,
    }


def _format_agenda_for_prompt(agenda: str) -> str:
    text = (agenda or "").strip()
    return text if text else "(empty)"


def _off_topic_shortcut() -> dict[str, Any]:
    return {
        "context_fit": 0.20,
        "on_topic": False,
        "reason": None,
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
    if agenda_unusable(agenda):
        return _off_topic_shortcut()

    prompt = _ADHERENCE_PROMPT.format(
        title=title or "(untitled)",
        agenda=_format_agenda_for_prompt(agenda),
        transcript=transcript,
    )

    raw = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
        raw = response.json()["response"]
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        parsed = json.loads(clean.strip())
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
