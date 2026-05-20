"""Text worker pipeline — transcript buffering, adherence analysis, Redis publish.

In-process state (module-level dicts) is safe because the text worker is a
single-process asyncio worker with no multi-threading on these dicts.
"""

from __future__ import annotations

from config import get_settings
from core.fusion_publisher import channel_text, publish_json
from infrastructure.redis_client import get_redis_client
from utils.logger import get_logger
from workers.text.adherence import analyzeAdherence, summarizeTranscript

log = get_logger(__name__)

# meeting_id → {title, agenda} — populated from gateway IPC messages
_meta: dict[str, dict] = {}

# meeting_id → list of transcript strings — flushed after threshold is reached
_buffers: dict[str, list[str]] = {}


def updateMeta(meeting_id: str, title: str, agenda: str) -> None:
    """Store meeting metadata received from the gateway IPC pipe."""
    _meta[meeting_id] = {"title": title, "agenda": agenda}
    log.debug("Meeting meta updated", meeting_id=meeting_id, title=title)


async def handleTranscript(meeting_id: str, offset_ms: int, transcript: str) -> None:
    """
    Accumulate a transcript chunk. When the buffer reaches
    ``text_transcript_ring_buffer_slots`` entries, flush it and call the LLM.
    """
    _buffers.setdefault(meeting_id, []).append(transcript)
    threshold = get_settings().text_transcript_ring_buffer_slots

    log.debug(
        "Transcript buffered",
        meeting_id=meeting_id,
        buffered=len(_buffers[meeting_id]),
        threshold=threshold,
    )

    if len(_buffers[meeting_id]) < threshold:
        return

    # Pop atomically before the async LLM call to prevent double-trigger
    chunks = _buffers.pop(meeting_id)
    full_transcript = " ".join(chunks)

    meta = _meta.get(meeting_id)
    if not meta:
        log.warning(
            "No meeting meta cached — skipping adherence analysis",
            meeting_id=meeting_id,
        )
        return

    settings = get_settings()
    log.info(
        "Triggering adherence analysis",
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        chunks=len(chunks),
        transcript_len=len(full_transcript),
    )

    result = await analyzeAdherence(
        title=meta["title"],
        agenda=meta["agenda"],
        transcript=full_transcript,
        ollama_url=settings.ollama_url,
        model=settings.ollama_model,
    )

    payload = {
        "meetingId": meeting_id,
        "offsetMs": offset_ms,
        "contextFit": result.get("context_fit"),
        "onTopic": result.get("on_topic"),
        "reason": result.get("reason"),
        "chunksAnalysed": len(chunks),
    }

    log.info(
        "Adherence result",
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        context_fit=payload["contextFit"],
        on_topic=payload["onTopic"],
        reason=payload["reason"],
    )

    try:
        r = get_redis_client()
        publish_json(r.raw, channel_text(meeting_id), payload)
    except Exception:
        log.error("Failed to publish adherence result to Redis", meeting_id=meeting_id, exc_info=True)


def evictMeeting(meeting_id: str) -> None:
    """Remove cached state when a meeting ends."""
    _meta.pop(meeting_id, None)
    _buffers.pop(meeting_id, None)


async def analyzeFullTranscriptAdherence(
    transcript: str,
    title: str,
    agenda: str,
) -> dict:
    """One-shot adherence analysis for a full recorded meeting transcript."""
    settings = get_settings()
    result = await analyzeAdherence(
        title=title or "Meeting",
        agenda=agenda or "",
        transcript=transcript,
        ollama_url=settings.ollama_url,
        model=settings.ollama_model,
    )
    score = result.get("context_fit")
    return {
        "score": float(score) if score is not None else None,
        "onTopic": result.get("on_topic"),
        "reason": result.get("reason"),
    }


async def summarizeFullTranscript(
    transcript: str,
    title: str,
    agenda: str,
) -> str:
    """One-shot AI summary for a full recorded meeting transcript."""
    if not transcript.strip():
        return ""
    settings = get_settings()
    return await summarizeTranscript(
        title=title or "Meeting",
        agenda=agenda or "",
        transcript=transcript,
        ollama_url=settings.ollama_url,
        model=settings.ollama_model,
    )
