"""Text worker entry point — asyncio process with two concurrent inputs:

  1. Pipe from gateway (metadata: title, agenda per meeting)
  2. Redis pub/sub pattern ``meeting:*:audio`` (transcripts from audio worker)
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Multiprocessing entry — must be module-level for Windows spawn pickling
# ---------------------------------------------------------------------------

def text_worker_process_target(recv_conn: Any, readiness_queue: Any) -> None:
    """Multiprocessing entry point (pickle-safe)."""
    text_worker_startup(recv_conn, readiness_queue)


def text_worker_startup(recv_conn: Any, readiness_queue: Any) -> None:
    """Validate imports, signal readiness, then run the asyncio event loop."""
    try:
        # Validate imports early so failures surface in the readiness queue
        from workers.text.pipeline import updateMeta, handleTranscript  # noqa: F401

        if readiness_queue is not None:
            readiness_queue.put({"module": "text", "ok": True})
    except Exception as e:
        if readiness_queue is not None:
            readiness_queue.put({"module": "text", "ok": False, "error": str(e)})
        traceback.print_exc()
        return

    asyncio.run(_textWorkerMain(recv_conn))


# ---------------------------------------------------------------------------
# Asyncio main
# ---------------------------------------------------------------------------

async def _textWorkerMain(conn: Any) -> None:
    loop = asyncio.get_running_loop()

    await asyncio.gather(
        _pipeTask(conn, loop),
        _redisTask(),
    )


async def _pipeTask(conn: Any, loop: asyncio.AbstractEventLoop) -> None:
    """
    Receive IPC messages from the gateway in a thread (conn.recv is blocking).
    Each message carries {meetingId, title, agenda} — stored in the pipeline.
    """
    from workers.text.pipeline import updateMeta

    while True:
        try:
            msg = await loop.run_in_executor(None, conn.recv)
        except EOFError:
            break
        if not isinstance(msg, dict):
            continue
        kind = msg.get("type") or msg.get("kind")
        if kind in ("shutdown", "stop"):
            break
        if kind not in ("chunk", "process_chunk"):
            continue
        meeting_id = msg.get("meetingId") or msg.get("meeting_id")
        title  = msg.get("title", "")
        agenda = msg.get("agenda", "")
        if meeting_id and (title or agenda):
            updateMeta(str(meeting_id), title, agenda)


async def _redisTask() -> None:
    """
    Subscribe to all audio channels (``meeting:*:audio``) via Redis pattern
    pub/sub. For each message that contains a transcript, forward it to the
    pipeline for buffering and eventual LLM analysis.
    """
    import redis.asyncio as aioredis
    from config import get_settings
    from workers.text.pipeline import handleTranscript

    settings = get_settings()
    r = aioredis.from_url(settings.redis_url)
    pubsub = r.pubsub()
    await pubsub.psubscribe("meeting:*:audio")

    async for message in pubsub.listen():
        if message["type"] != "pmessage":
            continue
        try:
            data = json.loads(message["data"])
        except Exception:
            continue

        transcript = (data.get("transcript") or "").strip()
        if not transcript:
            continue

        meeting_id = data.get("meetingId") or data.get("meeting_id") or ""
        offset_ms  = int(data.get("offsetMs", data.get("offset_ms", 0)))

        if meeting_id:
            await handleTranscript(meeting_id, offset_ms, transcript)
