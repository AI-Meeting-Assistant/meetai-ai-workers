"""Multiprocessing entry: receive ingest jobs from Gateway parent Pipe and run ``processLiveChunk``."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

# Repo root → ``import workers``, ``config``, ``infrastructure``
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def audio_worker_process_target(recv_conn: Any, readiness_queue: Any) -> None:
    """Module-level fork entry for ``multiprocessing.Process`` (Windows spawn-safe)."""
    audio_worker_startup(recv_conn, readiness_queue)


def audio_worker_startup(recv_conn: Any, readiness_queue: Any) -> None:
    """Warm up ASR/audio models once, publish readiness, then ``audioWorkerLoop``."""
    try:
        from models.audio_model import warmUpLiveAudioModels
        from workers.audio.pipeline import processLiveChunk  # validate imports early

        warmUpLiveAudioModels()
        if readiness_queue is not None:
            readiness_queue.put({"module": "audio", "ok": True})
    except Exception as e:
        if readiness_queue is not None:
            readiness_queue.put({"module": "audio", "ok": False, "error": str(e)})
        traceback.print_exc()
        return
    audioWorkerLoop(recv_conn, processLiveChunk)


def audioWorkerLoop(conn: Any, processLiveChunk: Any) -> None:
    """
    ``conn``: child end of a **Pipe** created by supervisor (``duplex=False``):
    Parent ``send``s dict messages; worker ``recv``s.

    Message shape::
        {\"type\": \"chunk\", \"meetingId\": uuid, \"offsetMs\": int, \"audioWebmBytes\": bytes}
        {\"type\": \"shutdown\"}
    """
    while True:
        try:
            msg = conn.recv()
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
        offset_ms = msg.get("offsetMs", msg.get("offset_ms"))
        webm = msg.get("audioWebmBytes", msg.get("audio_webm_bytes", b""))
        text_pipe = msg.get("textPipe") or msg.get("text_pipe")
        if meeting_id is None or offset_ms is None:
            continue
        try:
            processLiveChunk(
                meeting_id=str(meeting_id),
                offset_ms=int(offset_ms),
                audio_webm_bytes=webm if isinstance(webm, (bytes, bytearray)) else bytes(webm),
                text_pipe_send_end=text_pipe,
            )
        except Exception:
            traceback.print_exc()


def main() -> None:
    """Standalone debug: import and call audioWorkerLoop(conn) from your supervisor."""
    print("workers.audio.worker_main: import audio_worker_process_target from multiprocessing.")
