"""Multiprocessing entry: receive ingest jobs from Gateway parent Pipe and run ``processLiveChunk``."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Repo root → ``import workers``, ``config``, ``infrastructure``
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def audioWorkerLoop(conn) -> None:
    """
    ``conn``: child end of a **Pipe** created by supervisor (``duplex=False``):
    Parent ``send``s dict messages; worker ``recv``s.

    Message shape::
        {\"type\": \"chunk\", \"meetingId\": uuid, \"offsetMs\": int, \"audioWavBytes\": bytes}
        {\"type\": \"shutdown\"}
    """
    from workers.audio.pipeline import processLiveChunk

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
        wav = msg.get("audioWavBytes", msg.get("audio_wav_bytes", b""))
        text_pipe = msg.get("textPipe") or msg.get("text_pipe")
        if meeting_id is None or offset_ms is None:
            continue
        try:
            processLiveChunk(
                meeting_id=str(meeting_id),
                offset_ms=int(offset_ms),
                audio_wav_bytes=wav if isinstance(wav, (bytes, bytearray)) else bytes(wav),
                text_pipe_send_end=text_pipe,
            )
        except Exception:
            traceback.print_exc()


def main() -> None:
    """Standalone debug: import and call audioWorkerLoop(conn) from your supervisor."""
    print("workers.audio.worker_main: import and call audioWorkerLoop(conn) from your supervisor.")
