"""Multiprocessing entry: receive ingest jobs from gateway Pipe and run ``processLiveVisionChunk``."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

# Repo root → ``import workers``, ``config``, ``infrastructure``
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def video_worker_process_target(recv_conn: Any, readiness_queue: Any) -> None:
    """Module-level fork entry (pickle-safe on Windows)."""
    video_worker_startup(recv_conn, readiness_queue)


def video_worker_startup(recv_conn: Any, readiness_queue: Any) -> None:
    """Run stub warm-up, signal readiness, then ``videoWorkerLoop``."""
    try:
        from models.video_model import warmUpLiveVideoModels
        from workers.video.pipeline import processLiveVisionChunk  # validate imports early

        warmUpLiveVideoModels()
        if readiness_queue is not None:
            readiness_queue.put({"module": "video", "ok": True})
    except Exception as e:
        if readiness_queue is not None:
            readiness_queue.put({"module": "video", "ok": False, "error": str(e)})
        traceback.print_exc()
        return
    videoWorkerLoop(recv_conn, processLiveVisionChunk)


def videoWorkerLoop(conn: Any, processLiveVisionChunk: Any) -> None:
    """
    ``conn``: child (**recv**) end of ``Pipe(duplex=False)``.

    Message shape::
        {\"type\": \"chunk\", \"meetingId\": uuid, \"offsetMs\": int, \"videoFrames\": list[bytes]}
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
        frames = msg.get("videoFrames", msg.get("video_frames", []))
        if meeting_id is None or offset_ms is None:
            continue
        if not isinstance(frames, list):
            frames = []
        try:
            processLiveVisionChunk(
                meeting_id=str(meeting_id),
                offset_ms=int(offset_ms),
                jpeg_frames=[bytes(f) for f in frames],
            )
        except Exception:
            traceback.print_exc()


def main() -> None:
    print(
        "workers.video.worker_main: import video_worker_process_target(recv, queue) "
        "from multiprocessing.Process."
    )
