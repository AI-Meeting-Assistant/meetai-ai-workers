"""Spawn audio/video worker processes + Pipe IPC (PYTHON_WORKERS_IMPLEMENTATION.md §6, §8)."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from queue import Empty
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger(__name__)

READINESS_SECONDS = 90  # Whisper load on Windows spawn can take 30-60 s on first run


def _wait_worker_readiness(
    readiness_queue: Any,
    required_modules: tuple[str, ...],
    timeout_seconds: float,
) -> tuple[dict[str, bool], dict[str, str]]:
    """
    Consume readiness messages placed by worker starts.

    Returns:
        ``outcomes`` maps module → success (explicit ``False`` or ``True`` seen).
        ``errors`` maps module → diagnostic string when ``ok`` is ``False``.
    """
    deadline = time.monotonic() + timeout_seconds
    outcomes: dict[str, bool] = {}
    errors: dict[str, str] = {}

    while time.monotonic() < deadline:
        pending = []
        for m in required_modules:
            if m not in outcomes:
                pending.append(m)
        if not pending:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            item = readiness_queue.get(timeout=min(remaining, 0.25))
        except Empty:
            continue
        if not isinstance(item, dict):
            continue
        mod = item.get("module")
        ok = bool(item.get("ok"))
        if isinstance(mod, str):
            outcomes[mod] = ok
            if not ok and item.get("error"):
                errors[mod] = str(item.get("error"))

    for mod in required_modules:
        if mod not in outcomes:
            outcomes[mod] = False
            errors.setdefault(mod, "readiness deadline exceeded")

    return outcomes, errors


class WorkerSupervisor:
    """Owns multiprocessing Processes and parent-side Pipe send halves."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        import threading
        self.audio_send_lock = threading.Lock()
        self.video_send_lock = threading.Lock()
        self.text_send_lock  = threading.Lock()
        self._audio_send: Optional[Any] = None
        self._video_send: Optional[Any] = None
        self._text_send:  Optional[Any] = None
        self._audio_proc: Optional[mp.Process] = None
        self._video_proc: Optional[mp.Process] = None
        self._text_proc:  Optional[mp.Process] = None
        self.failed_modules: list[str] = []

    def _reset_state(self) -> None:
        self.failed_modules = []

    def start_blocking(self) -> None:
        """Spawn workers and collect readiness (**blocking** caller thread)."""
        self._reset_state()
        from multiprocessing.connection import Pipe

        from workers.audio.worker_main import audio_worker_process_target
        from workers.video.worker_main import video_worker_process_target
        from workers.text.worker_main  import text_worker_process_target

        readiness_queue = self._ctx.Queue()

        audio_recv_conn, audio_send_conn = Pipe(duplex=False)
        video_recv_conn, video_send_conn = Pipe(duplex=False)
        text_recv_conn,  text_send_conn  = Pipe(duplex=False)

        self._audio_send = audio_send_conn
        self._video_send = video_send_conn
        self._text_send  = text_send_conn

        self._audio_proc = self._ctx.Process(
            target=audio_worker_process_target,
            args=(audio_recv_conn, readiness_queue),
            name="meetaiAudioWorker",
        )
        self._video_proc = self._ctx.Process(
            target=video_worker_process_target,
            args=(video_recv_conn, readiness_queue),
            name="meetaiVideoWorker",
        )
        self._text_proc = self._ctx.Process(
            target=text_worker_process_target,
            args=(text_recv_conn, readiness_queue),
            name="meetaiTextWorker",
        )
        self._audio_proc.start()
        self._video_proc.start()
        self._text_proc.start()

        required = ("audio", "video", "text")
        log.info("Waiting for worker readiness", timeout_sec=READINESS_SECONDS, modules=required)
        outcomes, errors = _wait_worker_readiness(
            readiness_queue,
            required,
            READINESS_SECONDS,
        )
        for mod in required:
            if outcomes.get(mod) is not True:
                if mod not in self.failed_modules:
                    self.failed_modules.append(mod)
                log.error("Worker module failed readiness", module=mod, error=errors.get(mod))
            else:
                log.info("Worker module ready", module=mod)

        if self._audio_proc.exitcode not in (None, 0) and "audio" not in self.failed_modules:
            self.failed_modules.append("audio")
            log.error("Audio worker process exited unexpectedly", exitcode=self._audio_proc.exitcode)
        if self._video_proc.exitcode not in (None, 0) and "video" not in self.failed_modules:
            self.failed_modules.append("video")
            log.error("Video worker process exited unexpectedly", exitcode=self._video_proc.exitcode)
        if self._text_proc.exitcode not in (None, 0) and "text" not in self.failed_modules:
            self.failed_modules.append("text")
            log.error("Text worker process exited unexpectedly", exitcode=self._text_proc.exitcode)

    def shutdown_blocking(self, join_timeout_seconds: float = 8.0) -> None:
        """Graceful Pipe shutdown signals; join with timeout."""
        self._shutdown_send_shutdown()
        for proc in (self._audio_proc, self._video_proc, self._text_proc):
            if proc is None:
                continue
            proc.join(timeout=max(1.0, join_timeout_seconds * 0.5))
            if proc.is_alive():
                log.warning("Worker process did not exit, terminating", pid=proc.pid, name=proc.name)
                try:
                    proc.terminate()
                except Exception:
                    log.error("Failed to terminate worker process", exc_info=True)
                proc.join(timeout=2.0)
        self._audio_send = self._video_send = self._text_send = None
        self._audio_proc = self._video_proc = self._text_proc = None

    def _shutdown_send_shutdown(self) -> None:
        for conn in (self._audio_send, self._video_send, self._text_send):
            if conn is None:
                continue
            try:
                conn.send({"type": "shutdown"})
            except (BrokenPipeError, EOFError, OSError, ValueError):
                pass

    @property
    def audio_pipe_send_end(self) -> Any:
        if self._audio_send is None:
            raise RuntimeError("WorkerSupervisor not started")
        return self._audio_send

    @property
    def video_pipe_send_end(self) -> Any:
        if self._video_send is None:
            raise RuntimeError("WorkerSupervisor not started")
        return self._video_send

    @property
    def text_pipe_send_end(self) -> Any:
        if self._text_send is None:
            raise RuntimeError("WorkerSupervisor not started")
        return self._text_send

    def dead_modules(self) -> list[str]:
        """Modules whose OS process is gone (post warm-up crash or failed spawn)."""
        dead: list[str] = []
        if self._audio_proc is None or not self._audio_proc.is_alive():
            dead.append("audio")
        if self._video_proc is None or not self._video_proc.is_alive():
            dead.append("video")
        if self._text_proc is None or not self._text_proc.is_alive():
            dead.append("text")
        return dead

    def can_accept_ingest(self) -> bool:
        """Warm-up succeeded, processes alive, pipes configured."""
        if self.failed_modules:
            return False
        if (
            self._audio_proc is None
            or self._video_proc is None
            or self._text_proc is None
            or self._audio_send is None
            or self._video_send is None
            or self._text_send is None
        ):
            return False
        return (
            self._audio_proc.is_alive()
            and self._video_proc.is_alive()
            and self._text_proc.is_alive()
        )


def enqueue_ingest_to_workers_blocking(
    audio_send_conn: Any,
    video_send_conn: Any,
    text_send_conn: Any,
    audio_send_lock: Any,
    video_send_lock: Any,
    text_send_lock: Any,
    *,
    meeting_id: str,
    offset_ms: int,
    audio_webm_bytes: bytes,
    video_frames: list[bytes],
    title: str = "",
    agenda: str = "",
) -> None:
    """Send chunk payloads to audio, video, and text workers (**blocking** Pipe send, thread-safe)."""
    with audio_send_lock:
        audio_send_conn.send(
            {
                "type": "chunk",
                "meetingId": meeting_id,
                "offsetMs": offset_ms,
                "audioWebmBytes": audio_webm_bytes,
            }
        )
    with video_send_lock:
        video_send_conn.send(
            {
                "type": "chunk",
                "meetingId": meeting_id,
                "offsetMs": offset_ms,
                "videoFrames": video_frames,
            }
        )
    with text_send_lock:
        text_send_conn.send(
            {
                "type": "chunk",
                "meetingId": meeting_id,
                "offsetMs": offset_ms,
                "title": title,
                "agenda": agenda,
            }
        )
