"""Spawn audio/video worker processes + Pipe IPC (PYTHON_WORKERS_IMPLEMENTATION.md §6, §8)."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from queue import Empty
from typing import Any, Optional

READINESS_SECONDS = 15


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
        self._audio_send: Optional[Any] = None
        self._video_send: Optional[Any] = None
        self._audio_proc: Optional[mp.Process] = None
        self._video_proc: Optional[mp.Process] = None
        self.failed_modules: list[str] = []

    def _reset_state(self) -> None:
        self.failed_modules = []

    def start_blocking(self) -> None:
        """
        Spawn workers and collect readiness (**blocking** caller thread).

        ``text`` worker intentionally omitted (PYTHON_WORKERS_IMPLEMENTATION.md §3.2 placeholder).
        """
        self._reset_state()
        from multiprocessing.connection import Pipe

        from workers.audio.worker_main import audio_worker_process_target
        from workers.video.worker_main import video_worker_process_target

        readiness_queue = self._ctx.Queue()

        audio_recv_conn, audio_send_conn = Pipe(duplex=False)
        video_recv_conn, video_send_conn = Pipe(duplex=False)

        self._audio_send = audio_send_conn
        self._video_send = video_send_conn

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
        self._audio_proc.start()
        self._video_proc.start()

        required = ("audio", "video")
        outcomes, _ = _wait_worker_readiness(
            readiness_queue,
            required,
            READINESS_SECONDS,
        )
        for mod in required:
            if outcomes.get(mod) is not True:
                if mod not in self.failed_modules:
                    self.failed_modules.append(mod)

        if self._audio_proc.exitcode not in (None, 0) and "audio" not in self.failed_modules:
            self.failed_modules.append("audio")
        if self._video_proc.exitcode not in (None, 0) and "video" not in self.failed_modules:
            self.failed_modules.append("video")

    def shutdown_blocking(self, join_timeout_seconds: float = 8.0) -> None:
        """Graceful Pipe shutdown signals; join with timeout."""
        self._shutdown_send_shutdown()
        for proc in (self._audio_proc, self._video_proc):
            if proc is None:
                continue
            proc.join(timeout=max(1.0, join_timeout_seconds * 0.5))
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    traceback.print_exc()
                proc.join(timeout=2.0)
        self._audio_send = self._video_send = None
        self._audio_proc = self._video_proc = None

    def _shutdown_send_shutdown(self) -> None:
        for conn in (self._audio_send, self._video_send):
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

    def dead_modules(self) -> list[str]:
        """Modules whose OS process is gone (post warm-up crash or failed spawn)."""
        dead: list[str] = []
        if self._audio_proc is None or not self._audio_proc.is_alive():
            dead.append("audio")
        if self._video_proc is None or not self._video_proc.is_alive():
            dead.append("video")
        return dead

    def can_accept_ingest(self) -> bool:
        """Warm-up succeeded, processes alive, pipes configured."""
        if self.failed_modules:
            return False
        if (
            self._audio_proc is None
            or self._video_proc is None
            or self._audio_send is None
            or self._video_send is None
        ):
            return False
        return self._audio_proc.is_alive() and self._video_proc.is_alive()


def enqueue_ingest_to_workers_blocking(
    audio_send_conn: Any,
    video_send_conn: Any,
    *,
    meeting_id: str,
    offset_ms: int,
    audio_wav_bytes: bytes,
    video_bytes: bytes,
) -> None:
    """Send mirrored chunk payloads to audio and vision workers (**blocking** Pipe send)."""
    audio_send_conn.send(
        {
            "type": "chunk",
            "meetingId": meeting_id,
            "offsetMs": offset_ms,
            "audioWavBytes": audio_wav_bytes,
        }
    )
    video_send_conn.send(
        {
            "type": "chunk",
            "meetingId": meeting_id,
            "offsetMs": offset_ms,
            "videoBytes": video_bytes,
        }
    )
