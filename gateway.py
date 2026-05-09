"""FastAPI gateway — ticket validation, health, ingest dispatch (PYTHON_WORKERS_IMPLEMENTATION.md §1, §8–§11)."""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
import traceback
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()
multiprocessing.freeze_support()

from config import get_settings  # noqa: E402
from gateway_schemas import (  # noqa: E402
    error_envelope_json_response,
    ingest_success_response,
    success_envelope_json_response,
)
from gateway_supervisor import WorkerSupervisor, enqueue_ingest_to_workers_blocking  # noqa: E402
from infrastructure.redis_async_client import (  # noqa: E402
    close_async_redis,
    ping_redis,
    validate_stream_ticket,
)


def _health_failure_labels(redis_ok: bool, supervisor: WorkerSupervisor | None) -> list[str]:
    labels: list[str] = []
    if not redis_ok:
        labels.append("redis")
    if supervisor is None:
        labels.append("audio")
        labels.append("video")
        return labels
    for name in sorted(set(supervisor.failed_modules + supervisor.dead_modules())):
        labels.append(name)
    return labels


def _health_is_ready(redis_ok: bool, supervisor: WorkerSupervisor | None) -> bool:
    if not redis_ok or supervisor is None:
        return False
    return supervisor.can_accept_ingest()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    supervisor = WorkerSupervisor()
    await loop.run_in_executor(None, supervisor.start_blocking)
    app.state.worker_supervisor = supervisor
    yield
    await close_async_redis()
    await loop.run_in_executor(None, supervisor.shutdown_blocking)


app = FastAPI(title="meetai-ai-workers gateway", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_envelope_handler(_request: Any, exc: RequestValidationError) -> Any:
    return error_envelope_json_response(
        422,
        "multipart validation failed",
        http_status_code=422,
    )


@app.get("/health")
async def health() -> Any:
    redis_ok = await ping_redis()
    supervisor: WorkerSupervisor | None = getattr(app.state, "worker_supervisor", None)

    if _health_is_ready(redis_ok, supervisor):
        return success_envelope_json_response(
            message="AI workers ready",
            data={},
        )

    parts = sorted(set(_health_failure_labels(redis_ok, supervisor)))
    msg = "Service Unavailable: " + ", ".join(parts)
    return error_envelope_json_response(503, msg, http_status_code=503)


@app.post("/ingest")
async def ingest(
    meeting_id: Annotated[str, Form(alias="meetingId")],
    stream_ticket: Annotated[str, Form(alias="streamTicket")],
    offset_ms_field: Annotated[str, Form(alias="offsetMs")],
    audio_chunk: Annotated[UploadFile, File(alias="audioChunk")],
    video_chunk: Annotated[UploadFile, File(alias="videoChunk")],
) -> Any:
    supervisor: WorkerSupervisor | None = getattr(app.state, "worker_supervisor", None)

    redis_ok = await ping_redis()
    if not redis_ok:
        return error_envelope_json_response(
            503,
            "Redis unavailable — cannot authorize ingest.",
            http_status_code=503,
        )

    if not _health_is_ready(redis_ok, supervisor) or supervisor is None:
        parts = sorted(set(_health_failure_labels(redis_ok, supervisor)))
        msg = "Service Unavailable: " + ", ".join(parts)
        return error_envelope_json_response(503, msg, http_status_code=503)

    ticket_outcome = await validate_stream_ticket(meeting_id, stream_ticket)
    if ticket_outcome is None:
        return error_envelope_json_response(
            503,
            "Redis unavailable during ticket validation.",
            http_status_code=503,
        )
    if ticket_outcome is False:
        return error_envelope_json_response(401, "invalid stream ticket", http_status_code=401)

    try:
        offset_ms = int(offset_ms_field)
    except (TypeError, ValueError):
        return error_envelope_json_response(
            400,
            "offsetMs must be a base-10 integer",
            http_status_code=400,
        )

    stride = get_settings().media_chunk_duration_ms
    if stride <= 0 or offset_ms % stride != 0:
        return error_envelope_json_response(
            400,
            f"offsetMs must align to MEDIA_CHUNK_DURATION_MS ({stride} ms)",
            http_status_code=400,
        )

    audio_bytes, video_bytes = await asyncio.gather(audio_chunk.read(), video_chunk.read())

    enqueue_callable = partial(
        enqueue_ingest_to_workers_blocking,
        supervisor.audio_pipe_send_end,
        supervisor.video_pipe_send_end,
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        audio_wav_bytes=bytes(audio_bytes),
        video_bytes=bytes(video_bytes),
    )
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, enqueue_callable)
    except (
        BrokenPipeError,
        EOFError,
        OSError,
        ValueError,
        RuntimeError,
    ) as e:
        traceback.print_exc()
        return error_envelope_json_response(
            503,
            f"Ingest IPC failure: {type(e).__name__}",
            http_status_code=503,
        )

    return ingest_success_response()
