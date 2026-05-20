"""FastAPI gateway — ticket validation, health, ingest dispatch (PYTHON_WORKERS_IMPLEMENTATION.md §1, §8–§11)."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sys
import traceback
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()
multiprocessing.freeze_support()

from config import get_settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402
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

log = get_logger(__name__)


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
    log.info("Starting meetai-ai-workers gateway lifespan")
    loop = asyncio.get_running_loop()
    supervisor = WorkerSupervisor()
    await loop.run_in_executor(None, supervisor.start_blocking)
    app.state.worker_supervisor = supervisor
    if supervisor.can_accept_ingest():
        log.info("Worker processes started and ready")
    else:
        log.error(
            "Worker readiness incomplete; /health and /ingest return 503 until workers load",
            failed_modules=list(supervisor.failed_modules),
            dead_modules=supervisor.dead_modules(),
        )
    yield
    log.info("Shutting down meetai-ai-workers gateway")
    await close_async_redis()
    await loop.run_in_executor(None, supervisor.shutdown_blocking)
    log.info("Shutdown complete")


app = FastAPI(title="meetai-ai-workers gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


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
    video_frames: Annotated[list[UploadFile], File(alias="videoFrames[]")],
    title: Annotated[str, Form(alias="title")] = "",
    agenda: Annotated[str, Form(alias="agenda")] = "",
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
        log.warning("Ingest rejected: service unavailable", meeting_id=meeting_id, labels=parts)
        return error_envelope_json_response(503, msg, http_status_code=503)

    ticket_outcome = await validate_stream_ticket(meeting_id, stream_ticket)
    if ticket_outcome is None:
        log.error("Ticket validation failed due to Redis error", meeting_id=meeting_id)
        return error_envelope_json_response(
            503,
            "Redis unavailable during ticket validation.",
            http_status_code=503,
        )
    if ticket_outcome is False:
        log.warning("Ingest rejected: invalid stream ticket", meeting_id=meeting_id, submitted_ticket=stream_ticket)
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
        log.warning("Invalid offset alignment", meeting_id=meeting_id, offset_ms=offset_ms, stride=stride)
        return error_envelope_json_response(
            400,
            f"offsetMs must align to MEDIA_CHUNK_DURATION_MS ({stride} ms)",
            http_status_code=400,
        )

    log.debug("Ingest chunk accepted", meeting_id=meeting_id, offset_ms=offset_ms, n_video_frames=len(video_frames))

    audio_bytes, *frame_bytes_list = await asyncio.gather(
        audio_chunk.read(),
        *[f.read() for f in video_frames],
    )
    video_frames_bytes = [bytes(b) for b in frame_bytes_list]

    enqueue_callable = partial(
        enqueue_ingest_to_workers_blocking,
        supervisor.audio_pipe_send_end,
        supervisor.video_pipe_send_end,
        supervisor.text_pipe_send_end,
        supervisor.audio_send_lock,
        supervisor.video_send_lock,
        supervisor.text_send_lock,
        meeting_id=meeting_id,
        offset_ms=offset_ms,
        audio_webm_bytes=bytes(audio_bytes),
        video_frames=video_frames_bytes,
        title=title,
        agenda=agenda,
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
        log.error("Ingest IPC failure", meeting_id=meeting_id, offset_ms=offset_ms, exc_info=True)
        return error_envelope_json_response(
            503,
            f"Ingest IPC failure: {type(e).__name__}",
            http_status_code=503,
        )

    return ingest_success_response()


@app.post("/ingest-recorded")
async def ingest_recorded(
    background_tasks: BackgroundTasks,
    meeting_id: Annotated[str, Form(alias="meetingId")],
    stream_ticket: Annotated[str, Form(alias="streamTicket")],
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form(alias="title")] = "",
    agenda: Annotated[str, Form(alias="agenda")] = "",
) -> Any:
    """Accept a full audio/video file for offline recorded-meeting processing."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    redis_ok = await ping_redis()
    if not redis_ok:
        return error_envelope_json_response(
            503,
            "Redis unavailable — cannot authorize ingest.",
            http_status_code=503,
        )

    ticket_outcome = await validate_stream_ticket(meeting_id, stream_ticket)
    if ticket_outcome is None:
        return error_envelope_json_response(
            503,
            "Redis unavailable during ticket validation.",
            http_status_code=503,
        )
    if ticket_outcome is False:
        return error_envelope_json_response(401, "invalid stream ticket", http_status_code=401)

    original_name = file.filename or "upload.bin"
    ext = Path(original_name).suffix or ".bin"
    upload_dir = Path(settings.upload_dir) / meeting_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / f"original{ext}"

    total = 0
    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    dest_path.unlink(missing_ok=True)
                    return error_envelope_json_response(
                        400,
                        f"File exceeds maximum size of {settings.max_upload_size_mb} MB",
                        http_status_code=400,
                    )
                out.write(chunk)
    except Exception:
        dest_path.unlink(missing_ok=True)
        log.error("Failed to store recorded upload", meeting_id=meeting_id, exc_info=True)
        return error_envelope_json_response(500, "Failed to store upload", http_status_code=500)

    if total == 0:
        dest_path.unlink(missing_ok=True)
        return error_envelope_json_response(400, "Empty file upload", http_status_code=400)

    from workers.audio.recorded_processor import process_recorded_meeting_sync

    background_tasks.add_task(
        process_recorded_meeting_sync,
        meeting_id,
        str(dest_path),
        title,
        agenda,
    )

    log.info(
        "Recorded upload accepted for background processing",
        meeting_id=meeting_id,
        bytes=total,
        path=str(dest_path),
    )
    return success_envelope_json_response(
        message="Recorded upload accepted — processing started",
        data={"meetingId": meeting_id},
        status_code=202,
    )


class SummarizeRequest(BaseModel):
    meetingId: str = Field(alias="meetingId")
    streamTicket: str = Field(alias="streamTicket")
    transcript: str
    title: str = ""
    agenda: str = ""

    model_config = {"populate_by_name": True}


async def _run_summarize(meeting_id: str, transcript: str, title: str, agenda: str) -> None:
    from workers.text.pipeline import summarizeFullTranscript
    from core.fusion_publisher import channel_summary, publish_json
    from infrastructure.redis_client import get_redis_client

    summary = await summarizeFullTranscript(transcript, title, agenda)
    try:
        r = get_redis_client()
        publish_json(r.raw, channel_summary(meeting_id), {"meetingId": meeting_id, "summary": summary})
    except Exception:
        log.error("Failed to publish summary to Redis", meeting_id=meeting_id, exc_info=True)


@app.post("/summarize")
async def summarize(background_tasks: BackgroundTasks, body: SummarizeRequest) -> Any:
    redis_ok = await ping_redis()
    if not redis_ok:
        return error_envelope_json_response(503, "Redis unavailable", http_status_code=503)

    ticket_outcome = await validate_stream_ticket(body.meetingId, body.streamTicket)
    if ticket_outcome is None:
        return error_envelope_json_response(503, "Redis unavailable during ticket validation.", http_status_code=503)
    if ticket_outcome is False:
        return error_envelope_json_response(401, "invalid stream ticket", http_status_code=401)

    background_tasks.add_task(_run_summarize, body.meetingId, body.transcript, body.title, body.agenda)
    log.info("Summary generation started", meeting_id=body.meetingId)
    return success_envelope_json_response(
        message="Summary generation started",
        data={"meetingId": body.meetingId},
        status_code=202,
    )


if __name__ == "__main__":
    import uvicorn

    _port = int(os.getenv("PORT", "8000"))
    uvicorn.run("gateway:app", host="0.0.0.0", port=_port, reload=False)
