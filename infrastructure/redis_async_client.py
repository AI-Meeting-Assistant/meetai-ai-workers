"""Async Redis singleton for FastAPI gateway — ticket validation (PYTHON_WORKERS_IMPLEMENTATION.md §4, §6)."""

from __future__ import annotations

import os
from typing import Optional

from redis.asyncio import Redis
from redis.exceptions import RedisError

_redis_async: Optional[Redis] = None


def get_async_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379")


async def get_async_redis() -> Redis:
    global _redis_async
    if _redis_async is None:
        _redis_async = Redis.from_url(
            get_async_redis_url(),
            decode_responses=True,
            encoding="utf-8",
        )
    return _redis_async


async def close_async_redis() -> None:
    global _redis_async
    if _redis_async is not None:
        await _redis_async.aclose()
        _redis_async = None


async def ping_redis() -> bool:
    """Return True if Redis responds to PING."""
    r = await get_async_redis()
    try:
        return bool(await r.ping())
    except (RedisError, OSError):
        return False


def stream_ticket_key(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:ticket"


async def validate_stream_ticket(meeting_id: str, submitted_ticket: str) -> bool | None:
    """
    Exact string match against ``GET meeting:<meetingId>:ticket`` (§4).

    Returns:
        ``True`` Match.
        ``False`` Missing key / mismatch (**401** at HTTP).
        ``None`` Redis / transport failure (**503** ingest / mirror in health).
    """
    r = await get_async_redis()
    try:
        stored = await r.get(stream_ticket_key(meeting_id))
    except (RedisError, OSError):
        return None
    if stored is None:
        return False
    return str(stored) == str(submitted_ticket)


async def validateStreamTicket(meeting_id: str, submitted_ticket: str) -> bool | None:
    """Alias for PYTHON_WORKERS_IMPLEMENTATION.md wording (gateway uses snake_case callers)."""
    return await validate_stream_ticket(meeting_id, submitted_ticket)
