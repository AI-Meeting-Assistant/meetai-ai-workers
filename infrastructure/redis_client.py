"""Redis client(s) for the AI workers subsystem.

- ``RedisClient``: sync singleton used by **worker processes** (publish only).
- ``AsyncRedisClient``: async singleton used by the **gateway process** (ticket reads via redis.asyncio).
"""

from __future__ import annotations

import os
from typing import Any

import redis
import redis.asyncio as aioredis


# ---------------------------------------------------------------------------
# Sync client — worker processes (Pub/Sub publish)
# ---------------------------------------------------------------------------

class RedisClient:
    """Thin wrapper — one process-wide sync connection for Pub/Sub publish."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._r: redis.Redis | None = None

    @property
    def raw(self) -> redis.Redis:
        if self._r is None:
            self._r = redis.from_url(self._url, decode_responses=True)
        return self._r

    def publish(self, channel: str, message: str) -> int:
        return int(self.raw.publish(channel, message))


_client_singleton: RedisClient | None = None


def get_redis_client() -> RedisClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = RedisClient()
    return _client_singleton


# ---------------------------------------------------------------------------
# Async client — gateway process (ticket validation reads)
# ---------------------------------------------------------------------------

class AsyncRedisClient:
    """Async Redis wrapper for the gateway process (ticket validation only)."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._r: aioredis.Redis | None = None

    @property
    def raw(self) -> aioredis.Redis:
        if self._r is None:
            self._r = aioredis.from_url(self._url, decode_responses=True)
        return self._r

    async def get(self, key: str) -> str | None:
        return await self.raw.get(key)

    async def ping(self) -> Any:
        return await self.raw.ping()

    async def aclose(self) -> None:
        if self._r is not None:
            await self._r.aclose()
            self._r = None


_async_singleton: AsyncRedisClient | None = None


def get_async_redis_client() -> AsyncRedisClient:
    global _async_singleton
    if _async_singleton is None:
        _async_singleton = AsyncRedisClient()
    return _async_singleton
