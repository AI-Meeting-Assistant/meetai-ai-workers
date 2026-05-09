"""Sync Redis singleton for worker processes (publish). Gateway may use asyncio variant later."""

from __future__ import annotations

import os

import redis


class RedisClient:
    """Thin wrapper — one process-wide connection for Pub/Sub publish."""

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
