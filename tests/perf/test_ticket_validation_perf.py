# @trace NFR-PERF-01 — stream ticket validation lookup (micro-benchmark)
# @trace NFR-SEC-01 — ticket validation hot path

from __future__ import annotations

import time

import pytest

fakeredis = pytest.importorskip("fakeredis")

from infrastructure.redis_async_client import stream_ticket_key, validate_stream_ticket

WARMUP_ITERATIONS = 50
BENCHMARK_ITERATIONS = 2000
THRESHOLD_SECONDS = 1.0


@pytest.mark.asyncio
@pytest.mark.perf
async def test_validate_stream_ticket_within_threshold(monkeypatch) -> None:
    import fakeredis.aioredis

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    meeting_id = "meet-perf-ticket"
    ticket = "secret-ticket-uuid"
    await client.set(stream_ticket_key(meeting_id), ticket)

    async def fake_get_async_redis():
        return client

    import infrastructure.redis_async_client as mod

    monkeypatch.setattr(mod, "get_async_redis", fake_get_async_redis)

    for _ in range(WARMUP_ITERATIONS):
        assert await validate_stream_ticket(meeting_id, ticket) is True

    t0 = time.perf_counter()
    for _ in range(BENCHMARK_ITERATIONS):
        assert await validate_stream_ticket(meeting_id, ticket) is True
    elapsed = time.perf_counter() - t0

    assert elapsed < THRESHOLD_SECONDS
    await client.aclose()
