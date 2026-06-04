# @trace UC-01-NF-5 — stream ticket validation against Redis
# @trace NFR-SEC-01

from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from infrastructure.redis_async_client import stream_ticket_key, validate_stream_ticket


@pytest.mark.asyncio
async def test_validate_stream_ticket_match(monkeypatch):
    import fakeredis.aioredis

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    meeting_id = "meet-ticket-1"
    ticket = "secret-ticket-uuid"
    await client.set(stream_ticket_key(meeting_id), ticket)

    async def fake_get_async_redis():
        return client

    import infrastructure.redis_async_client as mod

    monkeypatch.setattr(mod, "get_async_redis", fake_get_async_redis)

    assert await validate_stream_ticket(meeting_id, ticket) is True
    assert await validate_stream_ticket(meeting_id, "wrong") is False
    await client.aclose()
