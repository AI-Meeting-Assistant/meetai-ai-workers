"""Redis Pub/Sub helpers for modality results (PYTHON_WORKERS_IMPLEMENTATION.md §5)."""

from __future__ import annotations

import json
from typing import Any, Protocol


class SupportsPublish(Protocol):
    def publish(self, channel: str, message: str) -> Any: ...


def channel_vision(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:vision"


def channel_audio(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:audio"


def channel_text(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:text"


def channel_recorded_complete(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:recorded-complete"


def channel_recorded_error(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:recorded-error"


def publish_json(redis_client: SupportsPublish, channel: str, data: dict[str, Any]) -> None:
    redis_client.publish(channel, json.dumps(data, ensure_ascii=False))
