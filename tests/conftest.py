"""Shared pytest fixtures for meetai-ai-workers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("MEDIA_CHUNK_DURATION_MS", "6000")
os.environ.setdefault("TEXT_TRANSCRIPT_RING_BUFFER_SLOTS", "4")


@pytest.fixture
def anyio_backend():
    return "asyncio"
