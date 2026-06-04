# @trace UC-02.4-ALT-4.1 — empty agenda bypasses LLM scoring
# @trace UC-02.4-NF — adherence tiers snapped to allowed values

from __future__ import annotations

import pytest

from workers.text.adherence import (
    agenda_unusable,
    normalize_adherence_result,
    snap_context_fit,
)


def test_agenda_unusable_empty():
    assert agenda_unusable("") is True
    assert agenda_unusable("   ") is True
    assert agenda_unusable("Sprint goals") is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.91, 0.90),
        (0.50, 0.40),
        (0.66, 0.65),
        (0.19, 0.20),
    ],
)
def test_snap_context_fit(raw: float, expected: float):
    assert snap_context_fit(raw) == expected


def test_normalize_clears_reason_when_on_topic():
    result = normalize_adherence_result({"context_fit": 0.90, "reason": "should drop"})
    assert result["context_fit"] == 0.90
    assert result["on_topic"] is True
    assert result["reason"] is None


def test_normalize_off_topic_tier():
    result = normalize_adherence_result({"context_fit": 0.20, "reason": None})
    assert result["on_topic"] is False
    assert result["reason"]
