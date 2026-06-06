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


def test_agenda_unusable_none_input():
    assert agenda_unusable(None) is True


def test_snap_context_fit_returns_none_for_out_of_range():
    assert snap_context_fit(1.5) is None
    assert snap_context_fit(-0.1) is None


def test_snap_context_fit_returns_none_for_non_numeric():
    assert snap_context_fit("bad") is None


def test_normalize_missing_context_fit_key():
    result = normalize_adherence_result({"context_fit": None, "reason": "x"})
    assert result["context_fit"] is None
    assert result["on_topic"] is None


def test_normalize_invalid_string_score():
    result = normalize_adherence_result({"context_fit": "invalid"})
    assert result["reason"] == "invalid_score"


def test_normalize_preserves_custom_reason_for_020():
    result = normalize_adherence_result({"context_fit": 0.20, "reason": "custom"})
    assert result["reason"] == "custom"
