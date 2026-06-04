# @trace SDD-DG4 — health reflects worker readiness
# @trace NFR-REL-02 — degraded modules reported in health

from __future__ import annotations

from unittest.mock import MagicMock

from gateway import _health_failure_labels, _health_is_ready


def test_health_ready_when_redis_and_supervisor_ok():
    sup = MagicMock()
    sup.failed_modules = []
    sup.dead_modules.return_value = []
    sup.can_accept_ingest.return_value = True
    assert _health_is_ready(True, sup) is True


def test_health_not_ready_without_redis():
    sup = MagicMock()
    sup.can_accept_ingest.return_value = True
    assert _health_is_ready(False, sup) is False


def test_failure_labels_include_failed_modules():
    sup = MagicMock()
    sup.failed_modules = ["text"]
    sup.dead_modules.return_value = ["audio"]
    labels = _health_failure_labels(True, sup)
    assert "text" in labels
    assert "audio" in labels
