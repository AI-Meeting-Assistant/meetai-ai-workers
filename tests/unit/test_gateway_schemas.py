# @trace UC-02 — gateway error envelope for validation failures

from __future__ import annotations

import json

import pytest

from gateway_schemas import (
    error_envelope_json_response,
    ingest_success_response,
    success_envelope_json_response,
)


def test_success_envelope_defaults():
    resp = success_envelope_json_response()
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["success"] is True
    assert body["data"] == {}
    assert body["message"] == ""


def test_success_envelope_custom_status_and_data():
    resp = success_envelope_json_response(data={"k": 1}, message="done", status_code=201)
    assert resp.status_code == 201
    body = json.loads(resp.body)
    assert body["data"] == {"k": 1}


def test_success_envelope_extra_key_collision_raises():
    with pytest.raises(ValueError, match="collide"):
        success_envelope_json_response(extra={"data": "bad"})


def test_error_envelope_code_as_status_when_omitted():
    resp = error_envelope_json_response(404, "not found")
    assert resp.status_code == 404


def test_ingest_success_envelope():
    resp = ingest_success_response()
    body = json.loads(resp.body)
    assert body["success"] is True
    assert "ingest" in body["message"].lower()


def test_error_envelope_maps_http_status():
    resp = error_envelope_json_response(401, "invalid stream ticket", http_status_code=401)
    assert resp.status_code == 401
    body = json.loads(resp.body)
    assert body["success"] is False
    assert body["error"]["code"] == 401
