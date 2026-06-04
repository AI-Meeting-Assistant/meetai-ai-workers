# @trace UC-02 — gateway error envelope for validation failures

from __future__ import annotations

import json

from gateway_schemas import error_envelope_json_response, ingest_success_response


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
