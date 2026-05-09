"""HTTP response envelope helpers (PYTHON_WORKERS_IMPLEMENTATION.md §10)."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class GatewayErrorBody(BaseModel):
    code: int
    message: str


class GatewayErrorEnvelope(BaseModel):
    success: bool = False
    error: GatewayErrorBody


def success_envelope_json_response(
    *,
    data: Any = None,
    message: str = "",
    extra: Optional[MutableMapping[str, Any]] = None,
    status_code: int = 200,
) -> JSONResponse:
    body: dict[str, Any] = {"success": True}
    body["data"] = {} if data is None else data
    body["message"] = message
    if extra:
        enc = dict(jsonable_encoder(extra))
        overlap = {"success", "data", "message"}.intersection(enc.keys())
        if overlap:
            raise ValueError(f"extra keys must not collide with envelope: {overlap}")
        body.update(enc)
    return JSONResponse(content=jsonable_encoder(body), status_code=status_code)


def error_envelope_json_response(
    code: int,
    message: str,
    http_status_code: Optional[int] = None,
) -> JSONResponse:
    status = http_status_code if http_status_code is not None else code
    body = GatewayErrorEnvelope(success=False, error=GatewayErrorBody(code=code, message=message))
    return JSONResponse(
        status_code=status,
        content=jsonable_encoder(body.model_dump()),
    )


def ingest_success_response() -> JSONResponse:
    return success_envelope_json_response(
        message="AI workers ingest accepted",
        data={},
    )
