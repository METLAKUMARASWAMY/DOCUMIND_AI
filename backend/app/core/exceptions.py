from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("documind.api")


class DocuMindError(Exception):
    def __init__(
        self, message: str, *, status_code: int = 400, code: str = "request_error"
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocuMindError)
    async def handle_documind_error(request: Request, exc: DocuMindError) -> JSONResponse:
        return _json_error(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("code", _CODE_BY_STATUS.get(exc.status_code, "request_error")))
            message = str(detail.get("message", "Request failed"))
        else:
            code = _CODE_BY_STATUS.get(exc.status_code, "request_error")
            message = str(detail) if detail else "Request failed"
        return _json_error(
            request,
            exc.status_code,
            code,
            message,
            # `detail` is part of the wire contract, not decoration: the approval
            # policy returns a structured blocker list through it, and the
            # frontend's parseResponse() reads it for the display message. It is
            # passed through verbatim rather than flattened into the envelope.
            detail=detail,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {
                "code": "validation_error",
                "message": str(item.get("msg", "Invalid value")),
                "field": ".".join(str(part) for part in item.get("loc", ())[1:]) or None,
            }
            for item in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {
                    "success": False,
                    "case_id": None,
                    "data": None,
                    "errors": errors
                    or [{"code": "validation_error", "message": "Invalid request"}],
                    "request_id": _request_id(request),
                    # Preserved for consumers that read FastAPI's native shape.
                    "detail": exc.errors(),
                }
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Never leak internals; never lose the trace.

        The client gets the same envelope every other error uses plus the
        correlation id. The stack trace goes to the log only, keyed by that same
        id, so an operator can find it without the caller ever seeing it.
        """

        request_id = _request_id(request)
        LOGGER.exception(
            "unhandled exception",
            extra={
                "context": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "exception_type": type(exc).__name__,
                }
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "case_id": None,
                "data": None,
                "errors": [
                    {
                        "code": "internal_error",
                        "message": (
                            "The request could not be completed. Quote the request id "
                            "when reporting this."
                        ),
                    }
                ],
                "request_id": request_id,
            },
        )


_CODE_BY_STATUS = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    503: "service_unavailable",
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None) or request.headers.get("x-request-id")


def _json_error(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    *,
    detail: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = _error_payload(code, message)
    payload["request_id"] = _request_id(request)
    if detail is not None:
        payload["detail"] = jsonable_encoder(detail)
    if status_code >= 500:
        LOGGER.error(
            "request failed",
            extra={
                "context": {
                    "request_id": payload["request_id"],
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "code": code,
                }
            },
        )
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "case_id": None,
        "data": None,
        "errors": [{"code": code, "message": message}],
    }
