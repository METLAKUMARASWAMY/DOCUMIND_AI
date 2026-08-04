from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class DocuMindError(Exception):
    def __init__(self, message: str, *, status_code: int = 400, code: str = "request_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocuMindError)
    async def handle_documind_error(_: Request, exc: DocuMindError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message),
        )


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "case_id": None,
        "data": None,
        "errors": [{"code": code, "message": message}],
    }
