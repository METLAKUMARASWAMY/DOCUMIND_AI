from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.models.schemas import (
    AuthTokenResponse,
    AuthUser,
    ForgotPasswordRequest,
    LoginRequest,
    RefreshTokenRequest,
    SignupRequest,
)
from app.services.auth import auth_service, require_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
@router.post("/signup")
async def signup(request: SignupRequest) -> dict[str, Any]:
    try:
        user = auth_service.signup(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope({"user": user.model_dump(mode="json")})


@router.post("/login")
async def login(request: LoginRequest) -> dict[str, Any]:
    try:
        access, refresh, expires_in, refresh_expires_in, user = auth_service.login(request)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    response = AuthTokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=expires_in,
        refresh_expires_in=refresh_expires_in,
        user=user,
    )
    return _envelope(response.model_dump(mode="json"))


@router.post("/refresh")
async def refresh(request: RefreshTokenRequest) -> dict[str, Any]:
    try:
        access, refresh_token, expires_in, refresh_expires_in, user = auth_service.refresh(
            request.refresh_token
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    response = AuthTokenResponse(
        access_token=access,
        refresh_token=refresh_token,
        expires_in=expires_in,
        refresh_expires_in=refresh_expires_in,
        user=user,
    )
    return _envelope(response.model_dump(mode="json"))


@router.get("/me")
async def me(user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    return _envelope(user.model_dump(mode="json"))


@router.post("/logout")
async def logout(
    request: RefreshTokenRequest | None = None,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    access_token = None
    if authorization and authorization.lower().startswith("bearer "):
        access_token = authorization.split(" ", 1)[1].strip()
    try:
        auth_service.logout(access_token, request.refresh_token if request else None)
    except ValueError:
        pass
    return _envelope({"status": "logged_out"})


@router.post("/forgot-password")
async def forgot_password(_: ForgotPasswordRequest, request: Request) -> dict[str, Any]:
    # An external mail provider can consume this audit-safe event in production.
    return _envelope(
        {
            "status": "accepted",
            "message": "If the account exists, password reset instructions will be sent.",
            "request_id": request.headers.get("x-request-id"),
        }
    )


def _envelope(data: Any, errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "success": not errors,
        "case_id": None,
        "data": data,
        "errors": errors or [],
    }
