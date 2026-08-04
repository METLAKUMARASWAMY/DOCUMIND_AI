from __future__ import annotations

import secrets

from fastapi.testclient import TestClient

from app.models.schemas import SignupRequest, UserRole
from app.services.auth import auth_service

_RUN_ID = secrets.token_hex(6)
_PASSWORD = f"Test-{secrets.token_urlsafe(18)}-Aa1!"
_USERS: dict[UserRole, tuple[str, str]] = {}


def auth_headers(client: TestClient, role: UserRole = UserRole.reviewer) -> dict[str, str]:
    email, password = _ensure_user(role)
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    token = response.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _ensure_user(role: UserRole) -> tuple[str, str]:
    existing = _USERS.get(role)
    if existing:
        return existing
    email = f"pytest-{role.value.lower()}-{_RUN_ID}@documind.test"
    auth_service.create_bootstrap_user(
        SignupRequest(
            name=f"Pytest {role.value.title()}",
            email=email,
            password=_PASSWORD,
            role=role,
            organisation_name=f"Pytest {_RUN_ID}",
        )
    )
    _USERS[role] = (email, _PASSWORD)
    return email, _PASSWORD
