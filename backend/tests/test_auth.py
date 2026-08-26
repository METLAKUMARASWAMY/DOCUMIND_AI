from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import LoginRequest, SignupRequest, UserRole
from app.services.auth import AuthService

client = TestClient(create_app())


def test_security_headers_request_id_and_auth_cache_policy():
    request_id = f"security-test-{uuid4().hex}"
    expected_security_headers = {
        "x-content-type-options": "nosniff",
        "x-frame-options": "DENY",
        "referrer-policy": "no-referrer",
        "permissions-policy": "camera=(), microphone=(), geolocation=()",
        "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
    }

    health = client.get("/health/live", headers={"X-Request-ID": request_id})
    assert health.status_code == 200
    assert health.headers["x-request-id"] == request_id
    for header, expected in expected_security_headers.items():
        assert health.headers[header] == expected

    reset = client.post(
        "/api/v1/auth/forgot-password",
        headers={"X-Request-ID": request_id},
        json={"email": "security-header-check@documind.test"},
    )
    assert reset.status_code == 200
    assert reset.headers["cache-control"] == "no-store"
    assert reset.headers["x-request-id"] == request_id
    for header, expected in expected_security_headers.items():
        assert reset.headers[header] == expected


def test_privileged_users_require_explicit_secure_bootstrap(tmp_path):
    service = AuthService(
        state_dir=tmp_path / "auth",
        secret="test-secret-at-least-thirty-two-bytes",
    )
    with pytest.raises(ValueError, match="Invalid email or password"):
        service.login(LoginRequest(email="admin@documind.ai", password="Admin@123"))

    password = f"Bootstrap-{uuid4().hex}-Aa1!"
    request = SignupRequest(
        name="Bootstrap Admin",
        email="bootstrap-admin@documind.test",
        password=password,
        role=UserRole.admin,
        organisation_name="Bootstrap Test",
    )

    user = service.create_bootstrap_user(request)
    *_, authenticated = service.login(LoginRequest(email=request.email, password=password))

    assert user.role == UserRole.admin
    assert authenticated.user_id == user.user_id


def test_login_audit_is_persisted_and_never_stores_plain_email_or_password(tmp_path):
    state_dir = tmp_path / "auth-audit"
    password = f"Audit-{uuid4().hex}-Aa1!"
    service = AuthService(
        state_dir=state_dir,
        secret="test-secret-at-least-thirty-two-bytes",
    )
    user = service.create_bootstrap_user(
        SignupRequest(
            name="Audit Reviewer",
            email="audit-reviewer@documind.test",
            password=password,
            role=UserRole.reviewer,
            organisation_name="Audit Test",
        )
    )

    with pytest.raises(ValueError, match="Invalid email or password"):
        service.login(LoginRequest(email=user.email, password="Wrong-Password-Aa1!"))
    service.login(LoginRequest(email=user.email, password=password))

    restarted = AuthService(
        state_dir=state_dir,
        secret="test-secret-at-least-thirty-two-bytes",
    )
    events = restarted.audit_events()
    assert [event["event_type"] for event in events[-2:]] == [
        "login_failed",
        "login_succeeded",
    ]
    serialized = (state_dir / "audit_events.json").read_text(encoding="utf-8")
    assert user.email not in serialized
    assert password not in serialized
    assert events[-1]["user_id"] == user.user_id
    assert events[-1]["tenant_id"] == user.tenant_id


def _registration() -> dict[str, str]:
    suffix = uuid4().hex[:10]
    return {
        "name": "New Viewer",
        "email": f"viewer-{suffix}@documind.test",
        "password": f"Viewer-{suffix}-Strong1!",
        "role": "ADMIN",
        "organisation_name": f"Auth Test {suffix}",
    }


def test_register_login_refresh_rotation_me_and_logout():
    registration = _registration()
    signup = client.post("/api/v1/auth/register", json=registration)
    assert signup.status_code == 200
    assert signup.json()["data"]["user"]["role"] == "VIEWER"

    login = client.post(
        "/api/v1/auth/login",
        json={"email": registration["email"], "password": registration["password"]},
    )
    assert login.status_code == 200
    payload = login.json()["data"]
    assert payload["access_token"]
    assert payload["refresh_token"]

    me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["data"]["email"] == registration["email"]

    refreshed = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": payload["refresh_token"]},
    )
    assert refreshed.status_code == 200
    rotated = refreshed.json()["data"]
    assert rotated["refresh_token"] != payload["refresh_token"]

    replay = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": payload["refresh_token"]},
    )
    assert replay.status_code == 401

    logout = client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {rotated['access_token']}"},
        json={"refresh_token": rotated["refresh_token"]},
    )
    assert logout.status_code == 200
    revoked_me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {rotated['access_token']}"},
    )
    assert revoked_me.status_code == 401


def test_password_policy_and_generic_password_reset_response():
    registration = _registration()
    registration["password"] = "weak"
    rejected = client.post("/api/v1/auth/signup", json=registration)
    assert rejected.status_code == 400

    reset = client.post("/api/v1/auth/forgot-password", json={"email": registration["email"]})
    assert reset.status_code == 200
    assert reset.json()["data"]["status"] == "accepted"
