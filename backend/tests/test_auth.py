from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())


def test_local_demo_admin_login_is_seeded():
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@documind.ai", "password": "Admin@123"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["user"]["role"] == "ADMIN"


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
