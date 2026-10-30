from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerifyMismatchError
from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings
from app.models.schemas import AuthUser, LoginRequest, SignupRequest, UserRole


@dataclass
class StoredUser:
    user: AuthUser
    password_hash: str


class AuthService:
    """Local auth provider with the same contract used by a future identity service.

    Passwords are Argon2id hashes. Access and refresh tokens are signed JWTs; refresh
    tokens are single-use and their JWT IDs are persisted as hashes for rotation.
    """

    def __init__(self, state_dir: str | Path | None = None, secret: str | None = None) -> None:
        self._state_dir = Path(state_dir or settings.auth_state_dir)
        self._users_file = self._state_dir / "users.json"
        self._sessions_file = self._state_dir / "refresh_sessions.json"
        self._audit_file = self._state_dir / "audit_events.json"
        self._users: dict[str, StoredUser] = {}
        self._refresh_sessions: dict[str, dict[str, Any]] = {}
        self._audit_events: list[dict[str, Any]] = []
        self._revoked_access_jtis: dict[str, int] = {}
        self._lock = threading.RLock()
        self._password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
        self._secret = secret or settings.auth_secret or secrets.token_urlsafe(48)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def signup(self, request: SignupRequest, *, allow_privileged_role: bool = False) -> AuthUser:
        email = self._normalize_email(request.email)
        self._validate_signup(request, email)
        with self._lock:
            if email in self._users:
                raise ValueError("User already exists")
            role = (
                request.role
                if allow_privileged_role
                else UserRole(settings.public_registration_role)
            )
            tenant_id = (
                self._tenant_id(request.organisation_name)
                if allow_privileged_role
                else self._isolated_tenant_id(request.organisation_name)
            )
            user = AuthUser(
                user_id=f"user-{uuid4().hex[:12]}",
                name=request.name.strip(),
                email=email,
                role=role,
                tenant_id=tenant_id,
            )
            self._users[email] = StoredUser(
                user=user,
                password_hash=self._password_hasher.hash(request.password),
            )
            self._persist_users()
        return user

    def create_bootstrap_user(self, request: SignupRequest) -> AuthUser:
        return self.signup(request, allow_privileged_role=True)

    def initialize_persistence(self) -> None:
        if not settings.enable_mongodb:
            return
        from app.services.vision.persistence import vision_persistence

        with self._lock:
            for payload in vision_persistence.load_auth_users():
                password_hash = payload.pop("password_hash", None)
                payload.pop("email_hash", None)
                if not password_hash:
                    continue
                try:
                    user = AuthUser.model_validate(payload)
                except ValueError:
                    continue
                self._users[self._normalize_email(user.email)] = StoredUser(
                    user=user,
                    password_hash=str(password_hash),
                )
            self._refresh_sessions.update(vision_persistence.load_refresh_sessions())
            self._persist_users_local()
            self._write_json(self._sessions_file, self._refresh_sessions)
            for stored in self._users.values():
                vision_persistence.save_auth_user(
                    stored.user.model_dump(mode="json"),
                    stored.password_hash,
                )
            vision_persistence.save_refresh_sessions(self._refresh_sessions)

    def login(self, request: LoginRequest) -> tuple[str, str, int, int, AuthUser]:
        email = self._normalize_email(request.email)
        stored = self._users.get(email)
        if (
            not stored
            or not stored.user.active
            or not self._verify_password(
                request.password,
                stored.password_hash,
            )
        ):
            self._record_auth_audit("login_failed", email=email, success=False)
            raise ValueError("Invalid email or password")
        self._record_auth_audit(
            "login_succeeded",
            email=email,
            success=True,
            user=stored.user,
        )
        return (*self._issue_token_pair(stored.user), stored.user)

    def audit_events(self) -> list[dict[str, Any]]:
        """Return redacted authentication events for local audit inspection."""
        return [dict(event) for event in self._audit_events]

    def reauthenticate(self, user: AuthUser, password: str) -> tuple[str, int, int]:
        stored = self._user_by_id(user.user_id)
        if (
            not stored
            or not stored.user.active
            or stored.user.tenant_id != user.tenant_id
            or not self._verify_password(password, stored.password_hash)
        ):
            raise ValueError("Reauthentication failed")
        now = int(time.time())
        expires_in = settings.review_reauthentication_ttl_seconds
        payload = {
            "sub": stored.user.user_id,
            "email": stored.user.email,
            "role": stored.user.role.value,
            "tenant_id": stored.user.tenant_id,
            "iss": "documind-vision-ai",
            "aud": "documind-api",
            "iat": now,
            "auth_time": now,
            "jti": uuid4().hex,
            "type": "reauth",
            "purpose": "sensitive_review_decision",
            "exp": now + expires_in,
        }
        return jwt.encode(payload, self._secret, algorithm="HS256"), expires_in, now

    def verify_reauthentication(
        self,
        token: str,
        user: AuthUser,
    ) -> dict[str, Any]:
        payload = self._decode_token(token, expected_type="reauth")
        if (
            payload.get("sub") != user.user_id
            or payload.get("tenant_id") != user.tenant_id
            or payload.get("email") != user.email
            or payload.get("purpose") != "sensitive_review_decision"
        ):
            raise ValueError("Reauthentication token does not match the current user")
        return payload

    def refresh(self, refresh_token: str) -> tuple[str, str, int, int, AuthUser]:
        payload = self._decode_token(refresh_token, expected_type="refresh")
        token_key = self._token_key(str(payload["jti"]))
        with self._lock:
            session = self._refresh_sessions.get(token_key)
            if not session or session.get("revoked_at") is not None:
                raise ValueError("Refresh token has already been used or revoked")
            if int(session.get("expires_at", 0)) < int(time.time()):
                raise ValueError("Refresh token has expired")
            stored = self._user_by_id(str(payload["sub"]))
            if not stored or not stored.user.active:
                raise ValueError("User no longer exists")
            session["revoked_at"] = int(time.time())
            tokens = self._issue_token_pair(stored.user)
            session["replaced_by"] = self._token_key(
                str(self._decode_token(tokens[1], expected_type="refresh")["jti"])
            )
            self._persist_sessions()
        return (*tokens, stored.user)

    def me(self, token: str) -> AuthUser:
        payload = self._decode_token(token, expected_type="access")
        if str(payload.get("jti")) in self._revoked_access_jtis:
            raise ValueError("Token has been logged out")
        stored = self._user_by_id(str(payload["sub"]))
        if not stored or not stored.user.active:
            raise ValueError("User no longer exists")
        if stored.user.tenant_id != payload.get("tenant_id"):
            raise ValueError("Token tenant does not match the user")
        return stored.user

    def logout(self, access_token: str | None, refresh_token: str | None = None) -> None:
        now = int(time.time())
        with self._lock:
            if access_token:
                payload = self._decode_token(access_token, expected_type="access")
                self._revoked_access_jtis[str(payload["jti"])] = int(payload["exp"])
            if refresh_token:
                payload = self._decode_token(refresh_token, expected_type="refresh")
                session = self._refresh_sessions.get(self._token_key(str(payload["jti"])))
                if session:
                    session["revoked_at"] = now
                    self._persist_sessions()
            self._prune_revocations(now)

    def require_authorization(self, authorization: str | None) -> AuthUser:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization.split(" ", 1)[1].strip()
        try:
            return self.me(token)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    def list_users(self, tenant_id: str) -> list[AuthUser]:
        return [item.user for item in self._users.values() if item.user.tenant_id == tenant_id]

    def _issue_token_pair(self, user: AuthUser) -> tuple[str, str, int, int]:
        now = int(time.time())
        access_jti = uuid4().hex
        refresh_jti = uuid4().hex
        common = {
            "sub": user.user_id,
            "email": user.email,
            "role": user.role.value,
            "tenant_id": user.tenant_id,
            "iss": "documind-vision-ai",
            "aud": "documind-api",
            "iat": now,
        }
        access_payload = {
            **common,
            "jti": access_jti,
            "type": "access",
            "exp": now + settings.auth_token_ttl_seconds,
        }
        refresh_payload = {
            **common,
            "jti": refresh_jti,
            "type": "refresh",
            "exp": now + settings.auth_refresh_ttl_seconds,
        }
        access_token = jwt.encode(access_payload, self._secret, algorithm="HS256")
        refresh_token = jwt.encode(refresh_payload, self._secret, algorithm="HS256")
        self._refresh_sessions[self._token_key(refresh_jti)] = {
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "created_at": now,
            "expires_at": refresh_payload["exp"],
            "revoked_at": None,
            "replaced_by": None,
        }
        self._persist_sessions()
        return (
            access_token,
            refresh_token,
            settings.auth_token_ttl_seconds,
            settings.auth_refresh_ttl_seconds,
        )

    def _decode_token(self, token: str, *, expected_type: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                audience="documind-api",
                issuer="documind-vision-ai",
                options={"require": ["exp", "iat", "sub", "jti", "type", "tenant_id"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Token has expired") from exc
        except jwt.PyJWTError as exc:
            raise ValueError("Invalid token") from exc
        if payload.get("type") != expected_type:
            raise ValueError(f"Expected a {expected_type} token")
        return payload

    def _verify_password(self, password: str, password_hash: str) -> bool:
        try:
            valid = self._password_hasher.verify(password_hash, password)
            if valid and self._password_hasher.check_needs_rehash(password_hash):
                return True
            return valid
        except (VerifyMismatchError, InvalidHash):
            return False

    def _validate_signup(self, request: SignupRequest, email: str) -> None:
        if not request.name.strip() or len(request.name.strip()) > 120:
            raise ValueError("Name is required and must not exceed 120 characters")
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            raise ValueError("A valid email address is required")
        password = request.password
        if len(password) < 12 or not re.search(r"[A-Z]", password):
            raise ValueError("Password must be at least 12 characters and include uppercase text")
        if not re.search(r"[a-z]", password) or not re.search(r"\d", password):
            raise ValueError("Password must include lowercase text and a number")
        if not re.search(r"[^A-Za-z0-9]", password):
            raise ValueError("Password must include a symbol")
        if not request.organisation_name.strip():
            raise ValueError("Organisation name is required")

    def _load_state(self) -> None:
        users = self._read_json(self._users_file, {})
        for email, value in users.items():
            try:
                self._users[email] = StoredUser(
                    user=AuthUser.model_validate(value["user"]),
                    password_hash=str(value["password_hash"]),
                )
            except (KeyError, ValueError):
                continue
        sessions = self._read_json(self._sessions_file, {})
        if isinstance(sessions, dict):
            self._refresh_sessions = sessions
        audit_payload = self._read_json(self._audit_file, {"events": []})
        self._audit_events = [
            dict(event) for event in audit_payload.get("events", []) if isinstance(event, dict)
        ]

    def _record_auth_audit(
        self,
        event_type: str,
        *,
        email: str,
        success: bool,
        user: AuthUser | None = None,
    ) -> None:
        event = {
            "event_id": f"auth-audit-{uuid4().hex[:12]}",
            "event_type": event_type,
            "success": success,
            "email_hash": hashlib.sha256(email.encode("utf-8")).hexdigest(),
            "user_id": user.user_id if user else None,
            "tenant_id": user.tenant_id if user else None,
            "created_at": int(time.time()),
        }
        self._audit_events.append(event)
        self._write_json(self._audit_file, {"events": self._audit_events})

    def _persist_users(self) -> None:
        self._persist_users_local()
        if settings.enable_mongodb:
            from app.services.vision.persistence import vision_persistence

            for stored in self._users.values():
                vision_persistence.save_auth_user(
                    stored.user.model_dump(mode="json"),
                    stored.password_hash,
                )

    def _persist_users_local(self) -> None:
        self._write_json(
            self._users_file,
            {
                email: {
                    "user": stored.user.model_dump(mode="json"),
                    "password_hash": stored.password_hash,
                }
                for email, stored in self._users.items()
            },
        )

    def _persist_sessions(self) -> None:
        self._write_json(self._sessions_file, self._refresh_sessions)
        if settings.enable_mongodb:
            from app.services.vision.persistence import vision_persistence

            vision_persistence.save_refresh_sessions(self._refresh_sessions)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _read_json(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else default
        except (OSError, json.JSONDecodeError):
            return default

    def _user_by_id(self, user_id: str) -> StoredUser | None:
        return next((item for item in self._users.values() if item.user.user_id == user_id), None)

    def _normalize_email(self, email: str) -> str:
        return email.strip().lower()

    def _tenant_id(self, organisation_name: str) -> str:
        """Tenant for a *privileged* creation: named, and therefore joinable.

        Only reachable from ``create_bootstrap_user``, which is server-side
        provisioning. Public signup must not use this -- see
        ``_isolated_tenant_id``.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", organisation_name.strip().lower()).strip("-")
        return f"tenant-{slug[:48] or uuid4().hex[:12]}"

    def _isolated_tenant_id(self, organisation_name: str) -> str:
        """Tenant for a public self-registration: always a brand new one.

        Deriving the tenant from the organisation name alone made tenancy a
        function of a string the registrant types. Anyone could self-register
        with an existing customer's organisation name, land inside that tenant
        and read its documents -- verified end to end before this changed.

        There is no invitation or membership flow in the product, so no
        self-service request can legitimately join an existing tenant. Public
        signups therefore always get their own workspace; the slug is kept only
        so the id stays readable, and a random suffix guarantees separation
        even when two registrants type the same name.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", organisation_name.strip().lower()).strip("-")
        return f"tenant-{slug[:40] or 'org'}-{uuid4().hex[:12]}"

    def _token_key(self, token_id: str) -> str:
        return hashlib.sha256(token_id.encode("utf-8")).hexdigest()

    def _prune_revocations(self, now: int) -> None:
        self._revoked_access_jtis = {
            key: expires for key, expires in self._revoked_access_jtis.items() if expires >= now
        }


auth_service = AuthService()


async def require_user(authorization: str | None = Header(default=None)) -> AuthUser:
    return auth_service.require_authorization(authorization)


def require_roles(*allowed: UserRole):
    allowed_set = set(allowed)

    async def dependency(user: AuthUser = Depends(require_user)) -> AuthUser:
        if user.role not in allowed_set:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return dependency
