#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))

from app.models.schemas import SignupRequest, UserRole  # noqa: E402
from app.services.auth import auth_service  # noqa: E402

DEMO_USERS = [
    ("admin@documind.local", UserRole.admin, "Local Admin"),
    ("reviewer@documind.local", UserRole.reviewer, "Local Reviewer"),
    ("auditor@documind.local", UserRole.auditor, "Local Auditor"),
    ("processor@documind.local", UserRole.processor, "Local Processor"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local DocuMind auth users.")
    parser.add_argument("--email", default="admin@documind.local")
    parser.add_argument("--name", default="Local Admin")
    parser.add_argument("--role", default="ADMIN", choices=[role.value for role in UserRole])
    parser.add_argument("--organisation", default="DocuMind Local")
    parser.add_argument(
        "--all-demo-roles",
        action="store_true",
        help="Create admin, reviewer, auditor, and processor users with generated passwords.",
    )
    args = parser.parse_args()

    users = DEMO_USERS if args.all_demo_roles else [(args.email, UserRole(args.role), args.name)]
    configured_password = os.environ.get("DOCUMIND_BOOTSTRAP_PASSWORD")
    created: list[tuple[str, str]] = []
    existing: list[str] = []

    for email, role, name in users:
        password = configured_password or _generated_password()
        request = SignupRequest(
            name=name,
            email=email,
            password=password,
            role=role,
            organisation_name=args.organisation,
        )
        try:
            auth_service.create_bootstrap_user(request)
            created.append((f"{email} ({role.value})", password))
        except ValueError as exc:
            if "already exists" not in str(exc).lower():
                raise
            existing.append(f"{email} ({role.value})")

    print("DocuMind local auth bootstrap")
    print(f"State directory: {auth_service._state_dir}")
    if created:
        print("Created:")
        for item, password in created:
            print(f"- {item}")
            print(f"  Password: {password}")
    if existing:
        print("Already existed:")
        for item in existing:
            print(f"- {item}")
    if not created and existing:
        print("No passwords were changed.")


def _generated_password() -> str:
    return f"DocuMind-{secrets.token_urlsafe(12)}-A1!"


if __name__ == "__main__":
    main()
