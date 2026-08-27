"""No credential may reach a tracked file, and no default may be guessable.

Directive Phase 31. Verification work creates real, working accounts; the
passwords are typed into a terminal and can outlive the account in a shell
history or a transcript. Source, documentation and fixtures must stay clean of
them, and the local purge tool must be able to find them.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return [PROJECT_ROOT / name for name in result.stdout.split("\0") if name]


#: Accounts created by hand during live verification. These have real passwords.
VERIFICATION_ACCOUNTS = (
    "stack-admin@documind.local",
    "host-e2e@documind.local",
    "fresh-tenant@documind.local",
)


def test_no_verification_account_appears_in_a_tracked_file():
    offenders: list[str] = []
    for path in _tracked_files():
        if not path.is_file() or path.suffix in {".png", ".jpg", ".pdf", ".ico", ".woff2"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for account in VERIFICATION_ACCOUNTS:
            if account in text and path.name != "test_credential_hygiene.py":
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {account}")
    assert not offenders, offenders


def test_env_files_are_not_tracked():
    tracked = {path.name for path in _tracked_files()}
    assert ".env" not in tracked
    for path in _tracked_files():
        assert path.name != ".env", path


def test_local_state_directory_is_ignored():
    result = subprocess.run(
        ["git", "check-ignore", "var/auth/users.json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, "var/ must be gitignored; it holds real password hashes"


def test_generated_test_passwords_are_not_literals():
    """Test users must get a fresh random password, not a shared constant."""
    helpers = (PROJECT_ROOT / "backend/app/tests/auth_helpers.py").read_text()
    assert "secrets.token_urlsafe" in helpers, helpers[:400]
    assert not re.search(r'_PASSWORD\s*=\s*["\'][A-Za-z0-9@!#$%^&*]{6,}["\']', helpers)


def test_the_purge_tool_finds_the_manual_accounts(tmp_path):
    """The tool must classify a hand-made account as removable and a real one
    as keep — reversing those would delete production users."""
    state = tmp_path / "auth"
    state.mkdir()
    (state / "users.json").write_text(
        json.dumps(
            {
                "pytest-reviewer-abc@documind.test": {"password_hash": "x", "user": {}},
                "stack-admin@documind.local": {"password_hash": "x", "user": {}},
                "real.person@customer.example.org": {"password_hash": "x", "user": {}},
            }
        )
    )

    result = subprocess.run(
        [
            "python",
            str(PROJECT_ROOT / "scripts/purge_local_test_accounts.py"),
            "--state-dir",
            str(state),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["manual_verification_accounts"] == ["stack-admin@documind.local"]
    assert report["retained_accounts"] == ["real.person@customer.example.org"]
    assert report["generated_test_users"] == 1
    assert report["executed"] is False, "a dry run must never modify state"


def test_the_purge_tool_is_read_only_without_execute(tmp_path):
    state = tmp_path / "auth"
    state.mkdir()
    users = {"stack-admin@documind.local": {"password_hash": "x", "user": {}}}
    (state / "users.json").write_text(json.dumps(users))

    subprocess.run(
        [
            "python",
            str(PROJECT_ROOT / "scripts/purge_local_test_accounts.py"),
            "--state-dir",
            str(state),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert json.loads((state / "users.json").read_text()) == users


def test_the_purge_tool_backs_up_before_removing(tmp_path):
    state = tmp_path / "auth"
    state.mkdir()
    (state / "users.json").write_text(
        json.dumps(
            {
                "stack-admin@documind.local": {"password_hash": "x", "user": {}},
                "keep.me@customer.example.org": {"password_hash": "x", "user": {}},
            }
        )
    )

    result = subprocess.run(
        [
            "python",
            str(PROJECT_ROOT / "scripts/purge_local_test_accounts.py"),
            "--state-dir",
            str(state),
            "--execute",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["executed"] is True
    assert Path(report["backup"]).is_file(), "state was changed without a backup"
    remaining = json.loads((state / "users.json").read_text())
    assert list(remaining) == ["keep.me@customer.example.org"]


def test_the_purge_report_never_prints_a_password_hash(tmp_path):
    state = tmp_path / "auth"
    state.mkdir()
    (state / "users.json").write_text(
        json.dumps(
            {"stack-admin@documind.local": {"password_hash": "argon2-secret-hash", "user": {}}}
        )
    )

    result = subprocess.run(
        [
            "python",
            str(PROJECT_ROOT / "scripts/purge_local_test_accounts.py"),
            "--state-dir",
            str(state),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "argon2-secret-hash" not in result.stdout
    assert "argon2-secret-hash" not in result.stderr


@pytest.mark.parametrize(
    "email,expected",
    [
        ("pytest-admin-1@documind.test", "generated_test_user"),
        ("stack-admin@documind.local", "manual_verification_account"),
        ("someone@example.com", "manual_verification_account"),
        ("real@customer.co.uk", "keep"),
        ("REAL@Customer.Co.UK", "keep"),
        ("Stack-Admin@DocuMind.Local", "manual_verification_account"),
    ],
)
def test_classification_is_case_insensitive_and_conservative(email, expected):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "purge_tool", PROJECT_ROOT / "scripts/purge_local_test_accounts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._classify(email) == expected
