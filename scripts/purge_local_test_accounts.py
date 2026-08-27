#!/usr/bin/env python3
"""Report or remove non-production accounts from the local auth state.

Verification work creates real, working accounts. The per-run pytest users are
harmless clutter; the hand-made ones are a liability, because their passwords
were typed into a terminal and may sit in a shell history or a transcript long
after the account is forgotten.

The command is read-only unless ``--execute`` is supplied, it never prints a
password or a password hash, and it writes a timestamped backup before changing
anything. Production accounts are matched by *not* looking like test accounts,
so an unexpected domain is reported and kept rather than quietly removed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Domains this project uses for accounts that must never reach production.
TEST_DOMAINS = ("@documind.test", "@documind.local", "@example.com")


def _classify(email: str) -> str:
    lowered = email.lower()
    if lowered.endswith("@documind.test"):
        return "generated_test_user"
    if any(lowered.endswith(domain) for domain in TEST_DOMAINS):
        return "manual_verification_account"
    return "keep"


def _state_path(state_dir: Path) -> Path:
    return state_dir / "users.json"


def _load(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} does not hold an account map")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "var" / "auth",
        help="directory holding users.json (default: var/auth)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually remove the accounts; without this the command only reports",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="also remove the per-run @documind.test users, not just the manual ones",
    )
    args = parser.parse_args()

    path = _state_path(args.state_dir)
    accounts = _load(path)

    buckets: dict[str, list[str]] = {
        "generated_test_user": [],
        "manual_verification_account": [],
        "keep": [],
    }
    for email in accounts:
        buckets[_classify(email)].append(email)

    targets = list(buckets["manual_verification_account"])
    if args.include_generated:
        targets += buckets["generated_test_user"]

    report = {
        "state_file": str(path),
        "total_accounts": len(accounts),
        "generated_test_users": len(buckets["generated_test_user"]),
        "manual_verification_accounts": sorted(buckets["manual_verification_account"]),
        "retained_accounts": sorted(buckets["keep"]),
        "would_remove": len(targets),
        "executed": False,
    }

    if args.execute and targets:
        backup = path.with_suffix(
            f".backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        shutil.copy2(path, backup)
        for email in targets:
            accounts.pop(email, None)
        path.write_text(json.dumps(accounts, indent=2, sort_keys=True))
        report["executed"] = True
        report["backup"] = str(backup)
        report["remaining_accounts"] = len(accounts)

    print(json.dumps(report, indent=2))
    if targets and not args.execute:
        print(
            "\nNothing was changed. Re-run with --execute to remove them.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
