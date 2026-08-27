from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:password|token|authorization|secret|bank_account|account_number|iban|routing)",
    re.IGNORECASE,
)


def mask_sensitive_value(key: str, value: Any) -> Any:
    if not SENSITIVE_KEY_PATTERN.search(key):
        return value
    if value is None:
        return None
    raw = str(value)
    if "account" in key.lower() or "iban" in key.lower() or "routing" in key.lower():
        compact = re.sub(r"\s+", "", raw)
        return f"***{compact[-4:]}" if len(compact) >= 4 else "***"
    return "[redacted]"


def sanitize_log_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: mask_sensitive_value(key, sanitize_log_data(item)) for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_log_data(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_log_data(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = sanitize_log_data(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_documind_json", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._documind_json = True  # type: ignore[attr-defined]
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
