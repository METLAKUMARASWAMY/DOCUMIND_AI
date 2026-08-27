from datetime import UTC, datetime
from hashlib import sha256
from hmac import compare_digest


def hash_document(content: bytes) -> str:
    return sha256(content).hexdigest()


def constant_time_equal(left: str, right: str) -> bool:
    return compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def redact_pii(text: str) -> str:
    """Lightweight local redaction for logs and demo responses."""
    import re

    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    text = re.sub(r"\b(?:\+?\d[\s-]?){10,15}\b", "[phone]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ssn]", text)
    return text
