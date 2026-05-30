"""Output redaction helpers for SSH capability tools."""

from __future__ import annotations

import re
from typing import Final

REDACT_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE), "private_key"),
    (re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]{10,}){1,2}(?![A-Za-z0-9_-])"), "jwt"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b"), "bearer_token"),
    (re.compile(r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{8,}|api_[A-Za-z0-9_-]{8,})(?![A-Za-z0-9_-])", re.IGNORECASE), "api_key"),
    (re.compile(r"(?i)\bpassword\b\s*[:=]\s*[^,\s;]+"), "password"),
    (re.compile(r"(?i)\bcookie\b\s*[:=]\s*[^,\s;]+"), "cookie"),
    (re.compile(r"(?i)\bsession(?:_id)?\b\s*[:=]\s*[^,\s;]+"), "session"),
    (re.compile(r"(?i)\btoken\b\s*[:=]\s*[^,\s;]+"), "token"),
    (re.compile(r"(?i)\bBearer\b"), "bearer"),
    (re.compile(r"(?i)\bprivate\s+key\b"), "private_key"),
)


def _redact(text: str) -> str:
    redacted = text
    for pattern, label in REDACT_PATTERNS:
        redacted = pattern.sub(f"[REDACTED: {label}]", redacted)
    return redacted


def redact_journal_output(output: str) -> str:
    """Redact secrets from journalctl output."""
    return _redact(output or "")


def redact_command_output(output: str) -> str:
    """Redact secrets from generic command output."""
    return _redact(output or "")

