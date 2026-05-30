"""Security policy for controlled SSH capability tools."""

from __future__ import annotations

import os
import re
from typing import Final

ALLOWED_SERVICES: Final[list[str]] = ["hermes-mcp", "hermes-gateway", "wecom", "feishu"]

# The denylist intentionally stays small and obvious. The execution layer
# combines these patterns with a few write-sensitive special cases so that
# reads of /etc/passwd remain allowed while writes are blocked.
DENYLIST_PATTERNS: Final[tuple[str, ...]] = (
    r"\.env(?:\b|/|$)",
    r"\btoken\b",
    r"\bsecret\b",
    r"\bcookie\b",
    r"\bprivate\s+key\b",
    r"\bprintenv\b",
    r"\benv\b",
    r"\brm\s+-rf\b",
    r"/etc/shadow\b",
    r"/etc/passwd\b",
    r"\bssh\b",
    r"\bscp\b",
)

L4_AUTHORIZED_COMMANDS: Final[tuple[str, ...]] = (
    "ssh_restart_service",
    "ssh_reload_service",
    "ssh_repair_hermes_mcp",
    "ssh_repair_gateway",
    "ssh_repair_wecom",
    "ssh_repair_feishu",
    "ssh_exec_command",
)

DEFAULT_TIMEOUT: Final[int] = 30
MAX_TIMEOUT: Final[int] = 120
FULL_SSH_DEFAULT_ENABLED: Final[bool] = False

_WRITE_CONTEXT_RE = re.compile(
    r"(?:\b(?:>|>>|tee|cp|mv|install|truncate|sed\s+-i|perl\s+-i)\b|"
    r">|>>|\|\s*tee\b).{0,120}/etc/passwd\b",
    re.IGNORECASE | re.DOTALL,
)

_SERVICE_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
_DENYLIST_REGEXES = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in DENYLIST_PATTERNS]


def normalize_service_name(service: str) -> str:
    """Normalize a service/unit name for comparison."""
    return (service or "").strip()


def is_allowed_service(service: str) -> bool:
    """Return True when *service* is explicitly whitelisted."""
    normalized = normalize_service_name(service)
    return normalized in ALLOWED_SERVICES


def is_valid_service_name(service: str) -> bool:
    """Return True for a conservative systemd unit/service name."""
    normalized = normalize_service_name(service)
    return bool(normalized) and bool(_SERVICE_RE.fullmatch(normalized))


def clamp_timeout(value: int | float | str | None, default: int = DEFAULT_TIMEOUT) -> int:
    """Clamp a timeout to the allowed SSH bounds."""
    try:
        timeout = int(value) if value is not None else default
    except (TypeError, ValueError):
        timeout = default
    return max(1, min(timeout, MAX_TIMEOUT))


def is_l4_authorized() -> bool:
    """Return True when the caller has L4 authorization."""
    for env_name in ("SSH_L4_AUTHORIZED", "HERMES_SSH_L4_AUTHORIZED"):
        raw = os.getenv(env_name, "")
        if raw.lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def is_full_ssh_enabled() -> bool:
    """Return True when the dangerous exec capability is enabled."""
    if FULL_SSH_DEFAULT_ENABLED:
        return True
    raw = os.getenv("SSH_FULL_SSH_ENABLED", "")
    return raw.lower() in {"1", "true", "yes", "on"}


def requires_l4_auth(command_name: str) -> bool:
    """Return True when a capability requires L4 authorization."""
    return command_name in L4_AUTHORIZED_COMMANDS


def _has_write_to_passwd(command: str) -> bool:
    """Return True when a command appears to write to /etc/passwd."""
    return bool(_WRITE_CONTEXT_RE.search(command))


def is_denylisted_command(command: str) -> tuple[bool, str]:
    """Return (denied, reason) for a shell command string.

    Reads of /etc/passwd are allowed, but any obvious write path is blocked.
    """
    normalized = command or ""
    for regex, pattern in zip(_DENYLIST_REGEXES, DENYLIST_PATTERNS):
        if pattern == r"/etc/passwd\b":
            if "/etc/passwd" in normalized and _has_write_to_passwd(normalized):
                return True, "write access to /etc/passwd is blocked"
            continue
        if regex.search(normalized):
            return True, f"blocked by denylist pattern: {pattern}"
    return False, ""

