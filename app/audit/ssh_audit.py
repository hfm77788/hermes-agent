"""Structured audit logging for SSH capability tools."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Final

SSH_AUDIT_LOGGER: Final[str] = "ssh_audit"

logger = logging.getLogger(SSH_AUDIT_LOGGER)


def audit_event(
    command: str,
    user: str,
    result: str,
    duration: float,
    redacted_output: str,
) -> None:
    """Emit a structured audit log entry."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "command": command,
        "duration_ms": int(max(duration, 0.0) * 1000),
        "result_status": result,
        "output_redacted": redacted_output,
    }
    logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))

