"""Controlled SSH capability tools registered into the Hermes MCP server."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from typing import Any, Sequence

from app.audit.ssh_audit import audit_event
from app.security.redaction import redact_command_output, redact_journal_output
from app.security.ssh_policy import (
    ALLOWED_SERVICES,
    DEFAULT_TIMEOUT,
    MAX_TIMEOUT,
    clamp_timeout,
    is_allowed_service,
    is_denylisted_command,
    is_full_ssh_enabled,
    is_l4_authorized,
)

logger = logging.getLogger(__name__)


def _current_user() -> str:
    """Best-effort user label for audit records."""
    return os.getenv("USER") or os.getenv("USERNAME") or "unknown"


def _json_ok(
    *,
    tool: str,
    command: str,
    output: str,
    timeout_seconds: int,
    returncode: int | None,
    status: str,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "tool": tool,
        "status": status,
        "command": command,
        "timeout_seconds": timeout_seconds,
        "returncode": returncode,
        "output": output,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _combine_streams(stdout: str | None, stderr: str | None) -> str:
    parts = [part for part in (stdout or "", stderr or "") if part]
    return "\n".join(parts)


def _run_command(
    args: Sequence[str] | str,
    *,
    timeout: int,
    tool: str,
    redactor=redact_command_output,
    shell: bool = False,
    extra: dict[str, Any] | None = None,
) -> str:
    """Run a command, redact output, and return a JSON payload."""
    command = args if isinstance(args, str) else shlex.join(list(args))
    start = time.monotonic()
    status = "success"
    returncode: int | None = None
    raw_output = ""
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
            executable="/bin/bash" if shell else None,
            check=False,
        )
        returncode = completed.returncode
        raw_output = _combine_streams(completed.stdout, completed.stderr)
        if completed.returncode != 0:
            status = "error"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        returncode = None
        raw_output = _combine_streams(
            getattr(exc, "stdout", None),
            getattr(exc, "stderr", None),
        )
    except FileNotFoundError as exc:
        status = "error"
        returncode = None
        raw_output = str(exc)

    redacted = redactor(raw_output)
    duration = time.monotonic() - start
    audit_event(
        command=command,
        user=_current_user(),
        result=status,
        duration=duration,
        redacted_output=redacted,
    )
    return _json_ok(
        tool=tool,
        command=command,
        output=redacted,
        timeout_seconds=timeout,
        returncode=returncode,
        status=status,
        extra=extra,
    )


def _deny(tool: str, command: str, reason: str, timeout: int) -> str:
    redacted_reason = redact_command_output(reason)
    audit_event(
        command=command,
        user=_current_user(),
        result="denied",
        duration=0.0,
        redacted_output=redacted_reason,
    )
    return _json_ok(
        tool=tool,
        command=command,
        output=redacted_reason,
        timeout_seconds=timeout,
        returncode=None,
        status="denied",
    )


def _systemctl_command(action: str, service: str) -> list[str]:
    return ["systemctl", action, service]


def ssh_service_status(service: str = "hermes-mcp", timeout: int = DEFAULT_TIMEOUT) -> str:
    """Return `systemctl status` for an allowlisted service."""
    timeout = clamp_timeout(timeout)
    if not is_allowed_service(service):
        return _deny(
            "ssh_service_status",
            f"systemctl status {service}",
            f"service {service!r} is not on the allowed service list: {', '.join(ALLOWED_SERVICES)}",
            timeout,
        )
    return _run_command(
        _systemctl_command("status", service) + ["--no-pager", "--full"],
        timeout=timeout,
        tool="ssh_service_status",
    )


def ssh_health_check(timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a minimal connectivity/command-execution health check."""
    timeout = clamp_timeout(timeout)
    return _run_command(
        ["hostname"],
        timeout=timeout,
        tool="ssh_health_check",
    )


def ssh_journal_tail_redacted(
    service: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Return a redacted `journalctl -n 50` tail."""
    timeout = clamp_timeout(timeout)
    if service and not is_allowed_service(service):
        return _deny(
            "ssh_journal_tail_redacted",
            f"journalctl -u {service} -n 50 --no-pager",
            f"service {service!r} is not on the allowed service list: {', '.join(ALLOWED_SERVICES)}",
            timeout,
        )
    args = ["journalctl", "--no-pager", "-n", "50"]
    if service:
        args = ["journalctl", "-u", service, "--no-pager", "-n", "50"]
    return _run_command(
        args,
        timeout=timeout,
        tool="ssh_journal_tail_redacted",
        redactor=redact_journal_output,
    )


def ssh_list_processes(timeout: int = DEFAULT_TIMEOUT) -> str:
    """Return `ps aux` output."""
    timeout = clamp_timeout(timeout)
    return _run_command(
        ["ps", "aux"],
        timeout=timeout,
        tool="ssh_list_processes",
    )


def ssh_disk_usage(timeout: int = DEFAULT_TIMEOUT) -> str:
    """Return `df -h` output."""
    timeout = clamp_timeout(timeout)
    return _run_command(
        ["df", "-h"],
        timeout=timeout,
        tool="ssh_disk_usage",
    )


def ssh_port_check(timeout: int = DEFAULT_TIMEOUT) -> str:
    """Return listening sockets via `ss -tlnp` or `netstat`."""
    timeout = clamp_timeout(timeout)
    candidates: tuple[list[str], ...] = (["ss", "-tlnp"], ["netstat", "-tlnp"])
    last_error = ""
    for candidate in candidates:
        result = _run_command(
            candidate,
            timeout=timeout,
            tool="ssh_port_check",
        )
        parsed = json.loads(result)
        if parsed.get("status") == "success":
            parsed["used_command"] = shlex.join(candidate)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        last_error = parsed.get("output", "")
        if parsed.get("returncode") is not None:
            break
    return _json_ok(
        tool="ssh_port_check",
        command="ss -tlnp | netstat -tlnp",
        output=last_error or "no listening socket command was available",
        timeout_seconds=timeout,
        returncode=None,
        status="error",
    )


def ssh_restart_service(service: str, timeout: int = 60) -> str:
    """Restart an allowlisted service."""
    timeout = clamp_timeout(timeout)
    if not is_allowed_service(service):
        return _deny(
            "ssh_restart_service",
            f"systemctl restart {service}",
            f"service {service!r} is not on the allowed service list: {', '.join(ALLOWED_SERVICES)}",
            timeout,
        )
    if not is_l4_authorized():
        return _deny(
            "ssh_restart_service",
            f"systemctl restart {service}",
            "L4 authorization is required for service restarts",
            timeout,
        )
    return _run_command(
        _systemctl_command("restart", service),
        timeout=timeout,
        tool="ssh_restart_service",
    )


def ssh_reload_service(service: str, timeout: int = 60) -> str:
    """Reload an allowlisted service."""
    timeout = clamp_timeout(timeout)
    if not is_allowed_service(service):
        return _deny(
            "ssh_reload_service",
            f"systemctl reload {service}",
            f"service {service!r} is not on the allowed service list: {', '.join(ALLOWED_SERVICES)}",
            timeout,
        )
    if not is_l4_authorized():
        return _deny(
            "ssh_reload_service",
            f"systemctl reload {service}",
            "L4 authorization is required for service reloads",
            timeout,
        )
    return _run_command(
        _systemctl_command("reload", service),
        timeout=timeout,
        tool="ssh_reload_service",
    )


def _repair_service(service: str, tool: str, timeout: int) -> str:
    timeout = clamp_timeout(timeout)
    if not is_allowed_service(service):
        return _deny(
            tool,
            f"systemctl restart {service}",
            f"service {service!r} is not on the allowed service list: {', '.join(ALLOWED_SERVICES)}",
            timeout,
        )
    if not is_l4_authorized():
        return _deny(
            tool,
            f"systemctl restart {service}",
            "L4 authorization is required for repair operations",
            timeout,
        )
    restart = json.loads(ssh_restart_service(service=service, timeout=timeout))
    if restart.get("status") != "success":
        return json.dumps(restart, indent=2, ensure_ascii=False)
    status = json.loads(ssh_service_status(service=service, timeout=timeout))
    payload = {
        "tool": tool,
        "status": "success" if status.get("status") == "success" else status.get("status", "error"),
        "service": service,
        "restart": restart,
        "status_check": status,
    }
    audit_event(
        command=f"repair {service}",
        user=_current_user(),
        result=payload["status"],
        duration=0.0,
        redacted_output=json.dumps(payload, ensure_ascii=False),
    )
    return json.dumps(payload, indent=2, ensure_ascii=False)


def ssh_repair_hermes_mcp(timeout: int = 120) -> str:
    """Repair the Hermes MCP service."""
    return _repair_service("hermes-mcp", "ssh_repair_hermes_mcp", timeout)


def ssh_repair_gateway(timeout: int = 120) -> str:
    """Repair the Hermes gateway service."""
    return _repair_service("hermes-gateway", "ssh_repair_gateway", timeout)


def ssh_repair_wecom(timeout: int = 120) -> str:
    """Repair the WeCom adapter service."""
    return _repair_service("wecom", "ssh_repair_wecom", timeout)


def ssh_repair_feishu(timeout: int = 120) -> str:
    """Repair the Feishu adapter service."""
    return _repair_service("feishu", "ssh_repair_feishu", timeout)


def ssh_exec_command(
    command: str,
    timeout: int = MAX_TIMEOUT,
    *,
    allow_full_ssh: bool = False,
    user_authorization: str = "",
) -> str:
    """Execute an arbitrary command when full SSH is explicitly enabled.

    Triple-gate authorization:
      1. allow_full_ssh=True must be passed in the call payload
      2. user_authorization must be non-empty and describe command intent
      3. Environment variables SSH_FULL_SSH_ENABLED=1 + SSH_L4_AUTHORIZED=1
         must both be set

    This capability is disabled by default.  It is strictly bounded to the
    local Hermes MCP host and cannot be used for ssh/scp jump-host hopping.
    """
    timeout = clamp_timeout(timeout)
    if not is_full_ssh_enabled():
        return _deny(
            "ssh_exec_command",
            command,
            "ssh_exec_command is disabled by default",
            timeout,
        )
    if not is_l4_authorized():
        return _deny(
            "ssh_exec_command",
            command,
            "L4 authorization is required for arbitrary command execution",
            timeout,
        )
    if not allow_full_ssh:
        return _deny(
            "ssh_exec_command",
            command,
            "allow_full_ssh=true must be set in the call payload",
            timeout,
        )
    if not user_authorization or not user_authorization.strip():
        return _deny(
            "ssh_exec_command",
            command,
            "user_authorization must be non-empty and describe command intent",
            timeout,
        )
    denied, reason = is_denylisted_command(command)
    if denied:
        return _deny("ssh_exec_command", command, reason, timeout)
    return _run_command(
        ["bash", "-lc", command],
        timeout=timeout,
        tool="ssh_exec_command",
        shell=False,
        extra={"user_authorization": user_authorization.strip()},
    )


def register_ssh_tools(mcp: Any) -> None:
    """Register SSH capability tools on an existing FastMCP server."""
    mcp.tool()(ssh_service_status)
    mcp.tool()(ssh_health_check)
    mcp.tool()(ssh_journal_tail_redacted)
    mcp.tool()(ssh_list_processes)
    mcp.tool()(ssh_disk_usage)
    mcp.tool()(ssh_port_check)
    mcp.tool()(ssh_restart_service)
    mcp.tool()(ssh_reload_service)
    mcp.tool()(ssh_repair_hermes_mcp)
    mcp.tool()(ssh_repair_gateway)
    mcp.tool()(ssh_repair_wecom)
    mcp.tool()(ssh_repair_feishu)
    mcp.tool()(ssh_exec_command)
