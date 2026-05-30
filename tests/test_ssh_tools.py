"""Tests for controlled SSH capability tools."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

import pytest


class _FakeTool:
    def __init__(self, fn):
        self.name = fn.__name__
        self.fn = fn


class _FakeToolManager:
    def __init__(self):
        self._tools: dict[str, _FakeTool] = {}

    def add_tool(self, fn):
        self._tools[fn.__name__] = _FakeTool(fn)

    async def call_tool(self, name, args=None):
        return self._tools[name].fn(**(args or {}))


class _FakeFastMCP:
    def __init__(self, *args, **kwargs):
        self._tool_manager = _FakeToolManager()

    def tool(self):
        def decorator(fn):
            self._tool_manager.add_tool(fn)
            return fn

        return decorator


@pytest.fixture
def ssh_mod():
    import app.tools.ssh_tools as ssh_tools

    return ssh_tools


@pytest.fixture
def policy_mod():
    import app.security.ssh_policy as ssh_policy

    return ssh_policy


def _completed(args: Any, stdout: str = "ok\n", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def _parse(payload: str) -> dict[str, Any]:
    return json.loads(payload)


def test_registers_expected_tools():
    from app.tools.ssh_tools import register_ssh_tools

    mcp = _FakeFastMCP()
    register_ssh_tools(mcp)
    assert {
        "ssh_service_status",
        "ssh_health_check",
        "ssh_journal_tail_redacted",
        "ssh_list_processes",
        "ssh_disk_usage",
        "ssh_port_check",
        "ssh_restart_service",
        "ssh_reload_service",
        "ssh_repair_hermes_mcp",
        "ssh_repair_gateway",
        "ssh_repair_wecom",
        "ssh_repair_feishu",
        "ssh_exec_command",
    }.issubset(set(mcp._tool_manager._tools))


def test_read_only_command_success(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    monkeypatch.setattr(ssh_tools.subprocess, "run", lambda *a, **k: _completed(a[0], stdout="system ok\n"))
    result = _parse(ssh_tools.ssh_health_check())
    assert result["status"] == "success"
    assert "system ok" in result["output"]


def test_service_whitelist_success(monkeypatch):
    import app.tools.ssh_tools as ssh_tools
    monkeypatch.setattr(ssh_tools.subprocess, "run", lambda *a, **k: _completed(a[0], stdout="active (running)\n"))
    result = _parse(ssh_tools.ssh_service_status(service="hermes-mcp"))
    assert result["status"] == "success"
    assert result["command"].startswith("systemctl status hermes-mcp")


def test_non_whitelist_service_rejected(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    called = {"value": False}

    def _boom(*args, **kwargs):
        called["value"] = True
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(ssh_tools.subprocess, "run", _boom)
    result = _parse(ssh_tools.ssh_restart_service("nginx"))
    assert result["status"] == "denied"
    assert "not on the allowed service list" in result["output"]
    assert called["value"] is False


def test_denylist_command_rejected(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    monkeypatch.setenv("SSH_FULL_SSH_ENABLED", "1")
    monkeypatch.setenv("SSH_L4_AUTHORIZED", "1")

    called = {"value": False}

    def _boom(*args, **kwargs):
        called["value"] = True
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(ssh_tools.subprocess, "run", _boom)
    result = _parse(ssh_tools.ssh_exec_command("cat /etc/shadow"))
    assert result["status"] == "denied"
    assert "denylist" in result["output"] or "blocked" in result["output"]
    assert called["value"] is False


def test_write_to_passwd_is_blocked(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    monkeypatch.setenv("SSH_FULL_SSH_ENABLED", "1")
    monkeypatch.setenv("SSH_L4_AUTHORIZED", "1")
    result = _parse(ssh_tools.ssh_exec_command("echo test >> /etc/passwd"))
    assert result["status"] == "denied"
    assert "passwd" in result["output"].lower()


def test_secret_output_is_redacted(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    def _run(*args, **kwargs):
        return _completed(
            args[0],
            stdout="API key: sk-test-secret-1234567890\nAuthorization: Bearer abc.def.ghi\n",
        )

    monkeypatch.setattr(ssh_tools.subprocess, "run", _run)
    result = _parse(ssh_tools.ssh_service_status("hermes-mcp"))
    assert "sk-test-secret-1234567890" not in result["output"]
    assert "[REDACTED: api_key]" in result["output"]


def test_exec_command_default_denied(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    monkeypatch.delenv("SSH_FULL_SSH_ENABLED", raising=False)
    monkeypatch.delenv("SSH_L4_AUTHORIZED", raising=False)
    result = _parse(ssh_tools.ssh_exec_command("echo hi"))
    assert result["status"] == "denied"
    assert "disabled by default" in result["output"]


def test_l4_missing_rejects_privileged_commands(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    monkeypatch.delenv("SSH_L4_AUTHORIZED", raising=False)
    result = _parse(ssh_tools.ssh_restart_service("hermes-mcp"))
    assert result["status"] == "denied"
    assert "L4 authorization" in result["output"]


def test_timeout_is_reported(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=5, output="partial\n", stderr="still waiting\n")

    monkeypatch.setattr(ssh_tools.subprocess, "run", _timeout)
    result = _parse(ssh_tools.ssh_service_status("hermes-mcp", timeout=5))
    assert result["status"] == "timeout"
    assert "partial" in result["output"]


def test_l4_authorized_command_succeeds(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    monkeypatch.setenv("SSH_L4_AUTHORIZED", "1")
    monkeypatch.setattr(ssh_tools.subprocess, "run", lambda *a, **k: _completed(a[0], stdout="reloaded\n"))
    result = _parse(ssh_tools.ssh_reload_service("hermes-mcp"))
    assert result["status"] == "success"
    assert "reloaded" in result["output"]


def test_journal_output_is_redacted(monkeypatch):
    import app.tools.ssh_tools as ssh_tools

    def _run(*args, **kwargs):
        return _completed(
            args[0],
            stdout="session=abc123\npassword=letmein\n",
        )

    monkeypatch.setattr(ssh_tools.subprocess, "run", _run)
    result = _parse(ssh_tools.ssh_journal_tail_redacted(timeout=5))
    assert "[REDACTED: session]" in result["output"]
    assert "[REDACTED: password]" in result["output"]


def test_audit_logger_emits_structured_record(caplog):
    from app.audit.ssh_audit import SSH_AUDIT_LOGGER, audit_event

    caplog.set_level(logging.INFO, logger=SSH_AUDIT_LOGGER)
    audit_event("systemctl status hermes-mcp", "tester", "success", 1.234, "[REDACTED: api_key]")
    assert any("duration_ms" in record.message for record in caplog.records)

