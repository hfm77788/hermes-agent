import io
import json
from types import SimpleNamespace

from hermes_cli import native_automation_progress as nap


def _payloads(text, prefix):
    return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]


def test_quiet_native_progress_is_opt_in(monkeypatch):
    monkeypatch.delenv("HERMES_NATIVE_PROGRESS", raising=False)
    agent = SimpleNamespace(tool_progress_callback=None, tool_start_callback=None, tool_complete_callback=None)
    assert nap.install_quiet_native_progress(agent) is False
    assert agent.tool_start_callback is None


def test_progress_emits_bounded_tool_metadata_only(monkeypatch):
    monkeypatch.setenv("HERMES_NATIVE_PROGRESS", "1")
    stderr = io.StringIO()
    monkeypatch.setattr(nap.sys, "stderr", stderr)
    agent = SimpleNamespace(tool_progress_callback=None, tool_start_callback=None, tool_complete_callback=None)

    assert nap.install_quiet_native_progress(agent) is True
    agent.tool_start_callback("call-1", "write_file", {"secret": "DO_NOT_LEAK"})
    agent.tool_complete_callback("call-1", "write_file", {"secret": "DO_NOT_LEAK"}, "PRIVATE_RESULT")

    progress = _payloads(stderr.getvalue(), nap.PROGRESS_PREFIX)
    assert [item["sequence"] for item in progress] == [1, 2, 3]
    assert progress[1]["phase"] == "editing"
    assert progress[1]["summary"] == "tool_started:write_file"
    assert "DO_NOT_LEAK" not in stderr.getvalue()
    assert "PRIVATE_RESULT" not in stderr.getvalue()


def test_partial_terminal_is_machine_visible(monkeypatch):
    monkeypatch.setenv("HERMES_NATIVE_PROGRESS", "1")
    stderr = io.StringIO()
    monkeypatch.setattr(nap.sys, "stderr", stderr)
    agent = SimpleNamespace(tool_progress_callback=None, tool_start_callback=None, tool_complete_callback=None)
    nap.install_quiet_native_progress(agent)

    nap.emit_native_terminal({"partial": True, "final_response": "PRIVATE"}, agent=agent)

    terminal = _payloads(stderr.getvalue(), nap.TERMINAL_PREFIX)
    assert terminal == [{"status": "partial", "reason": "partial"}]
    assert "PRIVATE" not in stderr.getvalue()
