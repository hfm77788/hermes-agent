"""Machine-readable progress for quiet Hermes automation.

Activated only when HERMES_NATIVE_PROGRESS=1.  The channel intentionally emits
bounded lifecycle metadata only: never tool arguments, tool results, prompts, or
credentials.  stdout remains reserved for the quiet final response.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from typing import Any

PROGRESS_PREFIX = "@@HERMES_PROGRESS "
TERMINAL_PREFIX = "@@HERMES_TERMINAL "
_ENV = "HERMES_NATIVE_PROGRESS"
_SAFE_REASON = re.compile(r"^[a-zA-Z0-9_.:-]{1,80}$")


def native_progress_enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _write(prefix: str, payload: dict[str, Any]) -> None:
    try:
        sys.stderr.write(prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stderr.flush()
    except Exception:
        # Observability must never change the business result.
        pass


def _bounded_tool_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-zA-Z0-9_.:-]", "_", text)
    return text[:80] or "unknown"


def _phase_for_tool(name: str) -> str:
    low = name.lower()
    if any(token in low for token in ("write", "patch", "edit", "create_file", "update_file")):
        return "editing"
    if any(token in low for token in ("read", "search", "fetch", "list", "status", "inspect", "browser")):
        return "inspecting"
    return "executing"


class NativeProgressEmitter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(self, phase: str, summary: str) -> None:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        _write(PROGRESS_PREFIX, {
            "sequence": sequence,
            "phase": phase,
            "summary": str(summary or "")[:240],
            "waiting_on": None,
        })

    def tool_started(self, *args: Any, **kwargs: Any) -> None:
        name = _bounded_tool_name(args[1] if len(args) > 1 else kwargs.get("function_name"))
        self.emit(_phase_for_tool(name), f"tool_started:{name}")

    def tool_completed(self, *args: Any, **kwargs: Any) -> None:
        name = _bounded_tool_name(args[1] if len(args) > 1 else kwargs.get("function_name"))
        self.emit(_phase_for_tool(name), f"tool_completed:{name}")

    def tool_progress(self, *args: Any, **kwargs: Any) -> None:
        name = _bounded_tool_name(args[1] if len(args) > 1 else kwargs.get("function_name"))
        event = _bounded_tool_name(args[0] if args else kwargs.get("event"))
        self.emit(_phase_for_tool(name), f"{event}:{name}")


def install_quiet_native_progress(agent: Any) -> bool:
    """Install a stderr-only progress side-channel on a quiet one-shot agent."""
    if not native_progress_enabled():
        return False
    emitter = NativeProgressEmitter()
    setattr(agent, "_native_progress_emitter", emitter)
    agent.tool_progress_callback = emitter.tool_progress
    agent.tool_start_callback = emitter.tool_started
    agent.tool_complete_callback = emitter.tool_completed
    emitter.emit("initializing", "native_agent_ready")
    return True


def emit_native_terminal(result: Any, *, agent: Any = None) -> None:
    """Emit a bounded terminal envelope; no model text or tool payload is included."""
    if not native_progress_enabled():
        return
    data = result if isinstance(result, dict) else {}
    if data.get("failed"):
        status = "failed"
    elif data.get("partial"):
        status = "partial"
    else:
        status = "completed"

    raw_reason = data.get("failure_reason") or ("partial" if status == "partial" else status)
    reason = str(raw_reason or status)
    if not _SAFE_REASON.fullmatch(reason):
        reason = status

    emitter = getattr(agent, "_native_progress_emitter", None)
    if emitter is not None:
        emitter.emit("completed" if status == "completed" else "finalizing", f"terminal:{status}")
    _write(TERMINAL_PREFIX, {"status": status, "reason": reason})
