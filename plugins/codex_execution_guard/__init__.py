"""codex_execution_guard — Hermes plugin hard闸 for Codex-execution requests.

Hook: ``pre_tool_call``

When the user explicitly asks for Codex as the execution agent, Hermes native
write tools (``write_file``, ``patch``) and git-write terminal commands are
blocked until a real ``codex exec`` terminal call has been observed.

The guard is session-scoped: it resets when a new user message arrives so
that a fresh conversation starts clean.

Activation
----------
List the plugin in ``~/.hermes/config.yaml``:

    plugins:
      enabled:
        - codex_execution_guard

No env vars or credentials are required.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phrases that activate the Codex guard (checked against user message text)
# ---------------------------------------------------------------------------
_CODEX_TRIGGER_RE = re.compile(
    r"(?:"
    r"\b(?:用\s*Codex|Codex\s*执行|OpenAI\s*Codex|Codex\s*PR)\b"
    r"|\bcodex\s+(?:exec|task|chat)\b"
    r"|指定执行端[:：]\s*Codex"
    r"|你是\s*Codex\s*执行端"
    r"|\bcodex\s*--"
    r")",
    re.IGNORECASE,
)

# Tools that are always blocked once the guard is active
_ALWAYS_BLOCKED = {"write_file", "patch"}

# Terminal commands that count as "git write" and are blocked when guarded
_GIT_WRITE_RE = re.compile(
    r"^\s*(?:"
    r"git\s+(?:add|commit|push|pull|merge|rebase|checkout|reset|restore|rm|mv|lane|worktree\s+add)\b"
    r"|(?:mkdir|rm|mv|cp)\b.*_control\b"
    r"|(?:touch|mkdir)\b.*(?:PR|pr|pull)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Session state (thread-safe per session + task)
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_sessions: dict[str, "_SessionState"] = {}


def _make_key(session_id: str, task_id: str) -> str:
    """Compose a unique state key from session and task identifiers."""
    return f"{session_id or 'default'}::{task_id or 'default'}"


def _get_state(session_id: str, task_id: str = "") -> "_SessionState":
    key = _make_key(session_id, task_id)
    with _lock:
        if key not in _sessions:
            _sessions[key] = _SessionState()
        return _sessions[key]


# Alias for backward compat with tests that pass only session_key
def _get_state_by_session(session_key: str) -> _SessionState:
    return _get_state(session_key, "")


class _SessionState:
    """Per-session guard state.  Thread-safe via the module-level lock."""

    __slots__ = ("_guard_active", "_codex_exec_seen")

    def __init__(self) -> None:
        self._guard_active: bool = False
        self._codex_exec_seen: bool = False

    def activate(self) -> None:
        self._guard_active = True
        self._codex_exec_seen = False

    def note_codex_exec(self) -> None:
        self._codex_exec_seen = True

    @property
    def is_blocked(self) -> bool:
        return self._guard_active and not self._codex_exec_seen


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", pre_tool_call)


def pre_tool_call(
    tool_name: str,
    args: Optional[Dict] = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    user_message: str = "",
    **kwargs,
) -> dict | None:
    """Block mutating tools when a Codex execution guard is active.

    Returns ``None`` (allow) when:
      - session has never seen a Codex trigger phrase, OR
      - a ``codex exec`` terminal call has already been observed.

    Returns a block message when:
      - ``write_file`` or ``patch`` is called while guard is active, OR
      - a git-write terminal command is called while guard is active.

    Parameters
    ----------
    tool_name:
        The name of the tool being called (e.g. ``write_file``).
    args:
        The arguments passed to the tool (named ``args`` to match invoke_hook convention).
    session_id:
        Unique session identifier.  Used to isolate per-conversation state.
    user_message:
        The current user message text.  Checked for Codex trigger phrases.
    **kwargs:
        Forward compatibility for future hook parameters.

    Returns
    -------
    Optional[str]
        ``None`` to allow, or a block message string to deny with explanation.
    """
    state = _get_state(session_id or "default", task_id or "")

    # ── 1. Activate guard on first trigger phrase ───────────────────────────
    if user_message and _CODEX_TRIGGER_RE.search(user_message):
        if not state._guard_active:
            logger.info("[codex_execution_guard] activated for session=%s task=%s", session_id, task_id)
            state.activate()
        # If guard is already active, do NOT call activate() again — it would
        # reset _codex_exec_seen to False and break the unlock flow (T6).

    # ── 2. If guard is not active, allow everything ──────────────────────────
    if not state._guard_active:
        return None

    # ── 3. If codex exec was already seen, allow everything ─────────────────
    if state._codex_exec_seen:
        return None

    # ── 4. Terminal calls: check for codex exec or git-write patterns ─────────
    if tool_name == "terminal":
        command: str = (args or {}).get("command", "")
        # codex exec unblocks the guard
        if re.search(r"codex\s+exec", command, re.IGNORECASE):
            logger.info("[codex_execution_guard] codex exec seen — guard released")
            state.note_codex_exec()
            return None
        # Block git-write commands while guarded
        if _GIT_WRITE_RE.search(command):
            return {
                "action": "block",
                "message": (
                    "[codex_execution_guard] BLOCKED — Codex execution guard is active.  "
                    "You must use `codex exec ...` to make changes.  "
                    "Git write commands (git add/commit/push/...) are not allowed directly."
                ),
            }
        return None

    # ── 5. Block always-blocked mutating tools ────────────────────────────────
    if tool_name in _ALWAYS_BLOCKED:
        return {
            "action": "block",
            "message": (
                "[codex_execution_guard] BLOCKED — Codex execution guard is active.  "
                "Use `codex exec ...` to make changes.  "
                f"`{tool_name}` is not allowed while waiting for Codex."
            ),
        }

    return None
