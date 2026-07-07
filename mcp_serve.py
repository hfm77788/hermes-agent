"""
Hermes MCP Server — expose messaging conversations as MCP tools.

Starts a stdio MCP server that lets any MCP client (Claude Code, Cursor, Codex,
etc.) list conversations, read message history, send messages, poll for live
events, and manage approval requests across all connected platforms.

Matches OpenClaw's 9-tool MCP channel bridge surface:
  conversations_list, conversation_get, messages_read, attachments_fetch,
  events_poll, events_wait, messages_send, permissions_list_open,
  permissions_respond

Plus: channels_list (Hermes-specific extra)

Usage:
    hermes mcp serve
    hermes mcp serve --verbose

MCP client config (e.g. claude_desktop_config.json):
    {
        "mcpServers": {
            "hermes": {
                "command": "hermes",
                "args": ["mcp", "serve"]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("hermes.mcp_serve")

# ---------------------------------------------------------------------------
# Skill tools import (ChatGPT-Hermes high-trust preauthorization channel)
# ---------------------------------------------------------------------------
try:
    from tools.mcp_skill_tools import (
        hermes_health_check as _skill_health_check,
        resolve_skill_uri as _skill_resolve_uri,
        read_skill_bundle as _skill_read_bundle,
        read_skill_file_chunked as _skill_read_file_chunked,
        smoke_skill_access as _skill_smoke_skill_access,
        get_preauthorization_profile as _skill_get_preauth_profile,
        run_preauthorized_skill_patch as _skill_run_preauth_patch,
        rollback_skill_patch as _skill_rollback_patch,
    )
    _SKILL_TOOLS_AVAILABLE = True
except ImportError:
    _SKILL_TOOLS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Lazy MCP SDK import
# ---------------------------------------------------------------------------

_MCP_SERVER_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP

    _MCP_SERVER_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sessions_dir() -> Path:
    """Return the sessions directory using HERMES_HOME."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "sessions"
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "sessions"


def _get_session_db():
    """Get a SessionDB instance for reading message transcripts."""
    try:
        from hermes_state import SessionDB
        return SessionDB()
    except Exception as e:
        logger.debug("SessionDB unavailable: %s", e)
        return None


def _load_sessions_index() -> dict:
    """Load the gateway sessions.json index directly.

    Returns a dict of session_key -> entry_dict with platform routing info.
    This avoids importing the full SessionStore which needs GatewayConfig.
    """
    sessions_file = _get_sessions_dir() / "sessions.json"
    if not sessions_file.exists():
        return {}
    try:
        with open(sessions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop documentation/metadata sentinels (keys starting with "_", e.g.
        # the "_README" note the gateway writes into the index). They are not
        # session entries and would break consumers that treat every value as
        # an entry dict.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not str(k).startswith("_")}
        return {}
    except Exception as e:
        logger.debug("Failed to load sessions.json: %s", e)
        return {}


def _load_channel_directory() -> dict:
    """Load the cached channel directory for available targets."""
    try:
        from hermes_constants import get_hermes_home
        directory_file = get_hermes_home() / "channel_directory.json"
    except ImportError:
        directory_file = Path(
            os.environ.get("HERMES_HOME", Path.home() / ".hermes")
        ) / "channel_directory.json"

    if not directory_file.exists():
        return {}
    try:
        with open(directory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load channel_directory.json: %s", e)
        return {}


def _coerce_int(
    value,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Coerce value to int with fallback and clamping.

    Used at MCP tool boundaries to handle invalid types from external clients.
    Returns default if value cannot be converted to int.
    """
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(coerced, maximum))


def _extract_message_content(msg: dict) -> str:
    """Extract text content from a message, handling multi-part content."""
    content = msg.get("content", "")
    if isinstance(content, list):
        text_parts = [
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(text_parts)
    return str(content) if content else ""


def _extract_attachments(msg: dict) -> List[dict]:
    """Extract non-text attachments from a message.

    Finds: multi-part image/file content blocks, MEDIA: tags in text,
    image URLs, and file references.
    """
    attachments = []
    content = msg.get("content", "")

    # Multi-part content blocks (image_url, file, etc.)
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "image_url":
                url = part.get("image_url", {}).get("url", "") if isinstance(part.get("image_url"), dict) else ""
                if url:
                    attachments.append({"type": "image", "url": url})
            elif ptype == "image":
                url = part.get("url", part.get("source", {}).get("url", ""))
                if url:
                    attachments.append({"type": "image", "url": url})
            elif ptype not in {"text",}:
                # Unknown non-text content type
                attachments.append({"type": ptype, "data": part})

    # MEDIA: tags in text content
    text = _extract_message_content(msg)
    if text:
        media_pattern = re.compile(r'MEDIA:\s*(\S+)')
        for match in media_pattern.finditer(text):
            path = match.group(1)
            attachments.append({"type": "media", "path": path})

    return attachments


# ---------------------------------------------------------------------------
# Event Bridge — polls SessionDB for new messages, maintains event queue
# ---------------------------------------------------------------------------

QUEUE_LIMIT = 1000
POLL_INTERVAL = 0.2  # seconds between DB polls (200ms)


@dataclass
class QueueEvent:
    """An event in the bridge's in-memory queue."""
    cursor: int
    type: str  # "message", "approval_requested", "approval_resolved"
    session_key: str = ""
    data: dict = field(default_factory=dict)


class EventBridge:
    """Background poller that watches SessionDB for new messages and
    maintains an in-memory event queue with waiter support.

    This is the Hermes equivalent of OpenClaw's WebSocket gateway bridge.
    Instead of WebSocket events, we poll the SQLite database for changes.
    """

    def __init__(self):
        self._queue: List[QueueEvent] = []
        self._cursor = 0
        self._lock = threading.Lock()
        self._new_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_poll_timestamps: Dict[str, float] = {}  # session_key -> unix timestamp
        # In-memory approval tracking (populated from events)
        self._pending_approvals: Dict[str, dict] = {}
        # mtime cache — skip expensive work when files haven't changed
        self._sessions_json_mtime: float = 0.0
        self._state_db_mtime: float = 0.0
        self._cached_sessions_index: dict = {}

    def start(self):
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.debug("EventBridge started")

    def stop(self):
        """Stop the background polling thread."""
        self._running = False
        self._new_event.set()  # Wake any waiters
        if self._thread:
            self._thread.join(timeout=5)
        logger.debug("EventBridge stopped")

    def poll_events(
        self,
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """Return events since after_cursor, optionally filtered by session_key."""
        with self._lock:
            events = [
                e for e in self._queue
                if e.cursor > after_cursor
                and (not session_key or e.session_key == session_key)
            ][:limit]

        next_cursor = events[-1].cursor if events else after_cursor
        return {
            "events": [
                {"cursor": e.cursor, "type": e.type,
                 "session_key": e.session_key, **e.data}
                for e in events
            ],
            "next_cursor": next_cursor,
        }

    def wait_for_event(
        self,
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> Optional[dict]:
        """Block until a matching event arrives or timeout expires."""
        deadline = time.monotonic() + (timeout_ms / 1000.0)

        while time.monotonic() < deadline:
            with self._lock:
                for e in self._queue:
                    if e.cursor > after_cursor and (
                        not session_key or e.session_key == session_key
                    ):
                        return {
                            "cursor": e.cursor, "type": e.type,
                            "session_key": e.session_key, **e.data,
                        }

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._new_event.clear()
            self._new_event.wait(timeout=min(remaining, POLL_INTERVAL))

        return None

    def list_pending_approvals(self) -> List[dict]:
        """List approval requests observed during this bridge session."""
        with self._lock:
            return sorted(
                self._pending_approvals.values(),
                key=lambda a: a.get("created_at", ""),
            )

    def respond_to_approval(self, approval_id: str, decision: str) -> dict:
        """Resolve a pending approval (best-effort without gateway IPC)."""
        with self._lock:
            approval = self._pending_approvals.pop(approval_id, None)

        if not approval:
            return {"error": f"Approval not found: {approval_id}"}

        self._enqueue(QueueEvent(
            cursor=0,  # Will be set by _enqueue
            type="approval_resolved",
            session_key=approval.get("session_key", ""),
            data={"approval_id": approval_id, "decision": decision},
        ))

        return {"resolved": True, "approval_id": approval_id, "decision": decision}

    def _enqueue(self, event: QueueEvent) -> None:
        """Add an event to the queue and wake any waiters."""
        with self._lock:
            self._cursor += 1
            event.cursor = self._cursor
            self._queue.append(event)
            # Trim queue to limit
            while len(self._queue) > QUEUE_LIMIT:
                self._queue.pop(0)
        self._new_event.set()

    def _poll_loop(self):
        """Background loop: poll SessionDB for new messages."""
        db = _get_session_db()
        if not db:
            logger.warning("EventBridge: SessionDB unavailable, event polling disabled")
            return

        while self._running:
            try:
                self._poll_once(db)
            except Exception as e:
                logger.debug("EventBridge poll error: %s", e)
            time.sleep(POLL_INTERVAL)

    def _poll_once(self, db):
        """Check for new messages across all sessions.

        Uses mtime checks on sessions.json and state.db to skip work
        when nothing has changed — makes 200ms polling essentially free.
        """
        # Check if sessions.json has changed (mtime check is ~1μs).
        # Capture the previously-seen mtime *before* refreshing the cache so the
        # skip guard below can still tell whether sessions.json changed this tick.
        prev_sessions_json_mtime = self._sessions_json_mtime
        sessions_file = _get_sessions_dir() / "sessions.json"
        try:
            sj_mtime = sessions_file.stat().st_mtime if sessions_file.exists() else 0.0
        except OSError:
            sj_mtime = 0.0

        if sj_mtime != self._sessions_json_mtime:
            self._sessions_json_mtime = sj_mtime
            self._cached_sessions_index = _load_sessions_index()

        # Check if state.db has changed
        try:
            from hermes_constants import get_hermes_home
            db_file = get_hermes_home() / "state.db"
        except ImportError:
            db_file = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "state.db"

        try:
            db_mtime = db_file.stat().st_mtime if db_file.exists() else 0.0
        except OSError:
            db_mtime = 0.0

        # Skip only when NEITHER file changed since the last poll. Comparing
        # against ``prev_sessions_json_mtime`` (not the freshly-stored
        # ``self._sessions_json_mtime``) is essential: the latter was just set to
        # ``sj_mtime`` above, so using it would make the sessions.json term
        # always true and collapse the guard to a db-only check. That would
        # discard a tick where only sessions.json changed — e.g. a brand-new
        # conversation registered after its first message already landed in
        # state.db on an earlier tick — and the new chat's messages would never
        # be emitted until state.db happened to change again.
        if db_mtime == self._state_db_mtime and sj_mtime == prev_sessions_json_mtime:
            return  # Nothing changed since last poll — skip entirely

        self._state_db_mtime = db_mtime
        entries = self._cached_sessions_index

        for session_key, entry in entries.items():
            session_id = entry.get("session_id", "")
            if not session_id:
                continue

            last_seen = self._last_poll_timestamps.get(session_key, 0.0)

            try:
                messages = db.get_messages(session_id)
            except Exception:
                continue

            if not messages:
                continue

            # Normalize timestamps to float for comparison
            def _ts_float(ts) -> float:
                if isinstance(ts, (int, float)):
                    return float(ts)
                if isinstance(ts, str) and ts:
                    try:
                        return float(ts)
                    except ValueError:
                        # ISO string — parse to epoch
                        try:
                            from datetime import datetime
                            return datetime.fromisoformat(ts).timestamp()
                        except Exception:
                            return 0.0
                return 0.0

            # Find messages newer than our last seen timestamp
            new_messages = []
            for msg in messages:
                ts = _ts_float(msg.get("timestamp", 0))
                role = msg.get("role", "")
                if role not in {"user", "assistant"}:
                    continue
                if ts > last_seen:
                    new_messages.append(msg)

            for msg in new_messages:
                content = _extract_message_content(msg)
                if not content:
                    continue
                self._enqueue(QueueEvent(
                    cursor=0,
                    type="message",
                    session_key=session_key,
                    data={
                        "role": msg.get("role", ""),
                        "content": content[:500],
                        "timestamp": str(msg.get("timestamp", "")),
                        "message_id": str(msg.get("id", "")),
                    },
                ))

            # Update last seen to the most recent message timestamp
            all_ts = [_ts_float(m.get("timestamp", 0)) for m in messages]
            if all_ts:
                latest = max(all_ts)
                if latest > last_seen:
                    self._last_poll_timestamps[session_key] = latest


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_mcp_server(
    event_bridge: Optional[EventBridge] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    mount_path: str = "/",
    transport_security: Optional[dict] = None,
) -> "FastMCP":
    """Create and return the Hermes MCP server with all tools registered.

    Args:
        event_bridge: Optional EventBridge for conversation events.
        host: Listen address for SSE transport (default: 127.0.0.1).
              Passed to FastMCP at construction time so transport_security
              is initialized correctly.
        port: Listen port for SSE transport (default: 8000).
        mount_path: Mount path for SSE app (default: "/").
        transport_security: Optional dict with keys 'allowed_hosts' and
                           'allowed_origins' lists. When provided, overrides
                           the default FastMCP transport security for non-
                           localhost hosts. Set to None to use FastMCP auto-
                           detection (default for localhost addresses).
    """
    if not _MCP_SERVER_AVAILABLE:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            f"Install with: {sys.executable} -m pip install 'mcp'"
        )

    # Build TransportSecuritySettings if explicitly provided
    ts = None
    if transport_security is not None:
        try:
            from mcp.server.fastmcp.server import TransportSecuritySettings
            ts = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=transport_security.get("allowed_hosts", []),
                allowed_origins=transport_security.get("allowed_origins", []),
            )
        except ImportError:
            pass

    mcp = FastMCP(
        "hermes",
        instructions=(
            "Hermes Agent messaging bridge. Use these tools to interact with "
            "conversations across Telegram, Discord, Slack, WhatsApp, Signal, "
            "Matrix, and other connected platforms."
        ),
        host=host,
        port=port,
        mount_path=mount_path,
        transport_security=ts,
    )

    bridge = event_bridge or EventBridge()

    # -- conversations_list ------------------------------------------------

    @mcp.tool()
    def conversations_list(
        platform: Optional[str] = None,
        limit: int = 50,
        search: Optional[str] = None,
    ) -> str:
        """List active messaging conversations across connected platforms.

        Returns conversations with their session keys (needed for messages_read),
        platform, chat type, display name, and last activity time.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
            limit: Maximum number of conversations to return (default 50)
            search: Optional text to filter conversations by name
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        entries = _load_sessions_index()
        conversations = []

        for key, entry in entries.items():
            origin = entry.get("origin", {})
            entry_platform = entry.get("platform") or origin.get("platform", "")

            if platform and entry_platform.lower() != platform.lower():
                continue

            display_name = entry.get("display_name", "")
            chat_name = origin.get("chat_name", "")
            if search:
                search_lower = search.lower()
                if (search_lower not in display_name.lower()
                        and search_lower not in chat_name.lower()
                        and search_lower not in key.lower()):
                    continue

            conversations.append({
                "session_key": key,
                "session_id": entry.get("session_id", ""),
                "platform": entry_platform,
                "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
                "display_name": display_name,
                "chat_name": chat_name,
                "user_name": origin.get("user_name", ""),
                "updated_at": entry.get("updated_at", ""),
            })

        conversations.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        conversations = conversations[:limit]

        return json.dumps({
            "count": len(conversations),
            "conversations": conversations,
        }, indent=2)

    # -- conversation_get --------------------------------------------------

    @mcp.tool()
    def conversation_get(session_key: str) -> str:
        """Get detailed info about one conversation by its session key.

        Args:
            session_key: The session key from conversations_list
        """
        entries = _load_sessions_index()
        entry = entries.get(session_key)

        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        origin = entry.get("origin", {})
        return json.dumps({
            "session_key": session_key,
            "session_id": entry.get("session_id", ""),
            "platform": entry.get("platform") or origin.get("platform", ""),
            "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
            "display_name": entry.get("display_name", ""),
            "user_name": origin.get("user_name", ""),
            "chat_name": origin.get("chat_name", ""),
            "chat_id": origin.get("chat_id", ""),
            "thread_id": origin.get("thread_id"),
            "updated_at": entry.get("updated_at", ""),
            "created_at": entry.get("created_at", ""),
            "input_tokens": entry.get("input_tokens", 0),
            "output_tokens": entry.get("output_tokens", 0),
            "total_tokens": entry.get("total_tokens", 0),
        }, indent=2)

    # -- messages_read -----------------------------------------------------

    @mcp.tool()
    def messages_read(
        session_key: str,
        limit: int = 50,
    ) -> str:
        """Read recent messages from a conversation.

        Returns the message history in chronological order with role, content,
        and timestamp for each message.

        Args:
            session_key: The session key from conversations_list
            limit: Maximum number of messages to return (default 50, most recent)
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        entries = _load_sessions_index()
        entry = entries.get(session_key)
        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        session_id = entry.get("session_id", "")
        if not session_id:
            return json.dumps({"error": "No session ID for this conversation"})

        db = _get_session_db()
        if not db:
            return json.dumps({"error": "Session database unavailable"})

        try:
            all_messages = db.get_messages(session_id)
        except Exception as e:
            return json.dumps({"error": f"Failed to read messages: {e}"})

        filtered = []
        for msg in all_messages:
            role = msg.get("role", "")
            if role in {"user", "assistant"}:
                content = _extract_message_content(msg)
                if content:
                    filtered.append({
                        "id": str(msg.get("id", "")),
                        "role": role,
                        "content": content[:2000],
                        "timestamp": msg.get("timestamp", ""),
                    })

        messages = filtered[-limit:]

        return json.dumps({
            "session_key": session_key,
            "count": len(messages),
            "total_in_session": len(filtered),
            "messages": messages,
        }, indent=2)

    # -- attachments_fetch -------------------------------------------------

    @mcp.tool()
    def attachments_fetch(
        session_key: str,
        message_id: str,
    ) -> str:
        """List non-text attachments for a message in a conversation.

        Extracts images, media files, and other non-text content blocks
        from the specified message.

        Args:
            session_key: The session key from conversations_list
            message_id: The message ID from messages_read
        """
        entries = _load_sessions_index()
        entry = entries.get(session_key)
        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        session_id = entry.get("session_id", "")
        if not session_id:
            return json.dumps({"error": "No session ID for this conversation"})

        db = _get_session_db()
        if not db:
            return json.dumps({"error": "Session database unavailable"})

        try:
            all_messages = db.get_messages(session_id)
        except Exception as e:
            return json.dumps({"error": f"Failed to read messages: {e}"})

        # Find the target message
        target_msg = None
        for msg in all_messages:
            if str(msg.get("id", "")) == message_id:
                target_msg = msg
                break

        if not target_msg:
            return json.dumps({"error": f"Message not found: {message_id}"})

        attachments = _extract_attachments(target_msg)

        return json.dumps({
            "message_id": message_id,
            "count": len(attachments),
            "attachments": attachments,
        }, indent=2)

    # -- events_poll -------------------------------------------------------

    @mcp.tool()
    def events_poll(
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """Poll for new conversation events since a cursor position.

        Returns events that have occurred since the given cursor. Use the
        returned next_cursor value for subsequent polls.

        Event types: message, approval_requested, approval_resolved

        Args:
            after_cursor: Return events after this cursor (0 for all)
            session_key: Optional filter to one conversation
            limit: Maximum events to return (default 20)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        limit = _coerce_int(limit, default=20, minimum=1, maximum=200)
        result = bridge.poll_events(
            after_cursor=after_cursor,
            session_key=session_key,
            limit=limit,
        )
        return json.dumps(result, indent=2)

    # -- events_wait -------------------------------------------------------

    @mcp.tool()
    def events_wait(
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> str:
        """Wait for the next conversation event (long-poll).

        Blocks until a matching event arrives or the timeout expires.
        Use this for near-real-time event delivery without polling.

        Args:
            after_cursor: Wait for events after this cursor
            session_key: Optional filter to one conversation
            timeout_ms: Maximum wait time in milliseconds (default 30000)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        timeout_ms = _coerce_int(
            timeout_ms,
            default=30000,
            minimum=0,
            maximum=300000,
        )  # Cap at 5 minutes
        event = bridge.wait_for_event(
            after_cursor=after_cursor,
            session_key=session_key,
            timeout_ms=timeout_ms,
        )
        if event:
            return json.dumps({"event": event}, indent=2)
        return json.dumps({"event": None, "reason": "timeout"}, indent=2)

    # -- messages_send -----------------------------------------------------

    @mcp.tool()
    def messages_send(
        target: str,
        message: str,
    ) -> str:
        """Send a message to a platform conversation.

        The target format is "platform:chat_id" — same format used by the
        channels_list tool. You can also use human-friendly channel names
        that will be resolved automatically.

        Examples:
            target="telegram:6308981865"
            target="discord:#general"
            target="slack:#engineering"

        Args:
            target: Platform target in "platform:identifier" format
            message: The message text to send
        """
        if not target or not message:
            return json.dumps({"error": "Both target and message are required"})

        try:
            from tools.send_message_tool import send_message_tool
            result_str = send_message_tool(
                {"action": "send", "target": target, "message": message}
            )
            return result_str
        except ImportError:
            return json.dumps({"error": "Send message tool not available"})
        except Exception as e:
            return json.dumps({"error": f"Send failed: {e}"})

    # -- channels_list -----------------------------------------------------

    @mcp.tool()
    def channels_list(platform: Optional[str] = None) -> str:
        """List available messaging channels and targets across platforms.

        Returns channels that you can send messages to. The target strings
        returned here can be used directly with the messages_send tool.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
        """
        directory = _load_channel_directory()
        if not directory:
            entries = _load_sessions_index()
            targets = []
            seen = set()
            for key, entry in entries.items():
                origin = entry.get("origin", {})
                p = entry.get("platform") or origin.get("platform", "")
                chat_id = origin.get("chat_id", "")
                if not p or not chat_id:
                    continue
                if platform and p.lower() != platform.lower():
                    continue
                target_str = f"{p}:{chat_id}"
                if target_str in seen:
                    continue
                seen.add(target_str)
                targets.append({
                    "target": target_str,
                    "platform": p,
                    "name": entry.get("display_name") or origin.get("chat_name", ""),
                    "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
                })
            return json.dumps({"count": len(targets), "channels": targets}, indent=2)

        channels = []
        for plat, entries_list in directory.get("platforms", {}).items():
            if platform and plat.lower() != platform.lower():
                continue
            if isinstance(entries_list, list):
                for ch in entries_list:
                    if isinstance(ch, dict):
                        chat_id = ch.get("id", ch.get("chat_id", ""))
                        channels.append({
                            "target": f"{plat}:{chat_id}" if chat_id else plat,
                            "platform": plat,
                            "name": ch.get("name", ch.get("display_name", "")),
                            "chat_type": ch.get("type", ""),
                        })

        return json.dumps({"count": len(channels), "channels": channels}, indent=2)

    # -- permissions_list_open ---------------------------------------------

    @mcp.tool()
    def permissions_list_open() -> str:
        """List pending approval requests observed during this bridge session.

        Returns exec and plugin approval requests that the bridge has seen
        since it started. Approvals are live-session only — older approvals
        from before the bridge connected are not included.
        """
        approvals = bridge.list_pending_approvals()
        return json.dumps({
            "count": len(approvals),
            "approvals": approvals,
        }, indent=2)

    # -- permissions_respond -----------------------------------------------

    @mcp.tool()
    def permissions_respond(
        id: str,
        decision: str,
    ) -> str:
        """Respond to a pending approval request.

        Args:
            id: The approval ID from permissions_list_open
            decision: One of "allow-once", "allow-always", or "deny"
        """
        if decision not in {"allow-once", "allow-always", "deny"}:
            return json.dumps({
                "error": f"Invalid decision: {decision}. "
                         f"Must be allow-once, allow-always, or deny"
            })

        result = bridge.respond_to_approval(id, decision)
        return json.dumps(result, indent=2)

    # -- Skill Read Tools (ChatGPT-Hermes High-Trust Channel) ---------------

    if _SKILL_TOOLS_AVAILABLE:

        @mcp.tool()
        def hermes_health_check() -> str:
            """Check Hermes MCP + skills health status.

            Returns mcp_ok, htp_ok, skills_root_exists, skills_root_readable,
            version, and timestamp. Use this to verify the skill channel is
            healthy before attempting skill reads.
            """
            result = _skill_health_check()
            return json.dumps(result, indent=2)

        @mcp.tool()
        def resolve_skill_uri(skill_name: str) -> str:
            """Resolve a skill name to its canonical URI and filesystem path.

            Args:
                skill_name: Skill name, e.g. "reference-writing"

            Returns canonical_uri, local_path, category, exists, confidence.
            When multiple matches exist, returns candidates list without
            auto-selecting one.
            """
            result = _skill_resolve_uri(skill_name)
            return json.dumps(result, indent=2)

        @mcp.tool()
        def read_skill_bundle(skill_name: str) -> str:
            """Read a skill's SKILL.md summary, frontmatter, and manifest.

            Args:
                skill_name: Skill name or canonical URI like
                           "reference-writing" or
                           "skill:productivity/reference-writing"

            Returns SKILL.md line_count, frontmatter, references list,
            scripts list, and _index.md summary. Full content > 20KB
            returns a summary with chunk_required=true.
            """
            result = _skill_read_bundle(skill_name)
            return json.dumps(result, indent=2)

        @mcp.tool()
        def read_skill_file_chunked(
            canonical_uri: str,
            start_line: int = 1,
            end_line: int = 0,
            relative_path: Optional[str] = None,
        ) -> str:
            """Read a chunk of a skill file by line range.

            Args:
                canonical_uri: Canonical URI, e.g.
                              "skill:productivity/reference-writing"
                start_line: 1-indexed start line (default 1)
                end_line: 1-indexed end line inclusive. 0 means to end.
                relative_path: Optional relative path within skill_dir
                              for non-SKILL.md files, e.g.
                              "references/_index.md" or
                              "references/styles/guide.md".

            Returns chunk content, line numbers, and chunk_required flag.
            """
            end = end_line if end_line > 0 else None
            result = _skill_read_file_chunked(canonical_uri, start_line, end, relative_path)
            return json.dumps(result, indent=2)

        @mcp.tool()
        def smoke_skill_access(skill_name: str) -> str:
            """Check accessibility of a skill (PASS/WARN/FAIL).

            Args:
                skill_name: Skill name, e.g. "reference-writing"

            Checks SKILL.md, references/_index.md, references/ dir,
            and scripts/ dir. Returns overall status and per-check results.
            """
            result = _skill_smoke_skill_access(skill_name)
            return json.dumps(result, indent=2)

        # Phase 2: Preauthorized write tools

        @mcp.tool()
        def get_preauthorization_profile() -> str:
            """Return the preauthorization profile. P3_PR_CREATE is policy_declared_only."""
            return json.dumps(_skill_get_preauth_profile(), indent=2)

        @mcp.tool()
        def run_preauthorized_skill_patch(manifest: str) -> str:
            """Execute or dry-run a preauthorized skill file patch.
            Args: manifest JSON with skill_name, file_path, new_content,
                  action, and optional dry_run (bool, default false).
            Dry-run: validates + shows diff without writing."""
            try:
                manifest_dict = json.loads(manifest)
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid JSON manifest: {e}"})
            return json.dumps(_skill_run_preauth_patch(manifest_dict), indent=2)

        @mcp.tool()
        def rollback_skill_patch(backup_path: str) -> str:
            """Rollback a skill patch to a previous backup."""
            return json.dumps(_skill_rollback_patch(backup_path), indent=2)

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_mcp_server(
    verbose: bool = False,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    mount_path: str = "/",
    allowed_host: Optional[str] = None,
    auth_token_env: Optional[str] = None,
) -> None:
    """Start the Hermes MCP server.

    Args:
        verbose: Enable debug logging.
        transport: Transport mode - "stdio" (default) or "sse".
        host: Listen address for SSE (default: 127.0.0.1).
              Use "0.0.0.0" to bind all interfaces.
        port: Listen port for SSE (default: 8000).
        mount_path: Mount path for SSE app (default: "/").
        allowed_host: External hostname/IP for SSE clients. When set,
                      added to DNS rebinding allowlist regardless of
                      bind address.
        auth_token_env: Env var name containing Bearer auth token.
                       Required for non-loopback SSE binds.
                       Loopback (127.0.0.1/localhost/::1) allows
                       no-token access for local development.
    """
    if not _MCP_SERVER_AVAILABLE:
        print(
            "Error: MCP server requires the 'mcp' package.\n"
            f"Install with: {sys.executable} -m pip install 'mcp'",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    bridge = EventBridge()
    bridge.start()

    import asyncio

    if transport == "sse":
        # --- Auth check: non-loopback requires auth token ---
        _is_loopback = host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")
        auth_token = None
        if not _is_loopback:
            if not auth_token_env:
                print(
                    "Error: Non-loopback SSE bind requires --auth-token-env.\n"
                    "Public/internal network SSE endpoints must be authenticated.\n"
                    f"Add --auth-token-env HERMES_MCP_AUTH_TOKEN to your command.\n"
                    f"For local development use --host 127.0.0.1 instead.",
                    file=sys.stderr,
                )
                bridge.stop()
                sys.exit(1)
            auth_token = os.environ.get(auth_token_env, "")
            if not auth_token:
                print(
                    f"Error: Auth token env var '{auth_token_env}' is empty or unset.\n"
                    f"Set it to a bearer token value before starting the server.",
                    file=sys.stderr,
                )
                bridge.stop()
                sys.exit(1)

        # --- Build transport security allowlist ---
        ts_config = None
        if host not in ("127.0.0.1", "localhost", "::1"):
            # Base localhost entries (always present)
            ts_hosts = [
                "127.0.0.1:*", "localhost:*", "[::1]:*",
                "127.0.0.1", "localhost", "[::1]",
            ]
            ts_origins = [
                "http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
                "http://127.0.0.1", "http://localhost", "http://[::1]",
            ]

            # Bind address entries (unless 0.0.0.0, which is not a real hostname)
            if host != "0.0.0.0":
                ts_hosts.append(host)
                ts_hosts.append(f"{host}:*")
                ts_origins.append(f"http://{host}")
                ts_origins.append(f"https://{host}")
                ts_origins.append(f"http://{host}:*")
                ts_origins.append(f"https://{host}:*")

            # allowed_host — always added when provided, regardless of bind address
            if allowed_host:
                ts_hosts.append(allowed_host)
                ts_hosts.append(f"{allowed_host}:*")
                ts_origins.append(f"http://{allowed_host}")
                ts_origins.append(f"https://{allowed_host}")
                ts_origins.append(f"http://{allowed_host}:*")
                ts_origins.append(f"https://{allowed_host}:*")
            elif host == "0.0.0.0" and not allowed_host:
                print(
                    "Warning: --host 0.0.0.0 requires --allowed-host to "
                    "set DNS rebinding protection. Without it, external "
                    "clients may be rejected.\n"
                    "Usage: --host 0.0.0.0 --allowed-host <your-hostname>",
                    file=sys.stderr,
                )

            ts_config = {"allowed_hosts": ts_hosts, "allowed_origins": ts_origins}

        server = create_mcp_server(
            event_bridge=bridge,
            host=host,
            port=port,
            mount_path=mount_path,
            transport_security=ts_config,
        )

        # --- Build auth middleware if token is configured ---
        starlette_app = None
        if auth_token:
            starlette_app = _build_auth_middleware(server, mount_path, auth_token)

        async def _run_sse():
            try:
                if auth_token:
                    import uvicorn
                    app = starlette_app or server.sse_app(mount_path=mount_path)
                    config = uvicorn.Config(
                        app,
                        host=host,
                        port=port,
                        log_level="debug" if verbose else "warning",
                    )
                    uvicorn_server = uvicorn.Server(config)
                    await uvicorn_server.serve()
                else:
                    print(
                        f"Hermes MCP SSE server listening on http://{host}:{port}{mount_path}",
                        file=sys.stderr,
                    )
                    await server.run_sse_async(mount_path=mount_path)
            finally:
                bridge.stop()

        try:
            asyncio.run(_run_sse())
        except KeyboardInterrupt:
            bridge.stop()
    else:
        # stdio: use default FastMCP settings (no host/port needed)
        server = create_mcp_server(event_bridge=bridge)

        async def _run_stdio():
            try:
                await server.run_stdio_async()
            finally:
                bridge.stop()

        try:
            asyncio.run(_run_stdio())
        except KeyboardInterrupt:
            bridge.stop()


def _build_auth_middleware(
    server: "FastMCP",
    mount_path: str,
    token: str,
) -> "Starlette":
    """Build a Starlette app wrapping the FastMCP SSE server with Bearer auth.

    The middleware checks every request to the SSE and message endpoints
    for a valid Authorization: Bearer <token> header. Unauthenticated
    requests return 401.
    """
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    sse_app = server.sse_app(mount_path=mount_path)

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Only protect SSE and message endpoints
            path = request.url.path.rstrip("/")
            sse_path = (mount_path.rstrip("/") or "") + "/sse"
            msg_path = (mount_path.rstrip("/") or "") + "/messages"
            if path == sse_path or path.startswith(msg_path):
                auth_header = request.headers.get("authorization", "")
                if not auth_header.startswith("Bearer "):
                    return Response("Unauthorized", status_code=401)
                provided = auth_header[len("Bearer "):]
                if provided != token:
                    return Response("Unauthorized", status_code=401)
            return await call_next(request)

    return Starlette(
        routes=[
            Mount("/", app=sse_app),
        ],
        middleware=[
            Middleware(BearerAuthMiddleware),
        ],
    )
