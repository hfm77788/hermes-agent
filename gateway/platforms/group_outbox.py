"""Shared-group SQLite outbox: persistent FIFO with lease-based state machine.

Extracted from BasePlatformAdapter (base.py) to reduce the 6000-line monolith.
Mixed into BasePlatformAdapter via GroupOutboxMixin.

State machine: queued → leased → (acked=deleted | nacked→queued/failed)
Owner lease: chat-level exclusive worker ownership with TTL renewal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from hermes_constants import get_hermes_home

if TYPE_CHECKING:
    from gateway.platforms.base import MessageEvent

logger = logging.getLogger(__name__)


class GroupOutboxMixin:
    """Mixin providing shared-group outbox + worker lifecycle.

    Expects the host class to provide:
      - self.name: str (adapter name for logging)
      - self._process_message_background(event, session_key): coroutine
      - self._background_tasks: set[asyncio.Task]
      - self._active_sessions: Dict[str, asyncio.Event]
      - self._session_tasks: Dict[str, asyncio.Task]
      - self.config.extra: dict (for shared_group_session_chat_ids)
    """

    LEASE_TIMEOUT_SECONDS = 300  # 5 min lease before recovery
    OWNER_LEASE_TIMEOUT_SECONDS = 600  # 10 min owner lease before takeover

    def _init_group_outbox(self) -> None:
        """Initialize outbox state. Call from host __init__."""
        self._group_outbox_db_path: Optional[Path] = None
        self._group_wakeup: asyncio.Event = asyncio.Event()
        self._group_worker_tasks: Dict[str, asyncio.Task] = {}
        self._owner_token: Optional[str] = None
        self._owner_chat_id: Optional[str] = None
        self._outbox_metrics: Dict[str, int] = {
            "enqueue_inserted": 0, "enqueue_duplicate": 0, "enqueue_failed": 0,
            "dequeue_success": 0, "dequeue_failed": 0,
            "acked": 0, "nacked": 0, "failed": 0,
            "payload_corrupt": 0, "leases_recovered": 0,
        }

    # ── DB lifecycle ────────────────────────────────────────────────────────

    def _ensure_outbox_db(self) -> Path:
        if self._group_outbox_db_path is not None:
            return self._group_outbox_db_path
        db_path = get_hermes_home() / "data" / "group_session_outbox.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("""CREATE TABLE IF NOT EXISTS outbox (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued',
            payload TEXT NOT NULL,
            lease_token TEXT,
            owner_token TEXT,
            owner_expires REAL,
            created_at REAL NOT NULL,
            leased_at REAL,
            lease_expires REAL,
            retry_count INTEGER NOT NULL DEFAULT 0
        )""")
        for col in ("lease_token", "owner_token", "owner_expires"):
            try:
                conn.execute(f"ALTER TABLE outbox ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_msg ON outbox(chat_id, message_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_dequeue ON outbox(chat_id, state, seq)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_token ON outbox(lease_token)")
        conn.execute("""CREATE TABLE IF NOT EXISTS conversation_state (
            chat_id TEXT PRIMARY KEY,
            last_speaker_id TEXT,
            last_speaker_name TEXT,
            last_mention_at REAL,
            conversation_mode TEXT DEFAULT 'unknown',
            topic_summary TEXT,
            bystander_ids TEXT DEFAULT '[]',
            recent_turns TEXT DEFAULT '[]',
            active_threads TEXT DEFAULT '[]',
            updated_at REAL NOT NULL
        )""")
        # v1.9 §8.7: add active_threads column if missing (migration)
        try:
            conn.execute("ALTER TABLE conversation_state ADD COLUMN active_threads TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        # v1.9 §8.8: memory_catalog — stable document_id/slot_key current-fact registry
        conn.execute("""CREATE TABLE IF NOT EXISTS memory_catalog (
            document_id TEXT PRIMARY KEY,
            slot_key TEXT NOT NULL,
            bank_id TEXT NOT NULL,
            content_summary TEXT DEFAULT '',
            visibility TEXT DEFAULT 'internal',
            disclosure TEXT DEFAULT 'none',
            validity TEXT DEFAULT 'active',
            source TEXT DEFAULT '',
            source_message_id TEXT DEFAULT '',
            supersedes TEXT DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_catalog_slot ON memory_catalog(slot_key, validity)")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
        conn.close()
        os.chmod(str(db_path), 0o600)
        self._group_outbox_db_path = db_path
        logger.info("[%s] Group outbox DB initialized: %s", self.name, db_path)
        return db_path

    # ── Serialization ───────────────────────────────────────────────────────

    def _serialize_event(self, event: Any) -> str:
        """Serialize event to versioned JSON with full fields (v4)."""
        source = getattr(event, "source", None)
        payload = {
            "v": 4,
            "event_id": str(uuid.uuid4()),
            "msg_id": str(getattr(event, "message_id", "") or ""),
            "msg_type": str(getattr(event, "message_type", "text") or "text"),
            "text": str(getattr(event, "text", "") or ""),
            "reply_to": str(getattr(event, "reply_to_message_id", "") or ""),
            "media_urls": [],
            "media_types": [],
            "metadata": {},
        }
        media_urls = getattr(event, "media_urls", None) or []
        if isinstance(media_urls, (list, tuple)):
            payload["media_urls"] = [str(u) for u in media_urls[:10]]
        media_types = getattr(event, "media_types", None) or []
        if isinstance(media_types, (list, tuple)):
            payload["media_types"] = [str(t) for t in media_types[:10]]
        metadata = getattr(event, "metadata", None) or {}
        if isinstance(metadata, dict):
            payload["metadata"] = {str(k): str(v) for k, v in list(metadata.items())[:20]}
        if source is not None:
            pf = getattr(source, "platform", "")
            pf_str = pf.value if hasattr(pf, "value") else str(pf or "")
            payload["src"] = {
                "pf": pf_str, "cid": str(getattr(source, "chat_id", "") or ""),
                "ctype": str(getattr(source, "chat_type", "") or ""),
                "uid": str(getattr(source, "user_id", "") or ""),
                "uname": str(getattr(source, "user_name", "") or ""),
                "tid": str(getattr(source, "thread_id", "") or ""),
                "ualt": str(getattr(source, "user_id_alt", "") or ""),
                "ts": time.time(),
            }
        return json.dumps(payload, ensure_ascii=False)

    def _reconstruct_event(self, payload_json: str) -> Optional[Any]:
        """Reconstruct a MessageEvent from JSON. Returns None on parse failure."""
        try:
            p = json.loads(payload_json)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(p, dict):
            return None
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.session import SessionSource

        src_data = p.get("src", {})
        pf_str = src_data.get("pf", "feishu")
        try:
            platform = Platform(pf_str)
        except (ValueError, TypeError):
            platform = Platform.FEISHU
        source = SessionSource(
            platform=platform,
            chat_id=src_data.get("cid", ""),
            chat_type=src_data.get("ctype", ""),
            user_id=src_data.get("uid", ""),
            user_name=src_data.get("uname", ""),
            thread_id=src_data.get("tid", ""),
            user_id_alt=src_data.get("ualt", ""),
        )
        evt = MessageEvent(
            source=source,
            message_id=p.get("msg_id", ""),
            message_type=p.get("msg_type", "text"),
            text=p.get("text", ""),
            reply_to_message_id=p.get("reply_to", ""),
        )
        # Restore media_urls, media_types, metadata (v4 DTO)
        media_urls = p.get("media_urls") or []
        if isinstance(media_urls, list):
            evt.media_urls = media_urls
        media_types = p.get("media_types") or []
        if isinstance(media_types, list):
            evt.media_types = media_types
        metadata = p.get("metadata") or {}
        if isinstance(metadata, dict):
            evt.metadata = metadata
        return evt

    # ── Enqueue / Dequeue / Ack / Nack ──────────────────────────────────────

    def _enqueue_group_event(self, chat_id: str, event: Any) -> Dict[str, Any]:
        """Insert event into outbox. Returns {'status': 'inserted'|'duplicate'|'failed'}."""
        result: Dict[str, Any] = {"status": "failed", "event_id": "", "error": ""}
        try:
            msg_id = str(getattr(event, "message_id", "") or "").strip()
            if not msg_id:
                msg_id = f"gen_{uuid.uuid4().hex[:16]}"
                result["event_id"] = msg_id
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            payload = self._serialize_event(event)
            now = time.time()
            cur = conn.execute(
                "INSERT OR IGNORE INTO outbox (chat_id, message_id, state, payload, created_at) VALUES (?, ?, 'queued', ?, ?)",
                (chat_id, msg_id, payload, now),
            )
            conn.commit()
            inserted = cur.rowcount > 0
            conn.close()
            if inserted:
                self._outbox_metrics["enqueue_inserted"] += 1
                self._group_wakeup.set()
                result["status"] = "inserted"
            else:
                self._outbox_metrics["enqueue_duplicate"] += 1
                result["status"] = "duplicate"
        except sqlite3.OperationalError as e:
            self._outbox_metrics["enqueue_failed"] += 1
            result["status"] = "failed"
            result["error"] = f"sqlite_busy: {e}"
            logger.error("[%s] Outbox enqueue failed (busy) for %s: %s", self.name, chat_id, e)
        except Exception as e:
            self._outbox_metrics["enqueue_failed"] += 1
            result["status"] = "failed"
            result["error"] = str(e)[:200]
            logger.error("[%s] Outbox enqueue failed for %s: %s", self.name, chat_id, e)
        return result

    def _dequeue_group_event(self, chat_id: str, session_key: str) -> Optional[Any]:
        """Atomic lease: generate token, UPDATE...WHERE seq=MIN(seq), SELECT BY token."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            lease_token = str(uuid.uuid4())
            # Recover expired leases
            conn.execute(
                "UPDATE outbox SET state='queued', leased_at=NULL, lease_expires=NULL, lease_token=NULL,"
                " retry_count=retry_count+1 WHERE state='leased' AND lease_expires < ? AND chat_id=?",
                (now, chat_id),
            )
            # Atomic lease: UPDATE exactly one row by MIN(seq), set token
            conn.execute(
                "UPDATE outbox SET state='leased', leased_at=?, lease_expires=?, lease_token=?"
                " WHERE seq = (SELECT MIN(seq) FROM outbox WHERE chat_id=? AND state='queued')",
                (now, now + self.LEASE_TIMEOUT_SECONDS, lease_token, chat_id),
            )
            # Fetch the exact row by token
            row = conn.execute(
                "SELECT seq, payload FROM outbox WHERE lease_token=?",
                (lease_token,),
            ).fetchone()
            conn.commit()
            conn.close()
            if row is None:
                return None
            seq, payload_json = row
            event = self._reconstruct_event(payload_json)
            if event is None:
                self._nack_group_event(chat_id, seq, lease_token=lease_token, permanent=True)
                logger.error("[%s] Outbox payload corrupt for seq=%d chat_id=%s", self.name, seq, chat_id)
                return None
            event._outbox_seq = seq
            event._outbox_chat_id = chat_id
            event._outbox_token = lease_token
            return event
        except Exception:
            logger.error("[%s] Outbox dequeue failed for %s", self.name, chat_id, exc_info=True)
            return None

    def _ack_group_event(self, chat_id: str, seq: int, *, lease_token: str = "") -> None:
        """Delete the event after successful processing. Token-conditional."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            if lease_token:
                conn.execute("DELETE FROM outbox WHERE chat_id=? AND seq=? AND lease_token=?",
                             (chat_id, seq, lease_token))
            else:
                conn.execute("DELETE FROM outbox WHERE chat_id=? AND seq=?", (chat_id, seq))
            conn.commit()
            conn.close()
            self._outbox_metrics["acked"] += 1
        except Exception:
            logger.error("[%s] Outbox ack failed for seq=%d", self.name, seq, exc_info=True)

    def _nack_group_event(self, chat_id: str, seq: int, *, lease_token: str = "", permanent: bool = False) -> None:
        """Return event to queued state (or permanent fail). Token-conditional."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            if permanent:
                if lease_token:
                    conn.execute("UPDATE outbox SET state='failed', leased_at=NULL, lease_expires=NULL, lease_token=NULL"
                                 " WHERE chat_id=? AND seq=? AND lease_token=?",
                                 (chat_id, seq, lease_token))
                else:
                    conn.execute("UPDATE outbox SET state='failed', leased_at=NULL, lease_expires=NULL, lease_token=NULL"
                                 " WHERE chat_id=? AND seq=?", (chat_id, seq))
                self._outbox_metrics["failed"] += 1
            else:
                if lease_token:
                    conn.execute("UPDATE outbox SET state='queued', leased_at=NULL, lease_expires=NULL, lease_token=NULL,"
                                 " retry_count=retry_count+1 WHERE chat_id=? AND seq=? AND lease_token=?",
                                 (chat_id, seq, lease_token))
                else:
                    conn.execute("UPDATE outbox SET state='queued', leased_at=NULL, lease_expires=NULL, lease_token=NULL,"
                                 " retry_count=retry_count+1 WHERE chat_id=? AND seq=?", (chat_id, seq))
                self._outbox_metrics["nacked"] += 1
            conn.commit()
            conn.close()
        except Exception:
            logger.error("[%s] Outbox nack failed for seq=%d", self.name, seq, exc_info=True)

    # ── Owner lease ─────────────────────────────────────────────────────────

    def _try_acquire_owner_lease(self, chat_id: str) -> bool:
        """Try to acquire chat-level owner lease. Returns True if acquired."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            row = conn.execute("SELECT owner_token, owner_expires FROM outbox"
                               " WHERE chat_id=? AND owner_token IS NOT NULL LIMIT 1",
                               (chat_id,)).fetchone()
            if row and row[1] and row[1] > now:
                conn.rollback()
                conn.close()
                return False
            owner_token = str(uuid.uuid4())
            conn.execute("UPDATE outbox SET owner_token=?, owner_expires=? WHERE chat_id=?",
                         (owner_token, now + self.OWNER_LEASE_TIMEOUT_SECONDS, chat_id))
            conn.commit()
            conn.close()
            self._owner_token = owner_token
            self._owner_chat_id = chat_id
            return True
        except Exception:
            return False

    def _recover_expired_leases(self, chat_id: str) -> None:
        """Recover expired leases."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            now = time.time()
            conn.execute("UPDATE outbox SET state='queued', leased_at=NULL, lease_expires=NULL, lease_token=NULL,"
                         " retry_count=retry_count+1 WHERE state='leased' AND lease_expires < ? AND chat_id=?",
                         (now, chat_id))
            conn.commit()
            conn.close()
            self._outbox_metrics["leases_recovered"] += 1
        except Exception:
            pass

    def _renew_owner_lease(self, chat_id: str) -> None:
        """Renew the owner lease for the given chat (extend expiry)."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            now = time.time()
            conn.execute("UPDATE outbox SET owner_expires=? WHERE chat_id=? AND owner_token=?",
                         (now + self.OWNER_LEASE_TIMEOUT_SECONDS, chat_id, self._owner_token or ""))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _release_owner_lease(self, chat_id: str) -> None:
        """Release the owner lease for the given chat."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("UPDATE outbox SET owner_token=NULL, owner_expires=NULL WHERE chat_id=? AND owner_token=?",
                         (chat_id, self._owner_token or ""))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ── Worker lifecycle ────────────────────────────────────────────────────

    def _build_group_session_key(self, event: Any) -> str:
        """Build the same canonical key used by BasePlatformAdapter."""
        from gateway.session import build_session_key

        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        return build_session_key(
            event.source,
            group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
            shared_group_session_chat_ids=extra.get("shared_group_session_chat_ids"),
        )

    def _start_group_worker(self, chat_id: str) -> None:
        """Start the owner-leased worker that serially drains one group."""
        existing = self._group_worker_tasks.get(chat_id)
        if existing is not None and not existing.done():
            return
        if not self._try_acquire_owner_lease(chat_id):
            return
        adapter_ref = self  # type: ignore[assignment]

        async def _worker() -> None:
            session_key: Optional[str] = None
            try:
                self._recover_expired_leases(chat_id)
                while True:
                    adapter_ref._renew_owner_lease(chat_id)

                    # A processing task may drain the next row from its finally
                    # block.  Wait for that canonical session owner before
                    # leasing more work, otherwise two responses can run in
                    # parallel against the same conversation.
                    if session_key and session_key in self._active_sessions:
                        active_task = self._session_tasks.get(session_key)
                        if active_task is not None and not active_task.done():
                            try:
                                await asyncio.shield(active_task)
                            except asyncio.CancelledError:
                                current = asyncio.current_task()
                                if current is not None and current.cancelling():
                                    raise
                            except Exception:
                                logger.error(
                                    "[%s] Shared-group session task failed for %s",
                                    self.name,
                                    session_key,
                                    exc_info=True,
                                )
                            continue
                        # Heal an impossible/stale guard rather than deadlock
                        # the durable queue forever.
                        self._session_tasks.pop(session_key, None)
                        self._active_sessions.pop(session_key, None)

                    evt = self._dequeue_group_event(chat_id, session_key or "")
                    if evt is None:
                        # Wakeups are process-local.  Poll briefly as well so
                        # an enqueue from another adapter/process is not delayed
                        # by the full lease timeout.
                        try:
                            await asyncio.wait_for(
                                self._group_wakeup.wait(),
                                timeout=1.0,
                            )
                        except asyncio.TimeoutError:
                            pass
                        self._group_wakeup.clear()
                        continue

                    try:
                        session_key = self._build_group_session_key(evt)
                    except Exception:
                        seq = getattr(evt, "_outbox_seq", None)
                        token = getattr(evt, "_outbox_token", "")
                        if seq is not None:
                            self._nack_group_event(
                                chat_id,
                                seq,
                                lease_token=token,
                                permanent=True,
                            )
                        logger.error(
                            "[%s] Cannot build session key for shared-group row %s",
                            self.name,
                            seq,
                            exc_info=True,
                        )
                        continue

                    # A live handler may have claimed the same session between
                    # the loop check and the row lease.  Return the row and let
                    # that owner drain it from its normal completion path.
                    if session_key in self._active_sessions:
                        self._nack_group_event(
                            chat_id,
                            getattr(evt, "_outbox_seq"),
                            lease_token=getattr(evt, "_outbox_token", ""),
                        )
                        continue

                    try:
                        started = self._start_session_processing(evt, session_key)
                    except Exception:
                        started = False
                        logger.error(
                            "[%s] Failed to start shared-group session %s",
                            self.name,
                            session_key,
                            exc_info=True,
                        )
                    if not started:
                        self._nack_group_event(
                            chat_id,
                            getattr(evt, "_outbox_seq"),
                            lease_token=getattr(evt, "_outbox_token", ""),
                        )
                        await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
            finally:
                adapter_ref._release_owner_lease(chat_id)

        task = asyncio.create_task(_worker())
        self._group_worker_tasks[chat_id] = task
        try:
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except TypeError:
            pass

    # ── Conversation state (panorama persistence) ─────────────────────────

    def get_conversation_state(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Read persisted conversation state for a shared group."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute(
                "SELECT last_speaker_id, last_speaker_name, last_mention_at,"
                " conversation_mode, topic_summary, bystander_ids, recent_turns, updated_at"
                " FROM conversation_state WHERE chat_id=?",
                (chat_id,),
            ).fetchone()
            conn.close()
            if row is None:
                return None
            return {
                "chat_id": chat_id,
                "last_speaker_id": row[0],
                "last_speaker_name": row[1],
                "last_mention_at": row[2],
                "conversation_mode": row[3] or "unknown",
                "topic_summary": row[4],
                "bystander_ids": json.loads(row[5] or "[]"),
                "recent_turns": json.loads(row[6] or "[]"),
                "updated_at": row[7],
            }
        except Exception:
            logger.warning("[%s] get_conversation_state failed for %s", self.name, chat_id, exc_info=True)
            return None

    def save_conversation_state(self, chat_id: str, *,
                                last_speaker_id: str = "",
                                last_speaker_name: str = "",
                                last_mention_at: Optional[float] = None,
                                conversation_mode: str = "unknown",
                                topic_summary: str = "",
                                bystander_ids: Optional[list] = None,
                                recent_turns: Optional[list] = None) -> None:
        """Upsert conversation state for a shared group."""
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            now = time.time()
            conn.execute(
                "INSERT INTO conversation_state"
                " (chat_id, last_speaker_id, last_speaker_name, last_mention_at,"
                "  conversation_mode, topic_summary, bystander_ids, recent_turns, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(chat_id) DO UPDATE SET"
                "  last_speaker_id=excluded.last_speaker_id,"
                "  last_speaker_name=excluded.last_speaker_name,"
                "  last_mention_at=excluded.last_mention_at,"
                "  conversation_mode=excluded.conversation_mode,"
                "  topic_summary=excluded.topic_summary,"
                "  bystander_ids=excluded.bystander_ids,"
                "  recent_turns=excluded.recent_turns,"
                "  updated_at=excluded.updated_at",
                (
                    chat_id,
                    last_speaker_id,
                    last_speaker_name,
                    last_mention_at or now,
                    conversation_mode,
                    topic_summary,
                    json.dumps(bystander_ids or [], ensure_ascii=False),
                    json.dumps(recent_turns or [], ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.warning("[%s] save_conversation_state failed for %s", self.name, chat_id, exc_info=True)

    # ── v1.9 §8.7: Active thread management ──────────────────────────────

    def get_active_threads(self, chat_id: str) -> list:
        """Return up to 3 active/paused topic threads for a shared group.

        Each thread dict: {thread_id, watermark, boss_last_point,
        employee_last_point, pending_question, response_owner,
        pending_human_confirmation, status, updated_at}
        """
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute(
                "SELECT active_threads FROM conversation_state WHERE chat_id=?",
                (chat_id,),
            ).fetchone()
            conn.close()
            if row is None:
                return []
            threads = json.loads(row[0] or "[]")
            return threads[:3]
        except Exception:
            logger.warning("[%s] get_active_threads failed for %s", self.name, chat_id, exc_info=True)
            return []

    def save_active_threads(self, chat_id: str, threads: list) -> None:
        """Upsert active threads (max 3) into conversation_state.

        Prunes to 3 most-recently-updated threads. Must be reconstructable
        from group buffer / Feishu — does not store full chat content.
        """
        try:
            # Prune to 3, sorted by updated_at desc
            threads = sorted(threads, key=lambda t: t.get("updated_at", 0), reverse=True)[:3]
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            now = time.time()
            conn.execute(
                "INSERT INTO conversation_state (chat_id, active_threads, updated_at)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(chat_id) DO UPDATE SET"
                "  active_threads=excluded.active_threads,"
                "  updated_at=excluded.updated_at",
                (chat_id, json.dumps(threads, ensure_ascii=False), now),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.warning("[%s] save_active_threads failed for %s", self.name, chat_id, exc_info=True)

    # ── v1.9 §8.8: Memory catalog ────────────────────────────────────────

    def memory_catalog_get(self, slot_key: str) -> Optional[Dict[str, Any]]:
        """Look up the current (validity='active') catalog entry for a slot.

        Returns None if no active entry exists for this slot_key.
        """
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute(
                "SELECT document_id, slot_key, bank_id, content_summary,"
                " visibility, disclosure, validity, source, source_message_id,"
                " supersedes, created_at, updated_at"
                " FROM memory_catalog WHERE slot_key=? AND validity='active'",
                (slot_key,),
            ).fetchone()
            conn.close()
            if row is None:
                return None
            return {
                "document_id": row[0], "slot_key": row[1], "bank_id": row[2],
                "content_summary": row[3], "visibility": row[4],
                "disclosure": row[5], "validity": row[6], "source": row[7],
                "source_message_id": row[8], "supersedes": row[9],
                "created_at": row[10], "updated_at": row[11],
            }
        except Exception:
            logger.warning("[%s] memory_catalog_get failed for slot=%s", self.name, slot_key, exc_info=True)
            return None

    def memory_catalog_upsert(self, *, document_id: str, slot_key: str,
                              bank_id: str, content_summary: str = "",
                              visibility: str = "internal",
                              disclosure: str = "none",
                              source: str = "", source_message_id: str = "",
                              supersedes: str = "") -> Dict[str, Any]:
        """Insert or update a memory catalog entry.

        If an existing active entry for the same slot_key exists with a
        different document_id, the old one is marked superseded.
        Returns {success, document_id, superseded_old}.
        """
        try:
            db_path = self._ensure_outbox_db()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=5000")
            now = time.time()
            superseded_old = ""
            # Check for existing active entry on same slot
            existing = conn.execute(
                "SELECT document_id FROM memory_catalog"
                " WHERE slot_key=? AND validity='active' AND document_id!=?",
                (slot_key, document_id),
            ).fetchone()
            if existing:
                superseded_old = existing[0]
                conn.execute(
                    "UPDATE memory_catalog SET validity='superseded', updated_at=?"
                    " WHERE document_id=?",
                    (now, superseded_old),
                )
            conn.execute(
                "INSERT INTO memory_catalog"
                " (document_id, slot_key, bank_id, content_summary, visibility,"
                "  disclosure, validity, source, source_message_id, supersedes,"
                "  created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)"
                " ON CONFLICT(document_id) DO UPDATE SET"
                "  content_summary=excluded.content_summary,"
                "  visibility=excluded.visibility,"
                "  disclosure=excluded.disclosure,"
                "  validity='active',"
                "  source=excluded.source,"
                "  source_message_id=excluded.source_message_id,"
                "  supersedes=excluded.supersedes,"
                "  updated_at=excluded.updated_at",
                (document_id, slot_key, bank_id, content_summary,
                 visibility, disclosure, source, source_message_id,
                 supersedes or superseded_old, now, now),
            )
            conn.commit()
            conn.close()
            return {"success": True, "document_id": document_id, "superseded_old": superseded_old}
        except Exception:
            logger.warning("[%s] memory_catalog_upsert failed doc=%s", self.name, document_id, exc_info=True)
            return {"success": False, "document_id": document_id, "superseded_old": ""}

    def memory_catalog_check_visibility(self, slot_key: str,
                                        context: str = "group") -> Dict[str, Any]:
        """Check if a slot's current fact is visible in the given context.

        Returns {exists, visible, reason}. If visibility is
        personalize_only/never_disclose and context is 'group', visible=False.
        """
        entry = self.memory_catalog_get(slot_key)
        if entry is None:
            return {"exists": False, "visible": False, "reason": "no_active_entry"}
        vis = entry.get("visibility", "internal")
        disc = entry.get("disclosure", "none")
        if context == "group" and vis in ("personalize_only", "never_disclose"):
            return {"exists": True, "visible": False, "reason": f"visibility={vis}"}
        if context == "group" and disc == "never_disclose":
            return {"exists": True, "visible": False, "reason": "disclosure=never_disclose"}
        return {"exists": True, "visible": True, "reason": "ok"}

    # ── v1.9 §8.7: Response owner / turn control ─────────────────────────

    def check_response_owner(self, chat_id: str, *,
                             sender_id: str, sender_name: str,
                             boss_open_id: str, employee_open_id: str,
                             bot_open_id: str = "") -> Dict[str, Any]:
        """Programmatic turn control: determine if bot may answer.

        Rules:
        - If pending_question targets a human (not bot), bot must NOT answer
          on their behalf. Returns {bot_can_answer: False, response_owner,
          pending_question, pending_human_confirmation}.
        - If no pending question or question targets bot, bot_can_answer=True.
        - Reads from active_threads for the chat.
        """
        threads = self.get_active_threads(chat_id)
        for thread in threads:
            if thread.get("status") != "active":
                continue
            pending_q = thread.get("pending_question", "")
            resp_owner = thread.get("response_owner", "")
            pending_confirm = thread.get("pending_human_confirmation", False)
            # If response_owner is a human (not bot), bot cannot answer
            if resp_owner and resp_owner != bot_open_id and resp_owner != "bot":
                return {
                    "bot_can_answer": False,
                    "response_owner": resp_owner,
                    "pending_question": pending_q,
                    "pending_human_confirmation": pending_confirm,
                    "thread_id": thread.get("thread_id", ""),
                    "reason": "response_owner_is_human",
                }
            if pending_confirm and resp_owner in (boss_open_id, employee_open_id):
                return {
                    "bot_can_answer": False,
                    "response_owner": resp_owner,
                    "pending_question": pending_q,
                    "pending_human_confirmation": True,
                    "thread_id": thread.get("thread_id", ""),
                    "reason": "awaiting_human_confirmation",
                }
        return {
            "bot_can_answer": True,
            "response_owner": "",
            "pending_question": "",
            "pending_human_confirmation": False,
            "thread_id": "",
            "reason": "no_blocking_thread",
        }

    # ── Startup recovery ───────────────────────────────────────────────────

    def recover_group_outbox_on_startup(self) -> int:
        """Scan outbox for stale leases + pending messages; restart workers.

        Call once after adapter.connect() succeeds.  Returns the number of
        chat workers (re)started.  Safe to call with no outbox DB (no-op).
        """
        shared_ids = (self.config.extra.get("shared_group_session_chat_ids") or [])  # type: ignore[attr-defined]
        if not shared_ids:
            return 0

        recovered = 0
        for chat_id in shared_ids:
            try:
                db_path = self._ensure_outbox_db()
                conn = sqlite3.connect(str(db_path))
                conn.execute("PRAGMA busy_timeout=5000")
                now = time.time()
                # 1. Recover expired row-level leases
                cur = conn.execute(
                    "UPDATE outbox SET state='queued', leased_at=NULL, lease_expires=NULL, lease_token=NULL,"
                    " retry_count=retry_count+1 WHERE state='leased' AND lease_expires < ? AND chat_id=?",
                    (now, chat_id),
                )
                stale = cur.rowcount
                # 2. Check for any pending work (queued or just-recovered)
                pending = conn.execute(
                    "SELECT COUNT(*) FROM outbox WHERE chat_id=? AND state='queued'", (chat_id,)
                ).fetchone()[0]
                # 3. Clear expired owner leases so we can re-acquire
                conn.execute(
                    "UPDATE outbox SET owner_token=NULL, owner_expires=NULL"
                    " WHERE chat_id=? AND owner_expires IS NOT NULL AND owner_expires < ?",
                    (chat_id, now),
                )
                conn.commit()
                conn.close()

                if stale:
                    self._outbox_metrics["leases_recovered"] += stale
                    logger.info(
                        "[%s] Startup recovery: %d stale lease(s) recovered for chat %s",
                        self.name, stale, chat_id,
                    )
                if pending > 0:
                    self._start_group_worker(chat_id)
                    recovered += 1
                    logger.info(
                        "[%s] Startup recovery: worker started for chat %s (%d pending)",
                        self.name, chat_id, pending,
                    )
            except Exception:
                logger.warning(
                    "[%s] Startup recovery failed for chat %s", self.name, chat_id, exc_info=True
                )
        return recovered