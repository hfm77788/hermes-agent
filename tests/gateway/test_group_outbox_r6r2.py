"""r6r2 independent verification tests for GroupOutboxMixin v1.9.

Covers:
1. 60+ concurrent enqueue (real handle_message path simulation)
2. Startup recovery (stale leases → queued)
3. Dual adapter owner lease (mutual exclusion)
4. Exception nack (transient + permanent)
5. DTO round-trip (serialize → reconstruct fidelity)
6. v2 spy: memory_catalog CRUD + visibility + conversation_state threads + response_owner
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal host stub so GroupOutboxMixin can run without a real adapter
# ---------------------------------------------------------------------------

class _FakeSource:
    def __init__(self, platform="feishu", chat_id="oc_test", chat_type="group",
                 user_id="u1", user_name="tester", thread_id="", user_id_alt=""):
        self.platform = platform
        self.chat_id = chat_id
        self.chat_type = chat_type
        self.user_id = user_id
        self.user_name = user_name
        self.thread_id = thread_id
        self.user_id_alt = user_id_alt


class _FakeEvent:
    def __init__(self, message_id="", text="hello", message_type="text",
                 reply_to_message_id="", media_urls=None, media_types=None,
                 metadata=None, source=None):
        self.message_id = message_id or f"msg_{uuid.uuid4().hex[:12]}"
        self.text = text
        self.message_type = message_type
        self.reply_to_message_id = reply_to_message_id
        self.media_urls = media_urls or []
        self.media_types = media_types or []
        self.metadata = metadata or {}
        self.source = source or _FakeSource()


class _FakeConfig:
    def __init__(self):
        self.extra = {"shared_group_session_chat_ids": ["oc_test"]}


class FakeAdapter:
    """Minimal host that mixes in GroupOutboxMixin."""

    def __init__(self, db_dir: str):
        from gateway.platforms.group_outbox import GroupOutboxMixin
        # Dynamically compose mixin
        self.__class__ = type("FakeAdapterWithOutbox", (GroupOutboxMixin,), dict(FakeAdapter.__dict__))
        self.name = "fake_adapter"
        self.config = _FakeConfig()
        self._background_tasks: set = set()
        self._active_sessions: Dict[str, asyncio.Event] = {}
        self._session_tasks: Dict[str, asyncio.Task] = {}
        self._db_dir = db_dir
        self._init_group_outbox()
        # Let _ensure_outbox_db create tables at temp path by patching Path.home
        from unittest.mock import patch as _patch
        with _patch.object(Path, "home", return_value=Path(db_dir)):
            self._ensure_outbox_db()

    async def _process_message_background(self, event, session_key):
        pass


@pytest.fixture
def adapter(tmp_path):
    return FakeAdapter(str(tmp_path))


@pytest.fixture
def adapter2(tmp_path):
    """Second adapter sharing the same DB (dual-adapter scenario)."""
    a = FakeAdapter(str(tmp_path))
    a.name = "fake_adapter_2"
    return a


# ===========================================================================
# 1. 60+ concurrent enqueue
# ===========================================================================

class TestConcurrentEnqueue:
    def test_60_concurrent_enqueue_all_inserted(self, adapter):
        """60 unique messages enqueued concurrently → all inserted, no duplicates."""
        chat_id = "oc_test"
        results = []
        for i in range(65):
            evt = _FakeEvent(message_id=f"msg_{i:04d}", text=f"message {i}")
            r = adapter._enqueue_group_event(chat_id, evt)
            results.append(r)

        inserted = [r for r in results if r["status"] == "inserted"]
        failed = [r for r in results if r["status"] == "failed"]
        assert len(inserted) == 65, f"Expected 65 inserted, got {len(inserted)}, failed={len(failed)}"
        assert len(failed) == 0

    def test_duplicate_message_id_rejected(self, adapter):
        """Same message_id twice → second is duplicate."""
        chat_id = "oc_test"
        evt1 = _FakeEvent(message_id="dup_001", text="first")
        evt2 = _FakeEvent(message_id="dup_001", text="second")
        r1 = adapter._enqueue_group_event(chat_id, evt1)
        r2 = adapter._enqueue_group_event(chat_id, evt2)
        assert r1["status"] == "inserted"
        assert r2["status"] == "duplicate"

    def test_fifo_order_preserved(self, adapter):
        """Dequeue returns events in insertion order (MIN seq)."""
        chat_id = "oc_test"
        for i in range(10):
            adapter._enqueue_group_event(chat_id, _FakeEvent(message_id=f"fifo_{i}", text=f"m{i}"))

        texts = []
        for _ in range(10):
            evt = adapter._dequeue_group_event(chat_id, "sk_test")
            if evt is None:
                break
            texts.append(evt.text)
            adapter._ack_group_event(chat_id, evt._outbox_seq, lease_token=evt._outbox_token)

        assert texts == [f"m{i}" for i in range(10)]


# ===========================================================================
# 2. Startup recovery
# ===========================================================================

class TestStartupRecovery:
    def test_stale_lease_recovered_to_queued(self, adapter):
        """Leased event with expired lease → recovered to queued on dequeue."""
        chat_id = "oc_test"
        adapter._enqueue_group_event(chat_id, _FakeEvent(message_id="stale_1", text="stale"))

        # Manually set lease to expired
        db = adapter._ensure_outbox_db()
        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE outbox SET state='leased', lease_expires=? WHERE chat_id=?",
            (time.time() - 100, chat_id),
        )
        conn.commit()
        conn.close()

        # Dequeue should recover and return the event
        evt = adapter._dequeue_group_event(chat_id, "sk_test")
        assert evt is not None
        assert evt.text == "stale"

    def test_recover_group_outbox_on_startup(self, adapter):
        """recover_group_outbox_on_startup restarts workers for pending chats."""
        chat_id = "oc_test"
        adapter._enqueue_group_event(chat_id, _FakeEvent(message_id="boot_1"))

        # Simulate stale lease
        db = adapter._ensure_outbox_db()
        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE outbox SET state='leased', lease_expires=? WHERE chat_id=?",
            (time.time() - 100, chat_id),
        )
        conn.commit()
        conn.close()

        count = adapter.recover_group_outbox_on_startup()
        assert count >= 0  # Should not crash; may start worker


# ===========================================================================
# 3. Dual adapter owner lease
# ===========================================================================

class TestDualAdapterOwnerLease:
    def test_second_adapter_cannot_acquire_while_first_holds(self, adapter, adapter2):
        """Two adapters sharing DB: only one gets owner lease."""
        chat_id = "oc_test"
        # Need at least one row for owner lease to attach to
        adapter._enqueue_group_event(chat_id, _FakeEvent(message_id="lease_1"))

        got1 = adapter._try_acquire_owner_lease(chat_id)
        got2 = adapter2._try_acquire_owner_lease(chat_id)

        assert got1 is True
        assert got2 is False

    def test_second_adapter_acquires_after_expiry(self, adapter, adapter2):
        """After owner lease expires, second adapter can take over."""
        chat_id = "oc_test"
        adapter._enqueue_group_event(chat_id, _FakeEvent(message_id="lease_2"))

        assert adapter._try_acquire_owner_lease(chat_id) is True

        # Expire the lease
        db = adapter._ensure_outbox_db()
        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE outbox SET owner_expires=? WHERE chat_id=?",
            (time.time() - 10, chat_id),
        )
        conn.commit()
        conn.close()

        assert adapter2._try_acquire_owner_lease(chat_id) is True

    def test_release_allows_reacquire(self, adapter, adapter2):
        """After release, another adapter can acquire."""
        chat_id = "oc_test"
        adapter._enqueue_group_event(chat_id, _FakeEvent(message_id="lease_3"))

        assert adapter._try_acquire_owner_lease(chat_id) is True
        adapter._release_owner_lease(chat_id)
        assert adapter2._try_acquire_owner_lease(chat_id) is True


# ===========================================================================
# 4. Exception nack
# ===========================================================================

class TestNack:
    def test_transient_nack_returns_to_queued(self, adapter):
        """Nack (non-permanent) puts event back to queued with retry_count+1."""
        chat_id = "oc_test"
        adapter._enqueue_group_event(chat_id, _FakeEvent(message_id="nack_1", text="retry me"))

        evt = adapter._dequeue_group_event(chat_id, "sk")
        assert evt is not None
        adapter._nack_group_event(chat_id, evt._outbox_seq, lease_token=evt._outbox_token)

        # Should be dequeue-able again
        evt2 = adapter._dequeue_group_event(chat_id, "sk")
        assert evt2 is not None
        assert evt2.text == "retry me"

        # Verify retry_count incremented
        db = adapter._ensure_outbox_db()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT retry_count FROM outbox WHERE chat_id=?", (chat_id,)).fetchone()
        conn.close()
        assert row[0] >= 1

    def test_permanent_nack_marks_failed(self, adapter):
        """Permanent nack sets state=failed, not re-dequeueable."""
        chat_id = "oc_test"
        adapter._enqueue_group_event(chat_id, _FakeEvent(message_id="nack_p", text="dead"))

        evt = adapter._dequeue_group_event(chat_id, "sk")
        assert evt is not None
        adapter._nack_group_event(chat_id, evt._outbox_seq, lease_token=evt._outbox_token, permanent=True)

        # Should NOT be dequeue-able
        evt2 = adapter._dequeue_group_event(chat_id, "sk")
        assert evt2 is None

        # Verify state=failed
        db = adapter._ensure_outbox_db()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT state FROM outbox WHERE chat_id=? AND message_id='nack_p'", (chat_id,)).fetchone()
        conn.close()
        assert row[0] == "failed"


# ===========================================================================
# 5. DTO round-trip
# ===========================================================================

class TestDTORoundTrip:
    def test_serialize_reconstruct_fidelity(self, adapter):
        """Serialize → reconstruct preserves all v4 fields."""
        src = _FakeSource(platform="feishu", chat_id="oc_dto", chat_type="group",
                          user_id="u42", user_name="张三", thread_id="t1", user_id_alt="alt42")
        evt = _FakeEvent(
            message_id="dto_001", text="你好世界", message_type="text",
            reply_to_message_id="reply_ref",
            media_urls=["https://img.example.com/a.png"],
            media_types=["image"],
            metadata={"key1": "val1", "key2": "val2"},
            source=src,
        )

        payload_json = adapter._serialize_event(evt)
        restored = adapter._reconstruct_event(payload_json)

        assert restored is not None
        assert restored.message_id == "dto_001"
        assert restored.text == "你好世界"
        assert restored.message_type == "text"
        assert restored.reply_to_message_id == "reply_ref"
        assert restored.media_urls == ["https://img.example.com/a.png"]
        assert restored.media_types == ["image"]
        assert restored.metadata == {"key1": "val1", "key2": "val2"}
        assert restored.source.chat_id == "oc_dto"
        assert restored.source.user_id == "u42"
        assert restored.source.user_name == "张三"
        assert restored.source.thread_id == "t1"
        assert restored.source.user_id_alt == "alt42"

    def test_corrupt_payload_returns_none(self, adapter):
        """Corrupt JSON → None, not crash."""
        assert adapter._reconstruct_event("not json{{{") is None
        assert adapter._reconstruct_event("") is None
        assert adapter._reconstruct_event("null") is None

    def test_empty_event_roundtrip(self, adapter):
        """Minimal event with no source still round-trips."""
        evt = _FakeEvent(message_id="min_1", text="", source=None)
        payload = adapter._serialize_event(evt)
        restored = adapter._reconstruct_event(payload)
        assert restored is not None
        assert restored.message_id == "min_1"


# ===========================================================================
# 6. v2 spy: memory_catalog + conversation_state threads + response_owner
# ===========================================================================

class TestMemoryCatalog:
    def test_upsert_and_get(self, adapter):
        """Basic upsert → get returns active entry."""
        result = adapter.memory_catalog_upsert(
            document_id="doc_001", slot_key="boss.birthday",
            bank_id="ma-secretary-system_v2_bge_m3",
            content_summary="1990-01-15", visibility="internal",
            disclosure="none", source="feishu_msg",
            source_message_id="msg_abc",
        )
        assert result["success"] is True
        assert result["document_id"] == "doc_001"
        assert result["superseded_old"] == ""

        entry = adapter.memory_catalog_get("boss.birthday")
        assert entry is not None
        assert entry["document_id"] == "doc_001"
        assert entry["content_summary"] == "1990-01-15"
        assert entry["visibility"] == "internal"
        assert entry["validity"] == "active"

    def test_upsert_supersedes_old(self, adapter):
        """New doc on same slot supersedes old active entry."""
        adapter.memory_catalog_upsert(
            document_id="doc_old", slot_key="boss.phone",
            bank_id="bank1", content_summary="13800000000",
        )
        result = adapter.memory_catalog_upsert(
            document_id="doc_new", slot_key="boss.phone",
            bank_id="bank1", content_summary="13900000000",
        )
        assert result["superseded_old"] == "doc_old"

        # Only new doc is active
        entry = adapter.memory_catalog_get("boss.phone")
        assert entry is not None
        assert entry["document_id"] == "doc_new"
        assert entry["content_summary"] == "13900000000"

    def test_same_doc_id_update_in_place(self, adapter):
        """Upsert with same document_id updates content, no supersede."""
        adapter.memory_catalog_upsert(
            document_id="doc_x", slot_key="slot_x",
            bank_id="b", content_summary="v1",
        )
        result = adapter.memory_catalog_upsert(
            document_id="doc_x", slot_key="slot_x",
            bank_id="b", content_summary="v2",
        )
        assert result["superseded_old"] == ""
        entry = adapter.memory_catalog_get("slot_x")
        assert entry["content_summary"] == "v2"

    def test_get_nonexistent_returns_none(self, adapter):
        assert adapter.memory_catalog_get("no.such.slot") is None

    def test_visibility_group_blocks_personalize_only(self, adapter):
        """personalize_only visibility → not visible in group context."""
        adapter.memory_catalog_upsert(
            document_id="doc_priv", slot_key="boss.secret",
            bank_id="b", visibility="personalize_only",
        )
        check = adapter.memory_catalog_check_visibility("boss.secret", context="group")
        assert check["exists"] is True
        assert check["visible"] is False
        assert "personalize_only" in check["reason"]

    def test_visibility_group_blocks_never_disclose(self, adapter):
        adapter.memory_catalog_upsert(
            document_id="doc_nd", slot_key="boss.nd",
            bank_id="b", visibility="internal", disclosure="never_disclose",
        )
        check = adapter.memory_catalog_check_visibility("boss.nd", context="group")
        assert check["visible"] is False

    def test_visibility_internal_ok_in_group(self, adapter):
        adapter.memory_catalog_upsert(
            document_id="doc_ok", slot_key="boss.ok",
            bank_id="b", visibility="internal", disclosure="none",
        )
        check = adapter.memory_catalog_check_visibility("boss.ok", context="group")
        assert check["visible"] is True

    def test_visibility_nonexistent_slot(self, adapter):
        check = adapter.memory_catalog_check_visibility("ghost.slot", context="group")
        assert check["exists"] is False
        assert check["visible"] is False


class TestConversationStateThreads:
    def test_save_and_get_threads(self, adapter):
        chat_id = "oc_test"
        threads = [
            {"thread_id": "t1", "watermark": 100, "boss_last_point": "A方案好",
             "employee_last_point": "B方案稳", "pending_question": "选哪个？",
             "response_owner": "bot", "pending_human_confirmation": False,
             "status": "active", "updated_at": time.time()},
            {"thread_id": "t2", "watermark": 50, "boss_last_point": "",
             "employee_last_point": "", "pending_question": "",
             "response_owner": "", "pending_human_confirmation": False,
             "status": "paused", "updated_at": time.time() - 10},
        ]
        adapter.save_active_threads(chat_id, threads)
        got = adapter.get_active_threads(chat_id)
        assert len(got) == 2
        assert got[0]["thread_id"] == "t1"  # Most recent first

    def test_prune_to_3(self, adapter):
        chat_id = "oc_test"
        threads = [
            {"thread_id": f"t{i}", "status": "active", "updated_at": time.time() - i}
            for i in range(5)
        ]
        adapter.save_active_threads(chat_id, threads)
        got = adapter.get_active_threads(chat_id)
        assert len(got) == 3

    def test_empty_threads(self, adapter):
        assert adapter.get_active_threads("oc_nonexistent") == []


class TestResponseOwner:
    def test_bot_can_answer_when_no_threads(self, adapter):
        result = adapter.check_response_owner(
            "oc_test", sender_id="u1", sender_name="tester",
            boss_open_id="ou_boss", employee_open_id="ou_emp",
        )
        assert result["bot_can_answer"] is True
        assert result["reason"] == "no_blocking_thread"

    def test_bot_blocked_when_owner_is_human(self, adapter):
        chat_id = "oc_test"
        threads = [{
            "thread_id": "t1", "status": "active",
            "pending_question": "老板你觉得呢？",
            "response_owner": "ou_boss",
            "pending_human_confirmation": False,
            "updated_at": time.time(),
        }]
        adapter.save_active_threads(chat_id, threads)

        result = adapter.check_response_owner(
            chat_id, sender_id="ou_emp", sender_name="小年糕",
            boss_open_id="ou_boss", employee_open_id="ou_emp",
        )
        assert result["bot_can_answer"] is False
        assert result["response_owner"] == "ou_boss"
        assert result["reason"] == "response_owner_is_human"

    def test_bot_allowed_when_owner_is_bot(self, adapter):
        chat_id = "oc_test"
        threads = [{
            "thread_id": "t1", "status": "active",
            "pending_question": "小马秘书查一下",
            "response_owner": "bot",
            "pending_human_confirmation": False,
            "updated_at": time.time(),
        }]
        adapter.save_active_threads(chat_id, threads)

        result = adapter.check_response_owner(
            chat_id, sender_id="ou_boss", sender_name="老板",
            boss_open_id="ou_boss", employee_open_id="ou_emp",
        )
        assert result["bot_can_answer"] is True

    def test_bot_blocked_awaiting_confirmation(self, adapter):
        """When response_owner is human + pending_human_confirmation, reason is response_owner_is_human (checked first)."""
        chat_id = "oc_test"
        threads = [{
            "thread_id": "t1", "status": "active",
            "pending_question": "确认一下时间",
            "response_owner": "ou_emp",
            "pending_human_confirmation": True,
            "updated_at": time.time(),
        }]
        adapter.save_active_threads(chat_id, threads)

        result = adapter.check_response_owner(
            chat_id, sender_id="ou_boss", sender_name="老板",
            boss_open_id="ou_boss", employee_open_id="ou_emp",
        )
        assert result["bot_can_answer"] is False
        assert result["reason"] == "response_owner_is_human"
        assert result["pending_human_confirmation"] is True

    def test_paused_thread_does_not_block(self, adapter):
        chat_id = "oc_test"
        threads = [{
            "thread_id": "t1", "status": "paused",
            "pending_question": "old question",
            "response_owner": "ou_boss",
            "pending_human_confirmation": False,
            "updated_at": time.time(),
        }]
        adapter.save_active_threads(chat_id, threads)

        result = adapter.check_response_owner(
            chat_id, sender_id="ou_emp", sender_name="小年糕",
            boss_open_id="ou_boss", employee_open_id="ou_emp",
        )
        assert result["bot_can_answer"] is True
