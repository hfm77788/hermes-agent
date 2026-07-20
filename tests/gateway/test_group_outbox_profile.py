"""Regression coverage for profile-aware shared-group outbox state."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.group_outbox import GroupOutboxMixin
from gateway.session import SessionSource, build_session_key


class _OutboxHost(GroupOutboxMixin):
    pass


def test_group_outbox_uses_active_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    host = _OutboxHost()
    host._init_group_outbox()

    db_path = host._ensure_outbox_db()

    assert db_path == tmp_path / "data" / "group_session_outbox.db"
    assert db_path.exists()


@pytest.mark.asyncio
async def test_group_worker_uses_canonical_session_key_and_guard():
    chat_id = "oc_serial_worker"
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_type="group",
        user_id="ou_user",
        user_name="Test User",
        thread_id="",
        user_id_alt="",
    )
    event = SimpleNamespace(source=source, _outbox_seq=1, _outbox_token="lease")
    host = _OutboxHost()
    host.name = "test"
    host.config = SimpleNamespace(
        extra={"shared_group_session_chat_ids": [chat_id]}
    )
    host._group_worker_tasks = {}
    host._background_tasks = set()
    host._active_sessions = {}
    host._session_tasks = {}
    host._group_wakeup = asyncio.Event()
    host._owner_tokens = {}

    queued = [event]
    seen = []
    processed = asyncio.Event()
    host._try_acquire_owner_lease = lambda _chat_id: True
    host._renew_owner_lease = lambda _chat_id: None
    host._release_owner_lease = lambda _chat_id: None
    host._recover_expired_leases = lambda _chat_id: None
    host._dequeue_group_event = (
        lambda _chat_id, _session_key: queued.pop(0) if queued else None
    )
    host._nack_group_event = lambda *_args, **_kwargs: None

    def start_session(evt, session_key):
        guard = asyncio.Event()
        host._active_sessions[session_key] = guard

        async def run():
            try:
                seen.append((evt, session_key, session_key in host._active_sessions))
                processed.set()
            finally:
                host._active_sessions.pop(session_key, None)
                host._session_tasks.pop(session_key, None)

        task = asyncio.create_task(run())
        host._session_tasks[session_key] = task
        return True

    host._start_session_processing = start_session
    host._start_group_worker(chat_id)
    await asyncio.wait_for(processed.wait(), timeout=1.0)

    worker = host._group_worker_tasks[chat_id]
    worker.cancel()
    with suppress(asyncio.CancelledError):
        await worker

    expected_key = build_session_key(
        source,
        shared_group_session_chat_ids=[chat_id],
    )
    assert seen == [(event, expected_key, True)]


def test_owner_lease_is_per_chat_and_survives_an_empty_outbox(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = _OutboxHost()
    first.name = "first"
    first._init_group_outbox()
    first._ensure_outbox_db()

    second = _OutboxHost()
    second.name = "second"
    second._init_group_outbox()
    second._ensure_outbox_db()

    assert first._try_acquire_owner_lease("chat-a") is True
    assert first._try_acquire_owner_lease("chat-b") is True
    assert set(first._owner_tokens) == {"chat-a", "chat-b"}

    # No outbox row exists: ownership must still live in the dedicated table.
    assert second._try_acquire_owner_lease("chat-a") is False

    first._release_owner_lease("chat-a")
    assert second._try_acquire_owner_lease("chat-a") is True

    first._release_owner_lease("chat-b")
    second._release_owner_lease("chat-a")
