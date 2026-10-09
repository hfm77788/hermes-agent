"""Lazy SessionStore startup: routing stays hot while transcript/FTS DB stays cold."""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionEntry, SessionStore


def test_gateway_config_parses_nested_session_store_lazy_open():
    cfg = GatewayConfig.from_dict({"gateway": {"session_store_lazy_open": True}})
    assert cfg.session_store_lazy_open is True
    assert cfg.to_dict()["session_store_lazy_open"] is True


def test_lazy_open_restores_routing_without_opening_full_sessiondb(tmp_path, monkeypatch):
    home = tmp_path / "home"
    sessions = home / "sessions"
    home.mkdir()
    sessions.mkdir()

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)

    key = "agent:main:api_server:dm:user-1"
    now = datetime.now(timezone.utc)
    entry = SessionEntry(
        session_key=key,
        session_id="sid-live",
        created_at=now,
        updated_at=now,
    )

    db_path = home / "state.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE gateway_routing (
                scope TEXT NOT NULL,
                session_key TEXT NOT NULL,
                entry_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (scope, session_key)
            )"""
        )
        conn.execute(
            "INSERT INTO gateway_routing(scope, session_key, entry_json, updated_at) VALUES (?, ?, ?, ?)",
            (str(sessions.resolve()), key, json.dumps(entry.to_dict()), now.timestamp()),
        )
        conn.commit()
    finally:
        conn.close()

    store = SessionStore(
        sessions_dir=sessions,
        config=GatewayConfig(session_store_lazy_open=True),
    )
    assert store._db_handles == {}

    store._ensure_loaded()

    assert store._entries[key].session_id == "sid-live"
    assert store._routing_db_loaded is True
    assert store._db_handles == {}, "routing restore must not initialize full SessionDB"



@pytest.mark.asyncio
async def test_post_startup_warmup_opens_full_db_off_loop(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import asyncio

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(session_store_lazy_open=True)
    opener = MagicMock(return_value=object())
    runner.session_store = SimpleNamespace(_open_session_db_for_active_scope=opener)
    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    await runner._warm_lazy_session_db_after_startup()

    assert len(calls) == 1
    opener.assert_called_once_with()
