"""Regression tests for #97963 — hygiene turn-hold must not burn a
watermark-fenced compression attempt.

The 10s ``hygiene_max_turn_hold_seconds`` budget (#92318) releases the
arriving user turn while a thinking summary model is still streaming its
reasoning prefix. Before the fix, that release ALWAYS cancelled the commit
fence, so 100% of the summary attempt (including the full thinking prefix)
was discarded on every turn — auto-compression permanently failed for any
deployment whose summary model thinks longer than the hold.

The fix decouples the turn from the compression: when the worker's commit is
watermark-fenced (rows appended after compression start survive its commit
verbatim as concurrent tail), the detached worker KEEPS its commit admission
and the summary is adopted at its own watermark-fenced commit boundary. The
turn is still released at the same budget — the invariant pinned by
``test_session_hygiene_turn_hold_budget_abandons_streaming_wait`` (#90845)
is untouched (that test's worker is NOT watermark-fenced and still takes the
cancel path).
"""

import asyncio
import importlib
import sys
import threading
import time
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _make_history(n_messages: int, content_size: int = 100) -> list:
    history = []
    content = "x" * content_size
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": content, "timestamp": f"t{i}"})
    return history


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM
        )
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="x")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _write_turnhold_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_timeout_seconds: 60\n"
        "  hygiene_total_ceiling_seconds: 600\n"
        "  hygiene_max_turn_hold_seconds: 0.3\n"
        "  hygiene_failure_cooldown_seconds: 120\n"
    )


def _build_runner(gateway_run, adapter, fake_db):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")
        }
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:12345",
        session_id="sess-97963",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = _make_history(
        6, content_size=400
    )
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = SimpleNamespace(_db=fake_db)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


def _make_event():
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="12345",
        ),
        message_id="1",
    )


def _install_fakes(monkeypatch, gateway_run, tmp_path, agent_cls):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run.GatewayRunner, "_resolve_session_agent_runtime",
        lambda self, **_kwargs: ("test-model", {"api_key": "fake"}),
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length_async",
        AsyncMock(return_value=100),
    )


async def _drain_deferred(runner, timeout=10.0):
    tasks = getattr(runner, "_deferred_agent_cleanup_tasks", None) or set()
    if tasks:
        await asyncio.wait_for(
            asyncio.gather(*list(tasks), return_exceptions=True), timeout
        )


@pytest.mark.asyncio
async def test_background_only_frontend_never_awaits_hygiene_planning(monkeypatch):
    """Even a blocked model-metadata planner cannot hold the foreground turn."""
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    history = [{"role": "user", "content": "hello"} for _ in range(6)]
    entry = SimpleNamespace(session_id="safe-session")
    marker = MagicMock(return_value=history)
    runner._hmwa_enqueue_background_hygiene = marker
    runner._hmwa_hygiene_settings = AsyncMock(side_effect=AssertionError("foreground planned"))
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {
        "compression": {"hygiene_background_only": True},
    })
    start = time.monotonic()
    result = await asyncio.wait_for(
        runner._hmwa_run_session_hygiene(None, None, entry, "key", history, "q", 1),
        timeout=0.2,
    )
    assert time.monotonic() - start < 0.1
    assert result is history
    marker.assert_called_once()


@pytest.mark.asyncio
async def test_background_enqueue_coalesces_per_session_and_preserves_storage():
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db = object()
    runner._track_deferred_agent_worker = MagicMock()
    runner._evict_cached_agent = MagicMock()
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def blocked(*args):
        calls.append(args)
        started.set()
        await release.wait()

    runner._hmwa_hygiene_background_once = blocked
    entry = SimpleNamespace(session_id="same-session", last_prompt_tokens=190000)
    history = [{"role": "user", "content": "z" * 9000} for _ in range(80)]
    original = list(history)
    config = {"model": {"context_length": 200000}}
    first = runner._hmwa_enqueue_background_hygiene(None, entry, "key", history, config)
    second = runner._hmwa_enqueue_background_hygiene(None, entry, "key", history, config)
    assert len(runner._hygiene_background_jobs) == 1
    # Until a real summary commits, earlier facts cannot disappear silently.
    assert first is history and second is history
    assert first[0] is history[0] and first[-1] is history[-1]
    assert history == original
    await asyncio.wait_for(started.wait(), 2)
    assert len(calls) == 1
    # Concurrent foreground mutation must never re-target an admitted background worker.
    assert calls[0][1] is not entry
    entry.session_id = "rotated-during-turn"
    entry.last_prompt_tokens = 1
    assert calls[0][1].session_id == "same-session"
    assert calls[0][1].last_prompt_tokens == 190000
    release.set()
    await asyncio.wait_for(next(iter(runner._hygiene_background_jobs.values())), 2)
    await asyncio.sleep(0)
    assert not runner._hygiene_background_jobs
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_background_admission_caps_parallel_jobs_without_blocking_foreground():
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db = object()
    runner._evict_cached_agent = MagicMock()
    runner._track_deferred_agent_worker = MagicMock()
    loop = asyncio.get_running_loop()
    runner._hygiene_background_jobs = {str(i): loop.create_future() for i in range(4)}
    history = [{"role": "user", "content": "history"} for _ in range(6)]
    start = time.monotonic()
    result = runner._hmwa_enqueue_background_hygiene(
        None, SimpleNamespace(session_id="fifth", last_prompt_tokens=250000),
        "key", history, {},
    )
    assert time.monotonic() - start < 0.1
    assert result is history
    assert len(runner._hygiene_background_jobs) == 4
    runner._track_deferred_agent_worker.assert_not_called()


@pytest.mark.asyncio
async def test_deferred_commit_rebuilds_cache_and_usage_only_on_next_turn():
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db = None
    runner._evict_cached_agent = MagicMock()
    runner._hygiene_background_rebuild = {"key": "sess"}
    entry = SimpleNamespace(session_id="sess", last_prompt_tokens=205000)
    history = [{"role": "user", "content": "small"} for _ in range(6)]
    result = runner._hmwa_enqueue_background_hygiene(
        None, entry, "key", history, {"model": {"context_length": 200000}},
    )
    assert result is history
    assert entry.session_id == "sess"
    assert entry.last_prompt_tokens == 0
    assert runner._hygiene_background_rebuild == {}
    runner._evict_cached_agent.assert_called_once_with("key")
    assert getattr(runner, "_hygiene_background_jobs", {}) == {}


@pytest.mark.asyncio
async def test_short_next_turn_still_consumes_fenced_summary_rebuild():
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._hygiene_background_rebuild = {"key": "original"}
    runner._evict_cached_agent = MagicMock()
    entry = SimpleNamespace(session_id="original", last_prompt_tokens=210000)
    tiny = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "summary"}]
    result = await runner._hmwa_run_session_hygiene(None, None, entry, "key", tiny, "q", 1)
    assert result is tiny
    assert entry.session_id == "original"
    assert entry.last_prompt_tokens == 0
    runner._evict_cached_agent.assert_called_once_with("key")
    assert not runner._hygiene_background_rebuild


@pytest.mark.asyncio
async def test_background_payload_keeps_uncompressed_old_commitments():
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    old = {"role": "user", "content": "Important commitment from months ago: keep the grant application."}
    history = [old] + [
        {"role": "assistant", "content": "large but uncompressed " * 900}
        for _ in range(110)
    ] + [{"role": "user", "content": "Continue the grant application."}]
    assert runner._hmwa_background_model_tail(history, {}) is history
    # The original hard limit still bounds pathological message counts,
    # without rewriting or deleting any durable history row.
    very_long = [old] + [{"role": "assistant", "content": str(i)} for i in range(5001)]
    bounded = runner._hmwa_background_model_tail(very_long, {})
    assert bounded is not very_long
    assert len(bounded) <= 5000
    assert len(very_long) == 5002
    assert bounded[-1] is very_long[-1]


@pytest.mark.asyncio
async def test_background_rebuild_does_not_cross_session_rotation():
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db = None
    runner._evict_cached_agent = MagicMock()
    runner._hygiene_background_rebuild = {"key": "outdated-session"}
    entry = SimpleNamespace(session_id="new-session", last_prompt_tokens=32000)
    history = [{"role": "user", "content": "new conversation"} for _ in range(5)]
    result = runner._hmwa_enqueue_background_hygiene(None, entry, "key", history, {})
    assert result is history
    assert entry.last_prompt_tokens == 32000
    assert runner._hygiene_background_rebuild == {}
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_background_commit_never_rebinds_or_evicts_live_session(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._track_deferred_agent_worker = MagicMock()
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    runner._hmwa_hygiene_settings = AsyncMock(return_value=SimpleNamespace(
        compression_enabled=True, total_ceiling_seconds=5.0,
        failure_cooldown_seconds=30.0, data={},
    ))
    runner._hmwa_hygiene_plan = AsyncMock(return_value=SimpleNamespace(
        needs_compress=True, approx_tokens=250000,
    ))
    runner._resolve_session_agent_runtime = MagicMock(return_value=(
        "test-model", {"api_key": "fake"},
    ))
    runner._evict_cached_agent = MagicMock()
    runner._hmwa_hygiene_record_failure_cooldown = AsyncMock()
    agent = SimpleNamespace(
        context_compressor=SimpleNamespace(bind_session_state=MagicMock()),
        session_id="session-unchanged", _last_compaction_in_place=False,
    )

    def compact(_messages, _prompt, **kwargs):
        fence = kwargs["commit_fence"]
        fence.mark_commit_watermark_fenced()
        assert fence.begin_commit()
        agent._last_compaction_in_place = True
        fence.finish_commit()
        return ([{"role": "assistant", "content": "summary"}], None)

    agent._compress_context = compact
    runner._hmwa_hygiene_build_agent = AsyncMock(return_value=(agent, object()))
    monkeypatch.setattr(gateway_run, "_reset_hygiene_failure_streak", MagicMock())
    entry = SimpleNamespace(session_id="session-unchanged", last_prompt_tokens=250000)
    history = [{"role": "user", "content": "history"} for _ in range(6)]
    await runner._hmwa_hygiene_background_once(None, entry, "key", history)
    assert entry.session_id == "session-unchanged"
    assert entry.last_prompt_tokens == 250000
    runner._evict_cached_agent.assert_not_called()
    assert runner._hygiene_background_rebuild == {"key": "session-unchanged"}
    runner._hmwa_hygiene_record_failure_cooldown.assert_not_awaited()
    runner._cleanup_agent_resources_off_loop.assert_awaited_once()


@pytest.mark.asyncio
async def test_background_late_commit_refreshes_next_turn_without_waiting(monkeypatch):
    """An admitted commit finishing after the host timeout still marks safe rehydration."""
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._track_deferred_agent_worker = MagicMock()
    runner._hmwa_hygiene_defer_cleanup = MagicMock()
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    runner._hmwa_hygiene_settings = AsyncMock(return_value=SimpleNamespace(
        compression_enabled=True, total_ceiling_seconds=0.025,
        failure_cooldown_seconds=30.0, data={},
    ))
    runner._hmwa_hygiene_plan = AsyncMock(return_value=SimpleNamespace(
        needs_compress=True, approx_tokens=230000,
    ))
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("test-model", {"api_key": "fake"}),
    )
    worker_started = threading.Event()
    release_commit = threading.Event()
    agent = SimpleNamespace(
        context_compressor=SimpleNamespace(bind_session_state=MagicMock()),
        session_id="same-sid", _last_compaction_in_place=False,
    )

    def late_commit(_messages, _prompt, **kwargs):
        fence = kwargs["commit_fence"]
        fence.mark_commit_watermark_fenced()
        assert fence.begin_commit()
        worker_started.set()
        assert release_commit.wait(3)
        agent._last_compaction_in_place = True
        fence.finish_commit()
        return ([{"role": "assistant", "content": "summary"}], None)

    agent._compress_context = late_commit
    runner._hmwa_hygiene_build_agent = AsyncMock(return_value=(agent, object()))
    runner._hmwa_hygiene_record_failure_cooldown = AsyncMock()
    entry = SimpleNamespace(session_id="same-sid", last_prompt_tokens=230000)
    try:
        await asyncio.wait_for(runner._hmwa_hygiene_background_once(
            None, entry, "same-key",
            [{"role": "user", "content": "history"} for _ in range(6)],
        ), timeout=0.3)
        assert worker_started.is_set()
        assert entry.last_prompt_tokens == 230000
        assert not getattr(runner, "_hygiene_background_rebuild", {})
    finally:
        release_commit.set()
    for _ in range(30):
        if getattr(runner, "_hygiene_background_rebuild", {}) == {"same-key": "same-sid"}:
            break
        await asyncio.sleep(0.01)
    assert runner._hygiene_background_rebuild == {"same-key": "same-sid"}
    runner._hmwa_hygiene_defer_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_turn_hold_keeps_admission_and_adopts_watermark_fenced_summary(
    monkeypatch, tmp_path
):
    """A watermark-fenced worker keeps its commit admission at turn-hold
    expiry; its late summary is ADOPTED (committed), not discarded — while
    the turn itself is still released at the budget (#90845 invariant).
    """
    worker_started = threading.Event()
    release_worker = threading.Event()
    committed = threading.Event()
    cleanup_done = threading.Event()
    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None

    class FencedStreamingAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", "sess-97963")
            self._session_db = kwargs.get("session_db")
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
            )
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock(side_effect=cleanup_done.set)
            type(self).last_instance = self

        def _compress_context(
            self, messages, *_args, commit_fence=None, **_kwargs
        ):
            # Real compress_context marks the fence right after capturing
            # the active-row watermark under the durable compression lock.
            if commit_fence is not None:
                commit_fence.mark_commit_watermark_fenced()
            worker_started.set()
            # Thinking-model shape: continuous progress, no commit yet —
            # only the turn-hold budget can release the waiting turn.
            # Bounded spin: a failing assertion before release_worker.set()
            # must not leave this executor thread alive forever (pytest
            # would hang at interpreter exit joining executor threads).
            _spin_started = time.monotonic()
            while not release_worker.is_set():
                if time.monotonic() - _spin_started > 20:
                    return (messages, None)
                if commit_fence is not None:
                    commit_fence.touch_progress()
                time.sleep(0.01)
            if commit_fence is not None and not commit_fence.begin_commit():
                return (messages, None)
            try:
                self._session_db.archive_and_compact(
                    self.session_id,
                    [{"role": "assistant", "content": "summary"}],
                    watermark=6,
                )
                self._last_compaction_in_place = True
                committed.set()
                return ([{"role": "assistant", "content": "summary"}], None)
            finally:
                if commit_fence is not None:
                    commit_fence.finish_commit()

    gateway_run = importlib.import_module("gateway.run")
    _write_turnhold_config(tmp_path)
    _install_fakes(monkeypatch, gateway_run, tmp_path, FencedStreamingAgent)

    adapter = _CaptureAdapter()
    runner = _build_runner(gateway_run, adapter, fake_db)

    started = time.monotonic()
    result = await asyncio.wait_for(runner._handle_message(_make_event()), timeout=15)
    elapsed = time.monotonic() - started

    # #90845/#92318 invariant intact: the turn is released at the budget.
    assert result == "ok"
    assert elapsed < 5.0, f"turn held for {elapsed:.1f}s despite the turn-hold budget"
    assert worker_started.is_set()
    assert runner._run_agent.await_count == 1

    # (b) NO retry-after was armed while the attempt is still running —
    # arming it would block the agent-side preflight from adopting the
    # finished summary ("same-session cooldown active", #97963).
    assert not fake_db.record_compression_failure_cooldown.called, (
        "keep-admission path must not arm the retry-after while the "
        "detached attempt is still running"
    )

    # The detached worker finishes late; its commit is ADMITTED (adoption),
    # not refused — the summary attempt is no longer burned.
    release_worker.set()
    await asyncio.wait_for(asyncio.to_thread(committed.wait, 5), timeout=6)
    assert committed.is_set(), (
        "watermark-fenced worker must keep its commit admission after "
        "turn-hold expiry (fence was cancelled — attempt burned)"
    )
    fake_db.archive_and_compact.assert_called_once()
    # The commit went through the watermark-fenced path (concurrent tail
    # rows above the watermark survive the compaction).
    assert fake_db.archive_and_compact.call_args.kwargs.get("watermark") == 6

    await _drain_deferred(runner)
    await asyncio.wait_for(asyncio.to_thread(cleanup_done.wait, 5), timeout=6)
    FencedStreamingAgent.last_instance.close.assert_called_once()

    # Successful adoption resets the hygiene failure streak and still never
    # advances it (the deferral is not a failure).
    assert not fake_db.increment_hygiene_failure_streak.called
    assert fake_db.reset_hygiene_failure_streak.called
    # Deferral notice still reaches the user.
    sent = [m["content"] for m in adapter.sent]
    assert any(
        "deferred" in c.lower() or "still streaming" in c.lower() for c in sent
    ), f"turn-hold must send deferral notice, got: {sent}"


@pytest.mark.asyncio
async def test_turn_hold_kept_admission_arms_flat_retry_only_when_nothing_commits(
    monkeypatch, tmp_path
):
    """If the kept-admission worker ends WITHOUT committing (summary failed
    / attempt superseded), the flat non-escalating retry-after is restored so
    sustained traffic does not spawn-and-abandon a compressor every turn —
    but only AFTER the attempt truly ended, and without touching the streak.
    """
    worker_started = threading.Event()
    release_worker = threading.Event()
    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None

    class FencedNoCommitAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", "sess-97963")
            self._session_db = kwargs.get("session_db")
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
            )
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()

        def _compress_context(
            self, messages, *_args, commit_fence=None, **_kwargs
        ):
            if commit_fence is not None:
                commit_fence.mark_commit_watermark_fenced()
            worker_started.set()
            _spin_started = time.monotonic()
            while not release_worker.is_set():
                if time.monotonic() - _spin_started > 20:
                    return (messages, None)
                if commit_fence is not None:
                    commit_fence.touch_progress()
                time.sleep(0.01)
            # Summary failed — return unchanged, no commit.
            return (messages, None)

    gateway_run = importlib.import_module("gateway.run")
    _write_turnhold_config(tmp_path)
    _install_fakes(monkeypatch, gateway_run, tmp_path, FencedNoCommitAgent)

    adapter = _CaptureAdapter()
    runner = _build_runner(gateway_run, adapter, fake_db)

    result = await asyncio.wait_for(runner._handle_message(_make_event()), timeout=15)
    assert result == "ok"
    assert worker_started.is_set()
    # While the attempt still runs: no cooldown, so preflight adoption
    # stays possible.
    assert not fake_db.record_compression_failure_cooldown.called

    release_worker.set()
    await _drain_deferred(runner)
    # Let the done-callback fire.
    for _ in range(100):
        if fake_db.record_compression_failure_cooldown.called:
            break
        await asyncio.sleep(0.05)

    # Nothing committed → flat retry-after restored (spacing), streak intact.
    assert fake_db.record_compression_failure_cooldown.called, (
        "a kept-admission attempt that ends without committing must restore "
        "the flat turn-hold retry-after spacing"
    )
    args = fake_db.record_compression_failure_cooldown.call_args[0]
    retry = args[1] - time.time()
    assert retry <= 120, (
        f"retry-after must stay flat (~60s), got {retry:.0f}s"
    )
    assert "turn-hold" in (args[2] or "")
    assert not fake_db.increment_hygiene_failure_streak.called, (
        "turn-hold deferral must never advance the failure streak"
    )


@pytest.mark.asyncio
async def test_turn_hold_without_watermark_fence_still_cancels(
    monkeypatch, tmp_path
):
    """A worker whose commit is NOT watermark-fenced (no session_db /
    watermark capture failed) must still be cancelled at turn-hold expiry —
    a late unfenced commit could clobber newer turns. Never worse than the
    status quo. (Complements the pinned #90845 test, which exercises the
    same path through the public surface.)
    """
    worker_started = threading.Event()
    release_worker = threading.Event()
    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None

    class UnfencedStreamingAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", "sess-97963")
            self._session_db = kwargs.get("session_db")
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
            )
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()

        def _compress_context(
            self, messages, *_args, commit_fence=None, **_kwargs
        ):
            # Deliberately NO mark_commit_watermark_fenced().
            worker_started.set()
            _spin_started = time.monotonic()
            while not release_worker.is_set():
                if time.monotonic() - _spin_started > 20:
                    return (messages, None)
                if commit_fence is not None:
                    commit_fence.touch_progress()
                time.sleep(0.01)
            if commit_fence is not None and not commit_fence.begin_commit():
                return (messages, None)
            try:
                self._session_db.archive_and_compact(
                    self.session_id,
                    [{"role": "assistant", "content": "too late"}],
                )
                return ([{"role": "assistant", "content": "too late"}], None)
            finally:
                if commit_fence is not None:
                    commit_fence.finish_commit()

    gateway_run = importlib.import_module("gateway.run")
    _write_turnhold_config(tmp_path)
    _install_fakes(monkeypatch, gateway_run, tmp_path, UnfencedStreamingAgent)

    adapter = _CaptureAdapter()
    runner = _build_runner(gateway_run, adapter, fake_db)

    result = await asyncio.wait_for(runner._handle_message(_make_event()), timeout=15)
    assert result == "ok"
    assert worker_started.is_set()

    release_worker.set()
    await _drain_deferred(runner)
    await asyncio.sleep(0.2)
    # The unfenced late commit was refused — discard as before the fix.
    fake_db.archive_and_compact.assert_not_called()
    # Legacy path still records the flat retry-after immediately.
    assert fake_db.record_compression_failure_cooldown.called
    assert not fake_db.increment_hygiene_failure_streak.called
