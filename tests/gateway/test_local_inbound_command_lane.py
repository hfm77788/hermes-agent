"""Command-lane regression contracts for local-inbound injection (repair t_a032be60).

``run_local_inbound`` once hardcoded ``allow_gateway_control=False``, which kept
``MessageEvent.is_command()``/``get_command()`` permanently None for injected text
(diagnosis t_6196a57c). The hema teacher health check's ``/new`` therefore ran as
a normal model turn: the session never reset, the model answered with fake
「已开新会话。」 receipts, and ``/new`` persisted as a user message.

The gate now opens ONLY for slash text that names a registered gateway command,
mirroring the native chain's own entry predicate (``get_command`` parsing →
``resolve_command`` alias canonicalisation → ``is_gateway_known_command``).
Everything else — plain text, unknown names, path-like pseudo names, skill
slashes — keeps the pre-repair model-lane behaviour, and because the rendezvous
short-circuit shares this one predicate, such text also keeps the t_479a04ac
grace wait (t_6196a57c §2.2: a ``startswith("/")`` shortcut would skip it and
drop the receipt). Whitelist, slash-access gate and the API-lane forced False
are untouched.
"""
import asyncio

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.run_local_inbound import (
    _injected_slash_opens_command_lane,
    inject_local_inbound,
)
from gateway.run_turn import _publish_local_inbound_response
from gateway.session import SessionSource
from hermes_cli.commands import is_gateway_known_command

from tests.gateway.test_local_inbound import FakeRunner, _payload


def _dispatch_stub():
    """A stand-in ``self`` carrying the seams ``_hm_resolve_command`` and
    ``_hm_dispatch_canonical_command`` touch, so the REAL resolution methods run
    over the event that ``inject_local_inbound`` built."""
    dispatched: dict = {}

    async def _hooks(event, source, quick_key, command, canonical):
        return False, None, None  # no command hooks participate here

    async def _cmd_new(event, source, quick_key):
        dispatched["cmd_new"] = event
        return True, "新会话已开启。"

    async def _handle_compress(event):
        dispatched["compress"] = event
        return "压缩预览：无需压缩。"

    stub = GatewayRunner  # class attrs (canonical set, idle/plain command tuples)
    import contextlib
    from types import SimpleNamespace
    return SimpleNamespace(
        _HM_CANONICAL_COMMANDS=stub._HM_CANONICAL_COMMANDS,
        _hm_quick_commands=dict,
        _check_slash_access=lambda source, canonical: None,
        _hm_command_hooks=_hooks,
        _hm_drop_held_input=lambda event: dispatched.setdefault("drop_held", event),
        _gateway_plain_command_handlers=dict,
        _gateway_idle_command_handlers=lambda: {"compress": _handle_compress},
        _async_profile_scope_for_source=lambda source: contextlib.nullcontext(),
        _hm_cmd_new=_cmd_new,
    ), dispatched


async def _resolve_and_dispatch(stub, event):
    """Mirror the built-in idle-path slice of ``_hm_dispatch_idle_commands``."""
    handled, result, command, canonical = await GatewayRunner._hm_resolve_command(
        stub, event, event.source, "session-key"
    )
    if not handled:
        handled, result = await GatewayRunner._hm_dispatch_canonical_command(
            stub, event, event.source, "session-key", canonical
        )
    return handled, result, command, canonical


def test_injected_new_opens_the_command_lane_gate():
    runner = FakeRunner()
    result = asyncio.run(inject_local_inbound(runner, _payload(message_id="cmd-new-1", text="/new")))
    event = runner.adapter.seen
    assert result["accepted"] is True
    # The event the live runner receives now resolves as the native /new command.
    assert event.allow_gateway_control is True
    assert event.is_command() is True
    assert event.get_command() == "new"


def test_injected_new_resolves_and_dispatches_via_the_real_gateway_chain():
    # Proves the opened gate reaches the RESET lane: the real resolution +
    # canonical-dispatch methods route the injected event to _hm_cmd_new and the
    # turn is handled there — the agent/turn path (the only writer of a user
    # line) is never entered, so no /new text persists and no model turn runs.
    runner = FakeRunner()
    stub, dispatched = _dispatch_stub()

    async def command_lane_handler(event):
        handled, receipt, command, canonical = await _resolve_and_dispatch(stub, event)
        assert (handled, command, canonical) == (True, "new", "new")
        assert is_gateway_known_command(canonical)
        event.source._local_inbound_response_future.set_result(receipt)

    runner.adapter._message_handler = command_lane_handler
    result = asyncio.run(inject_local_inbound(runner, _payload(message_id="cmd-new-2", text="/new")))

    assert result["accepted"] is True
    assert result["response"] == "新会话已开启。"
    assert "reason" not in result
    assert "cmd_new" in dispatched  # reset lane owned the turn
    assert runner.calls == 0  # never the agent lane: no model turn, no user line


def test_plain_text_injection_keeps_the_control_gate_closed():
    # Default zero-change lock (mirrors the :94 anchor in test_local_inbound.py):
    # ordinary learner text from the sidecar may never control the gateway.
    runner = FakeRunner()
    asyncio.run(inject_local_inbound(runner, _payload(message_id="plain-1", text="4000米")))
    assert runner.adapter.seen.allow_gateway_control is False


def test_unknown_slash_text_still_falls_through_to_the_model_lane():
    # The whitelist is NOT widened: an unregistered name keeps the gate CLOSED,
    # so the event is not a command at all. The real chain entry (the actual
    # resolver) yields no command/canonical, and the unknown-slash rejection
    # branch stays unreachable for injected text — it lives inside the skill
    # rewrite, which early-returns on a falsy command. The text answers as an
    # agent turn, exactly like before the repair.
    runner = FakeRunner()
    stub, dispatched = _dispatch_stub()

    async def resolve_like_handler(event):
        assert event.allow_gateway_control is False
        assert event.is_command() is False
        handled, _result, command, canonical = await _resolve_and_dispatch(stub, event)
        assert (handled, command, canonical) == (False, None, None)
        assert GatewayRunner._hm_skill_slash_rewrite(
            stub, event, event.source, "session-key", None
        ) is None  # no rejection reply is produced for injected unknown text
        # fall-through semantics = the model lane answers (FakeRunner turns)
        await runner._handle_message(event)

    runner.adapter._message_handler = resolve_like_handler
    result = asyncio.run(inject_local_inbound(
        runner, _payload(message_id="unknown-1", text="/frobnicate 这个指令不存在")))

    event = runner.adapter.seen
    assert event.get_command() is None  # the command chain was never entered
    assert result["response"] == "答对了。下一题"
    assert runner.calls == 1  # …and the model lane runs, unchanged
    assert "cmd_new" not in dispatched


def test_unknown_slash_keeps_the_rendezvous_grace_wait(monkeypatch):
    # t_6196a57c §2.2 secondary hazard: unknown-slash text is a model turn, so
    # it must park the bounded grace like plain text does (a late receipt still
    # reaches the rendezvous). A ``startswith("/")`` shortcut would skip the
    # wait and silently drop the answer.
    from gateway import run_local_inbound as run_local_inbound_module

    monkeypatch.setattr(run_local_inbound_module, "_RESPONSE_RENDEZVOUS_GRACE_SECONDS", 0.2)
    runner = FakeRunner()

    async def model_lane_handler(event):
        # A real agent turn answers after the handler returns, via egress.
        runner.calls += 1
        runner.adapter.seen = event

    runner.adapter._message_handler = model_lane_handler
    result = asyncio.run(inject_local_inbound(
        runner, _payload(message_id="unknown-late-1", text="/frobnicate")))

    assert result["accepted"] is True
    assert result["reason"] == "response_timeout"  # it WAITED, like plain text
    assert result["response_wait_ms"] >= 150
    assert runner.calls == 1
    assert "unknown-late-1" not in runner._local_inbound_response_waiters


def test_file_path_slash_text_is_never_resolved_as_a_command():
    # 「拒绝带 / 伪命令名」现口径不变: get_command() rejects names containing "/"
    # (file paths), so path-like injections stay conversational.
    runner = FakeRunner()
    asyncio.run(inject_local_inbound(
        runner, _payload(message_id="path-1", text="/tmp/state.db 备份一下")))
    event = runner.adapter.seen
    assert event.allow_gateway_control is False
    assert event.get_command() is None


@pytest.mark.parametrize(("text", "opens"), [
    ("/new", True),  # the health check's reset verb
    ("/NEW", True),  # case-insensitive, like get_command
    ("/reset", True),  # alias, canonicalised to new like _hm_resolve_command
    ("/new@hema_bot", True),  # mention suffix cut, like get_command
    ("/compress --preview", True),  # args ride the command lane
    ("/help", True),
    ("/stop", True),
    ("/frobnicate 这个指令不存在", False),  # unknown name stays chat
    ("/new-thing", False),  # prefix lookalikes are not aliases
    ("/tmp/state.db 备份一下", False),  # path-like pseudo name
    ("/", False),
    ("/ new", False),
    ("", False),
    ("4000米", False),
    ("@image:/x /new", False),  # attachment-ref prefix: fail closed
])
def test_command_lane_gate_predicate_locks_the_whitelist_agreement(text, opens):
    assert _injected_slash_opens_command_lane(text) is opens


def _install_grace_park_spy(monkeypatch) -> list:
    # tests/AGENTS.md flake policy (codex P2 on PR #133): prove the command
    # lane skipped the rendezvous grace park by observing the wait primitive
    # itself, not by bounding wall-clock at 5 ms on a shared runner.
    from gateway import run_local_inbound as _run_local_inbound_module

    parked: list = []
    real_wait_for = asyncio.wait_for

    async def _spy_wait_for(future, timeout=None, **kwargs):
        parked.append(timeout)
        return await real_wait_for(future, timeout=timeout, **kwargs)

    monkeypatch.setattr(_run_local_inbound_module.asyncio, "wait_for", _spy_wait_for)
    return parked


def test_command_lane_receipt_hits_the_message_id_rendezvous(monkeypatch):
    # t_479a04ac mechanism, command-lane half: a receipt published through the
    # message-id registry (a source rebuilt from the durable snapshot carries no
    # future attribute) still resolves the inject — and never reports a timeout
    # even though command lanes skip the grace park.
    parked = _install_grace_park_spy(monkeypatch)
    runner = FakeRunner()

    async def command_lane_handler(event):
        assert event.get_command() == "new"
        rebuilt = SessionSource(
            platform=Platform.DINGTALK, chat_id="cid-math", chat_type="group",
            user_id="child", message_id=event.source.message_id,
        )
        _publish_local_inbound_response(rebuilt, {"final_response": "新会话已开启。"}, runner=runner)

    runner.adapter._message_handler = command_lane_handler
    result = asyncio.run(inject_local_inbound(runner, _payload(message_id="cmd-rdv-1", text="/new")))

    assert result["accepted"] is True
    assert result["response"] == "新会话已开启。"
    assert "reason" not in result
    assert parked == []  # the rendezvous wait primitive was never invoked
    assert runner._local_inbound_response_waiters == {}  # registry cleaned


def test_command_lane_without_publish_returns_clean_empty_result(monkeypatch):
    # A command lane that answers without a publish-side turn (the receipts ride
    # the native send path, not the future) must return "" with no fake content
    # and no response_timeout, and still clean the rendezvous entry.
    parked = _install_grace_park_spy(monkeypatch)
    runner = FakeRunner()

    async def command_lane_handler(event):
        assert event.get_command() == "new"  # answers nothing through the rendezvous

    runner.adapter._message_handler = command_lane_handler
    result = asyncio.run(inject_local_inbound(runner, _payload(message_id="cmd-silent-1", text="/new")))

    assert result["accepted"] is True
    assert result["response"] == ""
    assert "reason" not in result
    assert parked == []  # the rendezvous wait primitive was never invoked
    assert "cmd-silent-1" not in runner._local_inbound_response_waiters


def test_injected_compress_resolves_through_the_command_lane():
    runner = FakeRunner()
    stub, dispatched = _dispatch_stub()

    async def command_lane_handler(event):
        assert event.get_command() == "compress"
        dispatched["event"] = event
        handled, result, _command, canonical = await _resolve_and_dispatch(stub, event)
        assert (handled, canonical) == (True, "compress")
        event.source._local_inbound_response_future.set_result(result)

    runner.adapter._message_handler = command_lane_handler
    result = asyncio.run(inject_local_inbound(
        runner, _payload(message_id="cmd-compress-1", text="/compress --preview")))

    assert is_gateway_known_command("compress")
    assert "compress" in GatewayRunner._IDLE_COMMANDS
    assert dispatched["compress"] is dispatched["event"]  # idle dispatch table owned it
    assert result["response"] == "压缩预览：无需压缩。"
