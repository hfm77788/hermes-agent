"""Command-lane regression contracts for local-inbound injection (repair t_a032be60).

``run_local_inbound`` once hardcoded ``allow_gateway_control=False``, which kept
``MessageEvent.is_command()``/``get_command()`` permanently None for injected text
(diagnosis t_6196a57c). The hema teacher health check's ``/new`` therefore ran as
a normal model turn: the session never reset, the model answered with fake
「已开新会话。」 receipts, and ``/new`` persisted as a user message.

The gate opens ONLY for slash text that names a registered gateway command,
mirroring the native chain's own entry predicate (``get_command`` parsing →
``resolve_command`` alias canonicalisation → ``is_gateway_known_command``), and —
round-2 audit B2 — the plugin half of that whitelist is queried under the TARGET
profile's runtime scope, the same table the dispatched command reads. Everything
else — plain text, unknown names, path-like pseudo names, skill slashes — keeps
the pre-repair model-lane behaviour, and because the rendezvous short-circuit
shares the one predicate, such text also keeps the t_479a04ac grace wait
(t_6196a57c §2.2: a ``startswith("/")`` shortcut would skip it and drop the
receipt). Whitelist, slash-access gate and the API-lane forced False are
untouched.

Round-2 audit B1/B3 contracts: a command-lane receipt rides the handler's RETURN
value and is delivered by the PRODUCTION publish seam
(``run_turn._publish_local_inbound_response``, one-shot ``future.done()`` guarded
— an egress-published receipt is never overridden, and no test body may set the
future to prove the delivery); the branch itself is decided by the REAL
``_hm_dispatch_idle_commands`` orchestration, not a hand-picked resolver slice.
"""
import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

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


class _NativeIdleStub:
    """A stand-in runner whose IDLE-PATH ORCHESTRATION is the real production code.

    The resolution, canonical/quick/plugin dispatch and the skill-slash
    fall-through are the genuine ``GatewayRunner`` methods (audit B3: the branch
    must be decided by the real chain, including the quick/plugin legs the first
    round skipped). Only the leaf seams — slash access, command hooks, and the
    ``/new`` and ``/compress`` handlers — are stubs, recording what they were
    asked to run. No seam here ever touches the response future: delivering the
    receipt is the production inject path's job (audit B1).
    """

    _hm_resolve_command = GatewayRunner._hm_resolve_command
    _hm_dispatch_canonical_command = GatewayRunner._hm_dispatch_canonical_command
    _hm_dispatch_quick_and_plugin_commands = GatewayRunner._hm_dispatch_quick_and_plugin_commands
    _hm_skill_slash_rewrite = GatewayRunner._hm_skill_slash_rewrite
    _hm_dispatch_idle_commands = GatewayRunner._hm_dispatch_idle_commands
    _HM_CANONICAL_COMMANDS = GatewayRunner._HM_CANONICAL_COMMANDS

    def __init__(self, dispatched: dict):
        self._dispatched = dispatched
        self._draining = False

    def _route_stop_epoch(self, _session_key: str) -> int:
        return 0

    def _hm_quick_commands(self) -> dict:
        return {}

    def _check_slash_access(self, _source, _canonical):
        return None

    async def _hm_command_hooks(self, event, source, quick_key, command, canonical):
        return False, None, None  # no command hooks participate here

    def _hm_drop_held_input(self, event):
        self._dispatched["drop_held"] = event

    def _gateway_plain_command_handlers(self) -> dict:
        return {}

    def _gateway_idle_command_handlers(self) -> dict:
        return {"compress": self._handle_compress}

    def _async_profile_scope_for_source(self, _source):
        return contextlib.nullcontext()

    async def _hm_cmd_new(self, event, _source, _quick_key):
        self._dispatched["cmd_new"] = event
        return True, "新会话已开启。"

    async def _handle_compress(self, event):
        self._dispatched["compress"] = event
        return "压缩预览：无需压缩。"

    async def _hm_dispatch_plugin_command(self, event, source, command):
        self._dispatched["plugin"] = command
        return False, None, command

    async def _run_in_executor_with_context(self, func, *args):
        return func(*args)

    async def _admit_command_turn(self, event, source, session_key, stop_epoch):
        self._dispatched["admit_turn"] = event
        return True, None


def _native_idle_handler(runner: FakeRunner, stub: _NativeIdleStub):
    """A ``_message_handler`` clone of the real idle-path decision shape: the REAL
    ``_hm_dispatch_idle_commands`` owns the branch; a handled command answers with
    its return value, and fall-through runs the agent turn. The body NEVER touches
    the response future (that is the production seam's job, asserted end-to-end).
    """

    async def handler(event):
        stub._dispatched["seen"] = event
        handled, receipt = await stub._hm_dispatch_idle_commands(
            event, event.source, "session-key"
        )
        if handled:
            return receipt
        return await runner._handle_message(event)

    return handler


def test_injected_new_opens_the_command_lane_gate():
    runner = FakeRunner()
    result = asyncio.run(inject_local_inbound(runner, _payload(message_id="cmd-new-1", text="/new")))
    event = runner.adapter.seen
    assert result["accepted"] is True
    # The event the live runner receives now resolves as the native /new command.
    assert event.allow_gateway_control is True
    assert event.is_command() is True
    assert event.get_command() == "new"


def test_command_lane_receipt_delivered_by_production_publish_over_real_chain():
    # B1 + B3 in one: the real idle orchestration routes the injected event to the
    # reset lane and RETURNS its receipt; the injection's production code publishes
    # that return value into the rendezvous. The test body never touches the future,
    # so this fails if the dispatch return is discarded (the round-1/round-2 gap:
    # accepted=True, response=""). The agent lane (the only writer of a user line)
    # is never entered.
    runner = FakeRunner()
    dispatched: dict = {}
    runner.adapter._message_handler = _native_idle_handler(runner, _NativeIdleStub(dispatched))

    result = asyncio.run(
        inject_local_inbound(runner, _payload(message_id="cmd-new-prod-1", text="/new"))
    )

    assert result["accepted"] is True
    assert result["response"] == "新会话已开启。"
    assert "reason" not in result
    assert dispatched.get("cmd_new") is dispatched.get("seen")  # reset lane owned the turn
    assert "admit_turn" not in dispatched  # never rewritten into a model turn
    assert runner.calls == 0  # no model turn ran, so no /new user line can persist


def test_command_lane_return_never_overrides_an_egress_publish():
    # B1 double-delivery lock: ``_publish_local_inbound_response`` is one-shot. A
    # receipt already published by egress (e.g. a durable-admission replay of the
    # turn) must win verbatim over the handler return value.
    runner = FakeRunner()

    async def handler(event):
        event.source._local_inbound_response_future.set_result("EGRESS-FIRST")
        return "HANDLER-RETURN"

    runner.adapter._message_handler = handler
    result = asyncio.run(
        inject_local_inbound(runner, _payload(message_id="cmd-no-double-1", text="/new"))
    )

    assert result["response"] == "EGRESS-FIRST"
    assert "reason" not in result


def test_plain_text_injection_keeps_the_control_gate_closed():
    # Default zero-change lock (mirrors the :94 anchor in test_local_inbound.py):
    # ordinary learner text from the sidecar may never control the gateway.
    runner = FakeRunner()
    asyncio.run(inject_local_inbound(runner, _payload(message_id="plain-1", text="4000米")))
    assert runner.adapter.seen.allow_gateway_control is False


def test_unknown_slash_branch_decided_by_the_real_idle_orchestration():
    # B3: the WHOLE real idle chain — resolver, canonical dispatch, the
    # quick/plugin leg and the skill-slash rewrite — decides that a gate-closed
    # unknown name is NOT a command and falls through. Nothing in this test body
    # pre-selects the branch: the handler consults only the production
    # ``_hm_dispatch_idle_commands`` return. The answer arrives as an agent turn
    # with the rendezvous receipt intact, exactly like before the repair.
    runner = FakeRunner()
    dispatched: dict = {}
    stub = _NativeIdleStub(dispatched)
    runner.adapter._message_handler = _native_idle_handler(runner, stub)

    result = asyncio.run(inject_local_inbound(
        runner, _payload(message_id="unknown-real-1", text="/frobnicate 这个指令不存在")))

    event = runner.adapter.seen
    assert event.allow_gateway_control is False
    assert event.get_command() is None          # the command chain was never entered
    assert result["response"] == "答对了。下一题"    # the model lane answered via the future
    assert runner.calls == 1
    assert "cmd_new" not in dispatched
    assert "admit_turn" not in dispatched
    # the real quick/plugin orchestration method ran (recorded below); its plugin
    # LEAF only fires for a resolvable command, so for gate-closed unknown text the
    # proof is that the orchestration returned without inventing one.
    assert dispatched.get("seen") is event
    assert "admit_turn" not in dispatched  # no skill-rewrite turn was admitted either


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
    # A command lane whose handler genuinely answers with NO return value and NO
    # publish (e.g. receipts riding the native send path) must return "" with no
    # fake content and no response_timeout, and still clean the rendezvous entry.
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


def test_injected_compress_dispatches_via_real_idle_table_and_answers_with_receipt(monkeypatch):
    # The canonical-dispatch leg of the REAL chain owns /compress through the
    # idle-handler table, and the preview receipt reaches the sidecar via the
    # production publish of the handler return value (no manual set_result).
    parked = _install_grace_park_spy(monkeypatch)
    runner = FakeRunner()
    dispatched: dict = {}
    stub = _NativeIdleStub(dispatched)
    runner.adapter._message_handler = _native_idle_handler(runner, stub)

    result = asyncio.run(inject_local_inbound(
        runner, _payload(message_id="cmd-compress-1", text="/compress --preview")))

    assert is_gateway_known_command("compress")
    assert "compress" in GatewayRunner._IDLE_COMMANDS
    assert dispatched.get("compress") is dispatched.get("seen")  # idle dispatch table owned it
    assert result["response"] == "压缩预览：无需压缩。"
    assert parked == []  # command lane: never burned the grace window


class _SplitRouteRunner(FakeRunner):
    """FakeRunner whose routing seams mirror ``_make_profile_message_handler``:
    the canonicalized identity carries a RUNTIME home distinct from both the
    caller's home and the receiving bot's profile home (round-3 audit: resolving
    to the transport profile alone false-opens transport-only verbs and
    false-closes runtime-only ones)."""

    def __init__(self, profile_home, runtime_home):
        super().__init__()
        self._profile_home = profile_home
        self._runtime_home = runtime_home
        self.canonicalize_calls: list[tuple] = []

    def _routed_profile_home(self, _profile_name):
        return self._profile_home

    def _canonicalize(self, source, *, transport_profile=None, primary_home=None):
        self.canonicalize_calls.append((transport_profile, primary_home))
        return SimpleNamespace(runtime_home=self._runtime_home)

    def _routed_runtime_home(self, identity, profile_home):
        return identity.runtime_home if identity is not None else profile_home


def test_gate_reads_the_runtime_home_plugin_registry(tmp_path, monkeypatch):
    # B2 (rounds 2+3): the plugin half of ``is_gateway_known_command`` is per-home
    # (plugins.py caches one manager per ``_plugin_home_key``), while the command
    # dispatches under the canonicalized identity's RUNTIME scope. Neither the
    # caller's home nor the receiving bot's profile home is the right table when
    # they differ; the gate must query the registry inside the runtime home.
    # Built-ins answer from the static set and never bind.
    import hermes_cli.plugins as plugins_module
    from hermes_constants import get_hermes_home

    transport_home = (tmp_path / "profiles" / "transport-bot").expanduser().resolve()
    runtime_home = (tmp_path / "profiles" / "runtime-target").expanduser().resolve()
    transport_home.mkdir(parents=True)
    runtime_home.mkdir(parents=True)
    # The caller home is resolved INSIDE the probe: the autouse home-isolation
    # fixture may re-point HERMES_HOME per test, and any name captured before the
    # bind would key the table on a stale path.
    def _home_keyed_discovery():
        home = str(Path(get_hermes_home()).expanduser().resolve())
        if home == str(runtime_home):
            commands = {"runtimeonly": {"description": "runtime-profile verb", "handler": None}}
        elif home == str(transport_home):
            commands = {"transportonly": {"description": "transport-profile verb", "handler": None}}
        else:
            commands = {"launchonly": {"description": "launch-profile verb", "handler": None}}
        return SimpleNamespace(_plugin_commands=commands)

    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", _home_keyed_discovery)

    runner = _SplitRouteRunner(transport_home, runtime_home)

    # runtime-only verb: caller- or transport-scope queries both FALSE-CLOSE it.
    asyncio.run(inject_local_inbound(runner, _payload(message_id="b2-open", text="/runtimeonly")))
    assert runner.adapter.seen.allow_gateway_control is True
    assert runner.canonicalize_calls  # identity resolved through the real seam

    # transport-only verb: not registered in the runtime table — must NOT open.
    asyncio.run(inject_local_inbound(runner, _payload(message_id="b2-close", text="/transportonly")))
    assert runner.adapter.seen.allow_gateway_control is False

    # the scope bind must unwind; the injection returns to the ambient caller home
    assert Path(get_hermes_home()).expanduser().resolve() not in (runtime_home, transport_home)


def test_command_lane_receipt_is_sanitised_under_the_runtime_home(tmp_path, monkeypatch):
    # B1 (round-3): the publish seam's sanitisation is scope-sensitive (phone
    # policy reads the ACTIVE profile; the secret-source table keys on the active
    # hermes_home_key). A receipt returned after the handler left the target scope
    # must be published INSIDE that scope, never sanitised under the caller's.
    import gateway.run_turn as run_turn_module
    from hermes_constants import get_hermes_home

    runtime_home = tmp_path / "profiles" / "runtime-target"
    runtime_home.mkdir(parents=True)
    published_home: list[str] = []
    real_publish = run_turn_module._publish_local_inbound_response

    def spy_publish(source, response, runner=None):
        published_home.append(str(Path(get_hermes_home()).expanduser().resolve()))
        return real_publish(source, response, runner=runner)

    monkeypatch.setattr(run_turn_module, "_publish_local_inbound_response", spy_publish)

    dispatched: dict = {}
    runner = _SplitRouteRunner(runtime_home, runtime_home)  # transport == runtime home here
    runner.adapter._message_handler = _native_idle_handler(runner, _NativeIdleStub(dispatched))

    result = asyncio.run(inject_local_inbound(
        runner, _payload(message_id="b1-scope-1", text="/new")))

    assert result["response"] == "新会话已开启。"
    assert published_home == [str(runtime_home.resolve())]  # published under the TARGET scope


def test_gate_keeps_caller_scope_when_target_profile_does_not_resolve():
    # ``UNRESOLVED_PROFILE_HOME`` means the target's native handlers run unscoped
    # too — its dispatch reads the CALLER's table, so binding nothing IS the
    # alignment, and built-in commands keep opening the gate as before.
    from gateway.run_adapters import UNRESOLVED_PROFILE_HOME

    class _UnresolvableRunner(_SplitRouteRunner):
        def _routed_profile_home(self, _profile_name):
            return UNRESOLVED_PROFILE_HOME

        def _canonicalize(self, source, *, transport_profile=None, primary_home=None):
            return None  # no live provenance: nothing pinned, sentinel passthrough

    runner = _UnresolvableRunner(UNRESOLVED_PROFILE_HOME, None)
    asyncio.run(inject_local_inbound(runner, _payload(message_id="b2-unres-1", text="/new")))
    assert runner.adapter.seen.allow_gateway_control is True
