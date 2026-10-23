"""Fork-local control-socket verbs must register on the LIVE construction site.

v0.22.4 re-anchor moved the control-socket build into ``gateway.run_bootstrap``
(``_start_gateway_start_control_socket``); the old ``gateway.run`` copy became a
facade-only duplicate. When the fork registrations (``local-session-health``,
``inject-local-inbound``) stayed on the dead copy, the running gateway answered
"unknown verb" — every DingTalk-bridge injection and the daily hema-teacher
health probes degraded to gateway_unavailable (repair card t_b2316993, 2026-10-23).

This test drives the real bootstrap function with a capture stub and asserts the
fork verbs are present, so the next upstream reshuffle of the construction site
fails loudly here instead of silently in production.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

from gateway import control_socket
from gateway.run_bootstrap import _start_gateway_start_control_socket

FORK_VERBS = {"local-session-health", "inject-local-inbound"}


def _capture_server(monkeypatch):
    captured: dict[str, set[str]] = {}

    class FakeServer:
        def __init__(self, home=None, verb_handlers=None):
            captured["verbs"] = set(verb_handlers or {})

        ticket_store = None

        async def start(self):
            return True

        def cleanup_files(self):
            pass

    monkeypatch.setattr(control_socket, "GatewayControlServer", FakeServer)
    return captured


def _stub_runner():
    runner = MagicMock()
    runner.session_runtime_descriptor = {}
    return runner


def test_bootstrap_registers_fork_local_verbs(monkeypatch):
    captured = _capture_server(monkeypatch)

    async def scenario():
        return await _start_gateway_start_control_socket(_stub_runner())

    assert asyncio.run(scenario()) is not None
    assert FORK_VERBS <= captured["verbs"], (
        f"fork-local verbs missing from the live control-socket build: "
        f"{sorted(FORK_VERBS - captured['verbs'])}"
    )


def test_both_construction_sites_keep_fork_registrations():
    """Guard against a future move of the construction site (audit invariant).

    Both ``gateway.run`` (legacy copy) and ``gateway.run_bootstrap`` (live) must
    keep registering the fork verbs; whichever module the startup path binds to,
    the socket answers.
    """
    from gateway import run as run_module

    sources = {
        "gateway.run_bootstrap": inspect.getsource(_start_gateway_start_control_socket),
        "gateway.run": inspect.getsource(run_module._start_gateway_start_control_socket),
    }
    for name, src in sources.items():
        for verb in FORK_VERBS:
            assert f'"{verb}"' in src, f"{name} lost the fork-local {verb!r} registration"
