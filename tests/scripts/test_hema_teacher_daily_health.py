from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "scripts" / "ops" / "hema_teacher_daily_health.py"
SPEC = importlib.util.spec_from_file_location("hema_teacher_daily_health", MODULE_PATH)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_build_signals_flags_infra_drift_disk_and_prompt():
    services = {
        mod.GATEWAY_SERVICE: {"active": False, "state": "inactive"},
        mod.BRIDGE_SERVICE: {"active": True, "state": "active"},
    }
    runtime = {
        "gateway_phase": "stopped",
        "runtime_sha": "aaa",
        "origin_main_sha": "bbb",
        "disk_free_gb": 8.5,
    }
    footprint = {"ok": True, "system_bytes": 80001, "tool_bytes": 65001}
    logs = {
        mod.GATEWAY_SERVICE: {"error_hits": 4},
        mod.BRIDGE_SERVICE: {"error_hits": 0},
    }
    codes = {s.code for s in mod.build_signals(services, runtime, footprint, logs)}
    assert {
        "gateway_not_healthy",
        "runtime_sha_drift",
        "low_disk_space",
        "large_system_prompt",
        "repeated_recent_errors",
    } <= codes
    observations = [x for x in mod.build_signals(services, runtime, footprint, logs) if x.code == "tool_schema_observation"]
    assert len(observations) == 1
    assert observations[0].severity == "info"


def test_effective_log_since_ignores_previous_runtime_errors(monkeypatch):
    def fake_run(cmd, **_kwargs):
        assert cmd[:4] == ["systemctl", "--user", "show", mod.BRIDGE_SERVICE]
        return 0, "Tue 2026-10-06 17:16:10 CST", ""

    monkeypatch.setattr(mod, "run", fake_run)
    since = mod._effective_log_since(
        mod.BRIDGE_SERVICE,
        90,
        now=mod.datetime(2026, 10, 6, 17, 45, 0),
    )
    assert since == "2026-10-06 17:16:10"


def test_effective_log_since_keeps_rolling_window_for_old_process(monkeypatch):
    monkeypatch.setattr(
        mod,
        "run",
        lambda *_a, **_k: (0, "Tue 2026-10-06 10:00:00 CST", ""),
    )
    since = mod._effective_log_since(
        mod.BRIDGE_SERVICE,
        90,
        now=mod.datetime(2026, 10, 6, 17, 45, 0),
    )
    assert since == "2026-10-06 16:15:00"


def test_actionable_ignores_info_and_self_healed():
    signals = [
        mod.Signal("info", "context_precompressed", "ok", True),
        mod.Signal("high", "bridge_auto_restarted", "ok", True),
        mod.Signal("medium", "response_latency_yellow", "slow"),
    ]
    assert [x.code for x in mod.actionable(signals)] == ["response_latency_yellow"]


def test_context_pressure_triggers_control_compress_and_verifies(monkeypatch):
    route = {
        "profile": "hema-teacher",
        "chat_id": "c",
        "chat_name": "g",
        "skill": "huangshang-math-tutor",
        "user_id": "u",
        "user_id_alt": "u",
        "role": "child",
    }
    replies = iter([
        {
            "accepted": True,
            "found": True,
            "session_id": "s",
            "last_prompt_tokens": 80000,
            "context_length": 100000,
            "context_pct": 80.0,
            "model": "qwen",
        },
        {
            "accepted": True,
            "found": True,
            "session_id": "s",
            "compressed": True,
            "changed": True,
            "last_prompt_tokens": 20000,
            "context_length": 100000,
            "context_pct": 20.0,
            "model": "qwen",
        },
    ])
    actions = []

    def fake(_route, *, action="inspect", **_kwargs):
        actions.append(action)
        return next(replies)

    monkeypatch.setattr(mod, "session_health", fake)
    rows, signals = mod.check_and_compact_contexts([route], threshold_pct=60)
    assert actions == ["inspect", "compress"]
    assert rows[0]["compression_triggered"] is True
    assert rows[0]["after"]["pct"] == 20.0
    assert any(x.code == "context_precompressed" and x.self_healed for x in signals)


def test_context_below_threshold_does_not_compress(monkeypatch):
    route = {
        "profile": "hema-teacher",
        "chat_id": "c",
        "chat_name": "g",
        "skill": "huangshang-math-tutor",
        "user_id": "u",
        "user_id_alt": "u",
        "role": "child",
    }
    actions = []

    def fake(_route, *, action="inspect", **_kwargs):
        actions.append(action)
        return {
            "accepted": True,
            "found": True,
            "session_id": "s",
            "last_prompt_tokens": 30000,
            "context_length": 100000,
            "context_pct": 30.0,
            "model": "qwen",
        }

    monkeypatch.setattr(mod, "session_health", fake)
    rows, signals = mod.check_and_compact_contexts([route], threshold_pct=60)
    assert actions == ["inspect"]
    assert rows[0]["before"]["pct"] == 30.0
    assert signals == []


def test_busy_session_is_deferred_without_actionable_incident(monkeypatch):
    route = {
        "profile": "hema-teacher",
        "chat_id": "c",
        "chat_name": "g",
        "skill": "huangshang-math-tutor",
        "user_id": "u",
        "user_id_alt": "u",
        "role": "child",
    }
    actions = []

    def fake(_route, *, action="inspect", **_kwargs):
        actions.append(action)
        return {
            "accepted": True,
            "found": True,
            "busy": True,
            "session_id": "s",
            "last_prompt_tokens": 80000,
            "context_length": 100000,
            "context_pct": 80.0,
            "model": "qwen",
        }

    monkeypatch.setattr(mod, "session_health", fake)
    _rows, signals = mod.check_and_compact_contexts([route], threshold_pct=60)
    assert actions == ["inspect"]
    assert [x.code for x in signals] == ["context_probe_deferred_busy"]
    assert signals[0].self_healed is True
    assert mod.actionable(signals) == []


def test_unknown_context_window_is_actionable(monkeypatch):
    route = {
        "profile": "hema-teacher",
        "chat_id": "c",
        "chat_name": "g",
        "skill": "huangshang-math-tutor",
        "user_id": "u",
        "user_id_alt": "u",
        "role": "child",
    }
    monkeypatch.setattr(
        mod,
        "session_health",
        lambda *_a, **_k: {
            "accepted": True,
            "found": True,
            "session_id": "s",
            "last_prompt_tokens": 50000,
            "context_length": 0,
            "context_pct": None,
            "model": "qwen",
        },
    )
    _rows, signals = mod.check_and_compact_contexts([route], threshold_pct=60)
    assert [x.code for x in signals] == ["context_window_unknown"]


def test_incident_body_requires_post_repair_green(tmp_path):
    package = {
        "generated_at": "now",
        "overall": "red",
        "signals": [
            {
                "severity": "high",
                "code": "gateway_not_healthy",
                "evidence": "down",
                "self_healed": False,
            }
        ],
    }
    body = mod.incident_body(package, tmp_path / "latest.json")
    assert "branch→test→audit→PR→CI/review→exact-head merge→exact-main deploy" in body
    assert "overall=green" in body
    assert "--no-ticket" in body


def test_persist_writes_latest_and_history(tmp_path):
    package = {"schema": mod.SCHEMA, "generated_at": "now"}
    mod.persist(package, tmp_path)
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "history.jsonl").exists()
    assert mod.SCHEMA in (tmp_path / "history.jsonl").read_text()
