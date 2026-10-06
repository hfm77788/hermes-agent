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


def test_parse_context_accepts_locale_independent_numeric_shape():
    out = mod.parse_context("Context\nIn use: ~76,800 / 128,000 (~60%)\n")
    assert out == {"used": 76800, "total": 128000, "pct": 60}


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
        "large_tool_schema",
        "repeated_recent_errors",
    } <= codes


def test_actionable_ignores_info_and_self_healed():
    signals = [
        mod.Signal("info", "context_precompressed", "ok", True),
        mod.Signal("high", "bridge_auto_restarted", "ok", True),
        mod.Signal("medium", "response_latency_yellow", "slow"),
    ]
    assert [x.code for x in mod.actionable(signals)] == ["response_latency_yellow"]


def test_context_pressure_triggers_compress_and_verifies(monkeypatch):
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
            "response": "x 80,000 / 100,000 (80%)",
            "elapsed_ms": 5,
            "session_id": "s",
        },
        {
            "accepted": True,
            "response": "compressed",
            "elapsed_ms": 20,
            "session_id": "s",
        },
        {
            "accepted": True,
            "response": "x 20,000 / 100,000 (20%)",
            "elapsed_ms": 4,
            "session_id": "s",
        },
    ])
    monkeypatch.setattr(mod, "inject", lambda *a, **k: next(replies))
    rows, signals = mod.check_and_compact_contexts([route], threshold_pct=60)
    assert rows[0]["compression_triggered"] is True
    assert rows[0]["after"]["pct"] == 20
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
    calls = []

    def fake(*args, **kwargs):
        calls.append(args[1])
        return {
            "accepted": True,
            "response": "x 30,000 / 100,000 (30%)",
            "elapsed_ms": 4,
            "session_id": "s",
        }

    monkeypatch.setattr(mod, "inject", fake)
    rows, signals = mod.check_and_compact_contexts([route], threshold_pct=60)
    assert calls == ["/context"]
    assert rows[0]["before"]["pct"] == 30
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
    monkeypatch.setattr(
        mod,
        "inject",
        lambda *a, **k: {"accepted": False, "reason": "session_busy"},
    )
    rows, signals = mod.check_and_compact_contexts([route], threshold_pct=60)
    assert rows[0]["reason"] == "session_busy"
    assert [x.code for x in signals] == ["context_probe_deferred_busy"]
    assert signals[0].self_healed is True
    assert mod.actionable(signals) == []


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
