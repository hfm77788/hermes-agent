from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "ops" / "weekly_latency_governor.py"
SPEC = importlib.util.spec_from_file_location("weekly_latency_governor", MODULE_PATH)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(mod)


def test_summarize_speed_uses_last_record_per_day():
    rows = [
        {"date": "2026-09-20", "median": 30, "p90": 70, "status": "red"},
        {"date": "2026-09-20", "median": 20, "p90": 60, "status": "green"},
        {"date": "2026-09-21", "median": 10, "p90": 40, "status": "green"},
    ]
    out = mod.summarize_speed(rows, days=7)
    assert out["days"] == 2
    assert out["latest"]["date"] == "2026-09-21"
    assert out["median_daily_median_s"] == 15.0
    assert out["median_daily_p90_s"] == 50.0
    assert out["median_change_pct"] == -50.0
    assert out["status_counts"] == {"green": 2}


def test_context_tuner_counts_only_real_targets():
    rows = [{"ts": 1, "recs": [
        {"seat": "default", "target": 60000},
        {"seat": "chief-engineer", "target": None},
        {"seat": "office-secretary", "target": 0},
    ]}]
    out = mod.latest_context_tuner(rows)
    assert out["available"] is True
    assert out["move_candidates"] == 1
def test_build_signals_flags_runtime_speed_and_prompt_pressure():
    package = {
        "runtime": {
            "gateway_phase": "stopped",
            "head_sha": "aaa",
            "origin_main_sha": "bbb",
        },
        "speed": {
            "latest": {"status": "red", "median": 45, "p90": 130},
            "median_change_pct": 25,
        },
        "context_tuner": {"move_candidates": 1},
        "prompt_footprint": {
            "default": {"ok": True, "system_bytes": 80000, "tool_bytes": 45000},
            "hema-teacher": {"ok": True, "system_bytes": 50000, "tool_bytes": 70000},
            "broken": {"ok": False, "error": "probe failed"},
        },
        "previous_compare": {
            "profiles": {"default": {"skills_bytes_pct": 20, "system_bytes_pct": 0, "tool_bytes_pct": 0}}
        },
    }
    codes = [x["code"] for x in mod.build_signals(package)]
    assert "gateway_not_running" in codes
    assert "runtime_sha_drift" in codes
    assert "speed_monitor_red" in codes
    assert "median_regression" in codes
    assert "context_tuner_candidates" in codes
    assert "large_system_prompt" in codes
    assert "large_tool_schema" in codes
    assert "prompt_probe_failed" in codes
    assert "prompt_footprint_regression" in codes


def test_compare_previous_reports_profile_deltas():
    current = {"prompt_footprint": {
        "default": {"ok": True, "skills_bytes": 120, "system_bytes": 220, "tool_bytes": 300},
    }}
    previous = {"generated_at": "old", "prompt_footprint": {
        "default": {"ok": True, "skills_bytes": 100, "system_bytes": 200, "tool_bytes": 300},
    }}
    out = mod.compare_previous(current, previous)
    assert out["available"] is True
    assert out["profiles"]["default"]["skills_bytes_pct"] == 20.0
    assert out["profiles"]["default"]["system_bytes_pct"] == 10.0
    assert out["profiles"]["default"]["tool_bytes_pct"] == 0.0
def test_persist_writes_latest_and_history(tmp_path):
    package = {"schema": mod.SCHEMA, "generated_at": "now"}
    mod.persist(package, tmp_path)
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "history.jsonl").exists()
    assert '"schema":"hermes.weekly_latency_governor.v1"' in (tmp_path / "history.jsonl").read_text()


def test_no_signals_is_noop_candidate_shape():
    package = {
        "runtime": {"gateway_phase": "running", "head_sha": "aaa", "origin_main_sha": "aaa"},
        "speed": {"latest": {"status": "green"}, "median_change_pct": -10},
        "context_tuner": {"move_candidates": 0},
        "prompt_footprint": {"default": {"ok": True, "system_bytes": 50000, "tool_bytes": 45000}},
        "previous_compare": {"profiles": {}},
    }
    assert mod.build_signals(package) == []
