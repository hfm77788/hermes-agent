#!/usr/bin/env python3
"""Build a weekly, read-only evidence package for Hermes latency governance.

The script never changes profile config, restarts services, merges code, or sends
messages.  It combines the existing daily speed monitor, weekly context tuner,
current prompt footprints, Fast Lane configuration, and runtime SHA into one
package for the chief-engineer seat to judge.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "hermes.weekly_latency_governor.v1"
DEFAULT_HOME = Path.home() / ".hermes"
DEFAULT_REPO = DEFAULT_HOME / "hermes-agent"
DEFAULT_STATE = DEFAULT_HOME / "state" / "weekly-latency-governor"
PROFILE_PLATFORMS = {
    "default": "feishu",
    "chief-engineer": "feishu",
    "hema-teacher": "dingtalk",
    "office-director": "feishu",
    "office-secretary": "feishu",
    "office-runner": "feishu",
    "office-cashier": "feishu",
    "office-archivist": "feishu",
    "leqi": "dingtalk",
}
def _run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                out.append(value)
        except Exception:
            continue
    return out


def _last_per_day(rows: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    by_day: dict[str, dict[str, Any]] = {}
    for row in rows:
        day = str(row.get("date") or "")
        if day:
            by_day[day] = row
    return [by_day[k] for k in sorted(by_day)[-days:]]
def summarize_speed(rows: list[dict[str, Any]], days: int = 7) -> dict[str, Any]:
    recent = _last_per_day(rows, days)
    if not recent:
        return {"days": 0, "latest": None, "status_counts": {}, "median_daily_median_s": None,
                "median_daily_p90_s": None, "median_change_pct": None}
    medians = [float(r["median"]) for r in recent if isinstance(r.get("median"), (int, float))]
    p90s = [float(r["p90"]) for r in recent if isinstance(r.get("p90"), (int, float))]
    change = None
    if len(medians) >= 2 and medians[0]:
        change = round((medians[-1] / medians[0] - 1.0) * 100.0, 1)
    return {
        "days": len(recent),
        "latest": recent[-1],
        "status_counts": dict(Counter(str(r.get("status") or "unknown") for r in recent)),
        "median_daily_median_s": round(statistics.median(medians), 1) if medians else None,
        "median_daily_p90_s": round(statistics.median(p90s), 1) if p90s else None,
        "median_change_pct": change,
        "history": recent,
    }


def latest_context_tuner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "move_candidates": 0, "recs": []}
    latest = rows[-1]
    recs = latest.get("recs") if isinstance(latest.get("recs"), list) else []
    return {
        "available": True,
        "ts": latest.get("ts"),
        "move_candidates": sum(1 for r in recs if isinstance(r, dict) and r.get("target")),
        "recs": recs,
    }


def _profile_home(home: Path, profile: str) -> Path:
    if profile == "default":
        return home
    nest = home.parent / ".hermes-nest" / profile
    if (nest / "config.yaml").exists():
        return nest
    return home / "profiles" / profile
def _read_fast_lane(home: Path, profile: str) -> dict[str, Any] | None:
    cfg = _profile_home(home, profile) / "config.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        value = ((raw.get("gateway") or {}).get("fast_lane") or {})
        return value if isinstance(value, dict) else None
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def prompt_footprint(profile: str, platform: str, *, repo: Path) -> dict[str, Any]:
    cmd = ["hermes"]
    if profile != "default":
        cmd += ["-p", profile]
    cmd += ["prompt-size", "--platform", platform, "--json"]
    env = os.environ.copy()
    env["HERMES_SESSION_PLATFORM"] = platform
    rc, out, err = _run(cmd, env=env, timeout=35)
    if rc != 0:
        return {"ok": False, "error": err or out or f"rc={rc}"}
    try:
        data = json.loads(out)
        return {
            "ok": True,
            "model": data.get("model"),
            "skills_count": len(data.get("skills_breakdown") or []),
            "skills_bytes": (data.get("skills_index") or {}).get("bytes"),
            "system_bytes": (data.get("system_prompt") or {}).get("bytes"),
            "tool_count": (data.get("tools") or {}).get("count"),
            "tool_bytes": (data.get("tools") or {}).get("json_bytes"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"json parse: {exc}"}


def runtime_state(home: Path, repo: Path) -> dict[str, Any]:
    life = _load_json(home / "state" / "gateway.lifecycle.json") or {}
    rc1, head, _ = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    rc2, origin, _ = _run(["git", "-C", str(repo), "rev-parse", "origin/main"])
    return {
        "gateway_phase": life.get("phase"),
        "gateway_pid": life.get("pid"),
        "gateway_started_at": life.get("started_at"),
        "head_sha": head if rc1 == 0 else None,
        "origin_main_sha": origin if rc2 == 0 else None,
    }
def _pct_delta(current: int | float | None, previous: int | float | None) -> float | None:
    if not isinstance(current, (int, float)) or not isinstance(previous, (int, float)) or previous == 0:
        return None
    return round((current / previous - 1.0) * 100.0, 1)


def compare_previous(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    if not previous:
        return {"available": False, "profiles": {}}
    prev_fp = previous.get("prompt_footprint") or {}
    deltas: dict[str, Any] = {}
    for profile, now in (current.get("prompt_footprint") or {}).items():
        before = prev_fp.get(profile) or {}
        if not now.get("ok") or not before.get("ok"):
            continue
        deltas[profile] = {
            "skills_bytes_pct": _pct_delta(now.get("skills_bytes"), before.get("skills_bytes")),
            "system_bytes_pct": _pct_delta(now.get("system_bytes"), before.get("system_bytes")),
            "tool_bytes_pct": _pct_delta(now.get("tool_bytes"), before.get("tool_bytes")),
        }
    return {"available": True, "generated_at": previous.get("generated_at"), "profiles": deltas}


def build_signals(package: dict[str, Any]) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    rt = package["runtime"]
    if rt.get("gateway_phase") != "running":
        signals.append({"severity": "high", "code": "gateway_not_running",
                        "evidence": f"phase={rt.get('gateway_phase')}"})
    if rt.get("head_sha") and rt.get("origin_main_sha") and rt["head_sha"] != rt["origin_main_sha"]:
        signals.append({"severity": "high", "code": "runtime_sha_drift",
                        "evidence": f"head={rt['head_sha']} origin={rt['origin_main_sha']}"})
    speed = package["speed"]
    latest = speed.get("latest") or {}
    if latest.get("status") == "red":
        signals.append({"severity": "high", "code": "speed_monitor_red",
                        "evidence": f"median={latest.get('median')} p90={latest.get('p90')}"})
    change = speed.get("median_change_pct")
    if isinstance(change, (int, float)) and change >= 20:
        signals.append({"severity": "medium", "code": "median_regression",
                        "evidence": f"7d first→latest +{change}%"})
    if package["context_tuner"].get("move_candidates", 0):
        signals.append({"severity": "medium", "code": "context_tuner_candidates",
                        "evidence": f"count={package['context_tuner']['move_candidates']}"})
    for profile, fp in package["prompt_footprint"].items():
        if not fp.get("ok"):
            signals.append({"severity": "medium", "code": "prompt_probe_failed",
                            "evidence": f"{profile}: {fp.get('error')}"})
            continue
        if (fp.get("system_bytes") or 0) > 75_000:
            signals.append({"severity": "medium", "code": "large_system_prompt",
                            "evidence": f"{profile}: {fp['system_bytes']}B"})
        if (fp.get("tool_bytes") or 0) > 65_000:
            signals.append({"severity": "medium", "code": "large_tool_schema",
                            "evidence": f"{profile}: {fp['tool_bytes']}B"})
    for profile, delta in (package.get("previous_compare") or {}).get("profiles", {}).items():
        for field in ("skills_bytes_pct", "system_bytes_pct", "tool_bytes_pct"):
            value = delta.get(field)
            if isinstance(value, (int, float)) and value >= 15:
                signals.append({"severity": "medium", "code": "prompt_footprint_regression",
                                "evidence": f"{profile} {field}=+{value}%"})
    return signals


def build_package(home: Path, repo: Path, state_dir: Path, days: int) -> dict[str, Any]:
    speed_rows = _load_jsonl(home / "state" / "speed-monitor" / "history.jsonl")
    ctx_rows = _load_jsonl(home / "state" / "ctx-tuner" / "history.jsonl")
    footprint: dict[str, Any] = {}
    fast_lane: dict[str, Any] = {}
    for profile, platform in PROFILE_PLATFORMS.items():
        footprint[profile] = prompt_footprint(profile, platform, repo=repo)
        fast_lane[profile] = _read_fast_lane(home, profile)
    package: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "state_path": str(state_dir / "latest.json"),
        "runtime": runtime_state(home, repo),
        "speed": summarize_speed(speed_rows, days),
        "context_tuner": latest_context_tuner(ctx_rows),
        "prompt_footprint": footprint,
        "fast_lane": fast_lane,
    }
    previous = _load_json(state_dir / "latest.json")
    package["previous_compare"] = compare_previous(package, previous)
    package["signals"] = build_signals(package)
    package["decision_owner"] = "chief-engineer"
    package["decision_hint"] = "REVIEW" if package["signals"] else "NOOP_CANDIDATE"
    package["guardrails"] = [
        "Evidence package is read-only; the chief-engineer owns the decision.",
        "Do not trade away factual verification, safety, privacy, or role-critical capabilities for speed.",
        "Config-only reversible changes require backup + semantic verification + post-change measurement.",
        "Code/runtime changes require branch→test→audit→PR→CI/review→exact-SHA merge→exact-main deploy→business smoke.",
        "If expected benefit is <1 second per turn or evidence is weak, prefer NOOP.",
        "Do not disable skills/toolsets from one-week footprint alone; require role-usage evidence and rollback path.",
    ]
    return package


def persist(package: dict[str, Any], state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_dir / "latest.json.tmp"
    tmp.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(state_dir / "latest.json")
    with (state_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(package, ensure_ascii=False, separators=(",", ":")) + "\n")


def render_text(package: dict[str, Any]) -> str:
    speed = package["speed"]
    latest = speed.get("latest") or {}
    fp = package["prompt_footprint"]
    biggest = sorted(
        ((name, row.get("system_bytes") or 0, row.get("tool_bytes") or 0) for name, row in fp.items() if row.get("ok")),
        key=lambda x: x[1] + x[2], reverse=True,
    )[:3]
    top = ", ".join(f"{n}:sys{s//1000}K/tool{t//1000}K" for n, s, t in biggest) or "n/a"
    return (
        f"[OPS/weekly-latency-governor][{package['decision_hint']}] "
        f"7d median={speed.get('median_daily_median_s')}s p90={speed.get('median_daily_p90_s')}s "
        f"latest={latest.get('status')} ctx_candidates={package['context_tuner'].get('move_candidates')} "
        f"signals={len(package['signals'])}\n"
        f"runtime={package['runtime'].get('head_sha')} gateway={package['runtime'].get('gateway_phase')}\n"
        f"largest_prompt={top}\n"
        f"evidence={package.get('state_path')}"
    )
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-state", action="store_true")
    args = parser.parse_args()
    package = build_package(args.home, args.repo, args.state_dir, max(1, args.days))
    if args.write_state:
        persist(package, args.state_dir)
    if args.json:
        print(json.dumps(package, ensure_ascii=False, indent=2))
    else:
        print(render_text(package))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
