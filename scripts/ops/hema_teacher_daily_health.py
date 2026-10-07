#!/usr/bin/env python3
"""Daily production health check for the hema-teacher DingTalk tutor.

Healthy runs are persisted only. Actionable failures create an idempotent
chief-engineer Kanban repair card. Low-risk self-heal is deliberately narrow:
restart the dedicated free-response bridge once when inactive, and proactively
/compress real tutoring sessions at or above the configured context threshold.

The script never blindly restarts the shared Gateway and never sends health
probe output into DingTalk. Gateway probes use trusted local inbound, whose
presentation is suppressed by gateway.run_local_inbound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from gateway.control_socket import inject_gateway_local_inbound, query_gateway_control

SCHEMA = "hermes.hema_teacher_daily_health.v1"
HOME = Path.home() / ".hermes"
REPO = HOME / "hermes-agent"
STATE_DIR = HOME / "state" / "hema-teacher-health"
BRIDGE_CONFIG = HOME / "learning" / "edu-agent" / "config" / "dingtalk-free-response-bridge.yaml"
GATEWAY_SERVICE = "hermes-gateway.service"
BRIDGE_SERVICE = "hema-dingtalk-free-response-bridge.service"
PROFILE = "hema-teacher"
HERMES_BIN = str(REPO / "venv" / "bin" / "hermes")
DEFAULT_CONTEXT_COMPRESS_PCT = 60
DEFAULT_LATENCY_WARN_MS = 12_000
DEFAULT_LATENCY_FAIL_MS = 25_000
#: Static prompt weight alone is not an incident. large_tool_schema only becomes
#: actionable when it accompanies latency regression or a significant jump vs
#: the historical baseline (weekly latency governor: NOOP without >=1s gain).
TOOL_SCHEMA_GROWTH_PCT = 25
ERROR_PATTERNS = re.compile(
    r"gateway injection unavailable|gateway injection rejected|"
    r"summary timed out|compression .*failed|Traceback|\bERROR\b|no progress|stalled",
    re.IGNORECASE,
)
#: Agent-level chatter that self-corrects inside the conversation and is not
#: service health: worker tool-call failures (tool_executor warnings, whose
#: payloads embed Python tracebacks) and auxiliary-client transient retries.
#: Infrastructure errors (injection rejections, compression failures, real
#: ERROR log records, stalled loops) stay counted by ERROR_PATTERNS untouched.
BENIGN_NOISE_PATTERNS = re.compile(
    r"agent\.tool_executor: tool \S+ (?:returned error|failed)"
    r"|auxiliary_client: .*transient transport error; retrying",
    re.IGNORECASE,
)


def error_hit_lines(text: str) -> list[str]:
    """Journal lines that signal a real service error, benign noise excluded."""
    return [
        line[-500:]
        for line in text.splitlines()
        if ERROR_PATTERNS.search(line) and not BENIGN_NOISE_PATTERNS.search(line)
    ]
_CONTEXT_RE = re.compile(
    r"(?P<used>\d[\d,]*)\s*/\s*(?P<total>\d[\d,]*)[^\n%]*"
    r"(?:\(|\s)(?:~)?(?P<pct>\d{1,3})%"
)


@dataclass
class Signal:
    severity: str
    code: str
    evidence: str
    self_healed: bool = False


def run(cmd: list[str], *, timeout: int = 30, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def service_state(service: str) -> dict[str, Any]:
    rc, out, err = run(["systemctl", "--user", "is-active", service], timeout=10)
    rc2, restarts, _ = run(
        ["systemctl", "--user", "show", service, "-p", "NRestarts", "--value"], timeout=10
    )
    return {
        "service": service,
        "active": rc == 0 and out == "active",
        "state": out or err or f"rc={rc}",
        "restart_count": int(restarts) if rc2 == 0 and restarts.isdigit() else None,
    }


def restart_bridge_once() -> tuple[bool, str]:
    rc, out, err = run(["systemctl", "--user", "restart", BRIDGE_SERVICE], timeout=30)
    if rc != 0:
        return False, err or out or f"rc={rc}"
    time.sleep(1.0)
    state = service_state(BRIDGE_SERVICE)
    return bool(state["active"]), state["state"]


def runtime_state(repo: Path = REPO, home: Path = HOME) -> dict[str, Any]:
    lifecycle = {}
    try:
        lifecycle = json.loads((home / "state" / "gateway.lifecycle.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    _, head, _ = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    _, origin, _ = run(["git", "-C", str(repo), "rev-parse", "origin/main"])
    disk = shutil.disk_usage(home)
    return {
        "gateway_phase": lifecycle.get("phase"),
        "gateway_pid": lifecycle.get("pid"),
        "runtime_sha": head or None,
        "origin_main_sha": origin or None,
        "disk_free_gb": round(disk.free / 1024**3, 2),
    }


def prompt_footprint() -> dict[str, Any]:
    rc, out, err = run(
        [HERMES_BIN, "-p", PROFILE, "prompt-size", "--platform", "dingtalk", "--json"],
        timeout=45,
        env={**os.environ, "HERMES_SESSION_PLATFORM": "dingtalk"},
    )
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


def _effective_log_since(
    service: str,
    minutes: int,
    *,
    now: datetime | None = None,
) -> str:
    """Start at the newer of the rolling window or the current service process.

    A repaired/restarted runtime must not stay yellow because errors from the previous
    process are still inside the rolling journal window.
    """
    local_now = (now or datetime.now()).replace(microsecond=0)
    window_start = local_now - timedelta(minutes=max(1, minutes))
    rc, out, _ = run(
        [
            "systemctl", "--user", "show", service,
            "-p", "ExecMainStartTimestamp", "--value",
        ],
        timeout=10,
    )
    if rc == 0 and out:
        match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", out)
        if match:
            try:
                process_start = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                window_start = max(window_start, process_start)
            except ValueError:
                pass
    return window_start.strftime("%Y-%m-%d %H:%M:%S")


def recent_error_summary(minutes: int = 90) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for service in (GATEWAY_SERVICE, BRIDGE_SERVICE):
        since = _effective_log_since(service, minutes)
        rc, out, err = run(
            [
                "journalctl", "--user", "-u", service,
                "--since", since, "--no-pager", "-n", "500",
            ],
            timeout=20,
        )
        text = out if rc == 0 else err
        hits = error_hit_lines(text)
        rows[service] = {
            "ok": rc == 0,
            "since": since,
            "error_hits": len(hits),
            "tail": hits[-8:],
        }
    return rows


def load_routes(config_path: Path = BRIDGE_CONFIG) -> list[dict[str, str]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profile = str(cfg.get("profile") or PROFILE)
    routes: list[dict[str, str]] = []
    for group in cfg.get("groups") or []:
        chat_id = str(group.get("chat_id") or "")
        skill = str(group.get("skill") or "")
        chat_name = str(group.get("name") or chat_id)
        for member in group.get("allowed_members") or []:
            user_id = str(member.get("open_dingtalk_id") or "")
            user_alt = str(member.get("hermes_session_user_id") or "")
            role = str(member.get("role") or "verified_member")
            if chat_id and user_id:
                routes.append({
                    "profile": profile,
                    "chat_id": chat_id,
                    "chat_name": chat_name,
                    "skill": skill,
                    "user_id": user_id,
                    "user_id_alt": user_alt,
                    "role": role,
                })
    return routes


def inject(route: dict[str, str], text: str, *, message_id: str, timeout: int = 190) -> dict[str, Any]:
    params = {
        "profile": route.get("profile") or PROFILE,
        "platform": "dingtalk",
        "chat_id": route["chat_id"],
        "chat_name": route.get("chat_name") or "healthcheck",
        "chat_type": "group",
        "user_id": route["user_id"],
        "user_id_alt": route.get("user_id_alt") or route["user_id"],
        "user_name": route.get("role") or "healthcheck",
        "message_id": message_id,
        "text": text,
        "skill": route.get("skill") or "huangshang-math-tutor",
        "wait_for_idle_seconds": 20,
        "timeout_seconds": min(timeout, 190),
    }
    result = inject_gateway_local_inbound(HOME, params, timeout=timeout + 5)
    return result if isinstance(result, dict) else {"accepted": False, "reason": "gateway_unavailable"}


def session_health(
    route: dict[str, str],
    *,
    action: str = "inspect",
    expected_message_prefix: str = "",
    expected_text: str = "/context",
    timeout: float | None = None,
) -> dict[str, Any]:
    params = {
        "profile": route.get("profile") or PROFILE,
        "platform": "dingtalk",
        "chat_id": route["chat_id"],
        "chat_name": route.get("chat_name") or "healthcheck",
        "chat_type": "group",
        "user_id": route["user_id"],
        "user_id_alt": route.get("user_id_alt") or route["user_id"],
        "user_name": route.get("role") or "healthcheck",
        "action": action,
    }
    if expected_message_prefix:
        params["expected_message_prefix"] = expected_message_prefix
        params["expected_text"] = expected_text
    result = query_gateway_control(
        HOME,
        "local-session-health",
        params=params,
        timeout=timeout if timeout is not None else (190.0 if action == "compress" else 25.0),
    )
    return result if isinstance(result, dict) else {
        "accepted": False,
        "reason": "session_health_control_unavailable",
    }


def check_and_compact_contexts(
    routes: list[dict[str, str]],
    *,
    threshold_pct: int = DEFAULT_CONTEXT_COMPRESS_PCT,
    allow_compaction: bool = True,
) -> tuple[list[dict[str, Any]], list[Signal]]:
    """Inspect existing tutoring sessions without adding a conversation turn."""
    results: list[dict[str, Any]] = []
    signals: list[Signal] = []
    for route in routes:
        route_key = short_hash(f"{route['chat_id']}|{route['user_id']}")
        before = session_health(route, action="inspect")
        row: dict[str, Any] = {
            "route": route_key,
            "accepted": bool(before.get("accepted")),
            "reason": before.get("reason"),
            "found": bool(before.get("found")),
            "session_id": (
                short_hash(str(before.get("session_id") or ""))
                if before.get("session_id") else None
            ),
            "busy": bool(before.get("busy")),
            "before": {
                "used": before.get("last_prompt_tokens"),
                "total": before.get("context_length"),
                "pct": before.get("context_pct"),
                "model": before.get("model"),
            } if before.get("found") else None,
        }
        if not before.get("accepted"):
            signals.append(
                Signal(
                    "high",
                    "context_probe_failed",
                    f"route={route_key} reason={before.get('reason')}",
                )
            )
            results.append(row)
            continue
        if not before.get("found"):
            results.append(row)
            continue
        if before.get("busy"):
            signals.append(
                Signal(
                    "info",
                    "context_probe_deferred_busy",
                    f"route={route_key} active lesson/session; no forced compaction",
                    self_healed=True,
                )
            )
            results.append(row)
            continue

        pct = before.get("context_pct")
        used = int(before.get("last_prompt_tokens") or 0)
        if used > 0 and pct is None:
            signals.append(
                Signal(
                    "medium",
                    "context_window_unknown",
                    f"route={route_key} used={used}",
                )
            )
            results.append(row)
            continue
        if not isinstance(pct, (int, float)) or pct < threshold_pct:
            results.append(row)
            continue

        row["compression_triggered"] = True
        if not allow_compaction:
            signals.append(
                Signal(
                    "medium",
                    "context_pressure",
                    f"route={route_key} pct={pct} threshold={threshold_pct}",
                )
            )
            results.append(row)
            continue

        comp = session_health(route, action="compress", timeout=190)
        row["compression_accepted"] = bool(comp.get("accepted"))
        row["compression_changed"] = bool(comp.get("changed"))
        row["compression_reason"] = comp.get("reason")
        row["after"] = {
            "used": comp.get("last_prompt_tokens"),
            "total": comp.get("context_length"),
            "pct": comp.get("context_pct"),
            "model": comp.get("model"),
        }
        after_pct = comp.get("context_pct")
        if (
            comp.get("accepted")
            and comp.get("compressed")
            and (
                after_pct is None
                or (isinstance(after_pct, (int, float)) and after_pct < threshold_pct)
            )
        ):
            signals.append(
                Signal(
                    "info",
                    "context_precompressed",
                    f"route={route_key} before={pct}% after={after_pct}",
                    self_healed=True,
                )
            )
        else:
            signals.append(
                Signal(
                    "high",
                    "proactive_compaction_unverified",
                    f"route={route_key} before={pct}% reason={comp.get('reason')} after={after_pct}",
                )
            )
        results.append(row)
    return results, signals


def latency_probe(
    *,
    warn_ms: int = DEFAULT_LATENCY_WARN_MS,
    fail_ms: int = DEFAULT_LATENCY_FAIL_MS,
) -> tuple[dict[str, Any], list[Signal]]:
    route = {
        "profile": PROFILE,
        "chat_id": "hema-teacher-healthcheck-synthetic",
        "chat_name": "河马老师健康检查",
        "skill": "huangshang-math-tutor",
        "user_id": "hema-healthcheck",
        "user_id_alt": "hema-healthcheck",
        "role": "healthcheck",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reset = inject(route, "/new", message_id=f"hema-health-reset-{stamp}", timeout=60)
    first = inject(
        route,
        "健康检查探针。只回复：OK",
        message_id=f"hema-health-latency-{stamp}-1",
        timeout=60,
    )
    attempts = [first]
    first_ms = int(first.get("elapsed_ms") or 999_999) if first.get("accepted") else 999_999
    if not first.get("accepted") or first_ms > warn_ms:
        inject(route, "/new", message_id=f"hema-health-reset-{stamp}-2", timeout=60)
        second = inject(
            route,
            "健康检查探针。只回复：OK",
            message_id=f"hema-health-latency-{stamp}-2",
            timeout=60,
        )
        attempts.append(second)

    accepted_ms = [
        int(x.get("elapsed_ms") or 0)
        for x in attempts
        if x.get("accepted") and str(x.get("response") or "").strip()
    ]
    result = {
        "reset_ok": bool(reset.get("accepted")),
        "attempts": [
            {
                "accepted": bool(x.get("accepted")),
                "reason": x.get("reason"),
                "elapsed_ms": x.get("elapsed_ms"),
                "nonempty_response": bool(str(x.get("response") or "").strip()),
            }
            for x in attempts
        ],
        "best_ms": min(accepted_ms) if accepted_ms else None,
        "worst_ms": max(accepted_ms) if accepted_ms else None,
    }
    signals: list[Signal] = []
    if not accepted_ms:
        signals.append(Signal("high", "latency_probe_failed", f"attempts={result['attempts']}"))
    elif min(accepted_ms) > fail_ms:
        signals.append(
            Signal("high", "response_latency_red", f"best={min(accepted_ms)}ms fail={fail_ms}ms")
        )
    elif min(accepted_ms) > warn_ms:
        signals.append(
            Signal("medium", "response_latency_yellow", f"best={min(accepted_ms)}ms warn={warn_ms}ms")
        )
    elif len(attempts) > 1 and first_ms > warn_ms:
        signals.append(
            Signal(
                "info",
                "response_latency_transient",
                f"first={first_ms}ms recovered={min(accepted_ms)}ms",
                True,
            )
        )
    return result, signals


def build_signals(
    services: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
    footprint: dict[str, Any],
    logs: dict[str, Any],
) -> list[Signal]:
    signals: list[Signal] = []
    gateway = services[GATEWAY_SERVICE]
    bridge = services[BRIDGE_SERVICE]
    if not gateway["active"] or runtime.get("gateway_phase") != "running":
        signals.append(
            Signal(
                "high",
                "gateway_not_healthy",
                f"service={gateway['state']} phase={runtime.get('gateway_phase')}",
            )
        )
    if not bridge["active"]:
        signals.append(Signal("high", "bridge_not_running", f"state={bridge['state']}"))
    if (
        runtime.get("runtime_sha")
        and runtime.get("origin_main_sha")
        and runtime["runtime_sha"] != runtime["origin_main_sha"]
    ):
        signals.append(
            Signal(
                "high",
                "runtime_sha_drift",
                f"runtime={runtime['runtime_sha']} main={runtime['origin_main_sha']}",
            )
        )
    if (runtime.get("disk_free_gb") or 0) < 10:
        signals.append(Signal("high", "low_disk_space", f"free={runtime.get('disk_free_gb')}GB"))
    if not footprint.get("ok"):
        signals.append(
            Signal("medium", "prompt_footprint_probe_failed", str(footprint.get("error")))
        )
    else:
        if (footprint.get("system_bytes") or 0) > 75_000:
            signals.append(Signal("medium", "large_system_prompt", f"{footprint['system_bytes']}B"))
        if (footprint.get("tool_bytes") or 0) > 65_000:
            # Static schema size alone is not a failure. Response latency is measured
            # independently below; keep this as an observation so a healthy fast tutor
            # does not generate a repair ticket just for having a rich tool surface.
            signals.append(
                Signal("info", "tool_schema_observation", f"{footprint['tool_bytes']}B")
            )
    for service, row in logs.items():
        if int(row.get("error_hits") or 0) >= 3:
            signals.append(
                Signal("medium", "repeated_recent_errors", f"{service} hits={row['error_hits']}")
            )
    return signals


def persist(package: dict[str, Any], state_dir: Path = STATE_DIR) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_dir / "latest.json.tmp"
    tmp.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(state_dir / "latest.json")
    with (state_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(package, ensure_ascii=False, separators=(",", ":")) + "\n")


def actionable(signals: list[Signal]) -> list[Signal]:
    return [
        signal
        for signal in signals
        if signal.severity in {"medium", "high"} and not signal.self_healed
    ]


def incident_body(package: dict[str, Any], state_path: Path) -> str:
    issues = actionable([Signal(**row) for row in package["signals"]])
    lines = [
        "# 河马老师每日体检异常维修单",
        "",
        f"- generated_at: {package['generated_at']}",
        f"- evidence: {state_path}",
        f"- overall: {package['overall']}",
        "",
        "## 当前异常",
    ]
    lines.extend(f"- [{signal.severity}] {signal.code}: {signal.evidence}" for signal in issues)
    lines += [
        "",
        "## 必须执行",
        "1. 先回读真实状态和最新日志，不允许因历史错误盲目重启。",
        "2. 针对异常做最小必要维修；L1-L2 可直接处理，L3+ 必须 branch→test→audit→PR→CI/review→exact-head merge→exact-main deploy。",
        "3. 不得通过关闭压缩、放宽安全阈值、禁用必要 Skill/工具来换取表面速度。",
        f"4. 维修后重新运行 {REPO}/scripts/ops/hema_teacher_daily_health.py --write-state --no-ticket。",
        "5. 只有复检 overall=green，且 Gateway/bridge 正常、响应探针通过、需要压缩的学习会话已完成预压缩，才允许关闭维修单。",
        "",
        "## 目标",
        "确保河马老师在实际做题时不临时触发大上下文压缩，并保持稳定、快速、可响应。",
    ]
    return "\n".join(lines) + "\n"


def dispatch_chief_engineer(
    package: dict[str, Any],
    state_dir: Path = STATE_DIR,
) -> dict[str, Any]:
    issues = actionable([Signal(**row) for row in package["signals"]])
    if not issues:
        return {"needed": False}

    state_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = short_hash("|".join(sorted(f"{x.code}:{x.evidence}" for x in issues)))
    day = datetime.now().astimezone().strftime("%Y-%m-%d")
    idempotency_key = f"hema-teacher-health-{day}-{fingerprint}"
    body_path = state_dir / "incident-body.md"
    body_path.write_text(
        incident_body(package, state_dir / "latest.json"),
        encoding="utf-8",
    )
    rc, out, err = run(
        [
            HERMES_BIN, "kanban", "create", f"[自动维修] 河马老师体检异常 {day}",
            "--body-file", str(body_path),
            "--assignee", "chief-engineer",
            "--idempotency-key", idempotency_key,
            "--max-runtime", "45m",
            "--max-retries", "2",
            "--goal",
            "--goal-max-turns", "12",
            "--initial-status", "blocked",
            "--created-by", "hema-healthcheck",
            "--json",
        ],
        timeout=30,
    )
    if rc != 0:
        return {"needed": True, "created": False, "error": err or out or f"rc={rc}"}
    try:
        task = json.loads(out)
        task_id = str(task.get("id") or task.get("task_id") or "")
        status = str(task.get("status") or "")
    except Exception as exc:
        return {
            "needed": True,
            "created": False,
            "error": f"parse create: {exc}",
            "raw": out[-500:],
        }

    released = None
    release_error = None
    if task_id and status == "blocked":
        rc2, out2, err2 = run(
            [
                HERMES_BIN, "kanban", "unblock", task_id,
                "--reason", "daily hema health incident dispatch",
            ],
            timeout=20,
        )
        released = rc2 == 0
        release_error = None if rc2 == 0 else (err2 or out2 or f"rc={rc2}")

    return {
        "needed": True,
        "created": True,
        "task_id": task_id,
        "initial_status": status,
        "released_to_dispatcher": released,
        "release_error": release_error,
        "idempotency_key": idempotency_key,
    }


def run_health(
    *,
    threshold_pct: int = DEFAULT_CONTEXT_COMPRESS_PCT,
    allow_compaction: bool = True,
    allow_bridge_restart: bool = True,
    do_latency_probe: bool = True,
) -> dict[str, Any]:
    services = {
        GATEWAY_SERVICE: service_state(GATEWAY_SERVICE),
        BRIDGE_SERVICE: service_state(BRIDGE_SERVICE),
    }
    signals: list[Signal] = []

    if not services[BRIDGE_SERVICE]["active"] and allow_bridge_restart:
        ok, detail = restart_bridge_once()
        services[BRIDGE_SERVICE] = service_state(BRIDGE_SERVICE)
        if ok:
            signals.append(Signal("info", "bridge_auto_restarted", detail, self_healed=True))
            signals.append(
                Signal(
                    "medium",
                    "bridge_was_down",
                    "bridge auto-recovered; chief engineer must inspect root cause",
                )
            )
        else:
            signals.append(Signal("high", "bridge_restart_failed", detail))

    runtime = runtime_state()
    footprint = prompt_footprint()
    logs = recent_error_summary()
    signals.extend(build_signals(services, runtime, footprint, logs))

    contexts: list[dict[str, Any]] = []
    latency: dict[str, Any] = {"skipped": True}
    if services[GATEWAY_SERVICE]["active"] and runtime.get("gateway_phase") == "running":
        try:
            contexts, context_signals = check_and_compact_contexts(
                load_routes(),
                threshold_pct=threshold_pct,
                allow_compaction=allow_compaction,
            )
            signals.extend(context_signals)
        except Exception as exc:
            signals.append(
                Signal("high", "context_health_exception", f"{type(exc).__name__}: {exc}")
            )
        if do_latency_probe:
            try:
                latency, latency_signals = latency_probe()
                signals.extend(latency_signals)
            except Exception as exc:
                signals.append(
                    Signal("high", "latency_health_exception", f"{type(exc).__name__}: {exc}")
                )

    active = actionable(signals)
    overall = (
        "red"
        if any(signal.severity == "high" for signal in active)
        else "yellow"
        if active
        else "green"
    )
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "decision_owner": "chief-engineer",
        "services": services,
        "runtime": runtime,
        "prompt_footprint": footprint,
        "recent_errors": logs,
        "contexts": contexts,
        "latency": latency,
        "signals": [asdict(signal) for signal in signals],
        "policy": {
            "context_compress_pct": threshold_pct,
            "latency_warn_ms": DEFAULT_LATENCY_WARN_MS,
            "latency_fail_ms": DEFAULT_LATENCY_FAIL_MS,
            "shared_gateway_auto_restart": False,
            "bridge_single_restart": allow_bridge_restart,
            "healthy_report_policy": "persist_only",
            "actionable_report_policy": "chief-engineer-kanban",
        },
    }


def render(package: dict[str, Any]) -> str:
    issues = actionable([Signal(**row) for row in package["signals"]])
    latency = package.get("latency") or {}
    return (
        f"[HEMA-HEALTH][{package['overall'].upper()}] "
        f"gateway={'ok' if package['services'][GATEWAY_SERVICE]['active'] else 'down'} "
        f"bridge={'ok' if package['services'][BRIDGE_SERVICE]['active'] else 'down'} "
        f"latency_best={latency.get('best_ms')}ms "
        f"contexts={len(package.get('contexts') or [])} "
        f"actionable={len(issues)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-state", action="store_true")
    parser.add_argument("--no-ticket", action="store_true")
    parser.add_argument("--no-compaction", action="store_true")
    parser.add_argument("--no-bridge-restart", action="store_true")
    parser.add_argument("--no-latency-probe", action="store_true")
    parser.add_argument(
        "--context-compress-pct",
        type=int,
        default=DEFAULT_CONTEXT_COMPRESS_PCT,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    threshold = min(90, max(30, int(args.context_compress_pct)))
    package = run_health(
        threshold_pct=threshold,
        allow_compaction=not args.no_compaction,
        allow_bridge_restart=not args.no_bridge_restart,
        do_latency_probe=not args.no_latency_probe,
    )
    if args.write_state:
        persist(package)

    ticket = {"needed": False, "skipped": bool(args.no_ticket)}
    if not args.no_ticket and actionable([Signal(**row) for row in package["signals"]]):
        if not args.write_state:
            persist(package)
        ticket = dispatch_chief_engineer(package)
        package["chief_engineer_ticket"] = ticket
        persist(package)

    if args.json:
        print(json.dumps(package, ensure_ascii=False, indent=2))
    else:
        print(render(package))
        if ticket.get("needed"):
            print(f"chief_engineer_task={ticket.get('task_id') or 'create_failed'}")
    return 0 if package["overall"] == "green" else 2


if __name__ == "__main__":
    raise SystemExit(main())
