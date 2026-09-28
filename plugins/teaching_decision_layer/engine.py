from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

SCHEMA = "hermes.teaching_decision.v1"
STATUS_SCHEMA = "hermes.teaching_decision_status.v1"

PERCEPTION_CHECKS = (
    "is_student_handwriting",
    "is_original_content",
    "is_teacher_annotation",
    "ocr_uncertain",
    "answer_region_detected",
    "answer_complete",
)
PEDAGOGY_CHECKS = (
    "understanding_demonstrated",
    "guessing_likely",
    "needs_followup_question",
    "ready_to_advance",
    "learning_objective_met",
    "response_too_entertaining",
)

PERCEPTION_YES = 0.90
PERCEPTION_NO = 0.10
PEDAGOGY_YES = 0.80
PEDAGOGY_NO = 0.20
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 1.5
MAX_EVIDENCE_BYTES = 12_000
MAX_AUDIT_ROWS = 10_000

_QUESTIONS = {
    "is_student_handwriting": (
        "Based only on the structured visual evidence for the target region, "
        "is this region student handwriting rather than printed source content or teacher annotation?"
    ),
    "is_original_content": (
        "Based only on the structured visual evidence for the target region, "
        "is this region original printed/source question content rather than later-added handwriting or annotation?"
    ),
    "is_teacher_annotation": (
        "Based only on the structured visual evidence for the target region, "
        "is this region teacher annotation such as marking, correction, score, tick, cross, or comment?"
    ),
    "ocr_uncertain": (
        "Based only on the OCR/visual evidence, is the recognized text or symbol materially uncertain enough "
        "that another targeted visual check is needed before teaching from it?"
    ),
    "answer_region_detected": (
        "Based only on the visual layout evidence, has the student's actual answer region been located reliably?"
    ),
    "answer_complete": (
        "Based only on the visual evidence, is the student's answer capture complete rather than cropped, hidden, "
        "overwritten, or partially unreadable?"
    ),
    "understanding_demonstrated": (
        "Based only on the latest learning interaction evidence, has the learner demonstrated actual understanding "
        "of the target knowledge rather than merely giving a final answer?"
    ),
    "guessing_likely": (
        "Based only on the latest learning interaction evidence, is the learner likely guessing or pattern-matching "
        "without demonstrating the underlying reasoning?"
    ),
    "needs_followup_question": (
        "Based only on the latest learning interaction evidence, should the teacher ask one short follow-up question "
        "to verify understanding before advancing?"
    ),
    "ready_to_advance": (
        "Based only on the latest learning interaction evidence, is the learner ready to advance to the next problem "
        "or knowledge point without additional explanation or verification?"
    ),
    "learning_objective_met": (
        "Based only on the current session evidence, has the intended learning objective for this interaction been met?"
    ),
    "response_too_entertaining": (
        "Based only on the current interaction evidence, is the teaching response or dialogue becoming entertaining "
        "without enough progress toward the explicit learning objective?"
    ),
}

_SENSITIVE_KEYS = {
    "name",
    "student_name",
    "learner_name",
    "child_name",
    "full_name",
    "school",
    "school_name",
    "phone",
    "mobile",
    "email",
    "chat_id",
    "sender",
    "sender_id",
    "contact",
    "address",
    "token",
    "secret",
    "password",
    "authorization",
    "cookie",
    "credential",
    "api_key",
    "learner_id",
    "file_path",
    "image_path",
    "attachment_path",
}
_SENSITIVE_KEY_FRAGMENTS = (
    "password", "token", "secret", "authorization", "cookie", "credential", "api_key"
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/]+=*"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\b(?:token|api[_-]?key|password|secret)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
)
_LOCK = threading.RLock()


def _redact_text(value: str) -> str:
    out = value
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out[:1200]


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth >= 6:
        return "[BOUNDED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth + 1) for item in list(value)[:32]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (raw_key, child) in enumerate(value.items()):
            if index >= 48:
                break
            key = str(raw_key)[:120]
            low = key.lower()
            if low in _SENSITIVE_KEYS or any(
                fragment in low for fragment in _SENSITIVE_KEY_FRAGMENTS
            ):
                out[key] = "[REDACTED]"
            else:
                out[key] = _sanitize(child, depth + 1)
        return out
    return _redact_text(str(value))


def sanitize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    clean = _sanitize(evidence)
    if not isinstance(clean, dict):
        clean = {"evidence": clean}
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) <= MAX_EVIDENCE_BYTES:
        return clean
    summary: dict[str, Any] = {"bounded": True, "sha256": hashlib.sha256(raw).hexdigest()}
    for key, value in clean.items():
        if len(summary) >= 16:
            break
        if not (value is None or isinstance(value, (bool, int, float, str))):
            continue
        candidate = value
        if isinstance(candidate, str):
            candidate = candidate[:256]
        trial = {**summary, key: candidate}
        trial_bytes = len(
            json.dumps(
                trial,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if trial_bytes >= MAX_EVIDENCE_BYTES:
            continue
        summary[key] = candidate
    return summary


def evidence_hash(evidence: dict[str, Any]) -> str:
    raw = json.dumps(
        sanitize_evidence(evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _state_root() -> Path:
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    root = home / "state" / "teaching-decision-layer"
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def _audit(record: dict[str, Any]) -> None:
    path = _state_root() / "decisions.jsonl"
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _provider_call(
    *,
    state: dict[str, Any],
    checks: tuple[str, ...],
    model: str,
    timeout: float,
) -> tuple[str, dict[str, float]]:
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise RuntimeError("missing_typesafe_api_key")
    from typesafe_sdk import Noul, TypeSafeClient

    client = TypeSafeClient(model=model)
    questions = {name: Noul(instructions=_QUESTIONS[name]) for name in checks}
    response = client.system_one(
        state=sanitize_evidence(state),
        questions=questions,
        model=model,
        timeout=timeout,
    )
    return str(response.model or model), {
        name: float(response.answers[name].noul) for name in checks
    }


def _thresholds(layer: str) -> tuple[float, float]:
    if layer == "perception":
        return PERCEPTION_YES, PERCEPTION_NO
    return PEDAGOGY_YES, PEDAGOGY_NO


def _next_action(layer: str, check: str, verdict: str) -> str:
    if verdict == "gray":
        return "visual_recheck" if layer == "perception" else "teacher_model_review"
    yes = verdict == "yes"
    if layer == "perception":
        if check == "ocr_uncertain":
            return "visual_recheck" if yes else "accept_current_ocr"
        if check in {"answer_region_detected", "answer_complete"}:
            return "accept_visual_evidence" if yes else "visual_recheck"
        return "accept_classification" if yes else "reject_classification"
    mapping = {
        "understanding_demonstrated": ("count_as_understanding_evidence", "verify_understanding"),
        "guessing_likely": ("ask_feynman_followup", "no_guessing_signal"),
        "needs_followup_question": ("ask_followup", "followup_not_required"),
        "ready_to_advance": ("advance", "stay_on_objective"),
        "learning_objective_met": ("objective_met", "continue_objective"),
        "response_too_entertaining": ("return_to_learning_objective", "keep_current_balance"),
    }
    return mapping[check][0 if yes else 1]


def judge(
    *,
    layer: str,
    evidence: dict[str, Any],
    checks: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    provider_call: Callable[..., tuple[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    allowed = PERCEPTION_CHECKS if layer == "perception" else PEDAGOGY_CHECKS if layer == "pedagogy" else ()
    if not allowed:
        return {"ok": False, "error": "invalid_layer", "layer": layer}
    if not isinstance(evidence, dict) or not evidence:
        return {"ok": False, "error": "evidence_required", "layer": layer}

    selected = tuple(checks or allowed)
    if not selected or any(item not in allowed for item in selected):
        return {
            "ok": False,
            "error": "invalid_checks",
            "layer": layer,
            "allowed_checks": list(allowed),
        }

    model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    try:
        timeout = max(0.2, min(5.0, float(timeout)))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT

    yes_threshold, no_threshold = _thresholds(layer)
    clean = sanitize_evidence(evidence)
    hashed = evidence_hash(clean)
    started = time.perf_counter()
    resolved_model = ""
    fallback_reason = ""
    probabilities: dict[str, float] = {}
    try:
        caller = provider_call or _provider_call
        resolved_model, probabilities = caller(
            state=clean,
            checks=selected,
            model=model,
            timeout=timeout,
        )
    except Exception as exc:
        fallback_reason = f"provider_unavailable:{type(exc).__name__}"

    latency_ms = int((time.perf_counter() - started) * 1000)
    decisions: dict[str, Any] = {}
    for check in selected:
        probability = probabilities.get(check)
        if probability is None:
            verdict = "defer"
            next_action = "visual_recheck" if layer == "perception" else "teacher_model_review"
        elif probability >= yes_threshold:
            verdict = "yes"
            next_action = _next_action(layer, check, verdict)
        elif probability <= no_threshold:
            verdict = "no"
            next_action = _next_action(layer, check, verdict)
        else:
            verdict = "gray"
            next_action = _next_action(layer, check, verdict)
        decisions[check] = {
            "probability": round(float(probability), 4) if probability is not None else None,
            "verdict": verdict,
            "next_action": next_action,
        }

    record = {
        "schema": SCHEMA,
        "decision_id": "teach_" + uuid.uuid4().hex,
        "timestamp": time.time(),
        "layer": layer,
        "checks": list(selected),
        "requested_model": model,
        "resolved_model": resolved_model,
        "yes_threshold": yes_threshold,
        "no_threshold": no_threshold,
        "latency_ms": latency_ms,
        "fallback_used": bool(fallback_reason),
        "fallback_reason": fallback_reason,
        "evidence_hash": hashed,
        "decisions": decisions,
    }
    _audit(record)
    return {"ok": True, **record}


def status(*, model: str = DEFAULT_MODEL, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    path = _state_root() / "decisions.jsonl"
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-MAX_AUDIT_ROWS:]
    except OSError:
        lines = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    fallback = sum(bool(item.get("fallback_used")) for item in rows)
    models = sorted({str(item.get("resolved_model")) for item in rows if item.get("resolved_model")})
    provider_ready = bool(os.environ.get("TYPESAFE_API_KEY", "").strip()) and (
        importlib.util.find_spec("typesafe_sdk") is not None
    )
    return {
        "ok": True,
        "schema": STATUS_SCHEMA,
        "requested_model": model,
        "provider_ready": provider_ready,
        "timeout_seconds": timeout,
        "resolved_models": models,
        "decision_batches": len(rows),
        "fallback_batches": fallback,
        "fallback_rate": round(fallback / len(rows), 4) if rows else 0.0,
        "thresholds": {
            "perception": {"yes": PERCEPTION_YES, "no": PERCEPTION_NO},
            "pedagogy": {"yes": PEDAGOGY_YES, "no": PEDAGOGY_NO},
        },
        "checks": {
            "perception": list(PERCEPTION_CHECKS),
            "pedagogy": list(PEDAGOGY_CHECKS),
        },
        "privacy": {
            "raw_evidence_logged": False,
            "raw_images_logged": False,
            "credentials_logged": False,
            "audit_uses_evidence_hash": True,
        },
    }
