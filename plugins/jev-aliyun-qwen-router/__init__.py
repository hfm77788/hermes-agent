"""Jev-driven per-turn effort/model routing for Alibaba Cloud Qwen3.8.

The plugin is deliberately narrow:
- only OpenAI-compatible chat-completions requests on an allowlisted Aliyun provider;
- one Jev decision per Hermes turn, cached for all tool-loop follow-ups;
- only the latest user text is sent to Jev, after forced secret redaction and truncation;
- Jev selects low/medium/xhigh effort, while a separate high-confidence noul gate is
  required before qwen3.8-flash may escalate to qwen3.8-max-0902;
- provider/Jev/parse failures leave the original request unchanged.

Authorization and tool/business gates are outside this router and remain authoritative.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
_DEFAULT_JEV_URL = _TYPESAFE_URL
_DEFAULT_JEV_KEY_ENV = "TYPESAFE_API_KEY"
_DEFAULT_JEV_MODEL = "jev-latest"
_DEFAULT_FLASH_MODEL = "qwen3.8-flash"
_DEFAULT_MAX_MODEL = "qwen3.8-max-0902"
_DEFAULT_PROVIDER = "custom:aliyun_ws"
_VALID_EFFORTS = frozenset({"low", "medium", "xhigh"})
_EFFORT_RANK = {"low": 0, "medium": 1, "xhigh": 2}
_PROFILE_POLICY_OVERRIDE_KEYS = frozenset({
    "min_effort_confidence",
    "max_escalation_probability",
    "min_max_choice_confidence",
    "min_reasoning_effort",
    "allow_max",
})
_CACHE_LIMIT = 512

_CACHE_LOCK = threading.Lock()
_NO_ROUTE = object()
_TURN_CACHE: OrderedDict[str, RouteDecision | object] = OrderedDict()
_LAST_WARNING_AT = 0.0

_QUESTIONS = {
    "effort": {
        "type": "choice",
        "instructions": (
            "Choose the minimum Qwen3.8 reasoning effort that is likely to complete this user turn "
            "reliably. Prefer lower effort when sufficient. Judge semantic reasoning difficulty, "
            "ambiguity, planning depth, synthesis burden, and tool-use complexity; do not treat length "
            "alone as difficulty."
        ),
        "criteria": {
            "low": (
                "Trivial or mechanical: status/lookup, extraction, formatting, a direct factual answer, "
                "or a very bounded action with little ambiguity and no meaningful multi-step reasoning."
            ),
            "medium": (
                "Ordinary bounded work: moderate analysis, a small decision, normal tool use, or several "
                "straightforward dependent steps where standard reasoning is enough."
            ),
            "xhigh": (
                "Hard reasoning: substantial ambiguity, long dependency chains, complex debugging or "
                "architecture, conflicting evidence, difficult synthesis, or many interacting constraints."
            ),
        },
    },
    "model_class": {
        "type": "choice",
        "instructions": (
            "Choose the least expensive Qwen3.8 model class likely to complete this turn reliably. "
            "Prefer Flash unless stronger model capability is materially useful for correctness, "
            "not merely because the task is long, important, or uses tools."
        ),
        "criteria": {
            "flash": (
                "Qwen3.8 Flash is sufficient, including at xhigh reasoning: routine through hard bounded "
                "work, normal coding/debugging, tool use, analysis, and synthesis."
            ),
            "max": (
                "Qwen3.8 Max is materially warranted: frontier-level ambiguity or architecture, unusually "
                "difficult debugging, conflicting constraints/evidence, or deep multi-stage reasoning where "
                "stronger model capability is likely to improve correctness."
            ),
        },
    },
}


@dataclass(frozen=True)
class RouteDecision:
    model: str
    effort: str
    tier: str
    effort_confidence: float
    model_choice_confidence: float
    max_probability: float
    jev_model: str
    latency_ms: int


def _as_float(value: Any, default: float, *, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, parsed))


def _as_int(value: Any, default: int, *, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, parsed))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _active_profile_name() -> str:
    try:
        from hermes_constants import get_hermes_home, profile_name_for_home

        return profile_name_for_home(get_hermes_home()) or "default"
    except Exception:
        return "default"


def _root_plugin_settings() -> dict[str, Any]:
    """Read this plugin's central settings from the default Hermes root.

    Named profiles keep their own operational config, but model-routing policy has one
    authority at the root. The context-local home override makes this safe inside the
    multiplex gateway without mutating process-global HERMES_HOME.
    """
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_constants import (
            get_default_hermes_root,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(get_default_hermes_root())
        try:
            config = load_config_readonly() or {}
        finally:
            reset_hermes_home_override(token)
    except Exception:
        return {}
    if not isinstance(config, dict):
        return {}
    plugins = config.get("plugins")
    entries = plugins.get("entries") if isinstance(plugins, dict) else None
    entry = entries.get("jev-aliyun-qwen-router") if isinstance(entries, dict) else None
    settings = entry.get("settings") if isinstance(entry, dict) else None
    return dict(settings) if isinstance(settings, dict) else {}


def _settings(ctx: Any) -> dict[str, Any]:
    root_settings = _root_plugin_settings()
    centralized = _as_bool(root_settings.get("centralized_profile_policy"), False)
    profile = _active_profile_name()

    if centralized:
        source = root_settings
        default_policy = str(source.get("default_policy") or "standard").strip() or "standard"
        profile_policies = source.get("profile_policies")
        if not isinstance(profile_policies, dict):
            profile_policies = {}
        policy_name = str(profile_policies.get(profile) or default_policy).strip() or default_policy
        policies = source.get("policies")
        if not isinstance(policies, dict):
            policies = {}
        candidate = policies.get(policy_name)
        policy = candidate if isinstance(candidate, dict) else {}

        def value(key: str, default: Any) -> Any:
            if key in _PROFILE_POLICY_OVERRIDE_KEYS and key in policy:
                return policy[key]
            return source.get(key, default)
    else:
        policy_name = "legacy"
        policy = {}

        def value(key: str, default: Any) -> Any:
            return ctx.get_config(key, default)

    providers = value("providers", [_DEFAULT_PROVIDER])
    if isinstance(providers, str):
        providers = [part.strip() for part in providers.split(",") if part.strip()]
    if not isinstance(providers, list):
        providers = [_DEFAULT_PROVIDER]
    providers = [str(item).strip() for item in providers if str(item).strip()]

    min_reasoning_effort = str(value("min_reasoning_effort", "low") or "low").strip().lower()
    if min_reasoning_effort not in _VALID_EFFORTS:
        min_reasoning_effort = "low"

    return {
        "enabled": _as_bool(value("enabled", True), True),
        "providers": set(providers or [_DEFAULT_PROVIDER]),
        "flash_model": str(value("flash_model", _DEFAULT_FLASH_MODEL) or _DEFAULT_FLASH_MODEL).strip(),
        "max_model": str(value("max_model", _DEFAULT_MAX_MODEL) or _DEFAULT_MAX_MODEL).strip(),
        "jev_model": str(value("jev_model", _DEFAULT_JEV_MODEL) or _DEFAULT_JEV_MODEL).strip(),
        "jev_url": str(value("jev_url", _DEFAULT_JEV_URL) or _DEFAULT_JEV_URL).strip(),
        "jev_key_env": str(value("jev_key_env", _DEFAULT_JEV_KEY_ENV) or _DEFAULT_JEV_KEY_ENV).strip(),
        "timeout_seconds": _as_float(value("timeout_seconds", 1.5), 1.5, low=0.2, high=10.0),
        "min_effort_confidence": _as_float(
            value("min_effort_confidence", 0.50), 0.50, low=0.0, high=1.0
        ),
        "max_escalation_probability": _as_float(
            value("max_escalation_probability", 0.90), 0.90, low=0.5, high=1.0
        ),
        "min_max_choice_confidence": _as_float(
            value("min_max_choice_confidence", 0.80), 0.80, low=0.0, high=1.0
        ),
        "max_excerpt_chars": _as_int(value("max_excerpt_chars", 2400), 2400, low=128, high=8000),
        "respect_existing_reasoning": _as_bool(value("respect_existing_reasoning", False), False),
        "respect_manual_max": _as_bool(value("respect_manual_max", True), True),
        # Narrow max->flash downgrade channel: only fresh/small sessions where
        # Jev confidently picks flash with low max-probability may downgrade.
        "downgrade_channel": _as_bool(value("downgrade_channel", False), False),
        "downgrade_max_history_messages": _as_int(
            value("downgrade_max_history_messages", 6), 6, low=1, high=64
        ),
        "downgrade_max_context_chars": _as_int(
            value("downgrade_max_context_chars", 8000), 8000, low=512, high=64000
        ),
        "downgrade_min_effort_confidence": _as_float(
            value("downgrade_min_effort_confidence", 0.80), 0.80, low=0.0, high=1.0
        ),
        "downgrade_max_probability": _as_float(
            value("downgrade_max_probability", 0.10), 0.10, low=0.0, high=1.0
        ),
        "downgrade_strict_effort_confidence": _as_float(
            value("downgrade_strict_effort_confidence", 0.85), 0.85, low=0.0, high=1.0
        ),
        "downgrade_strict_max_probability": _as_float(
            value("downgrade_strict_max_probability", 0.05), 0.05, low=0.0, high=1.0
        ),
        "downgrade_recent_tail_chars": _as_int(
            value("downgrade_recent_tail_chars", 1200), 1200, low=0, high=8000
        ),
        "max_model_aliases": _string_map(value("max_model_aliases", {})),
        "min_reasoning_effort": min_reasoning_effort,
        "allow_max": _as_bool(value("allow_max", True), True),
        "centralized_profile_policy": centralized,
        "profile": profile,
        "policy_name": policy_name,
    }


def _normalized_base_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _configured_provider_base_urls() -> dict[str, str]:
    """Return non-secret configured custom-provider URLs keyed by provider id."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:
        return {}
    providers = config.get("providers") if isinstance(config, dict) else None
    if not isinstance(providers, dict):
        return {}
    result: dict[str, str] = {}
    for key, entry in providers.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            bare = key.strip()
            result[bare] = base_url
            if not bare.lower().startswith("custom:"):
                # Core normalizes named custom providers (model.provider
                # "custom:aliyun_qwen" -> provider="custom" + base_url) before
                # llm_request middleware, and allowlists use the prefixed id.
                # Index bare config keys under the prefixed id too so both
                # spellings match.
                result.setdefault(f"custom:{bare}", base_url)
    return result


def _string_map(raw: Any) -> dict[str, str]:
    """Mapping of allowlisted provider id (or base URL) -> alternate max model id."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in raw.items():
        key_s = str(key).strip()
        value_s = str(item).strip()
        if key_s and value_s:
            result[key_s] = value_s
    return result


def _matched_provider_id(
    *, provider: str, base_url: str, allowed_providers: set[str]
) -> str:
    """Return the allowlisted provider id this request belongs to ("" = no match).

    Core normalizes named custom providers such as custom:aliyun_ws to provider=custom
    before llm_request middleware. Exact ids still match directly; normalized custom
    providers must also have a base URL identical to the configured allowlisted
    provider, so another custom Qwen endpoint cannot accidentally inherit Max routing.
    """
    if provider in allowed_providers:
        return provider
    if provider != "custom":
        return ""
    runtime_base_url = _normalized_base_url(base_url)
    if not runtime_base_url:
        return ""
    configured = _configured_provider_base_urls()
    for key in allowed_providers:
        if not key.startswith("custom:"):
            continue
        # Profile configs are inconsistent about the "custom:" prefix (some name
        # the provider "aliyun_qwen", others "custom:aliyun_qwen"); accept both.
        bare = key.split(":", 1)[1]
        if configured.get(key) == runtime_base_url or configured.get(bare) == runtime_base_url:
            return key
    return ""


def _provider_matches(
    *, provider: str, base_url: str, allowed_providers: set[str]
) -> bool:
    """Backward-compatible boolean wrapper around _matched_provider_id."""
    return bool(
        _matched_provider_id(
            provider=provider, base_url=base_url, allowed_providers=allowed_providers
        )
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in {"text", "input_text"} and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def _latest_user_text(request: dict[str, Any]) -> str:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        return _content_text(message.get("content"))
    return ""


def _redacted_excerpt(text: str, max_chars: int) -> str:
    if not text:
        return ""
    # Bound work before regex redaction, then force the safety boundary regardless of
    # the user's ordinary security.redact_secrets preference.
    candidate = text[: max_chars * 2]
    try:
        from agent.redact import redact_sensitive_text

        candidate = redact_sensitive_text(
            candidate,
            force=True,
            redact_url_credentials=True,
        )
    except Exception:
        # Fail closed for external egress: if the core redactor is unavailable, do not
        # send the raw user text to Jev.
        return ""
    return candidate[:max_chars]


def _has_explicit_reasoning(request: dict[str, Any]) -> bool:
    value = request.get("reasoning_effort")
    if isinstance(value, str) and value.strip():
        return True
    if request.get("thinking_budget") is not None:
        return True
    extra = request.get("extra_body")
    if isinstance(extra, dict):
        if extra.get("thinking_budget") is not None:
            return True
        nested = extra.get("reasoning")
        if isinstance(nested, dict) and nested:
            return True
    return False


def _state_payload(request: dict[str, Any], excerpt: str, settings: dict[str, Any]) -> str:
    tools = request.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0
    state = {
        "latest_user_message": excerpt,
        "features": {
            "message_chars": len(excerpt),
            "tool_count": min(tool_count, 128),
            "has_tools": tool_count > 0,
        },
    }
    tail = _recent_tail(request, settings)
    if tail:
        state["recent_context_tail"] = tail
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _probability(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _typesafe_key(*, centralized: bool, key_env: str = _DEFAULT_JEV_KEY_ENV) -> str:
    """Resolve the Jev credential without copying it into every named profile.

    Legacy mode keeps profile isolation: only the current process/profile environment is used.
    Centralized profile-policy mode treats Jev as shared Decision Plane infrastructure and reads
    the key from the default Hermes root under a context-local home override. The value is never
    written into the named profile or process-global environment.
    """
    direct = os.environ.get(key_env, "").strip()
    if direct or not centralized:
        return direct
    try:
        from hermes_cli.config import get_env_value
        from hermes_constants import (
            get_default_hermes_root,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(get_default_hermes_root())
        try:
            return str(get_env_value(key_env) or "").strip()
        finally:
            reset_hermes_home_override(token)
    except Exception:
        return ""


def _call_jev(
    *, state: str, model: str, timeout: float, centralized: bool = False,
    url: str = _DEFAULT_JEV_URL, key_env: str = _DEFAULT_JEV_KEY_ENV,
) -> tuple[dict[str, Any], int]:
    key = _typesafe_key(centralized=centralized, key_env=key_env)
    if not key:
        raise RuntimeError("jev_key_missing")
    started = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"state": state, "model": model, "questions": _QUESTIONS},
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("jev_response_invalid")
    return payload, int((time.perf_counter() - started) * 1000)


# Reason code for a successful parse (empty string = no rejection).
_PARSE_OK = ""


def _parse_route_with_reason(
    response: dict[str, Any],
    *,
    settings: dict[str, Any],
    active_model: str,
    latency_ms: int,
    downgrade_ok: bool = False,
    strict_gate: bool = True,
    effective_max_model: str = "",
) -> tuple[RouteDecision | None, str]:
    """Parse Jev's answer into a RouteDecision, returning (decision, reason).

    reason is "" on success, else a stable skip-code naming the exact rejection
    branch. Per jev-model-router-ops discipline every return-None path must be
    individually attributable -- the old single "decision_unparsed" bucket lumped
    "Jev not confident" (the dominant, HEALTHY fail-safe that leaves the request
    on its original model) together with genuine malformed/contradictory answers,
    which made conservative skips look like a broken router in audits.
    """
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return None, "jev_answers_malformed"

    effort_answer = answers.get("effort")
    model_answer = answers.get("model_class")
    if not isinstance(effort_answer, dict) or not isinstance(model_answer, dict):
        return None, "jev_answers_malformed"

    effort = str(effort_answer.get("choice") or "").strip().lower()
    if effort not in _VALID_EFFORTS:
        return None, "effort_choice_invalid"

    confidence = _probability(effort_answer.get("confidence"))
    probabilities = effort_answer.get("probabilities")
    if confidence is None and isinstance(probabilities, dict):
        confidence = _probability(probabilities.get(effort))
    if confidence is None:
        return None, "effort_confidence_unparsable"
    if confidence < settings["min_effort_confidence"]:
        # Expected fail-safe, NOT a fault: Jev is not confident enough to route
        # this turn, so the request stays on its original model (max-start seats
        # stay Max). Dominant healthy outcome for continuation/complex/ambiguous
        # turns -- kept as its own reason so audits never misread it as broken.
        return None, "effort_confidence_low"

    model_choice = str(model_answer.get("choice") or "").strip().lower()
    model_probabilities = model_answer.get("probabilities")
    if model_choice not in {"flash", "max"}:
        return None, "model_choice_invalid"
    if not isinstance(model_probabilities, dict):
        return None, "model_probs_malformed"
    model_confidence = _probability(model_answer.get("confidence"))
    if model_confidence is None:
        model_confidence = _probability(model_probabilities.get(model_choice))
    max_probability = _probability(model_probabilities.get("max"))
    if model_confidence is None:
        return None, "model_confidence_unparsable"
    if max_probability is None:
        return None, "max_probability_unparsable"

    flash_model = settings["flash_model"]
    max_model = (effective_max_model or "").strip() or settings["max_model"]
    active = active_model.strip()

    # An explicit manual Max selection is never silently downgraded.
    manual_max = settings["respect_manual_max"] and active == max_model
    should_max = (
        settings["allow_max"]
        and model_choice == "max"
        and max_probability >= settings["max_escalation_probability"]
        and model_confidence >= settings["min_max_choice_confidence"]
    )

    # Contradictory "low effort but Max required" judgments are treated as uncertain
    # before a role policy applies its minimum effort floor.
    if should_max and effort == "low" and not manual_max:
        return None, "contradictory_low_max"

    minimum_effort = settings["min_reasoning_effort"]
    if _EFFORT_RANK[effort] < _EFFORT_RANK[minimum_effort]:
        effort = minimum_effort

    # v2 downgrade channel: eligibility is about the message being
    # self-contained (not a continuation), NOT about session length. Longer
    # sessions simply face stricter confidence gates. No absolute context cap:
    # flash handles long contexts (million-token class window).
    downgrade_allowed = False
    if (
        settings["downgrade_channel"]
        and downgrade_ok
        and not should_max
        and model_choice == "flash"
        and effort != "xhigh"
    ):
        if strict_gate:
            conf_gate = settings["downgrade_strict_effort_confidence"]
            prob_gate = settings["downgrade_strict_max_probability"]
        else:
            conf_gate = settings["downgrade_min_effort_confidence"]
            prob_gate = settings["downgrade_max_probability"]
        downgrade_allowed = confidence >= conf_gate and max_probability <= prob_gate
    model = max_model if ((manual_max and not downgrade_allowed) or should_max) else flash_model
    tier = ("max_" if model == max_model else "flash_") + effort
    return RouteDecision(
        model=model,
        effort=effort,
        tier=tier,
        effort_confidence=confidence,
        model_choice_confidence=model_confidence,
        max_probability=max_probability,
        jev_model=str(response.get("model") or settings["jev_model"]),
        latency_ms=latency_ms,
    ), _PARSE_OK


def _parse_route(
    response: dict[str, Any],
    *,
    settings: dict[str, Any],
    active_model: str,
    latency_ms: int,
    downgrade_ok: bool = False,
    strict_gate: bool = True,
    effective_max_model: str = "",
) -> RouteDecision | None:
    """Back-compat wrapper: decision only, for callers/tests that don't need the reason."""
    decision, _reason = _parse_route_with_reason(
        response,
        settings=settings,
        active_model=active_model,
        latency_ms=latency_ms,
        downgrade_ok=downgrade_ok,
        strict_gate=strict_gate,
        effective_max_model=effective_max_model,
    )
    return decision


def _cache_get(turn_id: str) -> tuple[bool, RouteDecision | None]:
    if not turn_id:
        return False, None
    with _CACHE_LOCK:
        if turn_id not in _TURN_CACHE:
            return False, None
        cached = _TURN_CACHE[turn_id]
        _TURN_CACHE.move_to_end(turn_id)
        if cached is _NO_ROUTE:
            return True, None
        return True, cached if isinstance(cached, RouteDecision) else None


def _cache_put(turn_id: str, decision: RouteDecision | None) -> None:
    if not turn_id:
        return
    with _CACHE_LOCK:
        _TURN_CACHE[turn_id] = _NO_ROUTE if decision is None else decision
        _TURN_CACHE.move_to_end(turn_id)
        while len(_TURN_CACHE) > _CACHE_LIMIT:
            _TURN_CACHE.popitem(last=False)


def _warn_rate_limited(message: str, *args: Any) -> None:
    global _LAST_WARNING_AT
    now = time.monotonic()
    if now - _LAST_WARNING_AT < 60.0:
        return
    _LAST_WARNING_AT = now
    logger.warning(message, *args)


_SKIP_LOG_AT: dict[str, float] = {}


def _skip_rate_limited(reason: str, message: str, *args: Any) -> None:
    """INFO-level early-return trace, rate-limited per reason (5 min window).

    Silent skips made it impossible to diagnose why a seat's traffic never
    reached the classifier; every early return in route_request must leave a
    rate-limited breadcrumb with its reason key.
    """
    now = time.monotonic()
    if now - _SKIP_LOG_AT.get(reason, 0.0) < 300.0:
        return
    _SKIP_LOG_AT[reason] = now
    logger.info(message, *args)


def _apply_decision(request: dict[str, Any], decision: RouteDecision) -> dict[str, Any]:
    rewritten = dict(request)
    rewritten["model"] = decision.model
    rewritten["reasoning_effort"] = decision.effort
    # Qwen3.8 rejects reasoning_effort + thinking_budget together. We only reach
    # this function when explicit reasoning was not present, but strip a stale
    # top-level budget defensively if a transport injected one after that check.
    rewritten.pop("thinking_budget", None)
    extra = rewritten.get("extra_body")
    if isinstance(extra, dict) and "thinking_budget" in extra:
        copied = dict(extra)
        copied.pop("thinking_budget", None)
        rewritten["extra_body"] = copied
    return rewritten


def _visible_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Conversation messages excluding system/developer standing prompts."""
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        m
        for m in messages
        if isinstance(m, dict)
        and str(m.get("role", "")).lower() not in {"system", "developer"}
    ]


def _context_stats(request: dict[str, Any]) -> tuple[int, int]:
    """(visible message count, visible char count) used for threshold tiering."""
    visible = _visible_messages(request)
    return len(visible), sum(len(_content_text(m.get("content"))) for m in visible)


# Pure continuation utterances carry no standalone meaning; they inherit the
# task context by definition and must never be downgraded, whatever Jev says.
_CONTINUATION_EXACT = {
    "继续", "推进", "接着", "好", "好的", "嗯", "嗯嗯", "行", "可以", "就这样",
    "按此", "按这", "照旧", "同前", "收到", "是", "对", "做", "发", "走",
    "ok", "okay", "go", "yes", "y", "1", "?", "？", "。", ".", "！", "!", "、",
}
_PUNCTUATION_CHARS = " \t\n\r?？!！.。,，、;；:：~～…—-_*#@+"

_CONTINUATION_PREFIXES = (
    "继续", "接着", "上面", "上述", "刚才", "刚说", "按你", "按上", "如上",
    "同前", "照旧", "就按", "还是按", "仍按",
)


def _is_continuation(text: str) -> bool:
    """True when the latest user message is context-dependent rather than
    self-contained (single chars, bare acknowledgements, anaphora openers)."""
    stripped = (text or "").strip()
    if len(stripped) <= 1:
        return True
    if stripped in _CONTINUATION_EXACT or stripped.lower() in _CONTINUATION_EXACT:
        return True
    if not stripped.strip(_PUNCTUATION_CHARS):
        return True
    return stripped.startswith(_CONTINUATION_PREFIXES)


def _recent_tail(request: dict[str, Any], settings: dict[str, Any]) -> str:
    """Last two visible messages before the latest user turn, truncated and
    redacted, so Jev can tell standalone questions from task continuations."""
    budget = settings.get("downgrade_recent_tail_chars", 1200)
    if budget <= 0:
        return ""
    visible = _visible_messages(request)
    if visible and str(visible[-1].get("role", "")).lower() == "user":
        visible = visible[:-1]
    # Privacy: tool results never enter the tail (they can carry sensitive
    # payloads); only user/assistant turns give Jev conversational context.
    visible = [m for m in visible if str(m.get("role", "")).lower() in {"user", "assistant"}]
    parts = []
    per_message = max(budget // 2, 64)
    for message in visible[-2:]:
        text = _redacted_excerpt(_content_text(message.get("content")), per_message)
        if not text:
            continue
        parts.append(f"{str(message.get('role') or 'unknown').lower()}: {text}")
    return "\n".join(parts)


def route_request(ctx: Any, **kwargs: Any) -> dict[str, Any] | None:
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return None

    settings = _settings(ctx)
    if not settings["enabled"]:
        return None

    provider = str(kwargs.get("provider") or "").strip()
    base_url = str(kwargs.get("base_url") or "").strip()
    api_mode = str(kwargs.get("api_mode") or "").strip()
    active_model = str(request.get("model") or kwargs.get("model") or "").strip()
    matched_provider = _matched_provider_id(
        provider=provider,
        base_url=base_url,
        allowed_providers=settings["providers"],
    )
    if not matched_provider or api_mode != "chat_completions":
        _skip_rate_limited(
            "provider_unmatched",
            "jev-aliyun-qwen-router: skip=provider_unmatched profile=%s provider=%r api_mode=%r base_url=%.80s",
            settings["profile"], provider, api_mode, base_url,
        )
        return None
    # Per-endpoint max alias: some providers (e.g. token-plan Beijing) do not
    # serve the pinned snapshot name; they escalate to their own max id instead.
    aliases = settings["max_model_aliases"]
    effective_max = (
        aliases.get(matched_provider)
        or aliases.get(_normalized_base_url(base_url))
        or settings["max_model"]
    )
    if active_model not in {settings["flash_model"], effective_max}:
        _skip_rate_limited(
            "model_not_routable",
            "jev-aliyun-qwen-router: skip=model_not_routable profile=%s active_model=%r flash=%r effective_max=%r",
            settings["profile"], active_model, settings["flash_model"], effective_max,
        )
        return None
    if settings["respect_existing_reasoning"] and _has_explicit_reasoning(request):
        return None

    turn_id = str(kwargs.get("turn_id") or "").strip()
    if not turn_id:
        # Stable per-turn binding is a hard requirement; do not make uncached routing calls.
        _skip_rate_limited(
            "turn_id_missing",
            "jev-aliyun-qwen-router: skip=turn_id_missing profile=%s active_model=%r platform=%r",
            settings["profile"], active_model, str(kwargs.get("platform") or ""),
        )
        return None

    cache_key = f"{settings['profile']}::{turn_id}"
    cached, decision = _cache_get(cache_key)
    if cached and decision is None:
        return None

    fresh_decision = False
    if not cached:
        user_text = _latest_user_text(request)
        excerpt = _redacted_excerpt(user_text, settings["max_excerpt_chars"])
        if not excerpt:
            _skip_rate_limited(
                "excerpt_empty",
                "jev-aliyun-qwen-router: skip=excerpt_empty profile=%s turn_id=%.40s",
                settings["profile"], turn_id,
            )
            _cache_put(cache_key, None)
            return None
        try:
            response, latency_ms = _call_jev(
                state=_state_payload(request, excerpt, settings),
                model=settings["jev_model"],
                timeout=settings["timeout_seconds"],
                centralized=settings["centralized_profile_policy"],
                url=settings["jev_url"],
                key_env=settings["jev_key_env"],
            )
            msg_count, ctx_chars = _context_stats(request)
            small_session = (
                msg_count <= settings["downgrade_max_history_messages"]
                and ctx_chars <= settings["downgrade_max_context_chars"]
            )
            decision, parse_reason = _parse_route_with_reason(
                response,
                settings=settings,
                active_model=active_model,
                latency_ms=latency_ms,
                downgrade_ok=not _is_continuation(user_text),
                strict_gate=not small_session,
                effective_max_model=effective_max,
            )
        except Exception as exc:
            _warn_rate_limited(
                "jev-aliyun-qwen-router: Jev unavailable; leaving request unchanged (%s)",
                type(exc).__name__,
            )
            _cache_put(cache_key, None)
            return None
        if decision is None:
            reason = parse_reason or "decision_unparsed"
            _skip_rate_limited(
                reason,
                "jev-aliyun-qwen-router: skip=%s profile=%s turn_id=%.40s (Jev answered; no route decision -- request left on original model)",
                reason, settings["profile"], turn_id,
            )
            _cache_put(cache_key, None)
            return None
        _cache_put(cache_key, decision)
        fresh_decision = True

    rewritten = _apply_decision(request, decision)
    if fresh_decision:
        logger.info(
            "jev-aliyun-qwen-router: profile=%s policy=%s tier=%s effort_conf=%.3f model_conf=%.3f max_p=%.3f latency_ms=%d",
            settings["profile"],
            settings["policy_name"],
            decision.tier,
            decision.effort_confidence,
            decision.model_choice_confidence,
            decision.max_probability,
            decision.latency_ms,
        )
    return {
        "request": rewritten,
        "source": "jev-aliyun-qwen-router",
        "reason": decision.tier,
    }


def register(ctx: Any) -> None:
    ctx.register_middleware("llm_request", lambda **kwargs: route_request(ctx, **kwargs))
    logger.info("jev-aliyun-qwen-router: registered Aliyun Qwen3.8 llm_request router")
