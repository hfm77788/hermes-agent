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
_TURN_CACHE: "OrderedDict[str, RouteDecision | object]" = OrderedDict()
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
            result[key.strip()] = base_url
    return result


def _provider_matches(
    *, provider: str, base_url: str, allowed_providers: set[str]
) -> bool:
    """Match a normalized custom provider back to an allowlisted provider id.

    Core normalizes named custom providers such as custom:aliyun_ws to provider=custom
    before llm_request middleware. Exact ids still match directly; normalized custom
    providers must also have a base URL identical to the configured allowlisted
    provider, so another custom Qwen endpoint cannot accidentally inherit Max routing.
    """
    if provider in allowed_providers:
        return True
    if provider != "custom":
        return False
    runtime_base_url = _normalized_base_url(base_url)
    if not runtime_base_url:
        return False
    configured = _configured_provider_base_urls()
    return any(
        key.startswith("custom:")
        and configured.get(key) == runtime_base_url
        for key in allowed_providers
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


def _state_payload(request: dict[str, Any], excerpt: str) -> str:
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


def _typesafe_key(*, centralized: bool) -> str:
    """Resolve the Jev credential without copying it into every named profile.

    Legacy mode keeps profile isolation: only the current process/profile environment is used.
    Centralized profile-policy mode treats Jev as shared Decision Plane infrastructure and reads
    the key from the default Hermes root under a context-local home override. The value is never
    written into the named profile or process-global environment.
    """
    direct = os.environ.get("TYPESAFE_API_KEY", "").strip()
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
            return str(get_env_value("TYPESAFE_API_KEY") or "").strip()
        finally:
            reset_hermes_home_override(token)
    except Exception:
        return ""


def _call_jev(
    *, state: str, model: str, timeout: float, centralized: bool = False
) -> tuple[dict[str, Any], int]:
    key = _typesafe_key(centralized=centralized)
    if not key:
        raise RuntimeError("typesafe_key_missing")
    started = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            _TYPESAFE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"state": state, "model": model, "questions": _QUESTIONS},
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("jev_response_invalid")
    return payload, int((time.perf_counter() - started) * 1000)


def _parse_route(
    response: dict[str, Any],
    *,
    settings: dict[str, Any],
    active_model: str,
    latency_ms: int,
) -> RouteDecision | None:
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return None

    effort_answer = answers.get("effort")
    model_answer = answers.get("model_class")
    if not isinstance(effort_answer, dict) or not isinstance(model_answer, dict):
        return None

    effort = str(effort_answer.get("choice") or "").strip().lower()
    if effort not in _VALID_EFFORTS:
        return None

    confidence = _probability(effort_answer.get("confidence"))
    probabilities = effort_answer.get("probabilities")
    if confidence is None and isinstance(probabilities, dict):
        confidence = _probability(probabilities.get(effort))
    if confidence is None or confidence < settings["min_effort_confidence"]:
        return None

    model_choice = str(model_answer.get("choice") or "").strip().lower()
    model_probabilities = model_answer.get("probabilities")
    if model_choice not in {"flash", "max"} or not isinstance(model_probabilities, dict):
        return None
    model_confidence = _probability(model_answer.get("confidence"))
    if model_confidence is None:
        model_confidence = _probability(model_probabilities.get(model_choice))
    max_probability = _probability(model_probabilities.get("max"))
    if model_confidence is None or max_probability is None:
        return None

    flash_model = settings["flash_model"]
    max_model = settings["max_model"]
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
        return None

    minimum_effort = settings["min_reasoning_effort"]
    if _EFFORT_RANK[effort] < _EFFORT_RANK[minimum_effort]:
        effort = minimum_effort

    model = max_model if (manual_max or should_max) else flash_model
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
    )


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
    if not _provider_matches(
        provider=provider,
        base_url=base_url,
        allowed_providers=settings["providers"],
    ) or api_mode != "chat_completions":
        return None
    if active_model not in {settings["flash_model"], settings["max_model"]}:
        return None
    if settings["respect_existing_reasoning"] and _has_explicit_reasoning(request):
        return None

    turn_id = str(kwargs.get("turn_id") or "").strip()
    if not turn_id:
        # Stable per-turn binding is a hard requirement; do not make uncached routing calls.
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
            _cache_put(cache_key, None)
            return None
        try:
            response, latency_ms = _call_jev(
                state=_state_payload(request, excerpt),
                model=settings["jev_model"],
                timeout=settings["timeout_seconds"],
                centralized=settings["centralized_profile_policy"],
            )
            decision = _parse_route(
                response,
                settings=settings,
                active_model=active_model,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            _warn_rate_limited(
                "jev-aliyun-qwen-router: Jev unavailable; leaving request unchanged (%s)",
                type(exc).__name__,
            )
            _cache_put(cache_key, None)
            return None
        if decision is None:
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
