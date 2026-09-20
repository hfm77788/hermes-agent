"""Primary rate-limit cooldown arming and per-session model rejection markers, shared by the
fallback walk (chat_completion_helpers) and restore_primary_runtime (agent_runtime_helpers)."""
import logging
import math
import time

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})


def _arm_rate_limit_cooldown(
    agent, reason: "FailoverReason | None", *, reset_at=None,
) -> int | None:
    """Arm the primary cooldown, honoring a provider-declared reset when present."""
    if reason not in _RATE_LIMIT_FAILOVER_REASONS:
        return None
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
    if getattr(agent, "_fallback_activated", False) and not (
        primary_provider and current_provider == primary_provider
    ):
        return None

    backoff_count = getattr(agent, "_rate_limit_backoff_count", 0)
    agent._rate_limit_backoff_count = backoff_count + 1
    generic_seconds = min(60 * (2 ** backoff_count), 14400)
    backoff_seconds = generic_seconds

    if reset_at is not None:
        try:
            from agent.retry_utils import reset_at_delay_seconds
            reset_delay = reset_at_delay_seconds(reset_at)
        except Exception:
            reset_delay = None
        if reset_delay is not None and 0 <= reset_delay <= 32 * 24 * 3600:
            backoff_seconds = max(backoff_seconds, int(math.ceil(reset_delay)) + 2)

    now_mono = time.monotonic()
    existing_remaining = max(
        0, int(math.ceil((getattr(agent, "_rate_limited_until", 0) or 0) - now_mono))
    )
    backoff_seconds = max(backoff_seconds, existing_remaining)
    agent._rate_limited_until = now_mono + backoff_seconds
    logger.info(
        "Rate-limit circuit open: cooldown %d s (generic=%d s, backoff#%d, reset_signal=%s)",
        backoff_seconds, generic_seconds, backoff_count + 1, reset_at is not None,
    )
    return backoff_seconds


# Codex ChatGPT-account entitlement 400 — the account can never use the named slug, so with
# nothing to rotate it is a config error, not a transient failure (#106475).
_CODEX_ACCOUNT_MODEL_ENTITLEMENT_MARKER = "model is not supported when using codex with a chatgpt account"


def _mark_entitlement_rejected_model(agent, api_error) -> bool:
    """Record a Codex ChatGPT-account 400 that rejects the current model for this account.

    With a single credential there is no pool to rotate (#71970 covers that case), so the
    (provider, model) pair is treated as dead for the session: the fallback walk skips it and
    restore_primary_runtime stops switching back — otherwise every turn re-fails on the primary,
    announces an unverified "Primary model restored", and oscillates forever (#106475).
    """
    if getattr(api_error, "status_code", None) != 400:
        return False
    pool = getattr(agent, "_credential_pool", None)
    if pool is not None and len(pool.entries()) > 1:
        return False  # another account in the pool may be entitled; leave rotation to it
    haystack = str(getattr(api_error, "message", "") or api_error).lower()
    if _CODEX_ACCOUNT_MODEL_ENTITLEMENT_MARKER not in haystack:
        return False
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    model = str(getattr(agent, "model", "") or "").strip()
    if not provider or not model:
        return False
    rejected = getattr(agent, "_entitlement_rejected_models", None)
    if rejected is None:
        rejected = agent._entitlement_rejected_models = set()
    if (provider, model) in rejected:
        return True
    rejected.add((provider, model))
    logger.warning(
        "Model entitlement rejection: this account is not entitled to %s via %s; "
        "treating it as unavailable for this session",
        model, provider,
    )
    agent._buffer_status(
        f"🚫 This account is not entitled to {model} via {provider}; it will be skipped "
        "until restart. Switch to an entitled model via /model or `hermes model`."
    )
    return True


def _is_entitlement_rejected(agent, provider: str, model: str) -> bool:
    """True when (provider, model) — as configured or normalized — was rejected as unentitled
    for this account (see _mark_entitlement_rejected_model)."""
    rejected = getattr(agent, "_entitlement_rejected_models", None) or ()
    if not rejected:
        return False
    if (provider, model) in rejected:
        return True
    from hermes_cli.model_normalize import normalize_model_for_provider
    return (provider, normalize_model_for_provider(model, provider)) in rejected
