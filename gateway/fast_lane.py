"""Conservative fast-lane admission for long gateway sessions.

The lane only decides whether prior conversation is required. It never routes tools,
changes approvals, or executes actions. Any ambiguity or classifier failure uses the
normal full-history path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger("gateway.run")


@dataclass(frozen=True)
class FastLaneDecision:
    use_fast_lane: bool
    reason: str
    prompt_tokens: int = 0
    history_rows: int = 0
    classifier_ms: int = 0
    confidence: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.use_fast_lane


def _config(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(config, Mapping):
        return None
    gateway = config.get("gateway", {})
    gateway = gateway if isinstance(gateway, Mapping) else {}
    raw = gateway.get("fast_lane", {})
    raw = raw if isinstance(raw, Mapping) else {}

    def as_int(name, default, lo, hi):
        try:
            value = int(raw.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(lo, min(hi, value))

    def as_float(name, default, lo, hi):
        try:
            value = float(raw.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(lo, min(hi, value))

    enabled = raw.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
    return {
        "enabled": bool(enabled),
        "min_history_rows": as_int("min_history_rows", 60, 20, 500),
        "min_prompt_tokens": as_int("min_prompt_tokens", 32000, 8000, 500000),
        "max_message_chars": as_int("max_message_chars", 1200, 64, 8000),
        "timeout_s": as_float("classifier_timeout_s", 2.5, 0.5, 8.0),
        "min_confidence": as_float("min_confidence", 0.98, 0.8, 1.0),
    }


def _decision(use, reason, tokens, rows, started=None, confidence=0.0):
    elapsed = 0 if started is None else max(0, int((time.monotonic() - started) * 1000))
    d = FastLaneDecision(use, reason, tokens, rows, elapsed, confidence)
    if reason != "short_history":
        logger.info(
            "gateway_fast_lane decision=%s reason=%s history_rows=%d last_prompt_tokens=%d "
            "classifier_ms=%d confidence=%.3f",
            "accepted" if use else "fallback", reason, rows, tokens, elapsed, confidence,
        )
    return d


def _static_block_reason(event, source, *, is_new_session, was_auto_reset, pending_sidecar):
    if is_new_session or was_auto_reset:
        return "session_boundary"
    if pending_sidecar or getattr(event, "auto_skill", None) or getattr(event, "internal", False):
        return "sidecar_context"
    if getattr(event, "reply_to_message_id", None) or getattr(event, "reply_to_text", None):
        return "reply_context"
    if getattr(event, "media_urls", None) or getattr(event, "media_types", None):
        return "media_context"
    if getattr(event, "prompt_response", None):
        return "prompt_response"
    if getattr(event, "channel_prompt", None) or getattr(event, "channel_context", None):
        return "channel_context"
    if getattr(event, "metadata", None):
        return "event_metadata"
    if callable(getattr(event, "is_command", None)) and event.is_command():
        return "gateway_command"
    message_type = getattr(
        getattr(event, "message_type", None), "value", getattr(event, "message_type", None)
    )
    if message_type not in (None, "text"):
        return "media_context"
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", None))
    if str(platform or "").lower() == "discord":
        return "dynamic_voice_context"
    return None


async def decide_fast_lane(*, event, source, history, session_entry, config,
                           was_auto_reset, is_new_session, pending_sidecar=False):
    cfg = _config(config)
    rows = len(history or [])
    tokens = int(getattr(session_entry, "last_prompt_tokens", 0) or 0)
    if cfg is None:
        return _decision(False, "config_unavailable", tokens, rows)
    if not cfg["enabled"]:
        return _decision(False, "disabled", tokens, rows)

    reason = _static_block_reason(
        event, source, is_new_session=is_new_session,
        was_auto_reset=was_auto_reset, pending_sidecar=pending_sidecar,
    )
    if reason:
        return _decision(False, reason, tokens, rows)

    text = getattr(event, "text", None)
    if not isinstance(text, str) or not text.strip() or len(text) > cfg["max_message_chars"]:
        return _decision(False, "message_shape", tokens, rows)

    if rows < cfg["min_history_rows"] and tokens < cfg["min_prompt_tokens"]:
        return _decision(False, "short_history", tokens, rows)

    classifier_prompt = (
        "Decide whether the CURRENT USER MESSAGE can be correctly understood and acted on "
        "without any earlier conversation. Do not answer it and do not choose or route tools. "
        "Return JSON only with keys self_contained, confidence, reason. "
        "reason must be self_contained, needs_prior_context, or ambiguous. "
        "True requires every referent, target, constraint, and requested continuation to be fully "
        "specified in the current message. If earlier dialogue could materially change the meaning, "
        "return false. Any uncertainty means ambiguous and false."
    )
    started = time.monotonic()
    try:
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
        response = await asyncio.wait_for(
            async_call_llm(
                task="gateway_fast_lane_classifier",
                messages=[
                    {"role": "system", "content": classifier_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=80,
                timeout=cfg["timeout_s"],
            ),
            timeout=cfg["timeout_s"] + 0.25,
        )
        raw = extract_content_or_reasoning(response).strip()
        fence = chr(96) * 3
        if raw.startswith(fence) and raw.endswith(fence):
            raw = raw.strip(chr(96)).strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        obj = json.loads(raw)
        self_contained = obj.get("self_contained")
        confidence = float(obj.get("confidence", 0.0))
        reason = str(obj.get("reason") or (
            "self_contained" if self_contained is True else "needs_prior_context"
        )).strip()
        if reason not in {"self_contained", "needs_prior_context", "ambiguous"}:
            return _decision(False, "classifier_error", tokens, rows, started, confidence)
        if self_contained is True and confidence >= cfg["min_confidence"] and reason == "self_contained":
            return _decision(True, "self_contained", tokens, rows, started, confidence)
        return _decision(False, "classifier_context_dependent", tokens, rows, started, confidence)
    except Exception:
        logger.debug("gateway fast-lane classifier failed; using full history", exc_info=True)
        return _decision(False, "classifier_error", tokens, rows, started, 0.0)
