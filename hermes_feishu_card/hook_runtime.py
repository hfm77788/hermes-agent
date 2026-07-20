from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)
_CARD_PLATFORMS = {"feishu", "lark"}


def build_event(event_name: str, local_vars: Optional[Dict[str, Any]] = None, preview: bool = False) -> Optional[Dict[str, Any]]:
    return {
        "event": event_name,
        "preview": bool(preview),
        "data": {},
    }


def emit_from_hermes_locals(local_vars: Dict[str, Any], event_name: str = "message.started") -> bool:
    return False


async def emit_from_hermes_locals_async(local_vars: Dict[str, Any], event_name: str = "message.completed") -> bool:
    if event_name != "message.completed":
        return False
    try:
        source = local_vars.get("source")
        event = local_vars.get("event")
        gateway = local_vars.get("self")
        answer = local_vars.get("answer") or local_vars.get("response") or ""
        if not gateway or not source or not answer:
            return False
        platform = getattr(source.platform, "value", source.platform)
        normalized = str(platform or "").lower()
        if normalized not in _CARD_PLATFORMS:
            return False
        adapter = None
        adapters = getattr(gateway, "adapters", {}) or {}
        adapter = adapters.get(getattr(source, "platform", None))
        if adapter is None:
            adapter = adapters.get(normalized)
        if adapter is None:
            for key, value in adapters.items():
                key_norm = str(getattr(key, "value", key) or "").lower()
                if key_norm == normalized:
                    adapter = value
                    break
        if adapter is None:
            return False
        reply_anchor = gateway._reply_anchor_for_event(event)
        metadata = gateway._thread_metadata_for_source(source, reply_anchor)
        result = await adapter.send(source.chat_id, answer, metadata=metadata)
        ok = bool(getattr(result, "success", False))
        if not ok:
            logger.warning("[hermes-feishu-card] completed-send error=%s", getattr(result, "error", None))
        return ok
    except Exception as exc:
        logger.warning("[hermes-feishu-card] emit_async exception: %s: %s", exc.__class__.__name__, exc)
        return False


def emit_from_hermes_locals_threadsafe(local_vars: Dict[str, Any], event_name: str = "message.progress") -> bool:
    return False


def should_suppress_native_response(platform: Any, card_delivered: bool, attachments: Any = None) -> bool:
    normalized = getattr(platform, "value", platform)
    return bool(card_delivered and normalized in _CARD_PLATFORMS)


def request_clarify_response_from_hermes_locals(local_vars: Dict[str, Any], prompt_id: str | None = None) -> bool:
    return False


def request_approval_choice_from_hermes_locals(local_vars: Dict[str, Any], approval_id: str | None = None) -> bool:
    return False


def emit_cron_delivery(local_vars: Dict[str, Any]) -> bool:
    return False
