"""Trusted local sidecar -> live Gateway inbound injection.

Used when a sidecar can observe platform traffic the native bot transport does not
receive (DingTalk non-mention group messages). The turn runs in the live profile
session; presentation stays muted so the sidecar remains the single delivery path.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import Platform


_MAX_TEXT_CHARS = 8000
_MAX_CONTEXT_CHARS = 12000
_MAX_ID_CHARS = 1024
_RESULT_CACHE_MAX = 500


def _clean(value: Any, *, limit: int = _MAX_ID_CHARS) -> str:
    return str(value or "").strip()[:limit]


def _profile_adapter(runner, profile: str, platform: Platform):
    primary = getattr(runner, "_primary_profile_name", None) or "default"
    if profile in {"default", primary}:
        return (getattr(runner, "adapters", None) or {}).get(platform)
    return ((getattr(runner, "_profile_adapters", None) or {}).get(profile) or {}).get(platform)


async def _wait_until_idle(runner, adapter, session_key: str, wait_seconds: float) -> bool:
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while True:
        adapter_busy = session_key in getattr(adapter, "_active_sessions", {})
        runner_busy = bool(
            callable(getattr(runner, "_is_session_running", None))
            and runner._is_session_running(session_key)
        )
        if not adapter_busy and not runner_busy:
            return True
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.1)
async def _execute_local_inbound(runner, params: dict[str, Any]) -> dict[str, Any]:
    profile = _clean(params.get("profile")) or "default"
    if _clean(params.get("platform")).lower() != "dingtalk":
        return {"accepted": False, "reason": "unsupported_platform"}

    text = _clean(params.get("text"), limit=_MAX_TEXT_CHARS)
    chat_id = _clean(params.get("chat_id"))
    user_id = _clean(params.get("user_id"))
    user_id_alt = _clean(params.get("user_id_alt"))
    message_id = _clean(params.get("message_id"))
    if not (text and chat_id and user_id and message_id):
        return {"accepted": False, "reason": "missing_required_field"}

    adapter = _profile_adapter(runner, profile, Platform.DINGTALK)
    if adapter is None:
        return {"accepted": False, "reason": "adapter_unavailable"}

    source = adapter.build_source(
        chat_id=chat_id,
        chat_name=_clean(params.get("chat_name")) or None,
        chat_type=_clean(params.get("chat_type")) or "group",
        user_id=user_id,
        user_name=_clean(params.get("user_name")) or user_id,
        user_id_alt=user_id_alt or None,
        message_id=message_id,
    )
    # Process-local trust markers. They never serialize into SessionSource.
    source._suppress_presentation = True
    source._trusted_context_tail = True

    skill = _clean(params.get("skill"))
    auto_skill = [skill] if skill else (
        adapter._resolve_channel_skills(chat_id)
        if hasattr(adapter, "_resolve_channel_skills") else None
    )
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        user_id=source.user_id,
        user_name=source.user_name,
        source=source,
        message_id=message_id,
        auto_skill=auto_skill,
        channel_prompt=(
            adapter._resolve_channel_prompt(chat_id)
            if hasattr(adapter, "_resolve_channel_prompt") else None
        ),
        allow_gateway_control=False,
    )

    recent_context = _clean(params.get("recent_context"), limit=_MAX_CONTEXT_CHARS)
    if bool(params.get("force_context")) and recent_context:
        event.channel_context = recent_context
    # Derive through the live adapter so profile namespace/user isolation match native ingress.
    session_key = adapter._event_session_key(event)
    try:
        wait_seconds = float(params.get("wait_for_idle_seconds") or 120.0)
    except (TypeError, ValueError):
        wait_seconds = 120.0
    wait_seconds = min(max(wait_seconds, 0.0), 180.0)
    if not await _wait_until_idle(runner, adapter, session_key, wait_seconds):
        return {"accepted": False, "reason": "session_busy", "session_key": session_key}

    started = time.monotonic()
    response = await runner._handle_message(event)
    final_entry = (
        getattr(getattr(runner, "session_store", None), "_entries", None) or {}
    ).get(session_key)
    return {
        "accepted": True,
        "session_key": session_key,
        "session_id": str(getattr(final_entry, "session_id", "") or ""),
        "response": str(response or "").strip(),
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }


async def inject_local_inbound(runner, params: dict[str, Any]) -> dict[str, Any]:
    """Idempotently execute/join one sidecar-observed message by platform message id."""
    message_id = _clean(params.get("message_id"))
    cache = getattr(runner, "_local_inbound_result_cache", None)
    if not isinstance(cache, dict):
        cache = runner._local_inbound_result_cache = {}
    if message_id and message_id in cache:
        return dict(cache[message_id])

    inflight = getattr(runner, "_local_inbound_inflight", None)
    if not isinstance(inflight, dict):
        inflight = runner._local_inbound_inflight = {}
    if message_id and message_id in inflight:
        return await asyncio.shield(inflight[message_id])

    task = asyncio.create_task(_execute_local_inbound(runner, params))
    if message_id:
        inflight[message_id] = task
    try:
        result = await asyncio.shield(task)
        if message_id:
            cache[message_id] = dict(result)
            while len(cache) > _RESULT_CACHE_MAX:
                cache.pop(next(iter(cache)))
        return result
    finally:
        if message_id and inflight.get(message_id) is task:
            inflight.pop(message_id, None)
def local_inbound_verb(runner, loop):
    """Control-socket handler factory; control handlers execute on a worker thread."""
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            inject_local_inbound(runner, params), loop
        )
        try:
            timeout = float(params.get("timeout_seconds") or 185.0)
        except (TypeError, ValueError):
            timeout = 185.0
        timeout = min(max(timeout, 1.0), 190.0)
        return future.result(timeout=timeout)

    return handler
