"""Trusted local sidecar -> live Gateway inbound injection.

Used when a sidecar can observe platform traffic the native bot transport does not
receive (DingTalk non-mention group messages). The turn runs in the live profile
session; presentation stays muted so the sidecar remains the single delivery path.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import Platform

logger = logging.getLogger(__name__)


_MAX_TEXT_CHARS = 8000
_MAX_CONTEXT_CHARS = 12000
_MAX_ID_CHARS = 1024
_RESULT_CACHE_MAX = 500
#: Bounded wait after the dispatching handler returns, covering a response future
#: that lands just after it. The egress publishes strictly before the authority
#: drain settles the admission waiter, so the happy path resolves ahead of this
#: wait; it only burns when the hand-off genuinely failed (repair t_479a04ac).
_RESPONSE_RENDEZVOUS_GRACE_SECONDS = 5.0
_MEDIA_MAX_ATTACHMENTS = 8
_IMAGE_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
}


def _clean(value: Any, *, limit: int = _MAX_ID_CHARS) -> str:
    return str(value or "").strip()[:limit]


def _injected_slash_opens_command_lane(text: str, *, plugin_home: Any = None) -> bool:
    """True only when sidecar-injected slash text targets a registered gateway command.

    Repair t_a032be60 (diagnosis t_6196a57c): the hardcoded gate False kept the
    health check's ``/new`` a model turn (fake 「已开新会话。」 receipts while the
    session never reset). The gate opens for exactly the text the native command
    chain itself would dispatch, mirroring that entry predicate end to end:
    ``MessageEvent.get_command`` parsing (platforms/event.py — first token,
    strip ``/``, lower, cut at ``@``, reject path-like pseudo names), alias
    canonicalisation (``resolve_command``, as ``_hm_resolve_command`` does in
    run_inbound.py), and the ``is_gateway_known_command`` whitelist. Plain text,
    unknown names and skill slashes keep the gate closed: they stay model-lane
    chat with the rendezvous grace wait intact. Opening the gate grants nothing —
    the resolved command still passes ``_check_slash_access``, command hooks and
    busy policy.

    Audit B2 (round 2): the PLUGIN half of that whitelist is resolved from the
    ACTIVE Hermes home's manager (plugins.py ``_plugin_home_key``), while the
    command dispatches under the TARGET profile's runtime scope
    (``run_adapters.py::_make_profile_message_handler``). Evaluating the gate
    under the caller's home alone false-opens launch-only plugin verbs and
    false-closes target-only ones. ``plugin_home`` (resolved by the caller from
    the routing chain) rebinds the plugin-table query into the target profile's
    scope, so gate and dispatch read the same registry; built-in names answer
    from the static set first and cost nothing. Round-3 audit: when the TARGET
    table itself cannot be read the gate CLOSES for plugin verbs — guessing from
    the caller's table could false-open a transport-only name, so an unreadable
    target is treated as unregistered (model lane, the pre-repair behaviour).
    """
    parts = (text or "").lstrip().split(maxsplit=1)
    if not parts or not parts[0].startswith("/"):
        return False
    raw = parts[0][1:].lower().split("@", 1)[0]
    if not raw or "/" in raw:
        return False
    from hermes_cli.commands import (
        GATEWAY_KNOWN_COMMANDS,
        is_gateway_known_command,
        resolve_command,
    )

    cmd_def = resolve_command(raw)
    name = cmd_def.name if cmd_def else raw
    if name in GATEWAY_KNOWN_COMMANDS or plugin_home is None:
        return is_gateway_known_command(name)
    try:
        from agent.runtime_scope import bind_profile_scope

        with bind_profile_scope(plugin_home, hydrate_secrets=False):
            return is_gateway_known_command(name)
    except Exception:
        logger.debug(
            "plugin registry scope bind refused; the plugin verb is treated as unregistered",
            exc_info=True,
        )
        return False


def _injection_dispatch_runtime_home(runner: Any, profile: str, source: Any) -> Path | None:
    """The RUNTIME home the injected command actually dispatches under; None = the
    caller's own scope IS that home.

    Round-3 audit (B2): ``run_adapters.py::_make_profile_message_handler``
    canonicalizes the source FIRST and dispatches under
    ``_routed_runtime_home(identity, profile_home)`` — a shared bot whose route
    targets a satellite runs in the ROUTED profile's scope, so the receiving
    bot's profile home alone is the wrong table (transport-only plugin verbs
    false-open, runtime-only verbs false-close). This mirrors the same seams for
    BOTH halves of the repair: the plugin-registry gate query and the receipt
    publish run in this home's scope, never in a proxy for it.

    None keeps caller-scope evaluation, which is exact when: the runner is a bare
    rig without the routing seams (its handler runs unscoped too), the resolved
    home IS this process's home (the common classroom path pays zero binding
    cost), or the routed home is the ``UNRESOLVED_PROFILE_HOME`` sentinel (the
    handler binds a nullcontext, so the caller's table IS the dispatch table). A
    resolution exception also answers None: the caller table is the strictly
    narrower pre-repair behaviour, never a widened surface.
    """
    from gateway.run_adapters import UNRESOLVED_PROFILE_HOME
    from hermes_constants import get_hermes_home, hermes_home_key

    canonicalize = getattr(runner, "_canonicalize", None)
    routed_profile = getattr(runner, "_routed_profile_home", None)
    routed_runtime = getattr(runner, "_routed_runtime_home", None)
    if not (callable(canonicalize) and callable(routed_profile) and callable(routed_runtime)):
        return None
    try:
        primary = getattr(runner, "_primary_profile_name", None) or "default"
        if profile in {"default", primary}:
            # ``_make_default_profile_message_handler`` shape: the admitted
            # primary source dispatches in its identity's runtime home.
            admit = getattr(runner, "_admit_primary_source", None)
            if not callable(admit):
                return None
            default_home = Path(get_hermes_home())
            home = admit(source, default_home) or default_home
        else:
            profile_home = routed_profile(profile)
            identity = canonicalize(source, transport_profile=profile)
            home = routed_runtime(identity, profile_home)
    except Exception:
        logger.debug("dispatch runtime home does not resolve; caller scope is used", exc_info=True)
        return None
    # ``UNRESOLVED_PROFILE_HOME`` and ``None`` both mean "no distinct scope to
    # bind" — isinstance is the narrowing, never a name equality with the
    # sentinel's class.
    if not isinstance(home, (str, Path)) or home is UNRESOLVED_PROFILE_HOME:
        return None
    try:
        if hermes_home_key(home) == hermes_home_key(get_hermes_home()):
            return None
    except Exception:
        return None
    return Path(home)


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
    raw_media = list(params.get("media_urls") or [])[:_MEDIA_MAX_ATTACHMENTS]
    raw_types = list(params.get("media_types") or [])[: len(raw_media)]
    media_root = (
        Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
        / "state"
        / "dingtalk-free-response-media"
    ).resolve()
    media_urls: list[str] = []
    media_types: list[str] = []
    for index, value in enumerate(raw_media):
        path = Path(_clean(value, limit=4096)).expanduser().resolve()
        if not path.is_relative_to(media_root) or not path.is_file():
            return {"accepted": False, "reason": "invalid_media_path"}
        media_type = _clean(
            raw_types[index] if index < len(raw_types) else "image",
            limit=64,
        ).lower()
        if media_type in {"", "image"}:
            media_type = _IMAGE_MIME_BY_SUFFIX.get(path.suffix.lower(), "")
        if not media_type.startswith("image/"):
            return {"accepted": False, "reason": "unsupported_media_type"}
        media_urls.append(str(path))
        media_types.append(media_type)
    if media_urls and not text:
        text = "请结合图片内容和当前课堂上下文回答。"

    chat_id = _clean(params.get("chat_id"))
    user_id = _clean(params.get("user_id"))
    user_id_alt = _clean(params.get("user_id_alt"))
    message_id = _clean(params.get("message_id"))
    if not ((text or media_urls) and chat_id and user_id and message_id):
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
    response_future = asyncio.get_running_loop().create_future()
    source._local_inbound_response_future = response_future

    skill = _clean(params.get("skill"))
    # Keep the trusted sidecar's explicit tutoring identity process-local. The
    # gateway uses it only to apply the fast-tutoring runtime guard; it is never
    # persisted into SessionSource or accepted from public platform payloads.
    source._local_inbound_skill = skill
    source._local_inbound_has_media = bool(media_urls)
    # Round-3 audit (B2): resolve the command lane's dispatch scope ONCE, through
    # the same identity seams the profile message handler uses, and keep BOTH the
    # gate query and the receipt publish inside it. Canonicalizing here is the
    # gateway's own ingress invariant (the handler's repeat call is a no-op on the
    # pinned identity).
    dispatch_home = _injection_dispatch_runtime_home(runner, profile, source)
    auto_skill = [skill] if skill else (
        adapter._resolve_channel_skills(chat_id)
        if hasattr(adapter, "_resolve_channel_skills") else None
    )
    event = MessageEvent(
        text=text,
        message_type=MessageType.PHOTO if media_urls else MessageType.TEXT,
        user_id=source.user_id,
        user_name=source.user_name,
        source=source,
        message_id=message_id,
        media_urls=media_urls,
        media_types=media_types,
        auto_skill=auto_skill,
        channel_prompt=(
            adapter._resolve_channel_prompt(chat_id)
            if hasattr(adapter, "_resolve_channel_prompt") else None
        ),
        # Command-lane gate (repair t_a032be60): opens ONLY for injected text
        # that names a registered gateway command, queried under the TARGET
        # profile's plugin registry (audit B2; see
        # _injected_slash_opens_command_lane). The hardcoded False made
        # ``is_command()``/``get_command()`` (platforms/event.py) permanently
        # None, so the health check's /new ran as chat text and the model
        # answered with fake 「已开新会话。」 receipts while the session never
        # reset (t_6196a57c). Unknown slash names, path-like text and plain
        # chat keep the pre-repair model-lane behaviour; every gated command
        # still faces the native ``is_gateway_known_command`` whitelist, the
        # ``_check_slash_access`` policy and command hooks in the chain.
        allow_gateway_control=_injected_slash_opens_command_lane(
            text, plugin_home=dispatch_home
        ),
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

    # Process-local response rendezvous. The session-authority lane re-executes this
    # turn from the durable snapshot (``session_envelope.restore_native`` rebuilds a
    # SessionSource via ``from_dict``), so the wire-invisible future attribute above
    # never reaches that turn's egress. The message id IS a wire field and survives
    # the snapshot: the publish falls back to this registry keyed by it (repair
    # t_479a04ac — the ``done()``-only read back "" on every durable-admission turn).
    waiters = getattr(runner, "_local_inbound_response_waiters", None)
    if not isinstance(waiters, dict):
        waiters = runner._local_inbound_response_waiters = {}
    if message_id:
        waiters[message_id] = response_future

    started = time.monotonic()
    # v0.22.4 native admission only mints provenance inside the adapter-wired
    # message handler (``_canonicalize`` + ``native_callback``). Calling
    # ``runner._handle_message`` directly bypasses that ingress: under
    # ``multiplex_profiles`` the gate stamps ``_authorization_profile_home``,
    # ``capture_provenance`` then sees none and the turn is refused with
    # ``invalid_params (authorization_home_without_provenance)`` — every bridge
    # injection and the daily health probe read back an empty response
    # (repair card t_b2316993, 2026-10-23). Dispatch through the same handler
    # instance the live platform delivers to; a bare test adapter carrying no
    # wired handler keeps the historical direct call.
    try:
        handler: Any = getattr(adapter, "_message_handler", None)
        if callable(handler):
            dispatch_receipt = await handler(event)
        else:
            dispatch_receipt = await runner._handle_message(event)
        # Single source of truth with the gate above: only text actually routed
        # into the command lane skips the grace park (it answers without a
        # publish-side agent turn — repair t_479a04ac). The old
        # ``text.startswith("/")`` predicate also skipped it for unknown-slash
        # model turns, dropping their receipts (t_6196a57c §2.2).
        _is_command_lane = bool(event.allow_gateway_control)
        if _is_command_lane and isinstance(dispatch_receipt, str) and dispatch_receipt:
            # Audit B1: a native command's answer rides the handler's RETURN value
            # (the profile handler forwards ``_handle_message``'s receipt); no
            # agent-turn egress publishes it, so the sidecar used to get
            # ``accepted=True, response=""`` after a real reset. Deliver it through
            # the SAME production seam the egress uses: one-shot guarded
            # (``future.done()`` — no double publish, no override of an egress
            # receipt) and run through the normal chat-boundary sanitisation.
            # Round-3 audit: the seam's sanitisation is scope-sensitive (the phone
            # policy reads the ACTIVE profile, the secret-source redaction table is
            # keyed by the active ``hermes_home_key``), and the handler has already
            # LEFT the target runtime scope by the time its receipt returns here.
            # Publish inside the same scope the command dispatched under — never a
            # caller-scope sanitisation of a target-profile receipt.
            from gateway.run_turn import _publish_local_inbound_response

            if dispatch_home is None:
                _publish_local_inbound_response(
                    event.source, {"final_response": dispatch_receipt}, runner=runner
                )
            else:
                from gateway.run import _async_profile_runtime_scope

                async with _async_profile_runtime_scope(dispatch_home):
                    _publish_local_inbound_response(
                        event.source, {"final_response": dispatch_receipt}, runner=runner
                    )
        _wait_started = time.monotonic()
        response_timed_out = False
        if not response_future.done() and not _is_command_lane:
            # Slash-command lanes answer without a publish-side agent turn — never
            # park the grace window on them (repair t_479a04ac).
            try:
                await asyncio.wait_for(
                    asyncio.shield(response_future),
                    timeout=_RESPONSE_RENDEZVOUS_GRACE_SECONDS,
                )
            except TimeoutError:
                response_timed_out = True
        response_wait_ms = round((time.monotonic() - _wait_started) * 1000)
        response = response_future.result() if response_future.done() else ""
        final_entry = (
            getattr(getattr(runner, "session_store", None), "_entries", None) or {}
        ).get(session_key)
        result = {
            "accepted": True,
            "session_key": session_key,
            "session_id": str(getattr(final_entry, "session_id", "") or ""),
            "response": str(response or "").strip(),
            "response_wait_ms": response_wait_ms,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
        if response_timed_out:
            result["reason"] = "response_timeout"
        return result
    finally:
        # A raising dispatch must not leave a rendezvous entry behind either —
        # the registry lives for the runner's lifetime (repair t_479a04ac).
        if message_id and waiters.get(message_id) is response_future:
            waiters.pop(message_id, None)


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
            # Only accepted results are idempotency-cacheable. A rejection
            # (e.g. session_busy after the idle wait) executed no user turn,
            # so caching it poisoned every later retry of the same message id
            # and dead-lettered the student message within one poll cadence
            # (hema repair 2026-10-14). Rejections must fall through to a
            # fresh attempt once the session goes idle.
            if result.get("accepted"):
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

def _resolver_figures(result: Any) -> tuple[Any, Any, Any]:
    """Normalise ``_resolve_context_figures`` arity for the health/maintenance verb.

    v0.22 returned ``(used, context_length, model)``; the v0.23 usage-window
    plan support (upstream ``15307dece7``, landed via the v0.23.0 merge) widened
    it to ``(used, context_length, model, window_suffix)``. Unpacking a fixed
    3-tuple from a 4-tuple raised ``ValueError: too many values to unpack`` in
    every ``local-session-health`` verb that touched a real session, which the
    verb handler surfaced as ``control_failed:ValueError`` and the daily hema
    health check logged as eight ``context_probe_failed`` signals (2026-11-07
    ticket t_4b63e535). The suffix is only a ``/context`` display decoration —
    take the first three figures and stay forward-compatible.
    """
    used, total, model = result[0], result[1], result[2]
    return used, total, model


async def _local_session_health(runner, params: dict[str, Any]) -> dict[str, Any]:
    """Inspect or maintain one existing local session without creating a user turn."""
    profile = _clean(params.get("profile")) or "default"
    if _clean(params.get("platform")).lower() != "dingtalk":
        return {"accepted": False, "reason": "unsupported_platform"}

    chat_id = _clean(params.get("chat_id"))
    user_id = _clean(params.get("user_id"))
    user_id_alt = _clean(params.get("user_id_alt"))
    if not (chat_id and user_id):
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
        message_id=None,
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.TEXT,
        user_id=source.user_id,
        user_name=source.user_name,
        source=source,
    )
    session_key = adapter._event_session_key(event)
    entries = getattr(getattr(runner, "session_store", None), "_entries", None) or {}
    entry = entries.get(session_key)
    if entry is None:
        return {"accepted": True, "found": False, "session_key": session_key}

    adapter_busy = session_key in getattr(adapter, "_active_sessions", {})
    runner_busy = bool(
        callable(getattr(runner, "_is_session_running", None))
        and runner._is_session_running(session_key)
    )
    busy = bool(adapter_busy or runner_busy)
    resident = (
        runner._resident_agent_for(session_key)
        if callable(getattr(runner, "_resident_agent_for", None))
        else None
    )
    ctx = getattr(resident, "context_compressor", None) if resident is not None else None
    resolver = getattr(runner, "_resolve_context_figures", None)
    if callable(resolver):
        used, total, model = _resolver_figures(await resolver(resident, ctx, entry, source))
    else:
        used = max(0, int(getattr(entry, "last_prompt_tokens", 0) or 0))
        total = 0
        model = ""
    used = max(0, int(used or 0))
    total = max(0, int(total or 0))

    def snapshot(**extra: Any) -> dict[str, Any]:
        return {
            "accepted": True,
            "found": True,
            "session_key": session_key,
            "session_id": str(getattr(entry, "session_id", "") or ""),
            "busy": busy,
            "last_prompt_tokens": used,
            "context_length": total,
            "context_pct": round((used / total) * 100, 1) if total else None,
            "model": str(model or ""),
            **extra,
        }

    action = _clean(params.get("action"), limit=64).lower() or "inspect"
    if action == "inspect":
        return snapshot()
    if busy:
        return snapshot(action=action, changed=False, reason="session_busy")

    store = getattr(runner, "async_session_store", None)
    if store is None:
        return snapshot(action=action, changed=False, reason="session_store_unavailable")

    if action != "compress":
        return snapshot(action=action, changed=False, reason="unsupported_action")

    try:
        history = await store.load_transcript(entry.session_id)
    except Exception as exc:
        return snapshot(
            action=action,
            changed=False,
            reason=f"transcript_read_failed:{type(exc).__name__}",
        )
    from agent.conversation_compression_manual import MIN_MESSAGES, parse_compress_args
    messages = [
        row for row in history
        if isinstance(row, dict) and row.get("role") in {"user", "assistant", "tool"}
    ]
    if len(messages) < MIN_MESSAGES:
        return snapshot(action=action, changed=False, reason="not_enough_messages")

    request = parse_compress_args("here 6")
    try:
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(runner._resolve_profile_home_for_source(source)):
            await runner._run_manual_compression(source, entry, history, request)
    except Exception as exc:
        return snapshot(
            action=action,
            changed=False,
            reason=f"compression_failed:{type(exc).__name__}",
        )

    refreshed_entry = (
        (getattr(getattr(runner, "session_store", None), "_entries", None) or {}).get(session_key)
        or entry
    )
    refreshed_resident = (
        runner._resident_agent_for(session_key)
        if callable(getattr(runner, "_resident_agent_for", None))
        else None
    )
    refreshed_ctx = (
        getattr(refreshed_resident, "context_compressor", None)
        if refreshed_resident is not None else None
    )
    if callable(resolver):
        after_used, after_total, after_model = _resolver_figures(await resolver(
            refreshed_resident, refreshed_ctx, refreshed_entry, source
        ))
    else:
        after_used = max(0, int(getattr(refreshed_entry, "last_prompt_tokens", 0) or 0))
        after_total = total
        after_model = model
    after_used = max(0, int(after_used or 0))
    after_total = max(0, int(after_total or 0))
    return {
        "accepted": True,
        "found": True,
        "session_key": session_key,
        "session_id": str(getattr(refreshed_entry, "session_id", "") or ""),
        "busy": False,
        "action": "compress",
        "changed": after_used < used,
        "compressed": after_used < used,
        "before_prompt_tokens": used,
        "last_prompt_tokens": after_used,
        "context_length": after_total,
        "context_pct": round((after_used / after_total) * 100, 1) if after_total else None,
        "model": str(after_model or ""),
    }


def local_session_health_verb(runner, loop):
    """Control socket handler for zero-presentation session inspection and maintenance."""
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        action = _clean(params.get("action"), limit=64).lower() or "inspect"
        timeout = 185.0 if action == "compress" else 20.0
        future = asyncio.run_coroutine_threadsafe(
            _local_session_health(runner, params), loop
        )
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            return {
                "accepted": False,
                "reason": f"control_failed:{type(exc).__name__}",
            }

    return handler
