"""Feishu platform package compatibility boundary."""

from __future__ import annotations

import asyncio
from typing import Any

from . import adapter as _adapter


async def _read_limited_webhook_body(request: Any, max_bytes: int) -> bytes:
    """Read a webhook body with a hard size cap across aiohttp and test adapters.

    Real ``aiohttp.web.Request`` objects expose ``content.readexactly`` and stay
    on the bounded streaming path. Lightweight request adapters used by tests
    and integrations may expose only ``read``; that fallback is still checked
    against the same byte limit before any JSON parsing occurs.
    """

    content = getattr(request, "content", None)
    readexactly = getattr(content, "readexactly", None)
    if callable(readexactly):
        try:
            body = await readexactly(max_bytes + 1)
        except asyncio.IncompleteReadError as exc:
            body = exc.partial
    else:
        read = getattr(request, "read", None)
        if not callable(read):
            raise ValueError("request body reader unavailable")
        body = await read()

    if not isinstance(body, (bytes, bytearray)):
        raise ValueError("request body must be bytes")
    if len(body) > max_bytes:
        raise ValueError("payload too large")
    return bytes(body)


async def _connect_webhook_without_duplicate_body_limit(self: Any) -> None:
    """Start webhook mode while keeping body limiting in one tested layer."""

    if not _adapter.FEISHU_WEBHOOK_AVAILABLE:
        raise RuntimeError("aiohttp not installed; webhook mode unavailable")
    domain = (
        _adapter.FEISHU_DOMAIN
        if self._domain_name != "lark"
        else _adapter.LARK_DOMAIN
    )
    self._client = self._build_lark_client(domain)
    self._event_handler = self._build_event_handler()
    if self._event_handler is None:
        raise RuntimeError("failed to build Feishu event handler")
    await self._hydrate_bot_identity()

    # _handle_webhook_request reads through _read_limited_webhook_body, so the
    # 1 MiB cap remains enforced without requiring every aiohttp-compatible
    # Application factory or test double to accept client_max_size.
    app = _adapter.web.Application()
    app.router.add_post(self._webhook_path, self._handle_webhook_request)
    self._webhook_runner = _adapter.web.AppRunner(app)
    await self._webhook_runner.setup()
    self._webhook_site = _adapter.web.TCPSite(
        self._webhook_runner,
        self._webhook_host,
        self._webhook_port,
    )
    await self._webhook_site.start()


_adapter._read_limited_feishu_webhook_body = _read_limited_webhook_body
_adapter.FeishuAdapter._connect_webhook = _connect_webhook_without_duplicate_body_limit

register = _adapter.register

__all__ = ["register"]
