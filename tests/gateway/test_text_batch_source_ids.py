"""INVARIANT tests for merged source message_id preservation (feishu-wd card 1).

Text-batch merging (``BasePlatformAdapter._enqueue_text_event`` and the Feishu
override) concatenates chunks into one dispatched event. The watchdog message
reconciliation (ce-20260917 spec, dedup_and_correlation.merge_input, ruling 0-08)
records ``inbound_seen`` per source message_id, so the merged survivor must carry
EVERY contributing id in ``metadata['source_message_ids']`` — otherwise merged-away
ids look "never delivered" and D2 misfires.

These tests load the adapters through their real import paths and exercise the
real merge/flush machinery (no source-text assertions).
"""

import asyncio
import os
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def _event(text: str, message_id: str) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, message_id=message_id,
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_1", chat_type="dm",
                             user_id="ou_u", message_id=message_id),
    )


class _RecordingAdapter(BasePlatformAdapter):
    """Base-class text batching with dispatch recorded."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.FEISHU)
        self._text_batch_delay_seconds = 0.0
        self._text_batch_split_delay_seconds = 0.0
        self.dispatched: list = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *a: Any, **k: Any) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}

    async def handle_message(self, event: MessageEvent) -> None:
        self.dispatched.append(event)


@pytest.mark.asyncio
async def test_base_batch_merge_keeps_all_source_ids():
    """Two merged messages dispatch once, text unchanged, survivor carries both ids."""
    adapter = _RecordingAdapter()
    adapter._enqueue_text_event(_event("part one", "om_1"))
    adapter._enqueue_text_event(_event("part two", "om_2"))
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())

    assert [e.text for e in adapter.dispatched] == ["part one\npart two"]
    merged = adapter.dispatched[0]
    assert merged.metadata["source_message_ids"] == ["om_1", "om_2"]
    # The survivor's primary id keeps its original semantics (base path never advanced it).
    assert merged.message_id == "om_1"


@pytest.mark.asyncio
async def test_base_single_message_not_enriched():
    """A lone (unmerged) batch adds no source_message_ids noise."""
    adapter = _RecordingAdapter()
    adapter._enqueue_text_event(_event("solo", "om_9"))
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())
    assert adapter.dispatched[0].metadata.get("source_message_ids") is None


@pytest.mark.asyncio
async def test_base_idless_merge_still_dispatches():
    """Legacy events without message_ids (unit-test shape) must not crash the merge."""
    adapter = _RecordingAdapter()
    e1 = _event("a", "")
    e2 = _event("b", "")
    e1.message_id = e2.message_id = None
    adapter._enqueue_text_event(e1)
    adapter._enqueue_text_event(e2)
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())
    assert [e.text for e in adapter.dispatched] == ["a\nb"]
    assert "source_message_ids" not in adapter.dispatched[0].metadata


@pytest.mark.asyncio
async def test_feishu_batch_merge_keeps_source_ids_through_real_dispatch():
    """Feishu's async override advances the survivor id; the merged-away id must survive
    in metadata, and a count-limit split must not leak ids across batches."""
    os.environ["HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES"] = "2"
    try:
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter.handle_message = AsyncMock()

        async def _sleep(_delay):
            return None

        async def _run() -> None:
            with patch("plugins.platforms.feishu.adapter.asyncio.sleep", side_effect=_sleep):
                for text, mid in (("A", "om_1"), ("B", "om_2"), ("C", "om_3")):
                    await adapter._dispatch_inbound_event(_event(text, mid))
                await asyncio.gather(*adapter._pending_text_batch_tasks.values(),
                                     return_exceptions=True)

        await _run()

        assert adapter.handle_message.await_count == 2
        first = adapter.handle_message.await_args_list[0].args[0]
        second = adapter.handle_message.await_args_list[1].args[0]
        # Existing semantics preserved: merge text, id advance, source anchor move.
        assert first.text == "A\nB"
        assert first.message_id == "om_2" and first.source.message_id == "om_2"
        # New invariant: both contributing source ids are recoverable from the survivor.
        assert first.metadata["source_message_ids"] == ["om_1", "om_2"]
        # Fresh batch after the limit split carries no stale ids.
        assert second.text == "C"
        assert second.metadata.get("source_message_ids") is None
    finally:
        os.environ.pop("HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES", None)
