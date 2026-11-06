# -*- coding: utf-8 -*-
"""F13 regression: thread-targeted sends must never hit create with receive_id_type="thread_id".

Root cause (1105 journal, 99992402 x3, live+standalone same root): the create API has no
"thread_id" enum — every unanchored thread send was silently lost. Fix: anchor a reply on the
thread's last message (audio-path trick generalized to all msg_types); no anchor → chat-level
send + WARN (visible beats lost). Upstream: hermes-agent#142955.
"""
import asyncio
import inspect
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter

THREAD = "omt_19894ddec48f1a50"
CHAT = "oc_b044615532e0d934ae8198743707549c"
ANCHOR = "om_anchor_last_in_thread"


def _ok():
    return SimpleNamespace(success=lambda: True, code=0, msg="ok",
                           data=SimpleNamespace(message_id="om_sent"))


def _make_adapter(fetch_anchor):
    adapter = FeishuAdapter(PlatformConfig())
    calls = {"reply": [], "create": []}

    class _MessageAPI:
        def reply(self, request):
            calls["reply"].append(request)
            return _ok()

        def create(self, request):
            calls["create"].append(request)
            return _ok()

    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI())))

    async def _fake_fetch(thread_id):
        assert thread_id == THREAD
        return fetch_anchor

    adapter._fetch_last_message_in_thread = _fake_fetch
    return adapter, calls


def _send_raw(adapter, **over):
    kwargs = dict(chat_id=CHAT, msg_type="text",
                  payload=json.dumps({"text": "F13回归"}, ensure_ascii=False),
                  reply_to=None, metadata={"thread_id": THREAD})
    kwargs.update(over)
    return asyncio.run(adapter._send_raw_message(**kwargs))


class TestF13ThreadRouting(unittest.TestCase):
    def test_thread_send_anchors_reply_not_create(self):
        """Unanchored thread send → reply on thread's last message; create never called."""
        adapter, calls = _make_adapter(ANCHOR)
        resp = _send_raw(adapter)
        self.assertTrue(resp.success())
        self.assertEqual(len(calls["reply"]), 1)
        self.assertEqual(calls["reply"][0].message_id, ANCHOR)
        self.assertTrue(calls["reply"][0].request_body.reply_in_thread)
        self.assertEqual(calls["reply"][0].request_body.msg_type, "text")
        self.assertEqual(calls["create"], [])

    def test_thread_send_no_anchor_degrades_chat_level(self):
        """Anchor unfetchable → chat-level create (WARN path); never receive_id_type=thread_id."""
        adapter, calls = _make_adapter(None)
        resp = _send_raw(adapter)
        self.assertTrue(resp.success())
        self.assertEqual(calls["reply"], [])
        self.assertEqual(len(calls["create"]), 1)
        self.assertEqual(calls["create"][0].receive_id_type, "chat_id")
        self.assertEqual(calls["create"][0].request_body.receive_id, CHAT)

    def test_audio_msg_type_also_anchors(self):
        """Fix generalizes beyond text: audio msg_type takes the same anchor path."""
        adapter, calls = _make_adapter(ANCHOR)
        _send_raw(adapter, msg_type="audio", payload=json.dumps({"file_key": "fk"}))
        self.assertEqual(len(calls["reply"]), 1)
        self.assertEqual(calls["reply"][0].request_body.msg_type, "audio")
        self.assertEqual(calls["create"], [])

    def test_explicit_reply_to_still_wins(self):
        """metadata reply_to_message_id keeps the pre-existing direct reply path (no anchor fetch)."""
        adapter, calls = _make_adapter(ANCHOR)
        fetched = []

        async def _spy(thread_id):
            fetched.append(thread_id)
            return ANCHOR
        adapter._fetch_last_message_in_thread = _spy
        _send_raw(adapter, metadata={"thread_id": THREAD, "reply_to_message_id": "om_explicit"})
        self.assertEqual(fetched, [])
        self.assertEqual(calls["reply"][0].message_id, "om_explicit")
        self.assertEqual(calls["create"], [])

    def test_non_thread_send_unchanged(self):
        """No thread_id → plain chat create, exactly as before."""
        adapter, calls = _make_adapter(ANCHOR)
        _send_raw(adapter, metadata=None)
        self.assertEqual(calls["reply"], [])
        self.assertEqual(len(calls["create"]), 1)
        self.assertEqual(calls["create"][0].receive_id_type, "chat_id")

    def test_buggy_enum_gone_from_source(self):
        """Source-level kill switch: _send_raw_message must never route create via thread_id enum."""
        src = inspect.getsource(FeishuAdapter._send_raw_message)
        self.assertNotIn('receive_id, receive_id_type = thread_id', src)
        self.assertIn("_fetch_last_message_in_thread", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
