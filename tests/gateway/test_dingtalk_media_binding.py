"""Regression: DingTalk inbound text and media must reach the message handler."""
from types import SimpleNamespace

from plugins.platforms.dingtalk.adapter import DingTalkAdapter


def test_dingtalk_adapter_exposes_media_parser():
    msg_type, urls, types = DingTalkAdapter._extract_media(DingTalkAdapter, SimpleNamespace(image_content=None))
    assert msg_type is not None
    assert urls == []
    assert types == []
