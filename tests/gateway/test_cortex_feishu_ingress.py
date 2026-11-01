"""Cortex owner group admission, front-of-busy routing, failure closure."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway import cortex_feishu_ingress as ingress


@pytest.fixture
def owner_group(tmp_path, monkeypatch):
    p = tmp_path / "group-registry.yaml"
    p.write_text("""
workspaces:
  - chat_id: oc_cortex_group
    chat_name: GPT 河马协作群
    status: active
    boss_open_id: ou_owner
    cortex_direct_dispatch: true
""", encoding="utf-8")
    monkeypatch.setattr(ingress, "_REGISTRY", p)
    return {
        "platform": "feishu", "chat_type": "group",
        "chat_id": "oc_cortex_group", "user_id": "ou_owner",
    }


def test_owner_oneoff_is_authenticated_without_text_auth(owner_group):
    value = ingress.authenticated_goal(
        owner_group, "这是一次性任务，请让专项 GPT Chat 执行", "om_x1234567890abc"
    )
    assert value == {
        "chat_id": "oc_cortex_group",
        "message_id": "om_x1234567890abc",
        "goal": "这是一次性任务，请让专项 GPT Chat 执行",
    }


@pytest.mark.parametrize("field,value", [
    ("platform", "wechat"), ("chat_type", "dm"),
    ("chat_id", "oc_other"), ("user_id", "ou_other"),
    ("is_bot", True),
])
def test_other_contexts_cannot_claim_owner_publishing(owner_group, field, value):
    altered = dict(owner_group, **{field: value})
    assert ingress.authenticated_goal(
        altered, "这是一次性任务，请分析", "om_x1234567890abc"
    ) is None


@pytest.mark.parametrize("msg", [
    "闲聊你好", "/model work", "出个简短解释", "",
])
def test_non_task_or_control_message_is_not_published(owner_group, msg):
    assert ingress.authenticated_goal(
        owner_group, msg, "om_x1234567890abc"
    ) is None


def test_feature_disabled_remains_opt_in(owner_group, monkeypatch, tmp_path):
    p = tmp_path / "disabled.yaml"
    p.write_text("""
workspaces:
  - chat_id: oc_cortex_group
    chat_name: GPT 河马协作群
    status: active
    boss_open_id: ou_owner
""", encoding="utf-8")
    monkeypatch.setattr(ingress, "_REGISTRY", p)
    assert ingress.authenticated_goal(
        owner_group, "这是一次性任务，请分析", "om_x1234567890abc"
    ) is None


@pytest.mark.asyncio
async def test_gateway_dispatches_before_busy_session(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig
    from unittest.mock import MagicMock
    source = SessionSource(
        platform=Platform.FEISHU, chat_id="oc_cortex_group", chat_type="group",
        user_id="ou_owner", user_name="owner",
    )
    event = MessageEvent(
        text="这是一次性任务，请分析", message_id="om_x1234567890abc", source=source,
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.FEISHU: PlatformConfig(enabled=True)})
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.FEISHU: adapter}
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner._is_user_authorized_for_source = lambda _s: True
    runner._admit_bot_message_for_source = lambda _s: True
    runner._delivery_adapter_for = lambda _s: adapter
    runner._intake_adapter_for = lambda _s: adapter
    runner._rescue_orphaned_overflow = lambda _k, _a: None
    runner._enqueue_fifo = MagicMock()
    runner._is_session_running = lambda _k: True
    runner._hm_handle_running_session_message = AsyncMock(return_value="incorrectly queued")
    mock = AsyncMock(return_value={"action": "handled", "reply": "已受理"})
    monkeypatch.setattr(ingress, "maybe_publish_owner_goal", mock)
    assert await runner._handle_message(event) == "已受理"
    runner._hm_handle_running_session_message.assert_not_awaited()
    assert mock.await_args.kwargs["message_id"] == "om_x1234567890abc"


@pytest.mark.asyncio
async def test_publisher_failure_is_not_passed_to_model(owner_group, monkeypatch):
    class BadProcess:
        returncode = 1
        async def communicate(self, value):
            return b"", b"publisher_error"
    creator = AsyncMock(return_value=BadProcess())
    monkeypatch.setattr(ingress.asyncio, "create_subprocess_exec", creator)
    ans = await ingress.maybe_publish_owner_goal(
        platform="feishu", source=owner_group,
        text="这是一次性任务，请分析", message_id="om_x1234567890abc",
    )
    assert ans["action"] == "handled"
    assert "受阻" in ans["reply"]
    assert creator.await_count == 1


@pytest.mark.asyncio
async def test_publish_result_acknowledges_not_model_activated(owner_group, monkeypatch):
    class GoodProcess:
        returncode = 0
        async def communicate(self, value):
            data = json.loads(value)
            assert data["message_id"] == "om_x1234567890abc"
            return (json.dumps({
                "task_id": "clt-cortex-chat-abcd1234567890abcd",
                "profile": "chat_handoff",
                "idempotent_replay": False,
            }).encode(), b"")
    monkeypatch.setattr(ingress.asyncio, "create_subprocess_exec", AsyncMock(return_value=GoodProcess()))
    ans = await ingress.maybe_publish_owner_goal(
        platform="feishu", source=owner_group,
        text="这是一次性任务，请分析", message_id="om_x1234567890abc",
    )
    assert "等待接管" in ans["reply"]
    assert "完成" not in ans["reply"]


def test_unparseable_registry_fails_closed(owner_group, monkeypatch, tmp_path):
    file = tmp_path / "corrupt.yaml"
    file.write_text("workspaces: [unterminated", encoding="utf-8")
    monkeypatch.setattr(ingress, "_REGISTRY", file)
    assert ingress.authenticated_goal(
        owner_group, "这是一次性任务，请分析", "om_x1234567890abc"
    ) is None

@pytest.fixture
def wechat_owner_group(tmp_path, monkeypatch):
    p = tmp_path / "registry.yaml"
    p.write_text("""
workspaces:
  - chat_id: oc_cortex_group
    chat_name: GPT 河马协作群
    status: active
    boss_open_id: ou_owner
    cortex_direct_dispatch: true
  - chat_id: oc_wechat_secretary
    chat_name: 微信 ai 秘书
    status: active
    boss_open_id: ou_director_app_owner
    seat_binding: director
    cortex_direct_dispatch: true
""", encoding="utf-8")
    monkeypatch.setattr(ingress, "_REGISTRY", p)
    return {
        "platform": "feishu", "chat_type": "group",
        "chat_id": "oc_wechat_secretary", "user_id": "ou_director_app_owner",
    }


def test_wechat_secretary_owner_task_admitted(wechat_owner_group):
    got = ingress.authenticated_goal(
        wechat_owner_group, "任务：只读检查微信秘书健康", "om_x1234567890abc"
    )
    assert got == {
        "chat_id": "oc_wechat_secretary",
        "message_id": "om_x1234567890abc",
        "goal": "任务：只读检查微信秘书健康",
    }


@pytest.mark.parametrize("change", [
    {"user_id": "ou_another"},
    {"chat_id": "oc_cortex_group"},
    {"chat_type": "dm"},
    {"is_bot": True},
])
def test_wechat_owner_rejects_wrong_source(wechat_owner_group, change):
    assert ingress.authenticated_goal(
        {**wechat_owner_group, **change}, "任务：只读检查", "om_x1234567890abc"
    ) is None


@pytest.mark.parametrize("message", ["你好", "请解释一段技术", "/retry", ""])
def test_wechat_secretary_chat_not_published(wechat_owner_group, message):
    assert ingress.authenticated_goal(
        wechat_owner_group, message, "om_x1234567890abc"
    ) is None


@pytest.mark.asyncio
async def test_wechat_secretary_publisher_replay_shows_same_id(
    wechat_owner_group, monkeypatch
):
    class Process:
        returncode = 0
        async def communicate(self, body):
            obj = json.loads(body)
            assert obj["chat_id"] == "oc_wechat_secretary"
            assert obj["message_id"] == "om_x1234567890abc"
            return json.dumps({
                "task_id": "gpt-inbox-agent-cortex-1234567890abcd123456",
                "profile": "general", "idempotent_replay": True,
            }).encode(), b""
    monkeypatch.setattr(
        ingress.asyncio, "create_subprocess_exec",
        AsyncMock(return_value=Process()),
    )
    result = await ingress.maybe_publish_owner_goal(
        platform="feishu", source=wechat_owner_group,
        text="每天：只读核查健康", message_id="om_x1234567890abc",
    )
    assert "gpt-inbox-agent-cortex-1234567890abcd123456" in result["reply"]
    assert "复用已有任务" in result["reply"]
    assert "等待真实领取" in result["reply"]
    assert "已完成" not in result["reply"]
