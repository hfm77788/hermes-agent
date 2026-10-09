import time
from types import SimpleNamespace

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


def _turn(message="推进闭环", enabled=True):
    ctx = TurnContext(
        source=SimpleNamespace(platform=Platform.DINGTALK, chat_id="cid"),
        message=message,
        user_config={
            "gateway": {
                "long_task_ack": {
                    "enabled": enabled,
                    "delay_seconds": 2.5,
                    "message": "已接单，正在查实际状态。",
                }
            }
        },
        _status_adapter=object(),
        _status_chat_id="cid",
        _run_still_current=lambda: True,
    )
    runner = TurnRunner(SimpleNamespace(), ctx)
    agent = SimpleNamespace(_turn_origin=None, _goal_manager=None)
    return runner, agent


def test_long_task_ack_settings_only_for_long_turns():
    runner, agent = _turn("推进闭环")
    assert runner._long_task_ack_settings(agent) == (
        2.5, "已接单，正在查实际状态。"
    )
    runner._ctx.message = "现在正常吗？"
    assert runner._long_task_ack_settings(agent) is None


def test_long_task_ack_disabled_by_config():
    runner, agent = _turn("推进闭环", enabled=False)
    assert runner._long_task_ack_settings(agent) is None


def test_delayed_ack_fires_once_without_real_output(monkeypatch):
    runner, agent = _turn()
    sent = []
    monkeypatch.setattr(runner, "_long_task_ack_settings", lambda _agent: (0.01, "ACK"))
    monkeypatch.setattr(
        runner, "_status_callback_sync", lambda kind, message: sent.append((kind, message))
    )
    _delta, _interim, visible, timer = runner._arm_long_task_ack(agent, None, None)
    time.sleep(0.04)
    runner._cancel_long_task_ack(visible, timer)
    assert sent == [("long_task_ack", "ACK")]


def test_real_stream_output_cancels_ack(monkeypatch):
    runner, agent = _turn()
    sent = []
    streamed = []
    monkeypatch.setattr(runner, "_long_task_ack_settings", lambda _agent: (0.03, "ACK"))
    monkeypatch.setattr(
        runner, "_status_callback_sync", lambda kind, message: sent.append((kind, message))
    )
    delta, _interim, visible, timer = runner._arm_long_task_ack(
        agent, streamed.append, None
    )
    delta("真实输出")
    time.sleep(0.06)
    runner._cancel_long_task_ack(visible, timer)
    assert streamed == ["真实输出"]
    assert sent == []


def test_real_interim_output_cancels_ack(monkeypatch):
    runner, agent = _turn()
    sent = []
    interim = []
    monkeypatch.setattr(runner, "_long_task_ack_settings", lambda _agent: (0.03, "ACK"))
    monkeypatch.setattr(
        runner, "_status_callback_sync", lambda kind, message: sent.append((kind, message))
    )

    def on_interim(text, *, already_streamed=False):
        interim.append((text, already_streamed))

    _delta, wrapped, visible, timer = runner._arm_long_task_ack(
        agent, None, on_interim
    )
    wrapped("已有中间结果", already_streamed=False)
    time.sleep(0.06)
    runner._cancel_long_task_ack(visible, timer)
    assert interim == [("已有中间结果", False)]
    assert sent == []
