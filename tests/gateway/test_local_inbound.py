import asyncio
from types import SimpleNamespace

from gateway.config import Platform
from gateway.run_local_inbound import inject_local_inbound
from gateway.session import SessionSource


class FakeAdapter:
    def __init__(self):
        self._active_sessions = {}
        self.seen = None

    def build_source(self, **kwargs):
        return SessionSource(
            platform=Platform.DINGTALK,
            chat_id=kwargs["chat_id"],
            chat_name=kwargs.get("chat_name"),
            chat_type=kwargs.get("chat_type") or "group",
            user_id=kwargs.get("user_id"),
            user_id_alt=kwargs.get("user_id_alt"),
            user_name=kwargs.get("user_name"),
            message_id=kwargs.get("message_id"),
            profile="hema-teacher",
        )

    def _resolve_channel_skills(self, _chat_id):
        return ["math-skill"]

    def _resolve_channel_prompt(self, _chat_id):
        return None

    def _event_session_key(self, _event):
        return "session-key"
class FakeRunner:
    def __init__(self):
        self.adapter = FakeAdapter()
        self._primary_profile_name = "default"
        self.adapters = {}
        self._profile_adapters = {
            "hema-teacher": {Platform.DINGTALK: self.adapter}
        }
        self.session_store = SimpleNamespace(
            _entries={"session-key": SimpleNamespace(session_id="sid-live")}
        )
        self.calls = 0

    def _is_session_running(self, _key):
        return False

    async def _handle_message(self, event):
        self.calls += 1
        self.adapter.seen = event
        return "答对了。下一题"


def _payload(message_id="m1", text="4000米", **extra):
    return {
        "profile": "hema-teacher",
        "platform": "dingtalk",
        "chat_id": "cid-math",
        "chat_name": "小马快跑（数学）",
        "chat_type": "group",
        "user_id": "raw-sender",
        "user_id_alt": "hard-participant",
        "user_name": "learner_xiaoma",
        "message_id": message_id,
        "text": text,
        "skill": "math-skill",
        **extra,
    }
def test_injected_turn_reuses_native_identity_and_stays_human_input():
    runner = FakeRunner()
    result = asyncio.run(inject_local_inbound(
        runner,
        _payload(
            recent_context="河马老师: R2 求多少米？",
            force_context=True,
        ),
    ))
    event = runner.adapter.seen
    assert result["accepted"] is True
    assert result["response"] == "答对了。下一题"
    assert result["session_id"] == "sid-live"
    assert event.source.user_id == "raw-sender"
    assert event.source.user_id_alt == "hard-participant"
    assert event.source.role_authorized is False
    assert event.source._suppress_presentation is True
    assert event.source._trusted_context_tail is True
    assert event.auto_skill == ["math-skill"]
    assert event.channel_context == "河马老师: R2 求多少米？"
    assert event.allow_gateway_control is False


def test_duplicate_message_id_returns_cached_result_without_second_turn():
    runner = FakeRunner()

    async def scenario():
        first = await inject_local_inbound(runner, _payload())
        second = await inject_local_inbound(runner, _payload())
        return first, second

    first, second = asyncio.run(scenario())
    assert first == second
    assert runner.calls == 1
def test_unsupported_platform_fails_closed():
    runner = FakeRunner()
    payload = _payload()
    payload["platform"] = "feishu"
    result = asyncio.run(inject_local_inbound(runner, payload))
    assert result == {"accepted": False, "reason": "unsupported_platform"}



def test_sidecar_source_mutes_gateway_presentation_only():
    from gateway.run_turn import _turn_presentation_muted

    source = SessionSource(
        platform=Platform.DINGTALK,
        chat_id="cid-math",
        chat_type="group",
        user_id="raw-sender",
    )
    source._suppress_presentation = True
    assert _turn_presentation_muted({}, Platform.DINGTALK, {}, source) is True
