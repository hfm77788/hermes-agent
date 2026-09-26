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
        event.source._local_inbound_response_future.set_result("答对了。下一题")
        return None


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


def test_run_turn_publishes_authoritative_final_response():
    from gateway.run_turn import _publish_local_inbound_response

    async def scenario():
        source = SessionSource(
            platform=Platform.DINGTALK,
            chat_id="cid-math",
            chat_type="group",
            user_id="raw-sender",
        )
        future = asyncio.get_running_loop().create_future()
        source._local_inbound_response_future = future
        _publish_local_inbound_response(source, {"final_response": "最终正文"})
        return await future

    assert asyncio.run(scenario()) == "最终正文"



def test_injected_image_uses_photo_event_from_controlled_media_cache(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageType

    home = tmp_path / "hermes-home"
    media = home / "state" / "dingtalk-free-response-media" / "job" / "page.jpg"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"jpeg-bytes")
    monkeypatch.setenv("HERMES_HOME", str(home))

    runner = FakeRunner()
    result = asyncio.run(inject_local_inbound(
        runner,
        _payload(
            message_id="img-1",
            text="讲第二题",
            media_urls=[str(media)],
            media_types=["image"],
        ),
    ))
    event = runner.adapter.seen
    assert result["accepted"] is True
    assert event.message_type is MessageType.PHOTO
    assert event.media_urls == [str(media.resolve())]
    assert event.media_types == ["image/jpeg"]


def test_injected_image_rejects_path_outside_controlled_media_cache(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "state" / "dingtalk-free-response-media").mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"jpeg-bytes")
    monkeypatch.setenv("HERMES_HOME", str(home))

    runner = FakeRunner()
    result = asyncio.run(inject_local_inbound(
        runner,
        _payload(
            message_id="img-2",
            text="讲题",
            media_urls=[str(outside)],
            media_types=["image"],
        ),
    ))
    assert result == {"accepted": False, "reason": "invalid_media_path"}
    assert runner.calls == 0


def test_injected_media_rejects_non_image_type_inside_cache(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    media = home / "state" / "dingtalk-free-response-media" / "job" / "page.jpg"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"jpeg-bytes")
    monkeypatch.setenv("HERMES_HOME", str(home))

    runner = FakeRunner()
    result = asyncio.run(inject_local_inbound(
        runner,
        _payload(
            message_id="img-bad-type",
            text="讲题",
            media_urls=[str(media)],
            media_types=["application/octet-stream"],
        ),
    ))
    assert result == {"accepted": False, "reason": "unsupported_media_type"}
    assert runner.calls == 0
