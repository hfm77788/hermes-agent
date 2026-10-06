import asyncio
from types import SimpleNamespace

from gateway.config import Platform
from gateway.run_local_inbound import inject_local_inbound, _local_session_health
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
        self.entry = SimpleNamespace(session_id="sid-live", last_prompt_tokens=60000)
        self.session_store = SimpleNamespace(_entries={"session-key": self.entry})
        self.async_session_store = SimpleNamespace()
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
    assert event.source._local_inbound_skill == "math-skill"
    assert event.source._local_inbound_has_media is False
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


def test_local_session_health_inspects_without_creating_turn():
    runner = FakeRunner()

    async def resolve(_agent, _ctx, _entry, _source):
        return 60000, 100000, "qwen-test"

    runner._resident_agent_for = lambda _key: None
    runner._resolve_context_figures = resolve

    result = asyncio.run(_local_session_health(runner, _payload(action="inspect")))
    assert result["accepted"] is True
    assert result["found"] is True
    assert result["last_prompt_tokens"] == 60000
    assert result["context_length"] == 100000
    assert result["context_pct"] == 60.0
    assert runner.calls == 0


def test_local_session_health_busy_compress_is_fail_closed():
    runner = FakeRunner()

    async def resolve(_agent, _ctx, _entry, _source):
        return 70000, 100000, "qwen-test"

    runner._resident_agent_for = lambda _key: None
    runner._resolve_context_figures = resolve
    runner._is_session_running = lambda _key: True

    result = asyncio.run(_local_session_health(runner, _payload(action="compress")))
    assert result["accepted"] is True
    assert result["busy"] is True
    assert result["changed"] is False
    assert result["reason"] == "session_busy"
    assert runner.calls == 0


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
    assert event.source._local_inbound_has_media is True


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


def _tutor_source(*, skill="huangshang-math-tutor", has_media=False):
    source = SessionSource(
        platform=Platform.DINGTALK,
        chat_id="cid-math",
        chat_type="group",
        user_id="child",
    )
    source._trusted_context_tail = True
    source._local_inbound_skill = skill
    source._local_inbound_has_media = has_media
    return source


def test_fast_tutoring_guard_blocks_tools_for_mid_lesson_short_answer():
    from gateway.run_turn import _local_inbound_fast_tutoring_no_tools

    source = _tutor_source()
    history = [{"role": "assistant", "content": "C3：这道题是多少？"}]
    assert _local_inbound_fast_tutoring_no_tools(source, "5亿元", history) is True
    assert _local_inbound_fast_tutoring_no_tools(
        source, "我最开始算成75升，为啥错啦？", history
    ) is True


def test_fast_tutoring_guard_keeps_tools_for_lifecycle_media_and_final_closeout():
    from gateway.run_turn import _local_inbound_fast_tutoring_no_tools

    source = _tutor_source()
    history = [{"role": "assistant", "content": "C3：这道题是多少？"}]
    assert _local_inbound_fast_tutoring_no_tools(source, "开始", history) is False
    assert _local_inbound_fast_tutoring_no_tools(source, "搜集数学资料", history) is False

    image_source = _tutor_source(has_media=True)
    assert _local_inbound_fast_tutoring_no_tools(image_source, "讲第二题", history) is False

    final_history = [{"role": "assistant", "content": "E1（最后一题）：请说说理由。"}]
    assert _local_inbound_fast_tutoring_no_tools(source, "因为单位1变了", final_history) is False

    weak_final_prompt = [{"role": "assistant", "content": "最后一题：3+5="}]
    assert _local_inbound_fast_tutoring_no_tools(source, "8", weak_final_prompt) is False


def test_fast_tutoring_guard_ignores_stale_final_progress_phrase_for_state_correction():
    from gateway.run_turn import _local_inbound_fast_tutoring_no_tools

    source = _tutor_source()
    stale_history = [{"role": "assistant", "content": "今晚只剩最后一题，做完就结束。"}]

    assert _local_inbound_fast_tutoring_no_tools(source, "今天还没学啊", stale_history) is True


def test_fast_tutoring_guard_is_scoped_to_trusted_learning_group():
    from gateway.run_turn import _local_inbound_fast_tutoring_no_tools

    source = _tutor_source(skill="unrelated-skill")
    assert _local_inbound_fast_tutoring_no_tools(
        source, "5亿元", [{"role": "assistant", "content": "C3"}]
    ) is False

    dm = _tutor_source()
    dm.chat_type = "dm"
    assert _local_inbound_fast_tutoring_no_tools(
        dm, "5亿元", [{"role": "assistant", "content": "C3"}]
    ) is False
