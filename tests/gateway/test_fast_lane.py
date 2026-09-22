import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from gateway.fast_lane import FastLaneDecision, decide_fast_lane
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def _source():
    return SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u")


def _history(n=100):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": "x"} for i in range(n)]


def _run(event, history=None, tokens=60000):
    return asyncio.run(decide_fast_lane(
        event=event,
        source=_source(),
        history=history if history is not None else _history(),
        session_entry=SimpleNamespace(last_prompt_tokens=tokens),
        config={},
        was_auto_reset=False,
        is_new_session=False,
        pending_sidecar=False,
    ))


def test_long_self_contained_turn_is_accepted():
    response={"choices":[{"message":{"content":'{"self_contained":true,"confidence":0.99}'}}]}
    event=MessageEvent(text="List all active reminders.", message_type=MessageType.TEXT, source=_source())
    with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock(return_value=response)) as call:
        decision=_run(event)
    assert decision.use_fast_lane is True
    assert call.await_count == 1
    kwargs = call.await_args.kwargs
    system_prompt = kwargs["messages"][0]["content"]
    payload = json.loads(kwargs["messages"][1]["content"])
    assert payload == {"message_to_classify": "List all active reminders."}
    assert "EARLIER CHAT MESSAGES" in system_prompt
    assert "authenticated account" in system_prompt
    assert "current account/profile/tool environment" in system_prompt
    assert kwargs["max_tokens"] == 40
    assert kwargs["timeout"] == 4.0
    assert kwargs["reasoning_config"] == {"enabled": False}


def test_context_dependent_turn_is_rejected():
    response={"choices":[{"message":{"content":'{"self_contained":false,"confidence":0.99}'}}]}
    event=MessageEvent(text="Continue that.", message_type=MessageType.TEXT, source=_source())
    with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock(return_value=response)):
        decision=_run(event)
    assert decision.use_fast_lane is False
    assert decision.reason == "classifier_context_dependent"


def test_reply_context_falls_back_without_classifier():
    event=MessageEvent(
        text="Change it.", message_type=MessageType.TEXT, source=_source(),
        reply_to_message_id="m1", reply_to_text="prior",
    )
    with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock()) as call:
        decision=_run(event)
    assert decision.reason == "reply_context"
    assert call.await_count == 0


def test_media_sidecar_and_channel_context_fail_closed_without_classifier():
    cases=[
        MessageEvent(text="x", message_type=MessageType.PHOTO, source=_source(), media_urls=["/tmp/x"]),
        MessageEvent(text="x", message_type=MessageType.TEXT, source=_source(), channel_prompt="special"),
    ]
    for event in cases:
        with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock()) as call:
            assert _run(event).use_fast_lane is False
            assert call.await_count == 0


def test_short_history_avoids_classifier():
    event=MessageEvent(text="List reminders.", message_type=MessageType.TEXT, source=_source())
    with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock()) as call:
        decision=_run(event, history=_history(4), tokens=1200)
    assert decision.reason == "short_history"
    assert call.await_count == 0


def test_classifier_invalid_fails_closed():
    event=MessageEvent(text="List reminders.", message_type=MessageType.TEXT, source=_source())
    response={"choices":[{"message":{"content":"not-json"}}]}
    with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock(return_value=response)):
        decision=_run(event)
    assert decision.use_fast_lane is False
    assert decision.reason == "classifier_error"


def test_prepare_turn_fast_lane_skips_hygiene_and_preserves_loaded_history_view_for_preprocessing():
    from gateway.run_turn import GatewayTurnMixin

    source=_source()
    event=MessageEvent(text="List reminders.", message_type=MessageType.TEXT, source=source)
    loaded=_history()
    seen={}
    evicted=[]

    class Store:
        async def load_transcript(self, _sid):
            return loaded

    class Runner(GatewayTurnMixin):
        config=SimpleNamespace(multiplex_profiles=False)
        async_session_store=Store()
        async def _hmwa_open_session(self, *_a): return (False, False)
        def _set_session_env(self, _ctx): return ()
        def _pinned_session_context_prompt(self, *_a): return ""
        async def _hmwa_acquire_turn_lease(self, *_a): return None
        async def _mark_durable_active_turn(self, *_a): return True
        async def _hmwa_run_session_hygiene(self, *_a):
            raise AssertionError("hygiene must be skipped")
        async def _hmwa_first_contact_notes(self, _source, history, _notes):
            seen["first"]=history
        def _voice_channel_sidecar_note(self, *_a): return None
        async def _prepare_profile_scoped_inbound_message_text(self, *, history, **_kw):
            seen["preprocess"]=history
            return event.text
        def _hmwa_apply_message_timestamp(self, _event, text):
            return text, text, None
        def _bind_adapter_run_generation(self, *_a): return None
        def _delivery_adapter_for(self, _source): return None
        def _adapter_for_source(self, _source): return None
        def _peek_session_state(self, _key): return None
        def _evict_cached_agent(self, key): evicted.append(key)

    entry=SimpleNamespace(session_id="sid", session_key="skey", last_prompt_tokens=60000)
    accepted=FastLaneDecision(True, "self_contained", 60000, len(loaded), 1)
    with (
        patch("gateway.run_turn.build_session_context", return_value=SimpleNamespace()),
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("gateway.fast_lane.decide_fast_lane", new=AsyncMock(return_value=accepted)),
    ):
        prepared,_=asyncio.run(Runner()._hmwa_prepare_turn(event, source, entry, "skey", "qk", 1))
    assert prepared.history == []
    assert prepared.durable_history is loaded
    assert prepared.fast_lane is True
    assert evicted == ["skey"]
    assert seen["first"] is loaded
    assert seen["preprocess"] is loaded
    assert len(loaded) == 100



def test_fast_lane_persistence_uses_durable_history_and_appends_only_current_turn():
    from gateway.run_turn import GatewayTurnMixin

    source=_source()
    event=MessageEvent(text="List reminders.", message_type=MessageType.TEXT, source=source, message_id="m1")
    durable=_history()
    writes=[]

    class Store:
        async def append_to_transcript(self, _sid, entry, **_kw):
            writes.append(dict(entry))
        async def update_session(self, *_a, **_kw):
            return None

    class Runner(GatewayTurnMixin):
        async_session_store=Store()
        _session_db=None
        async def _refresh_agent_cache_message_count(self, *_a, **_kw):
            return None

    prepared=Runner._PreparedTurn(
        [], "", event.text, event.text, None, None, "sid", "owner", durable, True,
    )
    current=[
        {"role":"user","content":event.text},
        {"role":"assistant","content":"You have 2 reminders."},
    ]
    result={
        "agent_persisted":False,
        "messages":current,
        "history_offset":0,
        "last_prompt_tokens":120,
        "tools":[],
    }
    entry=SimpleNamespace(session_id="sid", session_key="skey")
    asyncio.run(Runner()._hmwa_persist_turn_transcript(
        event=event, source=source, session_entry=entry, session_key="skey",
        agent_result=result, agent_messages=current, prepared=prepared,
        response="You have 2 reminders.", agent_failed_early=False,
        hidden_reasoning_incomplete=False, is_context_overflow_failure=False,
    ))
    assert durable == _history()
    assert [row["role"] for row in writes] == ["user", "assistant"]
    assert all(row.get("role") != "session_meta" for row in writes)

def test_classifier_string_high_confidence_is_normalized():
    response={"choices":[{"message":{"content":'{"self_contained":true,"confidence":"high","reason":"self_contained"}'}}]}
    event=MessageEvent(text="List all active reminders.", message_type=MessageType.TEXT, source=_source())
    with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock(return_value=response)):
        decision=_run(event)
    assert decision.use_fast_lane is True
    assert decision.confidence == 1.0


def test_classifier_string_high_ambiguous_still_fails_closed():
    response={"choices":[{"message":{"content":'{"self_contained":false,"confidence":"high","reason":"ambiguous"}'}}]}
    event=MessageEvent(text="Do the thing.", message_type=MessageType.TEXT, source=_source())
    with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock(return_value=response)):
        decision=_run(event)
    assert decision.use_fast_lane is False
    assert decision.reason == "classifier_context_dependent"

def test_verbose_reason_does_not_override_high_confidence_true_classification():
    response={"choices":[{"message":{"content":'{"self_contained":true,"confidence":1.0,"reason":"This standalone request does not rely on earlier chat."}'}}]}
    event=MessageEvent(text="List all active reminders.", message_type=MessageType.TEXT, source=_source())
    with patch("agent.auxiliary_client.async_call_llm", new=AsyncMock(return_value=response)):
        decision=_run(event)
    assert decision.use_fast_lane is True
    assert decision.reason == "self_contained"
