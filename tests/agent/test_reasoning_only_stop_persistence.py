"""Regression: provider reasoning must never become an answer or replayable content.

Incident: 2026-10-09 22:18, qwen3.8-flash reasoning-only clean stop emitted
an English internal monologue to a DingTalk student.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

PRIVATE = "The user said R3. Wait, am I the user? Let me reread the conversation."
PUBLIC = "已收到，你这题答对了。下面做 C1。"


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.deepseek.com/v1",
            model="qwen3.8-flash",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are a tutoring assistant."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _run(agent, responses, user_message="R3 50.24平方厘米", conversation_history=None):
    agent.client.chat.completions.create.side_effect = list(responses)
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(user_message, conversation_history=conversation_history)


def test_reasoning_only_clean_stop_requires_visible_followup(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    result = _run(loop_agent, [
        _mock_response(content="", finish_reason="stop", reasoning_content=PRIVATE),
        _mock_response(content=PUBLIC, finish_reason="stop"),
    ])

    assert result["final_response"] == PUBLIC
    assert result["api_calls"] == 2
    assert PRIVATE not in result["final_response"]
    assert all("api_content" not in r for r in result["messages"])
    assert all(PRIVATE not in str(r.get("content") or "") for r in result["messages"])

    # A subsequent student message must not see a false previous assistant reply
    # made from reasoning_content; only the real visible answer may replay.
    _run(loop_agent, [_mock_response(content="继续做题。", finish_reason="stop")],
         user_message="下一题", conversation_history=result["messages"])
    wire = loop_agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert all(PRIVATE not in str(r.get("content") or "") for r in wire)
    assert any(r.get("content") == PUBLIC for r in wire if r.get("role") == "assistant")


def test_reasoning_only_with_tools_recovers_without_exposing_private_text(loop_agent):
    from tests.agent.test_run_agent import _mock_response
    loop_agent.valid_tool_names = {"read_file"}

    result = _run(loop_agent, [
        _mock_response(content="", finish_reason="stop", reasoning_content=PRIVATE),
        _mock_response(content=PUBLIC, finish_reason="stop"),
    ])

    assert result["api_calls"] == 2
    assert result["final_response"] == PUBLIC
    assert all("api_content" not in r for r in result["messages"])


@pytest.mark.parametrize("reasoning", [
    "Let me batch the terminal calls and run them in parallel.",
    "Wait, who was the user? Let me reread the raw conversation order.",
    "The answer is 42, but this is private reasoning.",
])
def test_any_reasoning_only_text_is_private_even_if_answer_like(loop_agent, reasoning):
    from tests.agent.test_run_agent import _mock_response
    loop_agent.valid_tool_names = {"terminal", "read_file"}

    result = _run(loop_agent, [
        _mock_response(content=None, finish_reason="stop", reasoning_content=reasoning),
        _mock_response(content=PUBLIC, finish_reason="stop"),
    ])
    assert result["api_calls"] == 2
    assert result["final_response"] == PUBLIC
    assert reasoning not in result["final_response"]


def test_repeated_reasoning_only_does_not_leak_on_retry_exhaustion(loop_agent):
    from tests.agent.test_run_agent import _mock_response
    loop_agent.valid_tool_names = {"terminal"}
    repeating = _mock_response(content="", finish_reason="stop", reasoning_content=PRIVATE)

    result = _run(loop_agent, [repeating] * 16)
    assert 1 < result["api_calls"] < 16
    assert PRIVATE not in result["final_response"]
    assert "api_content" not in str(result["messages"])
    assert "未能生成可用" in result["final_response"]


def test_visible_response_with_separate_reasoning_returns_only_visible(loop_agent):
    from tests.agent.test_run_agent import _mock_response
    result = _run(loop_agent, [
        _mock_response(content=PUBLIC, finish_reason="stop", reasoning_content=PRIVATE),
    ])
    assert result["final_response"] == PUBLIC
    assert result["api_calls"] == 1
