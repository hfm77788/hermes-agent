"""Foreground optional compaction must defer to the fenced Gateway background worker.

Regression for the observed 2026-11-03 13:47 WeChat office-director
200629-token / 30-second post-tool foreground hold and 300-second cooldown.
Provider-proven overflow and unknown/overfull model windows remain fail-closed.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_context_compaction import (
    CompactionOutcome, _defer_optional_hygiene_to_gateway,
)
from agent.turn_preflight import (
    compress_after_tool_results, run_preflight_compression,
)


def _agent(tokens=200629, window=256000, enabled=True):
    compressor = SimpleNamespace(
        threshold_tokens=200000, context_length=window,
        last_prompt_tokens=tokens,
        awaiting_real_usage_after_compression=False,
        protect_first_n=3, protect_last_n=8,
    )
    return SimpleNamespace(
        _gateway_background_hygiene_only=enabled,
        _persist_disabled=False,
        compression_enabled=True,
        context_compressor=compressor, tools=[],
        _usage_anchor=None, model="qwen3.8-max-0902",
        _compress_context=MagicMock(
            side_effect=AssertionError("optional foreground compression was called")
        ),
    )


@pytest.mark.parametrize("tokens,window,enabled,persist_disabled,expected", [
    (200629, 256000, True, False, True),
    (255999, 256000, True, False, True),
    (256001, 256000, True, False, False),
    (200629, None, True, False, False),
    (200629, 256000, False, False, False),
    (200629, 256000, True, True, False),
    ("200629", 256000, True, False, False),
])
def test_only_confirmed_fitting_gateway_turns_defer(
    tokens, window, enabled, persist_disabled, expected,
):
    agent = _agent(enabled=enabled, window=window)
    agent._persist_disabled = persist_disabled
    assert _defer_optional_hygiene_to_gateway(agent, tokens) is expected


def test_real_200629_token_post_tool_turn_does_not_wait_for_summary():
    agent = _agent()
    history = [
        {"role": "user", "content": "Preserve earlier instructions"},
        {"role": "assistant", "content": "Do not lose the original task"},
        {"role": "tool", "content": "New information"},
        {"role": "user", "content": "Continue the draft"},
    ]
    verdict = compress_after_tool_results(
        agent,
        messages=history, system_message="system",
        user_message=history[-1],
        active_system_prompt="system", conversation_history=history,
        compression_attempts=0, max_compression_attempts=3,
        effective_task_id="test-session", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=3,
    )
    assert not verdict.end_turn
    assert verdict.messages is history
    assert verdict.conversation_history is history
    assert verdict.compression_attempts == 0
    agent._compress_context.assert_not_called()


def test_fitting_pre_api_turn_skips_optional_summary_but_preserves_provider_request():
    agent = _agent()
    verdict = SimpleNamespace(
        messages=[{"role": "user", "content": "original"},
                  {"role": "tool", "content": "tool output"}],
        compression_attempts=0,
        result=None, action=None,
    )
    out = run_preflight_compression(
        agent, verdict, compressor=agent.context_compressor,
        request_pressure_tokens=200629,
        provider_overflow_preflight=False,
        defer_preflight=lambda _tokens: False,
        moa_prepared_request=None, system_message="system",
        user_message=verdict.messages[0], max_compression_attempts=3,
        effective_task_id="test-session",
    )
    assert out.action == "fallthrough"
    assert out.messages is verdict.messages
    agent._compress_context.assert_not_called()


def test_provider_proven_overflow_is_not_allowed_to_use_optional_defer():
    agent = _agent()
    agent.context_compressor.context_length = 198000
    assert not _defer_optional_hygiene_to_gateway(agent, 200629)
    # The actual provider-overflow gate is evaluated independently by
    # run_preflight_compression(provider_overflow_preflight=True).


def test_turn_start_threshold_must_defer_fitting_session(monkeypatch):
    from agent import turn_context as turn_ctx
    from agent import turn_context_compaction as compaction

    agent = _agent()
    history = [{"role": "user", "content": "prior value"},
               {"role": "assistant", "content": "prior commitment"}] * 5
    out = CompactionOutcome(
        messages=history, active_system_prompt="system",
        conversation_history=history, current_turn_user_idx=0,
    )
    monkeypatch.setattr(turn_ctx, "_review_fork_first_request_pending",
                        lambda _agent: False)
    monkeypatch.setattr(turn_ctx, "_should_run_preflight_estimate",
                        lambda *_a: True)
    monkeypatch.setattr(turn_ctx, "_preflight_request_tokens",
                        lambda *_a: 200629)
    compaction._preflight_compression(
        agent, out, "system", history[0], "test-session",
    )
    assert out.messages is history
    assert not out.compressed
    agent._compress_context.assert_not_called()


def test_non_opted_in_agent_retains_original_compression_gate():
    agent = _agent(enabled=False)
    assert not _defer_optional_hygiene_to_gateway(agent, 200629)
