from types import SimpleNamespace

from agent.conversation_compression import (
    context_compression_timed_out,
    mark_context_compression_timed_out,
)
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression


def test_fit_window_foreground_timeout_falls_through(monkeypatch):
    import agent.conversation_loop as loop
    import agent.turn_preflight as preflight

    monkeypatch.setattr(preflight, "ensure_compression_feasibility_checked", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_maybe_grow_local_window", lambda *_a, **_k: None)

    compressor = SimpleNamespace(
        context_length=1_000,
        threshold_tokens=1,
        get_active_compression_failure_cooldown=lambda: None,
        should_compress=lambda _tokens: True,
    )
    agent = SimpleNamespace(
        compression_enabled=True,
        context_compressor=compressor,
        model="test-model",
        _emit_status=lambda *_a, **_k: None,
        iteration_budget=SimpleNamespace(refund=lambda: None),
    )

    def _compress(messages, _system_message, **_kwargs):
        mark_context_compression_timed_out(agent)
        return messages, "sys"

    agent._compress_context = _compress

    v = PreflightGateVerdict(
        action="fallthrough",
        pending_moa_prepared_request=None,
        messages=[
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "new"},
        ],
        active_system_prompt="sys",
        conversation_history=[],
        api_call_count=1,
        compression_attempts=0,
        final_response="",
        failed=False,
        _turn_exit_reason=None,
        _compression_timeout_exhausted=False,
        _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=False,
        _last_preflight_pressure=None,
    )

    prepared = object()
    out = run_preflight_compression(
        agent,
        v,
        compressor=compressor,
        request_pressure_tokens=100,
        provider_overflow_preflight=False,
        defer_preflight=lambda _tokens: False,
        moa_prepared_request=prepared,
        system_message="sys",
        user_message="new",
        max_compression_attempts=3,
        effective_task_id="default",
    )

    assert out.action == "fallthrough"
    assert out.failed is False
    assert out.pending_moa_prepared_request is None
    assert context_compression_timed_out(agent) is False
