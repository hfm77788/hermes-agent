from types import SimpleNamespace

from agent.adaptive_turn_budget import (
    apply_adaptive_turn_budget,
    classify_turn,
    configure_adaptive_turn_budget,
    effective_turn_max_iterations,
    partition_tool_calls,
    tool_budget_exhausted_content,
)


def _agent(max_iterations=90):
    return SimpleNamespace(
        max_iterations=max_iterations,
        _turn_origin=None,
        _goal_manager=None,
    )


def _enable(agent):
    configure_adaptive_turn_budget(agent, {
        "adaptive_turn_budget": {
            "enabled": True,
            "simple_iterations": 3,
            "simple_tool_calls": 3,
            "normal_iterations": 8,
            "normal_tool_calls": 8,
            "research_iterations": 18,
            "research_tool_calls": 24,
        }
    })
    return agent


def test_simple_status_turn_gets_small_budget():
    agent = _enable(_agent())
    budget = apply_adaptive_turn_budget(agent, "进度")
    assert budget.tier == "simple"
    assert budget.iterations == 3
    assert budget.tool_calls == 3
    assert agent.iteration_budget.max_total == 3
    assert effective_turn_max_iterations(agent) == 3


def test_normal_short_question_gets_normal_budget():
    agent = _enable(_agent())
    budget = apply_adaptive_turn_budget(agent, "佳音的学习群设置好了吗")
    assert budget.tier == "normal"
    assert budget.iterations == 8
    assert budget.tool_calls == 8


def test_research_turn_gets_research_budget():
    agent = _enable(_agent())
    budget = apply_adaptive_turn_budget(agent, "帮我调研一下 Hermes 当前延迟")
    assert budget.tier == "research"
    assert budget.iterations == 18
    assert budget.tool_calls == 24


def test_long_commands_keep_configured_budget():
    for text in ("继续", "推进闭环", "彻底修复", "把这个任务推进到闭环"):
        agent = _enable(_agent())
        budget = apply_adaptive_turn_budget(agent, text)
        assert budget.tier == "long"
        assert budget.iterations == 90
        assert budget.tool_calls is None


def test_kanban_context_never_shrinks(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    agent = _enable(_agent())
    budget = apply_adaptive_turn_budget(agent, "进度")
    assert budget.tier == "long"
    assert budget.iterations == 90
    assert budget.reason == "kanban_task"


def test_disabled_policy_preserves_configured_budget():
    agent = _agent(37)
    configure_adaptive_turn_budget(agent, {"adaptive_turn_budget": {"enabled": False}})
    budget = apply_adaptive_turn_budget(agent, "进度")
    assert budget.tier == "disabled"
    assert budget.iterations == 37
    assert budget.tool_calls is None


def test_tool_budget_partitions_and_counts():
    agent = _enable(_agent())
    apply_adaptive_turn_budget(agent, "进度")
    calls = [SimpleNamespace(function=SimpleNamespace(name=f"t{i}")) for i in range(5)]
    allowed, skipped = partition_tool_calls(agent, calls)
    assert [c.function.name for c in allowed] == ["t0", "t1", "t2"]
    assert [c.function.name for c in skipped] == ["t3", "t4"]
    assert agent._adaptive_tool_calls_used == 3
    allowed2, skipped2 = partition_tool_calls(agent, calls[:1])
    assert allowed2 == []
    assert len(skipped2) == 1


def test_tool_budget_exhaustion_result_is_protocol_safe_json():
    import json

    agent = _enable(_agent())
    apply_adaptive_turn_budget(agent, "进度")
    calls = [SimpleNamespace(function=SimpleNamespace(name=f"t{i}")) for i in range(3)]
    partition_tool_calls(agent, calls)
    payload = json.loads(tool_budget_exhausted_content(agent, "t4"))
    assert payload["error"] == "adaptive_tool_budget_exhausted"
    assert payload["tool"] == "t4"
    assert payload["used"] == 3
    assert payload["limit"] == 3


def test_classify_question_and_explicit_long_command():
    agent = _agent()
    assert classify_turn(agent, "现在正常吗？")[0] == "simple"
    assert classify_turn(agent, "继续")[0] == "long"
