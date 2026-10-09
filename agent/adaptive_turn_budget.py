"""Adaptive per-turn iteration and tool budgets for fast simple turns.

The configured agent.max_turns remains the hard ceiling and is never mutated.
A smaller per-turn IterationBudget is selected only for clearly bounded human
turns; long-running, kanban, goal and delegated work keeps the configured cap.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

from agent.iteration_budget import IterationBudget


_LONG_EXACT = {
    "继续", "推进", "推进闭环", "继续推进", "修", "修复", "彻底修复",
    "落地", "落地执行", "执行", "做吧", "改进吧", "优化", "部署", "收尾",
}
_SIMPLE_EXACT = {
    "?", "？", "进度", "进度如何", "怎么样", "如何了", "正常吗", "完成了吗",
    "闭环了吗", "好了没", "怎么回事", "啥意思", "什么意思", "现在呢", "结果呢",
}
_LONG_MARKERS = (
    "推进到闭环", "全权推进", "彻底解决", "根因修复", "根因解决", "落地执行",
    "合并并部署", "部署到生产", "全量评审", "正式评审", "批量处理", "全量处理",
    "完整执行", "持续推进", "做完", "完成任务", "按流程修", "维修", "系统维护",
    "升级改造", "实现并部署", "从头到尾", "business smoke", "业务验收",
)
_RESEARCH_MARKERS = (
    "调研", "分析一下", "查一下", "检查一下", "核查", "对比", "评估", "总结",
    "搜索", "找一下", "看看", "研究", "排查",
)
_SIMPLE_PREFIXES = (
    "现在", "目前", "是不是", "是否", "为什么", "怎么", "多少", "哪个", "什么",
    "能不能", "有没有", "还有吗",
)


@dataclass(frozen=True)
class AdaptiveTurnBudget:
    tier: str
    iterations: int
    tool_calls: int | None
    reason: str


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def configure_adaptive_turn_budget(agent: Any, agent_section: dict[str, Any] | None) -> None:
    raw = (agent_section or {}).get("adaptive_turn_budget")
    cfg = raw if isinstance(raw, dict) else {}
    agent._adaptive_turn_budget_cfg = {
        "enabled": bool(cfg.get("enabled", False)),
        "simple_iterations": _positive_int(cfg.get("simple_iterations"), 3),
        "simple_tool_calls": _positive_int(cfg.get("simple_tool_calls"), 3),
        "normal_iterations": _positive_int(cfg.get("normal_iterations"), 8),
        "normal_tool_calls": _positive_int(cfg.get("normal_tool_calls"), 8),
        "research_iterations": _positive_int(cfg.get("research_iterations"), 18),
        "research_tool_calls": _positive_int(cfg.get("research_tool_calls"), 24),
    }


def _long_context(agent: Any, text: str) -> str | None:
    if os.environ.get("HERMES_KANBAN_TASK"):
        return "kanban_task"
    if getattr(agent, "_turn_origin", None):
        return "nested_or_background_turn"
    if getattr(agent, "_goal_manager", None) is not None:
        state = getattr(getattr(agent, "_goal_manager", None), "state", None)
        if getattr(state, "active", False):
            return "active_goal"
    stripped = text.strip()
    lowered = stripped.lower()
    if stripped in _LONG_EXACT:
        return "explicit_long_command"
    if any(marker in stripped for marker in _LONG_MARKERS):
        return "long_task_marker"
    if any(marker in lowered for marker in ("task_id", "branch", "pull request", " pr ", "sha=", "run_id")):
        return "task_control_marker"
    return None


def classify_turn(agent: Any, user_message: Any) -> tuple[str, str]:
    text = str(user_message or "").strip()
    long_reason = _long_context(agent, text)
    if long_reason:
        return "long", long_reason
    if text in _SIMPLE_EXACT:
        return "simple", "exact_status_or_ack"
    if len(text) <= 48 and (
        text.endswith(("?", "？"))
        or text.startswith(_SIMPLE_PREFIXES)
    ):
        return "simple", "short_question"
    if any(marker in text for marker in _RESEARCH_MARKERS):
        return "research", "bounded_research"
    if len(text) <= 240:
        return "normal", "bounded_human_turn"
    return "research", "longer_single_turn"


def apply_adaptive_turn_budget(agent: Any, user_message: Any) -> AdaptiveTurnBudget:
    base = max(1, int(getattr(agent, "max_iterations", 1) or 1))
    cfg = getattr(agent, "_adaptive_turn_budget_cfg", {}) or {}
    if not cfg.get("enabled", False):
        budget = AdaptiveTurnBudget("disabled", base, None, "disabled")
    else:
        tier, reason = classify_turn(agent, user_message)
        if tier == "long":
            budget = AdaptiveTurnBudget(tier, base, None, reason)
        elif tier == "simple":
            budget = AdaptiveTurnBudget(
                tier, min(base, cfg["simple_iterations"]), cfg["simple_tool_calls"], reason
            )
        elif tier == "normal":
            budget = AdaptiveTurnBudget(
                tier, min(base, cfg["normal_iterations"]), cfg["normal_tool_calls"], reason
            )
        else:
            budget = AdaptiveTurnBudget(
                tier, min(base, cfg["research_iterations"]), cfg["research_tool_calls"], reason
            )
    agent._effective_turn_max_iterations = budget.iterations
    agent._adaptive_tool_call_limit = budget.tool_calls
    agent._adaptive_tool_calls_used = 0
    agent._adaptive_turn_budget_tier = budget.tier
    agent._adaptive_turn_budget_reason = budget.reason
    agent.iteration_budget = IterationBudget(budget.iterations)
    return budget


def effective_turn_max_iterations(agent: Any) -> int:
    value = getattr(agent, "_effective_turn_max_iterations", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return max(1, int(getattr(agent, "max_iterations", 1) or 1))


def partition_tool_calls(
    agent: Any, tool_calls: Sequence[Any]
) -> tuple[list[Any], list[Any]]:
    """Reserve the remaining per-turn tool-call budget in original order."""
    calls = list(tool_calls or [])
    limit = getattr(agent, "_adaptive_tool_call_limit", None)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return calls, []
    used = max(0, int(getattr(agent, "_adaptive_tool_calls_used", 0) or 0))
    remaining = max(0, limit - used)
    allowed, skipped = calls[:remaining], calls[remaining:]
    agent._adaptive_tool_calls_used = used + len(allowed)
    return allowed, skipped


def tool_budget_exhausted_content(agent: Any, tool_name: str) -> str:
    payload = {
        "error": "adaptive_tool_budget_exhausted",
        "tool": tool_name,
        "message": (
            "This turn tool-call budget is exhausted. Do not call more tools in this "
            "turn; answer from the evidence already gathered, or state the remaining limitation."
        ),
        "used": int(getattr(agent, "_adaptive_tool_calls_used", 0) or 0),
        "limit": getattr(agent, "_adaptive_tool_call_limit", None),
    }
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "AdaptiveTurnBudget",
    "apply_adaptive_turn_budget",
    "classify_turn",
    "configure_adaptive_turn_budget",
    "effective_turn_max_iterations",
    "partition_tool_calls",
    "tool_budget_exhausted_content",
]
