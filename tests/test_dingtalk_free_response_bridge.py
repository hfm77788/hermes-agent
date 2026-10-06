from __future__ import annotations

import pytest

from scripts.dingtalk_free_response_bridge import (
    _normalize_dingtalk_math_reply,
    _prepare_dingtalk_reply,
)

MATH_SKILLS = ("huangshang-math-tutor", "jiayin-math-tutor")


@pytest.mark.parametrize("skill", MATH_SKILLS)
def test_math_routes_render_user_sample_as_plain_text(skill: str) -> None:
    raw = r"""不等式练习
1. 解不等式 $\dfrac{2x-1}{x+3} \ge 1$
2. 已知 A = {x | x²−5x+6 = 0}，若 B\subseteq A，求 m。
3. q: x<1+a 或 x>1-a
4. $|2x+1| \ge |x-2|$
5. x + $\dfrac{4}{x-1}$
6. x∈[−1, 2]，使得 x²−2x−a \ge 0"""
    rendered = _prepare_dingtalk_reply(skill, raw)
    assert "(2x-1)/(x+3) ≥ 1" in rendered
    assert "B⊆ A" in rendered
    assert "|2x+1| ≥ |x-2|" in rendered
    assert "(4)/(x-1)" in rendered
    assert "x²−2x−a ≥ 0" in rendered
    assert "$" not in rendered
    assert "\\" not in rendered
    assert chr(96) not in rendered


@pytest.mark.parametrize("skill", MATH_SKILLS)
def test_math_routes_share_the_same_plain_text_gate(skill: str) -> None:
    assert _prepare_dingtalk_reply(skill, r"$x^2 \le 9$") == "x² ≤ 9"


def test_non_math_route_is_not_rewritten() -> None:
    raw = r"$E=mc^2$"
    assert _prepare_dingtalk_reply("jiayin-physics-tutor", raw) == raw


def test_supported_sqrt_and_markdown_are_rendered_without_markup() -> None:
    raw = "### 题目\n**求值**：$\\sqrt{9} \\times 2$\n\n| 项 | 值 |\n|---|---|\n| A | 6 |"
    rendered = _normalize_dingtalk_math_reply(raw)
    assert rendered.startswith("题目\n求值：")
    assert "√(9) × 2" in rendered
    assert "项  值" in rendered
    assert "A  6" in rendered
    assert "**" not in rendered
    assert "|" not in rendered


def test_unknown_latex_fails_closed_instead_of_sending_source() -> None:
    with pytest.raises(ValueError, match="dingtalk_math_markup_residual"):
        _normalize_dingtalk_math_reply(r"$\begin{cases}x>1\end{cases}$")
