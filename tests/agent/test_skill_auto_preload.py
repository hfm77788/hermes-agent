"""Tests for runtime auto skill preload routing."""

from agent import skill_commands


def test_auto_preload_pr_uses_deterministic_code_bundle():
    slugs = skill_commands._matched_auto_preload_slugs("请创建PR并push这个修复".lower())

    assert slugs[:2] == ["code-task-execution", "github-pr-workflow"]


def test_auto_preload_ignores_broad_new_keyword():
    slugs = skill_commands._matched_auto_preload_slugs("new idea for tomorrow".lower())

    assert "github-pr-workflow" not in slugs


def test_l1_compact_classification_requires_fast_keyword_without_risk():
    assert skill_commands._is_l1_fast_message("小修 一个 typo") is True
    assert skill_commands._is_l1_fast_message("小修 但要创建PR") is False
    assert skill_commands._is_l1_fast_message("quick fix auth regression") is False


def test_build_auto_preload_uses_compact_l1(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    text = skill_commands.build_auto_skill_preload_context("小修：修一个 typo")

    assert "AUTO-PRELOADED-COMPACT" in text
    assert "L1 Compact Fast Path" in text
    assert "[skill: code-task-execution]" in text


def test_build_auto_preload_full_bundle_for_pr(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    seen = []

    def fake_build_skill_invocation_message(cmd_key, **kwargs):
        seen.append(cmd_key)
        return f"loaded {cmd_key}"

    monkeypatch.setattr(skill_commands, "build_skill_invocation_message", fake_build_skill_invocation_message)

    text = skill_commands.build_auto_skill_preload_context("创建PR并push")

    assert seen == ["/code-task-execution", "/github-pr-workflow"]
    assert "loaded /code-task-execution" in text
    assert "loaded /github-pr-workflow" in text
