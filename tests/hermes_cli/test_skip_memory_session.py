"""Session-scoped memory isolation leaves Hermes policy context and normal sessions unchanged."""
from __future__ import annotations

import os
import sys
import types

from hermes_cli._parser import build_top_level_parser


def test_skip_memory_flag_is_independent_of_rules():
    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["chat", "--skip-memory"])
    assert args.skip_memory is True
    assert args.ignore_rules is False
    assert args.ignore_user_config is False
    normal = parser.parse_args(["chat"])
    assert not getattr(normal, "skip_memory", False)


def test_cmd_chat_skip_memory_sets_only_memory_env(monkeypatch):
    import hermes_cli.main as main_mod

    for name in ("HERMES_SKIP_MEMORY", "HERMES_IGNORE_RULES",
                 "HERMES_IGNORE_USER_CONFIG", "HERMES_SAFE_MODE"):
        monkeypatch.delenv(name, raising=False)
    parser, _, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(["chat", "--skip-memory"])
    fake_cli = types.ModuleType("cli")
    recorded = {}
    def fake_main(**kwargs):
        recorded.update(kwargs)
    fake_cli.main = fake_main
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    try:
        main_mod.cmd_chat(args)
        assert os.environ.get("HERMES_SKIP_MEMORY") == "1"
        assert not os.environ.get("HERMES_IGNORE_RULES")
        assert not os.environ.get("HERMES_IGNORE_USER_CONFIG")
        assert recorded["ignore_rules"] is False
        assert recorded["ignore_user_config"] is False
    finally:
        os.environ.pop("HERMES_SKIP_MEMORY", None)


def test_agent_setup_keep_rules_when_skipping_memory():
    import inspect
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    source = inspect.getsource(CLIAgentSetupMixin)
    assert "skip_context_files=self.ignore_rules" in source
    assert 'os.environ.get("HERMES_SKIP_MEMORY") == "1"' in source
    assert "skip_memory=(self.ignore_rules or " in source
