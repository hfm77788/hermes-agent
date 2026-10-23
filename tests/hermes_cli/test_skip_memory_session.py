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


def test_agent_construction_skips_memory_but_keeps_rules(monkeypatch):
    """Inspect actual AIAgent constructor arguments, not source formatting."""
    from unittest.mock import MagicMock
    import run_agent
    import cli
    import hermes_cli.mcp_startup as mcp_startup
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    fake = MagicMock()
    fake.agent = None
    fake._session_db = object()
    fake._resumed = False
    fake._single_query_mode = True
    fake._auto_load_skills_result = None
    fake._pending_title = None
    fake.ignore_rules = False
    fake.system_prompt = ""
    fake.model = "unit-test/model"
    fake.prefill_messages = None
    fake._inline_diffs_enabled = False
    fake._ensure_runtime_credentials.return_value = True
    args = {}

    def fake_agent(**kwargs):
        args.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(run_agent, "AIAgent", fake_agent)
    # Upstream moved CLI turns onto the gateway agent: _init_agent imports GatewayAgent
    # under the AIAgent name, so patch the symbol it actually constructs.
    import hermes_cli.gateway_agent as gateway_agent_mod
    monkeypatch.setattr(gateway_agent_mod, "GatewayAgent", fake_agent)
    monkeypatch.setattr(cli, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **kw: None)
    monkeypatch.setattr(cli, "_active_agent_ref", None)
    monkeypatch.setenv("HERMES_SKIP_MEMORY", "1")
    success = CLIAgentSetupMixin._init_agent(
        fake, model_override="unit-test/model",
        runtime_override={"provider": "auto", "api_mode": "openai"})
    assert success is True
    assert args["skip_memory"] is True
    assert args["skip_context_files"] is False


def test_tui_skip_memory_rejected_before_agent_creation(monkeypatch):
    import pytest
    import hermes_cli.main as main_mod

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["chat", "--skip-memory", "--tui"])
    monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda args: None)
    monkeypatch.setattr(main_mod, "_apply_user_config_bypass", lambda args: None)
    monkeypatch.setattr(main_mod, "_guard_noninteractive_user_config", lambda args: None)
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda args: True)
    with pytest.raises(SystemExit, match="--skip-memory requires --cli"):
        main_mod.cmd_chat(args)
