from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "jev-aliyun-qwen-router" / "__init__.py"


def _load_plugin():
    name = "test_jev_aliyun_qwen_router_plugin"
    spec = importlib.util.spec_from_file_location(name, PLUGIN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod(monkeypatch):
    module = _load_plugin()
    module._TURN_CACHE.clear()
    monkeypatch.setattr(module, "_root_plugin_settings", lambda: {})
    monkeypatch.setattr(module, "_active_profile_name", lambda: "default")
    return module


class Ctx:
    def __init__(self, settings=None):
        self.settings = dict(settings or {})
        self.middleware = []

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_middleware(self, kind, callback):
        self.middleware.append((kind, callback))


def _request(model="qwen3.8-flash", text="Analyze this bounded task.", **extra):
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def _response(
    effort="medium",
    confidence=0.94,
    max_probability=0.1,
    model_choice=None,
    model_confidence=0.95,
):
    if model_choice is None:
        model_choice = "max" if max_probability >= 0.5 else "flash"
    return {
        "model": "jev-1.13.0",
        "answers": {
            "effort": {
                "type": "choice",
                "choice": effort,
                "confidence": confidence,
                "probabilities": {
                    "low": 0.02 if effort != "low" else confidence,
                    "medium": 0.02 if effort != "medium" else confidence,
                    "xhigh": 0.02 if effort != "xhigh" else confidence,
                },
            },
            "model_class": {
                "type": "choice",
                "choice": model_choice,
                "confidence": model_confidence,
                "probabilities": {
                    "flash": 1.0 - max_probability,
                    "max": max_probability,
                },
            },
        },
    }


def _route(
    mod,
    ctx,
    request,
    *,
    turn_id="turn-1",
    provider="custom:aliyun_ws",
    base_url="",
):
    return mod.route_request(
        ctx,
        request=request,
        provider=provider,
        base_url=base_url,
        api_mode="chat_completions",
        model=request.get("model", ""),
        turn_id=turn_id,
        session_id="session-1",
        api_call_count=1,
    )


def test_registers_only_llm_request_middleware(mod):
    ctx = Ctx()
    mod.register(ctx)
    assert len(ctx.middleware) == 1
    assert ctx.middleware[0][0] == "llm_request"
    assert callable(ctx.middleware[0][1])


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("low", "low"), ("medium", "medium"), ("xhigh", "xhigh")],
)
def test_routes_flash_effort(mod, monkeypatch, effort, expected):
    monkeypatch.setattr(mod, "_call_jev", lambda **_: (_response(effort=effort), 123))
    result = _route(mod, Ctx(), _request())
    assert result["request"]["model"] == "qwen3.8-flash"
    assert result["request"]["reasoning_effort"] == expected
    assert result["reason"] == f"flash_{expected}"


def test_escalates_to_max_only_above_high_confidence_gate(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_call_jev",
        lambda **_: (_response(effort="xhigh", confidence=0.97, max_probability=0.95), 234),
    )
    result = _route(mod, Ctx(), _request())
    assert result["request"]["model"] == "qwen3.8-max-0902"
    assert result["request"]["reasoning_effort"] == "xhigh"
    assert result["reason"] == "max_xhigh"




def test_max_choice_requires_choice_confidence_too(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_call_jev",
        lambda **_: (
            _response(
                effort="xhigh",
                confidence=0.98,
                max_probability=0.96,
                model_choice="max",
                model_confidence=0.60,
            ),
            1,
        ),
    )
    result = _route(mod, Ctx(), _request())
    assert result["request"]["model"] == "qwen3.8-flash"
    assert result["reason"] == "flash_xhigh"

def test_does_not_escalate_max_in_gray_zone(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_call_jev",
        lambda **_: (_response(effort="xhigh", confidence=0.97, max_probability=0.89), 234),
    )
    result = _route(mod, Ctx(), _request())
    assert result["request"]["model"] == "qwen3.8-flash"
    assert result["reason"] == "flash_xhigh"


def test_low_effort_plus_max_required_is_treated_as_inconsistent(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_call_jev",
        lambda **_: (_response(effort="low", confidence=0.97, max_probability=0.99), 111),
    )
    assert _route(mod, Ctx(), _request()) is None


def test_low_effort_confidence_fails_open_to_original_request(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_call_jev",
        lambda **_: (_response(effort="medium", confidence=0.49, max_probability=0.1), 111),
    )
    original = _request()
    assert _route(mod, Ctx(), original) is None
    assert "reasoning_effort" not in original
    assert original["model"] == "qwen3.8-flash"


def test_fallback_is_sticky_for_the_whole_turn(mod, monkeypatch):
    calls = []

    def uncertain(**_):
        calls.append(1)
        return _response(effort="medium", confidence=0.49, max_probability=0.1), 10

    monkeypatch.setattr(mod, "_call_jev", uncertain)
    assert _route(mod, Ctx(), _request(), turn_id="fallback-turn") is None
    assert _route(mod, Ctx(), _request(), turn_id="fallback-turn") is None
    assert len(calls) == 1


def test_provider_failure_fails_open_to_original_request(mod, monkeypatch):
    def boom(**_):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(mod, "_call_jev", boom)
    original = _request()
    assert _route(mod, Ctx(), original) is None
    assert original == _request()


def test_same_turn_reuses_one_jev_decision_across_tool_loop(mod, monkeypatch):
    calls = []

    def fake(**_):
        calls.append(1)
        return _response(effort="medium", max_probability=0.1), 88

    monkeypatch.setattr(mod, "_call_jev", fake)
    first = _route(mod, Ctx(), _request(), turn_id="stable-turn")
    second = _route(
        mod,
        Ctx(),
        {
            "model": "qwen3.8-flash",
            "messages": [
                {"role": "user", "content": "Analyze this bounded task."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
                {"role": "tool", "content": "private tool output that must never reach Jev"},
            ],
        },
        turn_id="stable-turn",
    )
    assert len(calls) == 1
    assert first["request"]["reasoning_effort"] == "medium"
    assert second["request"]["reasoning_effort"] == "medium"


def test_only_latest_user_text_is_sent_and_secrets_are_force_redacted(mod, monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured["state"] = kwargs["state"]
        return _response(), 50

    monkeypatch.setattr(mod, "_call_jev", fake)
    secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
    request = {
        "model": "qwen3.8-flash",
        "messages": [
            {"role": "user", "content": "old message"},
            {"role": "assistant", "content": "old response"},
            {"role": "tool", "content": "PRIVATE_TOOL_OUTPUT"},
            {"role": "user", "content": f"new task with token {secret}"},
        ],
    }
    result = _route(mod, Ctx(), request)
    assert result is not None
    state = captured["state"]
    assert secret not in state
    assert "PRIVATE_TOOL_OUTPUT" not in state
    assert "old message" not in state
    decoded = json.loads(state)
    assert "new task" in decoded["latest_user_message"]


def test_wrong_provider_and_missing_turn_id_are_not_routed(mod, monkeypatch):
    monkeypatch.setattr(mod, "_call_jev", lambda **_: (_response(), 1))
    assert _route(mod, Ctx(), _request(), provider="custom:aliyun_qwen") is None
    assert _route(mod, Ctx(), _request(), turn_id="") is None


def test_normalized_custom_provider_matches_allowlisted_configured_base_url(mod, monkeypatch):
    monkeypatch.setattr(mod, "_call_jev", lambda **_: (_response(effort="medium"), 1))
    monkeypatch.setattr(
        mod,
        "_configured_provider_base_urls",
        lambda: {"custom:aliyun_ws": "https://ws.example.com/compatible-mode/v1"},
    )
    result = _route(
        mod,
        Ctx(),
        _request(),
        provider="custom",
        base_url="https://ws.example.com/compatible-mode/v1/",
    )
    assert result is not None
    assert result["source"] == "jev-aliyun-qwen-router"
    assert result["reason"] == "flash_medium"


def test_normalized_custom_provider_rejects_other_custom_endpoint(mod, monkeypatch):
    calls = []

    def fake(**_):
        calls.append(1)
        return _response(effort="xhigh", max_probability=0.99), 1

    monkeypatch.setattr(mod, "_call_jev", fake)
    monkeypatch.setattr(
        mod,
        "_configured_provider_base_urls",
        lambda: {
            "custom:aliyun_ws": "https://ws.example.com/compatible-mode/v1",
            "custom:aliyun_qwen": "https://beijing.example.com/compatible-mode/v1",
        },
    )
    assert _route(
        mod,
        Ctx(),
        _request(),
        provider="custom",
        base_url="https://beijing.example.com/compatible-mode/v1",
    ) is None
    assert calls == []


def test_provider_default_medium_is_routeable_by_default(mod, monkeypatch):
    monkeypatch.setattr(mod, "_call_jev", lambda **_: (_response(effort="low"), 1))
    request = _request(reasoning_effort="medium")
    result = _route(mod, Ctx(), request)
    assert result["request"]["reasoning_effort"] == "low"


def test_existing_reasoning_can_be_respected_when_operator_opts_in(mod, monkeypatch):
    monkeypatch.setattr(mod, "_call_jev", lambda **_: (_response(), 1))
    request = _request(reasoning_effort="xhigh")
    assert _route(mod, Ctx({"respect_existing_reasoning": True}), request) is None


def test_existing_thinking_budget_can_be_replaced_when_explicitly_configured(mod, monkeypatch):
    monkeypatch.setattr(mod, "_call_jev", lambda **_: (_response(effort="medium"), 1))
    ctx = Ctx({"respect_existing_reasoning": False})
    request = _request(thinking_budget=1234, extra_body={"thinking_budget": 1234, "keep": True})
    result = _route(mod, ctx, request)
    assert result["request"]["reasoning_effort"] == "medium"
    assert "thinking_budget" not in result["request"]
    assert result["request"]["extra_body"] == {"keep": True}


def test_manual_max_is_never_silently_downgraded(mod, monkeypatch):
    monkeypatch.setattr(
        mod, "_call_jev", lambda **_: (_response(effort="medium", max_probability=0.01), 1)
    )
    result = _route(mod, Ctx(), _request(model="qwen3.8-max-0902"))
    assert result["request"]["model"] == "qwen3.8-max-0902"
    assert result["request"]["reasoning_effort"] == "medium"
    assert result["reason"] == "max_medium"


def test_custom_thresholds_and_models_are_honored(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_call_jev",
        lambda **_: (_response(effort="medium", confidence=0.81, max_probability=0.85), 1),
    )
    ctx = Ctx(
        {
            "min_effort_confidence": 0.80,
            "max_escalation_probability": 0.84,
            "flash_model": "qwen-flash-custom",
            "max_model": "qwen-max-custom",
        }
    )
    request = _request(model="qwen-flash-custom")
    result = _route(mod, ctx, request)
    assert result["request"]["model"] == "qwen-max-custom"
    assert result["request"]["reasoning_effort"] == "medium"



def _central_settings():
    return {
        "centralized_profile_policy": True,
        "providers": ["custom:aliyun_ws"],
        "flash_model": "qwen3.8-flash",
        "max_model": "qwen3.8-max-0902",
        "min_effort_confidence": 0.50,
        "max_escalation_probability": 0.90,
        "min_max_choice_confidence": 0.80,
        "default_policy": "standard",
        "profile_policies": {
            "chief-engineer": "deep",
            "hema-teacher": "deep",
            "office-director": "deep",
        },
        "policies": {
            "standard": {"min_reasoning_effort": "low"},
            "deep": {"min_reasoning_effort": "medium"},
        },
    }


def test_central_policy_applies_profile_effort_floor_and_ignores_local_settings(mod, monkeypatch):
    monkeypatch.setattr(mod, "_root_plugin_settings", _central_settings)
    monkeypatch.setattr(mod, "_active_profile_name", lambda: "chief-engineer")
    monkeypatch.setattr(mod, "_call_jev", lambda **_: (_response(effort="low"), 1))
    local = Ctx({"flash_model": "wrong-local-model", "min_reasoning_effort": "xhigh"})
    result = _route(mod, local, _request(), turn_id="deep-turn")
    assert result["request"]["model"] == "qwen3.8-flash"
    assert result["request"]["reasoning_effort"] == "medium"
    assert result["reason"] == "flash_medium"


def test_central_policy_default_profile_keeps_low_effort(mod, monkeypatch):
    monkeypatch.setattr(mod, "_root_plugin_settings", _central_settings)
    monkeypatch.setattr(mod, "_active_profile_name", lambda: "worker-general")
    monkeypatch.setattr(mod, "_call_jev", lambda **_: (_response(effort="low"), 1))
    result = _route(mod, Ctx(), _request(), turn_id="standard-turn")
    assert result["request"]["reasoning_effort"] == "low"
    assert result["reason"] == "flash_low"


def test_central_policy_preserves_same_max_gate_for_deep_role(mod, monkeypatch):
    monkeypatch.setattr(mod, "_root_plugin_settings", _central_settings)
    monkeypatch.setattr(mod, "_active_profile_name", lambda: "office-director")
    monkeypatch.setattr(
        mod,
        "_call_jev",
        lambda **_: (_response(effort="xhigh", max_probability=0.89, model_choice="max"), 1),
    )
    result = _route(mod, Ctx(), _request(), turn_id="deep-gray-zone")
    assert result["request"]["model"] == "qwen3.8-flash"
    assert result["request"]["reasoning_effort"] == "xhigh"


def test_turn_cache_is_scoped_by_profile(mod, monkeypatch):
    monkeypatch.setattr(mod, "_root_plugin_settings", _central_settings)
    active = {"profile": "default"}
    monkeypatch.setattr(mod, "_active_profile_name", lambda: active["profile"])
    calls = []

    def fake(**_):
        calls.append(active["profile"])
        return _response(effort="low"), 1

    monkeypatch.setattr(mod, "_call_jev", fake)
    first = _route(mod, Ctx(), _request(), turn_id="shared-turn")
    active["profile"] = "chief-engineer"
    second = _route(mod, Ctx(), _request(), turn_id="shared-turn")
    assert calls == ["default", "chief-engineer"]
    assert first["request"]["reasoning_effort"] == "low"
    assert second["request"]["reasoning_effort"] == "medium"



def test_typesafe_key_legacy_mode_does_not_cross_profile_boundary(mod, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert mod._typesafe_key(centralized=False) == ""


def test_route_passes_centralized_flag_to_jev_call(mod, monkeypatch):
    monkeypatch.setattr(mod, "_root_plugin_settings", _central_settings)
    captured = {}

    def fake(**kwargs):
        captured["centralized"] = kwargs["centralized"]
        return _response(effort="medium"), 1

    monkeypatch.setattr(mod, "_call_jev", fake)
    result = _route(mod, Ctx(), _request(), turn_id="central-secret-turn")
    assert result is not None
    assert captured["centralized"] is True



def test_typesafe_key_centralized_reads_default_root_only(mod, monkeypatch):
    import hermes_cli.config as config_mod
    import hermes_constants as constants

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    seen = {}
    fake_token = object()

    monkeypatch.setattr(constants, "get_default_hermes_root", lambda: Path("/central/hermes"))

    def set_override(path):
        seen["set"] = str(path)
        return fake_token

    def reset_override(token):
        seen["reset"] = token

    monkeypatch.setattr(constants, "set_hermes_home_override", set_override)
    monkeypatch.setattr(constants, "reset_hermes_home_override", reset_override)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: "central-test-secret" if key == "TYPESAFE_API_KEY" else None)

    assert mod._typesafe_key(centralized=True) == "central-test-secret"
    assert seen == {"set": "/central/hermes", "reset": fake_token}
