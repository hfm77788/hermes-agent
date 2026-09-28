from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.teaching_decision_layer import register
from plugins.teaching_decision_layer import engine


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_perception_thresholds_and_actions(isolated_home: Path) -> None:
    def fake_provider(**_kwargs):
        return "jev-1.test", {
            "is_student_handwriting": 0.96,
            "is_original_content": 0.03,
            "ocr_uncertain": 0.50,
        }

    result = engine.judge(
        layer="perception",
        evidence={"ink_color": "blue", "font_match": 0.08},
        checks=["is_student_handwriting", "is_original_content", "ocr_uncertain"],
        provider_call=fake_provider,
    )

    assert result["ok"] is True
    assert result["resolved_model"] == "jev-1.test"
    assert result["yes_threshold"] == 0.90
    assert result["no_threshold"] == 0.10
    assert result["decisions"]["is_student_handwriting"] == {
        "probability": 0.96,
        "verdict": "yes",
        "next_action": "accept_classification",
    }
    assert result["decisions"]["is_original_content"]["verdict"] == "no"
    assert result["decisions"]["ocr_uncertain"] == {
        "probability": 0.5,
        "verdict": "gray",
        "next_action": "visual_recheck",
    }


def test_pedagogy_thresholds_and_actions(isolated_home: Path) -> None:
    def fake_provider(**_kwargs):
        return "jev-1.test", {
            "understanding_demonstrated": 0.91,
            "guessing_likely": 0.84,
            "needs_followup_question": 0.55,
            "ready_to_advance": 0.12,
            "learning_objective_met": 0.81,
            "response_too_entertaining": 0.85,
        }

    result = engine.judge(
        layer="pedagogy",
        evidence={
            "answer_correct": True,
            "reasoning_explained": False,
            "target_sentence_uses": 1,
        },
        provider_call=fake_provider,
    )

    assert result["yes_threshold"] == 0.80
    assert result["no_threshold"] == 0.20
    assert result["decisions"]["understanding_demonstrated"]["next_action"] == "count_as_understanding_evidence"
    assert result["decisions"]["guessing_likely"]["next_action"] == "ask_feynman_followup"
    assert result["decisions"]["needs_followup_question"] == {
        "probability": 0.55,
        "verdict": "gray",
        "next_action": "teacher_model_review",
    }
    assert result["decisions"]["ready_to_advance"]["next_action"] == "stay_on_objective"
    assert result["decisions"]["learning_objective_met"]["next_action"] == "objective_met"
    assert result["decisions"]["response_too_entertaining"]["next_action"] == "return_to_learning_objective"


def test_provider_failure_defers_without_blocking(isolated_home: Path) -> None:
    def failing_provider(**_kwargs):
        raise TimeoutError("slow")

    result = engine.judge(
        layer="pedagogy",
        evidence={"answer_correct": True},
        checks=["ready_to_advance"],
        provider_call=failing_provider,
    )

    assert result["ok"] is True
    assert result["fallback_used"] is True
    assert result["fallback_reason"] == "provider_unavailable:TimeoutError"
    assert result["decisions"]["ready_to_advance"] == {
        "probability": None,
        "verdict": "defer",
        "next_action": "teacher_model_review",
    }


def test_invalid_layer_checks_and_empty_evidence_do_not_call_provider(isolated_home: Path) -> None:
    def must_not_call(**_kwargs):
        raise AssertionError("provider must not run")

    assert engine.judge(layer="other", evidence={"x": 1}, provider_call=must_not_call)["error"] == "invalid_layer"
    assert engine.judge(layer="perception", evidence={}, provider_call=must_not_call)["error"] == "evidence_required"
    invalid = engine.judge(
        layer="perception",
        evidence={"x": 1},
        checks=["understanding_demonstrated"],
        provider_call=must_not_call,
    )
    assert invalid["error"] == "invalid_checks"


def test_privacy_redaction_and_audit_never_store_raw_evidence(
    isolated_home: Path,
) -> None:
    raw_name = "PRIVATE_STUDENT_NAME_123"
    raw_chat = "PRIVATE_CHAT_ID_456"
    raw_token = "sk-private-secret-789"

    clean = engine.sanitize_evidence(
        {
            "student_name": raw_name,
            "chat_id": raw_chat,
            "notes": f"Bearer abcDEF123 token={raw_token}",
            "visual": {"ink_color": "blue"},
        }
    )
    encoded = json.dumps(clean, ensure_ascii=False)
    assert raw_name not in encoded
    assert raw_chat not in encoded
    assert raw_token not in encoded
    assert "abcDEF123" not in encoded
    assert clean["visual"]["ink_color"] == "blue"
    assert engine.sanitize_evidence({"region_name": "answer_box"})["region_name"] == "answer_box"
    text_clean = engine.sanitize_evidence(
        {"note": "mail test@example.com phone 13812345678"}
    )
    assert "test@example.com" not in text_clean["note"]
    assert "13812345678" not in text_clean["note"]

    def fake_provider(**_kwargs):
        return "jev-1.test", {"answer_complete": 0.99}

    result = engine.judge(
        layer="perception",
        evidence={
            "student_name": raw_name,
            "chat_id": raw_chat,
            "notes": raw_token,
            "answer_area": "complete",
        },
        checks=["answer_complete"],
        provider_call=fake_provider,
    )
    assert result["ok"] is True

    audit = (
        isolated_home / "state" / "teaching-decision-layer" / "decisions.jsonl"
    ).read_text(encoding="utf-8")
    assert raw_name not in audit
    assert raw_chat not in audit
    assert raw_token not in audit
    assert '"evidence_hash"' in audit
    assert '"answer_area"' not in audit


def test_oversized_evidence_is_bounded(isolated_home: Path) -> None:
    payload = {"answer_region_detected": True}
    payload.update({f"blob_{i}": "x" * 1200 for i in range(20)})
    clean = engine.sanitize_evidence(payload)
    assert clean["bounded"] is True
    assert len(json.dumps(clean, ensure_ascii=False).encode("utf-8")) < engine.MAX_EVIDENCE_BYTES
    assert clean["answer_region_detected"] is True


def test_status_reports_counts_without_evidence(isolated_home: Path) -> None:
    def fake_provider(**_kwargs):
        return "jev-1.test", {"ready_to_advance": 0.95}

    engine.judge(
        layer="pedagogy",
        evidence={"answer_correct": True},
        checks=["ready_to_advance"],
        provider_call=fake_provider,
    )
    status = engine.status(model="jev-latest", timeout=1.5)
    assert status["ok"] is True
    assert status["provider_ready"] is False
    assert status["decision_batches"] == 1
    assert status["fallback_batches"] == 0
    assert status["resolved_models"] == ["jev-1.test"]
    assert status["privacy"]["raw_evidence_logged"] is False
    assert "evidence" not in status


class _FakeContext:
    def __init__(self):
        self.tools = {}

    def get_config(self, key, default=None):
        return {
            "jev_model": "jev-custom",
            "timeout_seconds": 2.25,
        }.get(key, default)

    def register_tool(self, *, name, toolset, schema, handler, emoji):
        self.tools[name] = {
            "toolset": toolset,
            "schema": schema,
            "handler": handler,
            "emoji": emoji,
        }


def test_register_exposes_only_three_explicit_tools() -> None:
    ctx = _FakeContext()
    register(ctx)
    assert set(ctx.tools) == {
        "teaching_perception_judge",
        "teaching_pedagogy_judge",
        "teaching_decision_status",
    }
    assert all(item["toolset"] == "teaching_decision" for item in ctx.tools.values())
    assert ctx.tools["teaching_perception_judge"]["schema"]["parameters"]["required"] == ["evidence"]
    assert ctx.tools["teaching_decision_status"]["schema"]["parameters"]["additionalProperties"] is False


def test_check_catalog_has_exactly_twelve_judgments() -> None:
    all_checks = set(engine.PERCEPTION_CHECKS) | set(engine.PEDAGOGY_CHECKS)
    assert len(engine.PERCEPTION_CHECKS) == 6
    assert len(engine.PEDAGOGY_CHECKS) == 6
    assert len(all_checks) == 12
