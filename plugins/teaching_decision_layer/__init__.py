from __future__ import annotations

import json
from typing import Any

from .engine import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    PEDAGOGY_CHECKS,
    PERCEPTION_CHECKS,
    judge,
    status,
)


def _schema(name: str, description: str, checks: tuple[str, ...]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "evidence": {
                    "type": "object",
                    "description": (
                        "De-identified structured evidence only. Do not pass raw images, names, school/contact/chat IDs, "
                        "credentials, or full transcripts. Vision/OCR tools should inspect the image first and summarize "
                        "only the relevant layout, OCR, ink, region, answer, and interaction evidence here."
                    ),
                },
                "checks": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(checks)},
                    "uniqueItems": True,
                    "description": "Optional subset of judgments. Omit to run all checks in this layer.",
                },
            },
            "required": ["evidence"],
            "additionalProperties": False,
        },
    }


PERCEPTION_SCHEMA = _schema(
    "teaching_perception_judge",
    (
        "Teaching Decision Layer visual/perception judge. Use only after vision/OCR has produced structured evidence; "
        "this tool does not read images. It judges handwriting vs original content vs teacher annotation, OCR uncertainty, "
        "answer-region detection, and answer completeness. yes>=0.90, no<=0.10, gray=>visual_recheck."
    ),
    PERCEPTION_CHECKS,
)

PEDAGOGY_SCHEMA = _schema(
    "teaching_pedagogy_judge",
    (
        "Teaching Decision Layer pedagogy judge. Use only at learning decision hinges, not every message. It judges "
        "demonstrated understanding, likely guessing, need for follow-up, readiness to advance, objective completion, "
        "and whether interaction is becoming entertaining without enough learning progress. "
        "yes>=0.80, no<=0.20, gray=>teacher_model_review."
    ),
    PEDAGOGY_CHECKS,
)

STATUS_SCHEMA = {
    "name": "teaching_decision_status",
    "description": "Read Teaching Decision Layer model, thresholds, audit count, and fallback health without learner evidence.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def _settings(ctx) -> tuple[str, float]:
    model = str(ctx.get_config("jev_model", DEFAULT_MODEL) or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    raw_timeout = ctx.get_config("timeout_seconds", DEFAULT_TIMEOUT)
    try:
        timeout = max(0.2, min(5.0, float(raw_timeout)))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    return model, timeout


def register(ctx) -> None:
    model, timeout = _settings(ctx)

    def perception_handler(args: dict[str, Any], **_kw: Any) -> str:
        return json.dumps(
            judge(
                layer="perception",
                evidence=args.get("evidence") or {},
                checks=args.get("checks"),
                model=model,
                timeout=timeout,
            ),
            ensure_ascii=False,
        )

    def pedagogy_handler(args: dict[str, Any], **_kw: Any) -> str:
        return json.dumps(
            judge(
                layer="pedagogy",
                evidence=args.get("evidence") or {},
                checks=args.get("checks"),
                model=model,
                timeout=timeout,
            ),
            ensure_ascii=False,
        )

    def status_handler(_args: dict[str, Any], **_kw: Any) -> str:
        return json.dumps(status(model=model, timeout=timeout), ensure_ascii=False)

    ctx.register_tool(
        name="teaching_perception_judge",
        toolset="teaching_decision",
        schema=PERCEPTION_SCHEMA,
        handler=perception_handler,
        emoji="👁️",
    )
    ctx.register_tool(
        name="teaching_pedagogy_judge",
        toolset="teaching_decision",
        schema=PEDAGOGY_SCHEMA,
        handler=pedagogy_handler,
        emoji="🎓",
    )
    ctx.register_tool(
        name="teaching_decision_status",
        toolset="teaching_decision",
        schema=STATUS_SCHEMA,
        handler=status_handler,
        emoji="📊",
    )
