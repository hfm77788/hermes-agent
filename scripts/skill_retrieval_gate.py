#!/usr/bin/env python3
"""Deterministic Skill Retrieval acceptance gate for CI and deployed runtimes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "hermes.skill_retrieval_acceptance.v1"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "skill_retrieval_acceptance.json"
REGRESSION_NODES = [
    "tests/agent/test_skill_retrieval.py",
    "tests/agent/test_prompt_builder.py::test_names_only_all_keeps_catalog_but_omits_static_descriptions",
    "tests/agent/test_prompt_builder.py::test_retrieval_catalog_respects_disabled_skill_visibility",
    "tests/agent/test_session_reset_fix.py::test_skill_retrieval_catalog_cleared_on_session_reset",
    "tests/agent/test_codex_app_server_integration.py::test_codex_app_server_receives_per_turn_skill_context",
]


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "hermes.skill_retrieval_acceptance_fixture.v1":
        raise ValueError("skill_retrieval_gate_fixture_schema_invalid")
    return payload


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return float(ordered[index])


def _expanded_catalog(base: list[dict[str, Any]], target_size: int) -> list[dict[str, Any]]:
    catalog = [dict(entry) for entry in base]
    index = 0
    while len(catalog) < target_size:
        catalog.append(
            {
                "name": f"zz-filler-skill-{index:03d}",
                "category": "benchmark",
                "description": f"zzfiller{index:03d} qx{index:03d}",
                "semantic_terms": [f"zzfiller{index:03d}", f"qx{index:03d}"],
            }
        )
        index += 1
    return catalog[:target_size]


def _quality(catalog: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
    from agent.skill_retrieval import retrieve_skills

    rows: list[dict[str, Any]] = []
    for item in queries:
        query = str(item["query"])
        expected = str(item["expected"])
        hits = retrieve_skills(query, catalog, top_k=8)
        names = [str(hit.get("name") or "") for hit in hits]
        rank = names.index(expected) + 1 if expected in names else None
        rows.append({"query": query, "expected": expected, "rank": rank, "hits": names[:5]})
    total = len(rows)
    top1 = sum(row["rank"] == 1 for row in rows)
    top5 = sum(isinstance(row["rank"], int) and row["rank"] <= 5 for row in rows)
    return {
        "total": total,
        "top1": top1,
        "top5": top5,
        "top1_rate": (top1 / total) if total else 0.0,
        "top5_rate": (top5 / total) if total else 0.0,
        "rows": rows,
    }


def _dynamic_context(catalog: list[dict[str, Any]], query: str, expected: str) -> dict[str, Any]:
    from agent.skill_retrieval import build_skill_retrieval_context

    context = build_skill_retrieval_context(query, catalog, top_k=8)
    candidate_count = sum(1 for line in context.splitlines() if line.startswith("- "))
    return {
        "has_relevant_skills": "<relevant_skills>" in context,
        "has_skill_view_instruction": "skill_view(name)" in context,
        "has_expected_skill": expected in context,
        "candidate_count": candidate_count,
    }


def _fallback(catalog: list[dict[str, Any]], query: str, expected: str) -> dict[str, Any]:
    import agent.skill_retrieval as sr

    original = sr._build_semantic_state
    try:
        sr.clear_skill_retrieval_cache()
        sr._build_semantic_state = lambda _catalog: None
        hits = sr.retrieve_skills(query, catalog, top_k=5)
    finally:
        sr._build_semantic_state = original
        sr.clear_skill_retrieval_cache()
    names = [str(hit.get("name") or "") for hit in hits]
    top = hits[0] if hits else {}
    return {
        "expected": expected,
        "rank": names.index(expected) + 1 if expected in names else None,
        "semantic_score": float(top.get("semantic_score") or 0.0),
        "hits": names,
    }


def _latency(
    catalog: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    batches: int = 3,
    repeats: int = 4,
) -> dict[str, Any]:
    from agent.skill_retrieval import clear_skill_retrieval_cache, retrieve_skills

    query_texts = [str(item["query"]) for item in queries]
    clear_skill_retrieval_cache()
    start = time.perf_counter()
    retrieve_skills(query_texts[0], catalog, top_k=8)
    cold_ms = (time.perf_counter() - start) * 1000.0

    for query in query_texts:
        retrieve_skills(query, catalog, top_k=8)

    all_values: list[float] = []
    batch_p95: list[float] = []
    for _ in range(batches):
        values: list[float] = []
        for _repeat in range(repeats):
            for query in query_texts:
                started = time.perf_counter()
                retrieve_skills(query, catalog, top_k=8)
                values.append((time.perf_counter() - started) * 1000.0)
        all_values.extend(values)
        batch_p95.append(_percentile(values, 0.95))
    return {
        "cold_ms": cold_ms,
        "warm_p50_ms": statistics.median(all_values) if all_values else 0.0,
        "warm_p95_ms": statistics.median(batch_p95) if batch_p95 else 0.0,
        "warm_max_ms": max(all_values) if all_values else 0.0,
        "samples": len(all_values),
    }


def _synthetic_prompt(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache

    with tempfile.TemporaryDirectory(prefix="skill-retrieval-gate-") as tmp:
        skills_root = Path(tmp) / "skills"
        for entry in catalog:
            name = str(entry["name"])
            folder = skills_root / "benchmark" / name
            folder.mkdir(parents=True, exist_ok=True)
            description = str(entry.get("description") or "")
            tags = list(entry.get("semantic_terms") or [])
            (folder / "SKILL.md").write_text(
                "---\n"
                f"name: {json.dumps(name, ensure_ascii=False)}\n"
                f"description: {json.dumps(description, ensure_ascii=False)}\n"
                f"tags: {json.dumps(tags, ensure_ascii=False)}\n"
                "---\n",
                encoding="utf-8",
            )
        clear_skills_system_prompt_cache(clear_snapshot=True)
        built_catalog: list[dict[str, Any]] = []
        prompt = build_skills_system_prompt(
            skills_dir_override=skills_root,
            names_only_all=True,
            catalog_out=built_catalog,
        )
        clear_skills_system_prompt_cache(clear_snapshot=True)

    prompt_bytes = len(prompt.encode("utf-8"))
    base_descriptions = [
        str(entry.get("description") or "")
        for entry in catalog
        if str(entry.get("description") or "").strip()
    ]
    leaked = [description for description in base_descriptions if description in prompt]
    return {
        "skill_count": len(built_catalog),
        "skills_index_bytes": prompt_bytes,
        "bytes_per_skill": (prompt_bytes / len(built_catalog)) if built_catalog else float("inf"),
        "description_leak_count": len(leaked),
    }


def _run_regressions() -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *REGRESSION_NODES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "duration_ms": (time.perf_counter() - started) * 1000.0,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-12:]),
    }


def _ci_receipt(fixture: dict[str, Any], *, run_regressions: bool) -> dict[str, Any]:
    from agent.skill_retrieval import clear_skill_retrieval_cache

    cfg = fixture["ci"]
    catalog = _expanded_catalog(fixture["synthetic_skills"], int(cfg["catalog_size"]))
    queries = fixture["ci_queries"]
    clear_skill_retrieval_cache()
    quality = _quality(catalog, queries)
    context_item = fixture["ci_context_query"]
    context = _dynamic_context(catalog, context_item["query"], context_item["expected"])
    fallback_item = fixture["ci_fallback_query"]
    fallback = _fallback(catalog, fallback_item["query"], fallback_item["expected"])
    prompt = _synthetic_prompt(catalog)
    latency = _latency(catalog, queries)
    regressions = _run_regressions() if run_regressions else {"passed": True, "skipped": True}

    checks = {
        "quality_top1": quality["top1_rate"] >= float(cfg["min_top1_rate"]),
        "quality_top5": quality["top5_rate"] >= float(cfg["min_top5_rate"]),
        "dynamic_injection": (
            context["has_relevant_skills"]
            and context["has_skill_view_instruction"]
            and context["has_expected_skill"]
            and context["candidate_count"] <= int(cfg["max_context_candidates"])
        ),
        "lexical_fallback": fallback["rank"] == 1 and fallback["semantic_score"] == 0.0,
        "static_prompt_names_only": prompt["description_leak_count"] == 0,
        "prompt_budget": prompt["bytes_per_skill"] <= float(cfg["max_static_bytes_per_skill"]),
        "latency_cold": latency["cold_ms"] <= float(cfg["max_cold_ms"]),
        "latency_warm_p95": latency["warm_p95_ms"] <= float(cfg["max_warm_p95_ms"]),
        "focused_regressions": bool(regressions.get("passed")),
    }
    return {
        "mode": "ci",
        "catalog_count": len(catalog),
        "quality": quality,
        "dynamic_context": context,
        "fallback": fallback,
        "prompt": prompt,
        "latency": latency,
        "regressions": regressions,
        "thresholds": cfg,
        "checks": checks,
    }


def _runtime_receipt(fixture: dict[str, Any], *, run_regressions: bool) -> dict[str, Any]:
    from agent.skill_retrieval import clear_skill_retrieval_cache
    from agent.system_prompt import _skills_prompt
    from agent.turn_context import _skill_turn_retrieval
    from hermes_cli.prompt_size import _build_inspection_agent, compute_prompt_breakdown

    cfg = fixture["runtime"]
    agent = _build_inspection_agent("cli")
    skills_prompt = _skills_prompt(agent)
    catalog = list(getattr(agent, "_skill_retrieval_catalog", None) or [])
    queries = fixture["runtime_queries"]
    clear_skill_retrieval_cache()
    quality = _quality(catalog, queries)
    context_item = queries[1] if len(queries) > 1 else queries[0]
    turn_context = _skill_turn_retrieval(agent, context_item["query"])
    context = {
        "has_relevant_skills": "<relevant_skills>" in turn_context,
        "has_skill_view_instruction": "skill_view(name)" in turn_context,
        "has_expected_skill": context_item["expected"] in turn_context,
        "candidate_count": sum(1 for line in turn_context.splitlines() if line.startswith("- ")),
    }
    fallback_item = fixture["runtime_fallback_query"]
    fallback = _fallback(catalog, fallback_item["query"], fallback_item["expected"])
    latency = _latency(catalog, queries)
    prompt = compute_prompt_breakdown("cli")
    skills_index_bytes = int(prompt["skills_index"]["bytes"])
    system_prompt_bytes = int(prompt["system_prompt"]["bytes"])
    full_description_leaks = [
        str(entry.get("name") or "")
        for entry in catalog[:100]
        if len(str(entry.get("description") or "")) >= 24
        and str(entry.get("description") or "") in skills_prompt
    ]
    regressions = _run_regressions() if run_regressions else {"passed": True, "skipped": True}
    bytes_per_skill = (skills_index_bytes / len(catalog)) if catalog else float("inf")

    checks = {
        "catalog_present": len(catalog) >= int(cfg["min_catalog_count"]),
        "quality_top1": quality["top1_rate"] >= float(cfg["min_top1_rate"]),
        "quality_top5": quality["top5_rate"] >= float(cfg["min_top5_rate"]),
        "dynamic_injection": (
            context["has_relevant_skills"]
            and context["has_skill_view_instruction"]
            and context["has_expected_skill"]
            and context["candidate_count"] <= int(cfg["max_context_candidates"])
        ),
        "lexical_fallback": fallback["rank"] == 1 and fallback["semantic_score"] == 0.0,
        "static_prompt_names_only": not full_description_leaks,
        "prompt_budget": (
            bytes_per_skill <= float(cfg["max_static_bytes_per_skill"])
            and system_prompt_bytes <= int(cfg["max_system_prompt_bytes"])
        ),
        "latency_cold": latency["cold_ms"] <= float(cfg["max_cold_ms"]),
        "latency_warm_p95": latency["warm_p95_ms"] <= float(cfg["max_warm_p95_ms"]),
        "focused_regressions": bool(regressions.get("passed")),
    }
    return {
        "mode": "runtime",
        "catalog_count": len(catalog),
        "quality": quality,
        "dynamic_context": context,
        "fallback": fallback,
        "prompt": {
            "skills_index_bytes": skills_index_bytes,
            "system_prompt_bytes": system_prompt_bytes,
            "bytes_per_skill": bytes_per_skill,
            "description_leak_count": len(full_description_leaks),
        },
        "latency": latency,
        "regressions": regressions,
        "thresholds": cfg,
        "checks": checks,
    }


def apply_failure_injection(receipt: dict[str, Any], failure: str | None) -> dict[str, Any]:
    out = copy.deepcopy(receipt)
    if not failure:
        return out
    mapping = {
        "prompt-size": "prompt_budget",
        "latency": "latency_warm_p95",
        "dynamic-injection": "dynamic_injection",
    }
    out["checks"][mapping[failure]] = False
    out["injected_failure"] = failure
    return out


def run_gate(
    mode: str,
    fixture_path: Path = DEFAULT_FIXTURE,
    *,
    run_regressions: bool = True,
    inject_failure: str | None = None,
) -> dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    payload = (
        _ci_receipt(fixture, run_regressions=run_regressions)
        if mode == "ci"
        else _runtime_receipt(fixture, run_regressions=run_regressions)
    )
    payload = apply_failure_injection(payload, inject_failure)
    passed = all(bool(value) for value in payload["checks"].values())
    return {
        "schema": SCHEMA,
        "final": "PASS" if passed else "FAIL",
        "passed": passed,
        "generated_at_epoch_s": time.time(),
        **payload,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ci", "runtime"), required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--skip-regressions", action="store_true")
    parser.add_argument(
        "--inject-failure",
        choices=("prompt-size", "latency", "dynamic-injection"),
        help="Test-only fault injection proving the gate fails closed.",
    )
    args = parser.parse_args(argv)

    receipt = run_gate(
        args.mode,
        args.fixture,
        run_regressions=not args.skip_regressions,
        inject_failure=args.inject_failure,
    )
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
