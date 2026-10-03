from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts import skill_retrieval_gate as gate


def test_ci_gate_passes_without_nested_regressions():
    receipt = gate.run_gate("ci", run_regressions=False)
    assert receipt["schema"] == gate.SCHEMA
    assert receipt["final"] == "PASS"
    assert receipt["quality"]["top1"] == receipt["quality"]["total"]
    assert receipt["quality"]["top5"] == receipt["quality"]["total"]
    assert receipt["dynamic_context"]["candidate_count"] <= 8
    assert receipt["prompt"]["description_leak_count"] == 0


@pytest.mark.parametrize(
    ("failure", "check"),
    [
        ("prompt-size", "prompt_budget"),
        ("latency", "latency_warm_p95"),
        ("dynamic-injection", "dynamic_injection"),
    ],
)
def test_fault_injection_fails_closed(failure, check):
    receipt = gate.run_gate("ci", run_regressions=False)
    injected = gate.apply_failure_injection(receipt, failure)
    assert injected["checks"][check] is False
    assert not all(injected["checks"].values())


def test_cli_failure_injection_returns_nonzero_and_writes_receipt(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(gate.ROOT / "scripts" / "skill_retrieval_gate.py"),
            "--mode",
            "ci",
            "--skip-regressions",
            "--inject-failure",
            "dynamic-injection",
            "--json-out",
            str(receipt_path),
        ],
        cwd=gate.ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 1
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["final"] == "FAIL"
    assert receipt["checks"]["dynamic_injection"] is False
