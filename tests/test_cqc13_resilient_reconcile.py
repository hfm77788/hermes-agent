import importlib.util
import json
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ops/cqc13-review/pipeline_reconcile_v3.py"

def load_module(tmp_path):
    os.environ["CQC13_ROOT"] = str(tmp_path)
    spec = importlib.util.spec_from_file_location("cqc13_pipeline_reconcile", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def seed_formal(mod):
    mod.BATCHES.mkdir(parents=True, exist_ok=True)
    ids=[f"P{i:03d}" for i in range(200)]
    mod.atomic(mod.FORMAL, {"project_ids": ids, "batch_size": 40})
    return ids

def test_batch_local_failure_does_not_stop_later_batches(tmp_path, monkeypatch):
    mod=load_module(tmp_path)
    ids=seed_formal(mod)
    calls=[]
    monkeypatch.setattr(mod, "get_python", lambda: "/usr/bin/python3")
    monkeypatch.setattr(mod, "ensure_batch", lambda n, wanted: mod.BATCHES/f"batch_{n:03d}")

    def fake_reconcile(n, b, wanted):
        calls.append(n)
        if n == 2:
            raise RuntimeError("local_batch_problem")
        return "wait" if n == 3 else "closed"

    monkeypatch.setattr(mod, "reconcile_batch", fake_reconcile)
    monkeypatch.setattr(mod, "valid_results", lambda d, wanted: set(wanted[:10]))
    monkeypatch.setattr(mod, "final_audit", lambda formal: (_ for _ in ()).throw(AssertionError("not ready")))

    assert mod.reconcile_once() is False
    assert calls == [1, 2, 3, 4, 5]
    state=json.loads(mod.STATE.read_text(encoding="utf8"))
    assert state["phase"] == "progressing_with_quarantine"
    assert state["batch_status"]["2"]["status"] == "quarantined"
    assert state["batch_status"]["3"]["status"] == "wait"
    assert state["closed_batches"] == [1, 4, 5]
    assert state["total_scored"] == 50

def test_all_batches_closed_runs_final_audit(tmp_path, monkeypatch):
    mod=load_module(tmp_path)
    seed_formal(mod)
    monkeypatch.setattr(mod, "get_python", lambda: "/usr/bin/python3")
    monkeypatch.setattr(mod, "ensure_batch", lambda n, wanted: mod.BATCHES/f"batch_{n:03d}")
    monkeypatch.setattr(mod, "reconcile_batch", lambda n, b, wanted: "closed")
    monkeypatch.setattr(mod, "valid_results", lambda d, wanted: set(wanted))
    monkeypatch.setattr(mod, "final_audit", lambda formal: True)
    assert mod.reconcile_once() is True

def test_formal_manifest_integrity_is_global_blocker(tmp_path, monkeypatch):
    mod=load_module(tmp_path)
    mod.BATCHES.mkdir(parents=True, exist_ok=True)
    mod.atomic(mod.FORMAL, {"project_ids": ["P1"]*200, "batch_size": 40})
    monkeypatch.setattr(mod, "get_python", lambda: "/usr/bin/python3")
    try:
        mod.reconcile_once()
    except RuntimeError as e:
        assert str(e) == "formal_200_manifest_integrity_invalid"
    else:
        raise AssertionError("expected global blocker")


def test_recovery_phase_clears_stale_error_metadata(tmp_path, monkeypatch):
    mod=load_module(tmp_path)
    mod.STATE.parent.mkdir(parents=True, exist_ok=True)
    mod.atomic(mod.STATE, {
        "phase": "technical_blocker",
        "error": "RuntimeError",
        "detail": "old problem",
        "retry_in_seconds": 300,
        "repeated_error_count": 110,
    })
    mod.write_state(phase="waiting_blind_review", batch=2)
    state=json.loads(mod.STATE.read_text(encoding="utf8"))
    assert state["phase"] == "waiting_blind_review"
    for key in ("error","detail","retry_in_seconds","repeated_error_count"):
        assert key not in state
