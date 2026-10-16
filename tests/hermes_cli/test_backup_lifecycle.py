from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from hermes_cli.backup_lifecycle import (
    BackupItem,
    PlanEntry,
    Retention,
    apply_plan,
    choose_kept,
    discover,
    make_plan,
    retention_for_used_percent,
)


def _touch(path: Path, when: datetime, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def _mkdir(path: Path, when: datetime) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _touch(path / "payload", when, b"x" * 10)
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def test_pressure_tiers():
    assert retention_for_used_percent(70)[0] == "normal"
    assert retention_for_used_percent(82)[0] == "warning"
    assert retention_for_used_percent(87)[0] == "pressure"
    assert retention_for_used_percent(92)[0] == "emergency"


def test_discover_manages_only_known_families(tmp_path: Path):
    home = tmp_path / "home"
    now = datetime.now().astimezone()
    _mkdir(home / "hermes-agent-backups" / "agent-runtime-deploy-a", now)
    _mkdir(home / "hermes-agent-backups" / "conversation-lifecycle-v1", now)
    _touch(home / ".hermes/backups/state-db/state.db.auto.1.gz", now)
    _touch(home / ".hermes/backups/state-db/state.db.pre-restart.1.gz", now)
    _touch(home / ".hermes/backups/hindsight/a.sql.gz", now)
    names = {item.path.name for item in discover(home)}
    assert names == {"agent-runtime-deploy-a", "state.db.auto.1.gz", "a.sql.gz"}


def test_retention_keeps_recent_daily_weekly():
    now = datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0)
    items = [
        BackupItem("x", Path(f"/x/{i}"), (now - timedelta(days=i)).timestamp(), 1)
        for i in range(10)
    ]
    kept = choose_kept(items, Retention(recent=2, daily=3, weekly=2))
    assert Path("/x/0") in kept
    assert Path("/x/1") in kept
    assert Path("/x/2") in kept
    assert len(kept) >= 3


def test_protected_marker_always_kept(tmp_path: Path):
    home = tmp_path / "home"
    now = datetime.now().astimezone()
    root = home / "hermes-agent-backups"
    for i in range(5):
        _mkdir(root / f"agent-runtime-deploy-{i}", now - timedelta(hours=i))
    protected = root / "agent-runtime-deploy-4"
    (protected / ".keep").write_text("keep")
    plan = make_plan(discover(home), Retention(1, 1, 1))
    row = next(entry for entry in plan if entry.path == str(protected))
    assert row.action == "keep"
    assert row.reason == "protected_marker_or_symlink"


def test_apply_deletes_only_planned_known_copy(tmp_path: Path):
    home = tmp_path / "home"
    now = datetime.now().astimezone()
    root = home / "hermes-agent-backups"
    for i in range(5):
        _mkdir(root / f"agent-runtime-deploy-{i}", now - timedelta(hours=i))
    unknown = root / "conversation-lifecycle-v1"
    _mkdir(unknown, now - timedelta(days=30))
    plan = make_plan(discover(home), Retention(1, 1, 1))
    candidates = [e for e in plan if e.action == "delete"]
    assert candidates
    count, _bytes = apply_plan(plan, max_delete_bytes=1024**2)
    assert count == len(candidates)
    assert unknown.exists()
    assert (root / "agent-runtime-deploy-0").exists()


def test_delete_budget_fails_closed(tmp_path: Path):
    path = tmp_path / "old"
    _touch(path, datetime.now().astimezone(), b"x" * 100)
    plan = [PlanEntry("x", str(path), 100, path.stat().st_mtime, "delete", "expired")]
    with pytest.raises(RuntimeError, match="delete_budget_exceeded"):
        apply_plan(plan, max_delete_bytes=10)
    assert path.exists()
