import json
import time
import zipfile

import pytest

from hermes_state import SessionDB
from hermes_state_cold_archive import COLD_ARCHIVE_SCHEMA


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _old_archived(db, sid, *, days=120, pinned=False):
    old = time.time() - days * 86400
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", f"question from {sid}", timestamp=old)
    db.append_message(sid, "assistant", f"answer from {sid}", timestamp=old + 1)
    db.end_session(sid, "done")
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=?, last_activity_at=? WHERE id=?",
        (old, old + 2, old + 2, sid),
    )
    db._conn.commit()
    assert db.set_session_archived(sid, True)
    if pinned:
        assert db.set_session_pinned(sid, True)
    return old


def test_cold_candidates_are_conservative(tmp_path, db):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "safe")
    _old_archived(db, "pinned", pinned=True)

    old = time.time() - 120 * 86400
    db.create_session("open", source="cli")
    db.append_message("open", "user", "still open", timestamp=old)
    db._conn.execute(
        "UPDATE sessions SET started_at=?, last_activity_at=?, archived=1 WHERE id='open'",
        (old, old),
    )

    db.create_session("parent", source="cli")
    db.create_session("child", source="cli", parent_session_id="parent")
    for sid in ("parent", "child"):
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=?, last_activity_at=?, archived=1 WHERE id=?",
            (old, old + 1, old + 1, sid),
        )
    db._conn.commit()

    preview = db.cold_archive(
        older_than_days=90, dry_run=True, archive_dir=archive_dir
    )
    assert preview["candidate_ids"] == ["safe"]
    assert not archive_dir.exists()


def test_cold_archive_write_verify_delete_search_restore(tmp_path, db):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "cold-1")

    result = db.cold_archive(older_than_days=90, archive_dir=archive_dir)
    assert result["ok"] is True
    assert result["archived"] == result["deleted"] == 1
    assert db.get_session("cold-1") is None
    assert len(result["bundles"]) == 1

    bundle = next(archive_dir.glob("*.cold.zip"))
    with zipfile.ZipFile(bundle, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["schema"] == COLD_ARCHIVE_SCHEMA
        assert manifest["session_count"] == 1
        entry = manifest["sessions"][0]
        assert entry["session_id"] == "cold-1"
        assert entry["active_message_count"] == 2
        assert entry["history_message_count"] == 2

    listed = db.cold_list(archive_dir=archive_dir)
    assert [row["session_id"] for row in listed] == ["cold-1"]
    matches = db.cold_search("question from cold-1", archive_dir=archive_dir)
    assert matches and matches[0]["session_id"] == "cold-1"
    id_matches = db.cold_search("cold-1", archive_dir=archive_dir)
    assert id_matches and id_matches[0]["session_id"] == "cold-1"

    restored = db.cold_restore("cold-1", archive_dir=archive_dir)
    assert restored["ok"] is True
    assert db.get_session("cold-1") is not None
    assert [m["content"] for m in db.get_messages("cold-1")] == [
        "question from cold-1",
        "answer from cold-1",
    ]

    collision = db.cold_restore("cold-1", archive_dir=archive_dir)
    assert collision == {
        "ok": False,
        "restored": False,
        "error": "live_session_collision",
    }


def test_verification_failure_deletes_nothing(tmp_path, db, monkeypatch):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "keep-me")

    monkeypatch.setattr(
        db,
        "_verify_cold_entry_internal",
        lambda bundle, entry: {"ok": False, "reason": "test_failure"},
    )
    result = db.cold_archive(older_than_days=90, archive_dir=archive_dir)

    assert result["ok"] is False
    assert result["deleted"] == 0
    assert db.get_session("keep-me") is not None
    assert list(archive_dir.glob("*.cold.zip"))


def test_race_after_bundle_fails_closed(tmp_path, db, monkeypatch):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "raced")

    monkeypatch.setattr(db, "_verify_snapshot_still_current", lambda sid, entry: False)
    result = db.cold_archive(older_than_days=90, archive_dir=archive_dir)

    assert result["deleted"] == 0
    assert db.get_session("raced") is not None
    assert any(x["reason"] == "hot_store_diverged" for x in result["skipped"])


def test_rerun_reuses_existing_bundle_instead_of_duplicate(tmp_path, db, monkeypatch):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "retry")

    original_delete = db._delete_cold_snapshot
    monkeypatch.setattr(db, "_delete_cold_snapshot", lambda *a, **k: False)
    first = db.cold_archive(older_than_days=90, archive_dir=archive_dir)
    assert first["deleted"] == 0
    assert db.get_session("retry") is not None
    assert len(list(archive_dir.glob("*.cold.zip"))) == 1

    monkeypatch.setattr(db, "_delete_cold_snapshot", original_delete)
    second = db.cold_archive(older_than_days=90, archive_dir=archive_dir)
    assert second["deleted"] == 1
    assert second["bundles"] == []
    assert len(list(archive_dir.glob("*.cold.zip"))) == 1


def test_stale_verified_archive_is_superseded_by_new_revision(tmp_path, db, monkeypatch):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "stale-revision")

    original_delete = db._delete_cold_snapshot
    monkeypatch.setattr(db, "_delete_cold_snapshot", lambda *a, **k: False)
    first = db.cold_archive(older_than_days=90, archive_dir=archive_dir)
    assert first["deleted"] == 0
    assert len(list(archive_dir.glob("*.cold.zip"))) == 1
    assert db.get_session("stale-revision") is not None

    # Simulate a safe post-bundle metadata update while preserving old inactivity.
    db._conn.execute(
        "UPDATE sessions SET title=? WHERE id=?",
        ("newer title", "stale-revision"),
    )
    db._conn.commit()

    monkeypatch.setattr(db, "_delete_cold_snapshot", original_delete)
    second = db.cold_archive(older_than_days=90, archive_dir=archive_dir)

    assert second["ok"] is True
    assert second["deleted"] == 1
    assert len(second["bundles"]) == 1
    assert len(list(archive_dir.glob("*.cold.zip"))) == 2
    assert db.get_session("stale-revision") is None

    restored = db.cold_restore("stale-revision", archive_dir=archive_dir)
    assert restored["ok"] is True
    assert db.get_session("stale-revision")["title"] == "newer title"


def test_mutating_cold_archive_serializes_on_maintenance_lock(tmp_path, db):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "serialized")

    import hermes_state_repair

    held = hermes_state_repair._try_acquire_auto_maintenance_lock(db.db_path)
    assert held is not None
    try:
        blocked = db.cold_archive(older_than_days=90, archive_dir=archive_dir)
        assert blocked["ok"] is False
        assert blocked["deleted"] == 0
        assert blocked["skipped"] == [
            {"session_id": None, "reason": "maintenance_lock_busy"}
        ]
        assert db.get_session("serialized") is not None
        assert not archive_dir.exists()

        preview = db.cold_archive(
            older_than_days=90, dry_run=True, archive_dir=archive_dir
        )
        assert preview["candidate_ids"] == ["serialized"]
    finally:
        hermes_state_repair._release_auto_maintenance_lock(held)


def test_live_turn_guard_excludes_candidate(tmp_path, db):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "leased")
    assert db.try_acquire_session_turn_lease("leased", "test-holder", ttl_seconds=300)

    preview = db.cold_archive(
        older_than_days=90, dry_run=True, archive_dir=archive_dir
    )
    assert "leased" not in preview["candidate_ids"]


def test_final_delete_transaction_refuses_new_live_lease(tmp_path, db):
    _old_archived(db, "late-lease")
    snap = db._snapshot_for_cold("late-lease")
    assert snap is not None
    assert db.try_acquire_session_turn_lease("late-lease", "late-holder", ttl_seconds=300)

    assert db._delete_cold_snapshot("late-lease", snap) is False
    assert db.get_session("late-lease") is not None


def test_auto_cold_archive_throttles_independently(tmp_path, db):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "auto")
    first = db.maybe_auto_cold_archive(
        older_than_days=90,
        min_interval_hours=24,
        archive_dir=archive_dir,
    )
    assert first["deleted"] == 1
    second = db.maybe_auto_cold_archive(
        older_than_days=90,
        min_interval_hours=24,
        archive_dir=archive_dir,
    )
    assert second["skipped"] is True


def test_dry_run_is_strictly_read_only(tmp_path, db, monkeypatch):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "preview-only")

    def _forbid_write(*args, **kwargs):
        pytest.fail("cold-archive dry-run attempted a write transaction")

    monkeypatch.setattr(db, "_execute_write", _forbid_write)
    preview = db.cold_archive(
        older_than_days=90,
        dry_run=True,
        archive_dir=archive_dir,
    )

    assert preview["candidate_ids"] == ["preview-only"]
    assert not archive_dir.exists()


def test_auto_cold_archive_vacuums_only_after_safe_admission(
    tmp_path, db, monkeypatch
):
    archive_dir = tmp_path / "cold"
    _old_archived(db, "vacuum-me")

    import hermes_state_holders

    monkeypatch.setattr(
        hermes_state_holders, "foreign_state_db_holders", lambda path: []
    )
    monkeypatch.setattr(
        hermes_state_holders,
        "in_process_state_db_holders",
        lambda path, exclude=None: [],
    )
    monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.50)
    called = {"vacuum": 0}

    def _vacuum():
        called["vacuum"] += 1
        return 0

    monkeypatch.setattr(db, "vacuum", _vacuum)

    result = db.maybe_auto_cold_archive(
        older_than_days=90,
        min_interval_hours=0,
        archive_dir=archive_dir,
        vacuum=True,
        min_vacuum_interval_days=0,
        min_vacuum_freelist_ratio=0.10,
    )

    assert result["deleted"] == 1
    assert result["vacuumed"] is True
    assert result["freelist_ratio"] == pytest.approx(0.50)
    assert called["vacuum"] == 1
