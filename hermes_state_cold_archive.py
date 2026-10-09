"""Cold storage for old Hermes sessions.

Conversation Lifecycle v1 is intentionally conservative: only standalone sessions that
are already soft-archived, ended, unpinned, idle past the cutoff, and not protected by a
live transcript/compression lease are eligible.  Complex parent/child lineages remain in
the hot store until a later lifecycle version can move them atomically as a group.

Each cold batch is a compressed ZIP containing:
- manifest.json: schema/version, per-session file names, counts and SHA256 digests;
- sessions/*.json: canonical active-context export accepted by import_sessions();
- history/*.json: complete message-row audit history (including inactive/compacted rows).

The file is written temp -> fsync -> atomic rename -> directory fsync and read back before
any hot-store delete.  The existing delete_session transactional display snapshot fence is
then used so a raced write cannot be lost.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO, _sql_session_last_active

logger = logging.getLogger("hermes_state")

COLD_ARCHIVE_SCHEMA = "hermes.session-cold-archive.v1"
_BUNDLE_SUFFIX = ".cold.zip"
_MAX_SESSIONS_PER_BUNDLE = 250
_MAX_UNCOMPRESSED_BYTES_PER_BUNDLE = 20 * 1024 * 1024
_DEFAULT_LIST_LIMIT = 500


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SessionColdArchiveMixin:
    """Archive-before-delete lifecycle for old standalone sessions."""

    def _cold_archive_dir(self, archive_dir: Optional[Path] = None) -> Path:
        if archive_dir is not None:
            return Path(archive_dir)
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "session-archive"

    def _cold_bundle_paths(self, archive_dir: Optional[Path] = None) -> List[Path]:
        root = self._cold_archive_dir(archive_dir)
        try:
            return sorted(p for p in root.glob(f"*{_BUNDLE_SUFFIX}") if p.is_file())
        except OSError:
            return []

    @staticmethod
    def _read_manifest(bundle: Path) -> Dict[str, Any]:
        with zipfile.ZipFile(bundle, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("schema") != COLD_ARCHIVE_SCHEMA:
            raise ValueError(f"unsupported cold archive schema: {manifest.get('schema')!r}")
        return manifest

    def _cold_index(self, archive_dir: Optional[Path] = None) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
        """Newest valid manifest entry per session id."""
        out: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
        for bundle in self._cold_bundle_paths(archive_dir):
            try:
                manifest = self._read_manifest(bundle)
            except Exception:
                continue
            for entry in manifest.get("sessions") or []:
                sid = str(entry.get("session_id") or "")
                if sid:
                    out[sid] = (bundle, entry)
        return out

    def _cold_archive_candidate_rows(
        self, cutoff: float, *, reclaim_stale_guards: bool = True
    ) -> List[Dict[str, Any]]:
        """Return safe standalone candidates.

        v1 deliberately excludes every session with a parent or child. That avoids
        breaking compression/branch/delegate lineages and prevents delete_session's
        delegate cascade from removing a child that was never archived.

        Normal archive runs use the existing transactional guard check, which may reclaim
        expired/dead guard rows. Dry-runs use a conservative read-only guard check so a
        dry-run performs no writes.
        """
        last_active = _sql_session_last_active("s")
        query = f"""
            SELECT s.id, s.source, s.title, s.model, s.started_at,
                   {last_active} AS last_active, s.ended_at, s.message_count
            FROM sessions s
            WHERE s.archived = 1
              AND s.ended_at IS NOT NULL
              AND COALESCE(s.pinned, 0) = 0
              AND s.parent_session_id IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM sessions child WHERE child.parent_session_id = s.id
              )
              AND {last_active} < ?
            ORDER BY last_active ASC, s.started_at ASC
        """

        if not reclaim_stale_guards:
            now = time.time()
            with self._read_ctx() as conn:
                rows = conn.execute(query, (cutoff,)).fetchall()
                safe = []
                for row in rows:
                    sid = str(row["id"])
                    conversation_id = self._session_turn_lease_key_on_conn(conn, sid)
                    lease = conn.execute(
                        "SELECT expires_at FROM session_turn_leases WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                    if lease is not None and float(lease["expires_at"]) > now:
                        continue
                    lock = conn.execute(
                        "SELECT expires_at FROM compression_locks WHERE session_id = ?",
                        (sid,),
                    ).fetchone()
                    if lock is not None and float(lock["expires_at"]) > now:
                        continue
                    safe.append(dict(row))
                return safe

        def _select(conn):
            rows = conn.execute(query, (cutoff,)).fetchall()
            safe = []
            for row in rows:
                sid = str(row["id"])
                if self._write_guards_reject(
                    conn, sid, allow_closed_compression_parent=True
                ):
                    continue
                safe.append(dict(row))
            return safe

        return self._execute_write(_select) or []

    def _snapshot_for_cold(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Capture canonical restore payload + full audit history + delete fence."""
        # No cascade is allowed in v1, even if the topology changed after candidate selection.
        if self.get_session_delete_targets(session_id) != [session_id]:
            return None
        payload = self.export_session(session_id, include_compacted=False)
        if payload is None:
            return None
        # include_inactive=True returns every durable row, including rewind and compacted rows.
        history = self.get_messages(session_id, include_inactive=True)
        display = self.get_messages(session_id, include_compacted=True)
        return {
            "payload": payload,
            "history": history,
            "display": display,
            "payload_bytes": _json_bytes(payload),
            "history_bytes": _json_bytes(history),
        }

    @staticmethod
    def _entry_for_snapshot(session_id: str, index: int, snap: Dict[str, Any]) -> Dict[str, Any]:
        payload_part = f"sessions/{index:04d}.json"
        history_part = f"history/{index:04d}.json"
        return {
            "session_id": session_id,
            "payload_part": payload_part,
            "history_part": history_part,
            "payload_sha256": _sha256(snap["payload_bytes"]),
            "history_sha256": _sha256(snap["history_bytes"]),
            "active_message_count": len(snap["payload"].get("messages") or []),
            "history_message_count": len(snap["history"]),
            "title": snap["payload"].get("title"),
            "source": snap["payload"].get("source"),
            "started_at": snap["payload"].get("started_at"),
            "ended_at": snap["payload"].get("ended_at"),
        }

    def _write_cold_bundle(
        self,
        items: List[Tuple[str, Dict[str, Any]]],
        *,
        cutoff: float,
        older_than_days: float,
        archive_dir: Optional[Path] = None,
    ) -> Path:
        root = self._cold_archive_dir(archive_dir)
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.time()
        seq = 0
        while True:
            name = (
                f"cold-{time.strftime('%Y%m%dT%H%M%S', time.gmtime(stamp))}-"
                f"{int(stamp * 1000) % 1000:03d}-{os.getpid()}-{seq}{_BUNDLE_SUFFIX}"
            )
            final = root / name
            if not final.exists():
                break
            seq += 1
        tmp = root / f".{name}.tmp"

        entries = [
            self._entry_for_snapshot(sid, idx, snap)
            for idx, (sid, snap) in enumerate(items)
        ]
        manifest = {
            "schema": COLD_ARCHIVE_SCHEMA,
            "created_at": stamp,
            "cutoff": cutoff,
            "older_than_days": older_than_days,
            "session_count": len(entries),
            "active_message_count": sum(e["active_message_count"] for e in entries),
            "history_message_count": sum(e["history_message_count"] for e in entries),
            "sessions": entries,
        }
        try:
            with open(tmp, "w+b") as fh:
                with zipfile.ZipFile(
                    fh, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
                ) as zf:
                    zf.writestr("manifest.json", _json_bytes(manifest))
                    for entry, (_, snap) in zip(entries, items):
                        zf.writestr(entry["payload_part"], snap["payload_bytes"])
                        zf.writestr(entry["history_part"], snap["history_bytes"])
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
            try:
                dir_fd = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                # The file itself is already fsynced and atomically renamed; directory fsync
                # is unavailable on a few filesystems/platforms.
                pass
            return final
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _read_entry_bytes(
        zf: zipfile.ZipFile, entry: Dict[str, Any]
    ) -> Tuple[bytes, bytes]:
        return zf.read(entry["payload_part"]), zf.read(entry["history_part"])

    def _verify_cold_entry_internal(
        self, bundle: Path, entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify only archive bytes; safe for restore after hot rows are gone."""
        try:
            with zipfile.ZipFile(bundle, "r") as zf:
                payload_bytes, history_bytes = self._read_entry_bytes(zf, entry)
            if _sha256(payload_bytes) != entry.get("payload_sha256"):
                return {"ok": False, "reason": "payload_sha_mismatch"}
            if _sha256(history_bytes) != entry.get("history_sha256"):
                return {"ok": False, "reason": "history_sha_mismatch"}
            payload = json.loads(payload_bytes.decode("utf-8"))
            history = json.loads(history_bytes.decode("utf-8"))
            if str(payload.get("id") or "") != str(entry.get("session_id") or ""):
                return {"ok": False, "reason": "session_id_mismatch"}
            if len(payload.get("messages") or []) != int(entry.get("active_message_count") or 0):
                return {"ok": False, "reason": "active_count_mismatch"}
            if len(history) != int(entry.get("history_message_count") or 0):
                return {"ok": False, "reason": "history_count_mismatch"}
            return {
                "ok": True,
                "payload": payload,
                "history": history,
                "payload_bytes": payload_bytes,
                "history_bytes": history_bytes,
            }
        except Exception as exc:
            return {"ok": False, "reason": f"archive_unreadable:{type(exc).__name__}"}

    def _verify_snapshot_still_current(
        self, session_id: str, entry: Dict[str, Any]
    ) -> bool:
        """Strong pre-delete race check against active context and every durable row."""
        snap = self._snapshot_for_cold(session_id)
        if snap is None:
            return False
        return (
            _sha256(snap["payload_bytes"]) == entry.get("payload_sha256")
            and _sha256(snap["history_bytes"]) == entry.get("history_sha256")
        )

    def _delete_cold_snapshot(
        self,
        session_id: str,
        snap: Dict[str, Any],
        *,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        # Re-check topology immediately before the transactional delete.
        if self.get_session_delete_targets(session_id) != [session_id]:
            return False
        return bool(
            self.delete_session(
                session_id,
                sessions_dir=sessions_dir,
                expected_delete_ids=[session_id],
                expected_display_messages={session_id: snap["display"]},
                reject_active_write_guards=True,
            )
        )

    @staticmethod
    def _chunks(
        items: Iterable[Tuple[str, Dict[str, Any]]]
    ) -> Iterable[List[Tuple[str, Dict[str, Any]]]]:
        batch: List[Tuple[str, Dict[str, Any]]] = []
        size = 0
        for item in items:
            item_size = len(item[1]["payload_bytes"]) + len(item[1]["history_bytes"])
            if batch and (
                len(batch) >= _MAX_SESSIONS_PER_BUNDLE
                or size + item_size > _MAX_UNCOMPRESSED_BYTES_PER_BUNDLE
            ):
                yield batch
                batch, size = [], 0
            batch.append(item)
            size += item_size
        if batch:
            yield batch

    def cold_archive(
        self,
        *,
        older_than_days: float = 90,
        dry_run: bool = False,
        sessions_dir: Optional[Path] = None,
        archive_dir: Optional[Path] = None,
        limit: Optional[int] = None,
        _maintenance_lock_held: bool = False,
    ) -> Dict[str, Any]:
        """Archive old standalone sessions and then remove verified copies from hot SQLite.

        Mutating runs serialize through the existing cross-process state.db maintenance
        lock. Dry-runs remain read-only and lock-free. Internal auto-maintenance callers
        that already hold the same lock pass _maintenance_lock_held=True.
        """
        if older_than_days is None or older_than_days < 0:
            raise ValueError("older_than_days must be >= 0")
        if not dry_run and not _maintenance_lock_held:
            from hermes_state_repair import (
                _release_auto_maintenance_lock,
                _try_acquire_auto_maintenance_lock,
            )
            lock = _try_acquire_auto_maintenance_lock(self.db_path)
            if lock is None:
                return {
                    "ok": False,
                    "dry_run": False,
                    "candidates": 0,
                    "archived": 0,
                    "deleted": 0,
                    "bundles": [],
                    "skipped": [
                        {"session_id": None, "reason": "maintenance_lock_busy"}
                    ],
                }
            try:
                return self.cold_archive(
                    older_than_days=older_than_days,
                    dry_run=False,
                    sessions_dir=sessions_dir,
                    archive_dir=archive_dir,
                    limit=limit,
                    _maintenance_lock_held=True,
                )
            finally:
                _release_auto_maintenance_lock(lock)
        cutoff = time.time() - float(older_than_days) * 86400.0
        rows = self._cold_archive_candidate_rows(
            cutoff, reclaim_stale_guards=not dry_run
        )
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        result: Dict[str, Any] = {
            "ok": True,
            "dry_run": bool(dry_run),
            "candidates": len(rows),
            "archived": 0,
            "deleted": 0,
            "bundles": [],
            "skipped": [],
        }
        if dry_run:
            result["candidate_ids"] = [str(r["id"]) for r in rows]
            return result
        if not rows:
            return result

        existing = self._cold_index(archive_dir)
        new_items: List[Tuple[str, Dict[str, Any]]] = []
        snapshots: Dict[str, Dict[str, Any]] = {}

        # Existing verified archive entries can complete a previously fenced/aborted delete
        # without writing a duplicate bundle, but only when the hot row is byte-identical.
        for row in rows:
            sid = str(row["id"])
            snap = self._snapshot_for_cold(sid)
            if snap is None:
                result["skipped"].append({"session_id": sid, "reason": "topology_or_export_changed"})
                continue
            snapshots[sid] = snap
            prior = existing.get(sid)
            if prior is None:
                new_items.append((sid, snap))
                continue
            bundle, entry = prior
            verified = self._verify_cold_entry_internal(bundle, entry)
            if not verified.get("ok"):
                result["skipped"].append({"session_id": sid, "reason": "existing_archive_invalid"})
                result["ok"] = False
                continue
            if (
                _sha256(snap["payload_bytes"]) != entry.get("payload_sha256")
                or _sha256(snap["history_bytes"]) != entry.get("history_sha256")
            ):
                # A previously verified cold copy can become stale if the hot row changed
                # after the bundle was written but before its guarded delete completed.
                # Do not strand that session in the hot store forever: write a new immutable
                # revision from the current snapshot, verify it, then let the ordinary
                # archive-before-delete fences decide whether deletion is still safe.
                # _cold_index() already resolves the newest bundle per session id for restore.
                new_items.append((sid, snap))
                continue
            if self._delete_cold_snapshot(sid, snap, sessions_dir=sessions_dir):
                result["deleted"] += 1
                result["archived"] += 1
            else:
                result["skipped"].append({"session_id": sid, "reason": "delete_fence_rejected"})

        for batch in self._chunks(new_items):
            try:
                bundle = self._write_cold_bundle(
                    batch,
                    cutoff=cutoff,
                    older_than_days=float(older_than_days),
                    archive_dir=archive_dir,
                )
            except Exception as exc:
                result["ok"] = False
                for sid, _ in batch:
                    result["skipped"].append(
                        {"session_id": sid, "reason": f"bundle_write_failed:{type(exc).__name__}"}
                    )
                continue
            result["bundles"].append(str(bundle))
            try:
                manifest = self._read_manifest(bundle)
                entries = {
                    str(e.get("session_id") or ""): e
                    for e in manifest.get("sessions") or []
                }
            except Exception:
                entries = {}
            for sid, snap in batch:
                entry = entries.get(sid)
                verified = (
                    self._verify_cold_entry_internal(bundle, entry)
                    if entry is not None
                    else {"ok": False, "reason": "manifest_entry_missing"}
                )
                if not verified.get("ok"):
                    result["ok"] = False
                    result["skipped"].append(
                        {"session_id": sid, "reason": verified.get("reason", "verify_failed")}
                    )
                    continue
                # Compare every durable row with what was archived before deletion.
                if not self._verify_snapshot_still_current(sid, entry):
                    result["ok"] = False
                    result["skipped"].append({"session_id": sid, "reason": "hot_store_diverged"})
                    continue
                if self._delete_cold_snapshot(sid, snap, sessions_dir=sessions_dir):
                    result["deleted"] += 1
                    result["archived"] += 1
                else:
                    result["skipped"].append({"session_id": sid, "reason": "delete_fence_rejected"})
        return result

    def cold_list(
        self,
        *,
        archive_dir: Optional[Path] = None,
        limit: int = _DEFAULT_LIST_LIMIT,
    ) -> List[Dict[str, Any]]:
        limit = max(0, int(limit))
        if limit == 0:
            return []
        out: List[Dict[str, Any]] = []
        for bundle in reversed(self._cold_bundle_paths(archive_dir)):
            try:
                manifest = self._read_manifest(bundle)
            except Exception:
                continue
            for entry in reversed(manifest.get("sessions") or []):
                out.append(
                    {
                        "session_id": entry.get("session_id"),
                        "title": entry.get("title"),
                        "source": entry.get("source"),
                        "started_at": entry.get("started_at"),
                        "ended_at": entry.get("ended_at"),
                        "message_count": entry.get("active_message_count"),
                        "history_message_count": entry.get("history_message_count"),
                        "bundle": str(bundle),
                    }
                )
                if len(out) >= limit:
                    return out
        return out

    @staticmethod
    def _search_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def cold_search(
        self,
        query: str,
        *,
        archive_dir: Optional[Path] = None,
        max_bundles: int = 50,
        match_limit: int = 50,
        snippet_len: int = 180,
    ) -> List[Dict[str, Any]]:
        """Bounded substring search over manifests and complete archived message history."""
        term = (query or "").strip().lower()
        max_bundles = max(0, int(max_bundles))
        match_limit = max(0, int(match_limit))
        if not term or max_bundles == 0 or match_limit == 0:
            return []
        matches: List[Dict[str, Any]] = []
        bundles = self._cold_bundle_paths(archive_dir)[-max_bundles:]
        for bundle in reversed(bundles):
            try:
                manifest = self._read_manifest(bundle)
                with zipfile.ZipFile(bundle, "r") as zf:
                    for entry in reversed(manifest.get("sessions") or []):
                        sid = str(entry.get("session_id") or "")
                        title = str(entry.get("title") or "")
                        title_match = term in title.lower()
                        sid_match = term in sid.lower()
                        snippet = title if title_match else (sid if sid_match else "")
                        if not snippet:
                            history = json.loads(zf.read(entry["history_part"]).decode("utf-8"))
                            for msg in history:
                                text = self._search_text(msg.get("content"))
                                pos = text.lower().find(term)
                                if pos >= 0:
                                    snippet = text[max(0, pos - 50) : pos + snippet_len]
                                    break
                        if snippet:
                            matches.append(
                                {
                                    "session_id": sid,
                                    "title": title or None,
                                    "snippet": snippet,
                                    "bundle": str(bundle),
                                }
                            )
                            if len(matches) >= match_limit:
                                return matches
            except Exception:
                continue
        return matches

    def cold_restore(
        self, session_id: str, *, archive_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Restore canonical live context; full audit history stays preserved in the cold bundle."""
        sid = str(session_id or "").strip()
        if not sid:
            return {"ok": False, "restored": False, "error": "session_id_required"}
        if self.get_session(sid) is not None:
            return {"ok": False, "restored": False, "error": "live_session_collision"}

        indexed = self._cold_index(archive_dir)
        found = indexed.get(sid)
        if found is None:
            return {"ok": False, "restored": False, "error": "cold_session_not_found"}
        bundle, entry = found
        verified = self._verify_cold_entry_internal(bundle, entry)
        if not verified.get("ok"):
            return {
                "ok": False,
                "restored": False,
                "error": verified.get("reason", "archive_verification_failed"),
            }
        # Race-safe collision recheck immediately before import.
        if self.get_session(sid) is not None:
            return {"ok": False, "restored": False, "error": "live_session_collision"}
        imported = self.import_sessions([verified["payload"]])
        if not imported.get("ok") or int(imported.get("imported") or 0) != 1:
            return {
                "ok": False,
                "restored": False,
                "error": "import_failed_or_collision",
                "import_result": imported,
            }
        return {
            "ok": True,
            "restored": True,
            "session_id": sid,
            "bundle": str(bundle),
            "audit_history_preserved_in_cold_bundle": True,
        }

    def maybe_auto_cold_archive(
        self,
        *,
        older_than_days: int = 90,
        min_interval_hours: int = 24,
        sessions_dir: Optional[Path] = None,
        archive_dir: Optional[Path] = None,
        vacuum: bool = True,
        min_vacuum_interval_days: int = 30,
        min_vacuum_freelist_ratio: float = AUTO_VACUUM_MIN_FREELIST_RATIO,
    ) -> Dict[str, Any]:
        """Throttled auto pass; callers own the config enable switch. Never raises."""
        from hermes_state_repair import (
            _release_auto_maintenance_lock,
            _try_acquire_auto_maintenance_lock,
        )

        result: Dict[str, Any] = {
            "skipped": False,
            "archived": 0,
            "deleted": 0,
            "bundles": [],
            "vacuumed": False,
        }
        if older_than_days is None or older_than_days < 0:
            result["skipped"] = True
            return result
        lock = None
        try:
            lock = _try_acquire_auto_maintenance_lock(self.db_path)
            if lock is None:
                result["skipped"] = True
                return result
            now = time.time()
            try:
                last = float(self.get_meta("last_auto_cold_archive") or 0)
            except (TypeError, ValueError):
                last = 0.0
            if last and now - last < int(min_interval_hours) * 3600:
                result["skipped"] = True
                return result
            outcome = self.cold_archive(
                older_than_days=float(older_than_days),
                sessions_dir=sessions_dir,
                archive_dir=archive_dir,
                _maintenance_lock_held=True,
            )
            result.update(
                archived=int(outcome.get("archived") or 0),
                deleted=int(outcome.get("deleted") or 0),
                bundles=list(outcome.get("bundles") or []),
                skipped_sessions=list(outcome.get("skipped") or []),
            )
            if not outcome.get("ok", True):
                result["error"] = "one_or_more_sessions_failed_closed"

            # Cold storage may be enabled while auto_prune stays disabled. Reclaim free
            # SQLite pages with the same holder/freelist/time admission gates used by
            # ordinary maintenance so a successful cold move can actually shrink state.db.
            if vacuum and result["deleted"] > 0:
                try:
                    raw_last_vacuum = self.get_meta("last_vacuum")
                    since_vacuum = (
                        None if not raw_last_vacuum else now - float(raw_last_vacuum)
                    )
                except (TypeError, ValueError):
                    since_vacuum = None
                vacuum_due = (
                    since_vacuum is None
                    or since_vacuum >= int(min_vacuum_interval_days) * 86400
                )
                if vacuum_due:
                    result["freelist_ratio"] = ratio = self._freelist_ratio()
                    from hermes_state_holders import (
                        foreign_state_db_holders,
                        in_process_state_db_holders,
                    )
                    holders = (
                        foreign_state_db_holders(self.db_path)
                        + in_process_state_db_holders(self.db_path, exclude=self)
                    )
                    if holders:
                        result["vacuum_skipped_holders"] = len(holders)
                    elif ratio is None or ratio > float(min_vacuum_freelist_ratio):
                        try:
                            from hermes_startup_watchdog import report_startup_progress
                            report_startup_progress(
                                900.0, phase="state_db_auto_cold_vacuum"
                            )
                            self.vacuum()
                            result["vacuumed"] = True
                            self.set_meta("last_vacuum", str(now))
                        except Exception as exc:
                            logger.warning(
                                "state.db cold-archive VACUUM failed: %s", exc
                            )

            self.set_meta("last_auto_cold_archive", str(now))
            if result["archived"]:
                logger.info(
                    "state.db auto cold-archive: %d session(s) moved into %d bundle(s)%s",
                    result["archived"],
                    len(result["bundles"]),
                    " + VACUUM" if result["vacuumed"] else "",
                )
        except Exception as exc:
            logger.warning("state.db auto cold-archive failed: %s", exc)
            result["error"] = str(exc)
        finally:
            if lock is not None:
                _release_auto_maintenance_lock(lock)
        return result
