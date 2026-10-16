"""Bounded retention for high-growth Hermes backup families.

Unknown backup names are never deleted. The module is dry-run by default;
--apply is required for mutation. Protected markers always win.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Retention:
    recent: int
    daily: int
    weekly: int


@dataclass(frozen=True)
class BackupItem:
    family: str
    path: Path
    mtime: float
    size: int
    protected: bool = False


@dataclass(frozen=True)
class PlanEntry:
    family: str
    path: str
    size: int
    mtime: float
    action: str
    reason: str


def retention_for_used_percent(used_percent: float) -> tuple[str, Retention]:
    if used_percent >= 90:
        return "emergency", Retention(2, 2, 1)
    if used_percent >= 85:
        return "pressure", Retention(3, 3, 2)
    if used_percent >= 80:
        return "warning", Retention(3, 5, 3)
    return "normal", Retention(3, 7, 4)


def _tree_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                pass
    return total


def _protected(path: Path) -> bool:
    if path.is_symlink():
        return True
    if path.is_dir():
        return any((path / name).exists() for name in (".keep", ".protected", "PROTECTED"))
    return any(path.with_name(path.name + suffix).exists() for suffix in (".keep", ".protected"))


def discover(user_home: Path) -> list[BackupItem]:
    hermes_home = user_home / ".hermes"
    specs: list[tuple[str, Path, str, bool]] = [
        ("agent_runtime", user_home / "hermes-agent-backups", "agent-runtime-deploy-*", True),
        ("state_auto", hermes_home / "backups" / "state-db", "state.db.auto.*.gz", False),
        ("hindsight", hermes_home / "backups" / "hindsight", "*.sql.gz", False),
        ("hindsight", hermes_home / "backups" / "hindsight", "*.dump.gz", False),
    ]
    found: dict[Path, BackupItem] = {}
    for family, root, pattern, dirs_only in specs:
        if not root.is_dir():
            continue
        for path in root.glob(pattern):
            if dirs_only and not path.is_dir():
                continue
            if not dirs_only and not path.is_file():
                continue
            try:
                st = path.lstat()
                item = BackupItem(
                    family=family,
                    path=path,
                    mtime=st.st_mtime,
                    size=_tree_size(path),
                    protected=_protected(path),
                )
            except FileNotFoundError:
                continue
            found[path] = item
    return sorted(found.values(), key=lambda item: item.mtime, reverse=True)


def _calendar_keys(item: BackupItem) -> tuple[str, str]:
    dt = datetime.fromtimestamp(item.mtime).astimezone()
    iso = dt.isocalendar()
    return dt.strftime("%Y-%m-%d"), f"{iso.year}-W{iso.week:02d}"


def choose_kept(items: Iterable[BackupItem], retention: Retention) -> set[Path]:
    ordered = sorted(items, key=lambda item: item.mtime, reverse=True)
    kept: set[Path] = {item.path for item in ordered if item.protected}
    for item in ordered[: max(0, retention.recent)]:
        kept.add(item.path)

    seen_days: set[str] = set()
    for item in ordered:
        day, _week = _calendar_keys(item)
        if day in seen_days:
            continue
        if len(seen_days) >= max(0, retention.daily):
            break
        seen_days.add(day)
        kept.add(item.path)

    seen_weeks: set[str] = set()
    for item in ordered:
        _day, week = _calendar_keys(item)
        if week in seen_weeks:
            continue
        if len(seen_weeks) >= max(0, retention.weekly):
            break
        seen_weeks.add(week)
        kept.add(item.path)
    return kept


def make_plan(items: list[BackupItem], retention: Retention) -> list[PlanEntry]:
    result: list[PlanEntry] = []
    by_family: dict[str, list[BackupItem]] = {}
    for item in items:
        by_family.setdefault(item.family, []).append(item)
    for family, family_items in sorted(by_family.items()):
        kept = choose_kept(family_items, retention)
        for item in sorted(family_items, key=lambda x: x.mtime, reverse=True):
            if item.protected:
                action, reason = "keep", "protected_marker_or_symlink"
            elif item.path in kept:
                action, reason = "keep", "retention"
            else:
                action, reason = "delete", "expired_redundant_copy"
            result.append(PlanEntry(family, str(item.path), item.size, item.mtime, action, reason))
    return result


def apply_plan(plan: list[PlanEntry], *, max_delete_bytes: int) -> tuple[int, int]:
    candidates = [entry for entry in plan if entry.action == "delete"]
    requested = sum(entry.size for entry in candidates)
    if requested > max_delete_bytes:
        raise RuntimeError(f"delete_budget_exceeded:{requested}>{max_delete_bytes}")
    deleted_count = 0
    deleted_bytes = 0
    for entry in candidates:
        path = Path(entry.path)
        if not path.exists():
            continue
        if path.is_symlink() or _protected(path):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        if path.exists():
            raise RuntimeError(f"delete_verification_failed:{path}")
        deleted_count += 1
        deleted_bytes += entry.size
    return deleted_count, deleted_bytes


def _used_percent(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return 100.0 * usage.used / usage.total if usage.total else 0.0


def run(*, user_home: Path, apply: bool, max_delete_gib: float) -> dict:
    used_before = _used_percent(user_home)
    tier, retention = retention_for_used_percent(used_before)
    items = discover(user_home)
    plan = make_plan(items, retention)
    reclaimable = sum(entry.size for entry in plan if entry.action == "delete")
    deleted_count = 0
    deleted_bytes = 0
    if apply:
        deleted_count, deleted_bytes = apply_plan(
            plan,
            max_delete_bytes=int(max_delete_gib * (1024**3)),
        )
    return {
        "schema": "hermes.backup_lifecycle.v1",
        "mode": "apply" if apply else "dry_run",
        "pressure_tier": tier,
        "disk_used_percent_before": round(used_before, 2),
        "disk_used_percent_after": round(_used_percent(user_home), 2),
        "retention": asdict(retention),
        "managed_items": len(items),
        "reclaimable_bytes": reclaimable,
        "deleted_count": deleted_count,
        "deleted_bytes": deleted_bytes,
        "plan": [asdict(entry) for entry in plan],
        "generated_at": time.time(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded Hermes backup lifecycle manager")
    parser.add_argument("--apply", action="store_true", help="delete planned redundant copies")
    parser.add_argument("--user-home", default=str(Path.home()), help=argparse.SUPPRESS)
    parser.add_argument("--max-delete-gib", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        user_home=Path(args.user_home).expanduser(),
        apply=bool(args.apply),
        max_delete_gib=max(0.0, float(args.max_delete_gib)),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        gib = 1024**3
        print(
            f"backup lifecycle: {report['mode']} tier={report['pressure_tier']} "
            f"managed={report['managed_items']} reclaimable={report['reclaimable_bytes']/gib:.2f}GiB "
            f"deleted={report['deleted_bytes']/gib:.2f}GiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
