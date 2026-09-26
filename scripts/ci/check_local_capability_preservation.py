#!/usr/bin/env python3
"""Validate fork-local Hermes capability regression contracts before merge/deploy.

The manifest deliberately avoids source-text/symbol change detectors. Capability
preservation is proven by executable regression tests; this helper only validates
that the reviewed contract files and canonical test targets still exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "hermes.local_capability_preservation.v1"
DEFAULT_MANIFEST = Path(__file__).with_name("local_capabilities.json")


class ContractError(ValueError):
    """Raised when the preservation manifest itself is malformed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read manifest {path}: {exc}") from exc
    if data.get("schema") != SCHEMA:
        raise ContractError(f"unexpected schema: {data.get('schema')!r}")
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise ContractError("capabilities must be a non-empty list")
    seen: set[str] = set()
    for capability in capabilities:
        if not isinstance(capability, dict):
            raise ContractError("each capability must be an object")
        capability_id = capability.get("id")
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise ContractError("each capability requires a non-empty id")
        if capability_id in seen:
            raise ContractError(f"duplicate capability id: {capability_id}")
        seen.add(capability_id)
        for key in ("required_files", "pytest_targets"):
            value = capability.get(key)
            if not isinstance(value, list) or not value:
                raise ContractError(f"{capability_id}.{key} must be a non-empty string list")
            if not all(isinstance(item, str) and item for item in value):
                raise ContractError(f"{capability_id}.{key} must contain non-empty strings")
    return data


def _safe_path(root: Path, relpath: str) -> Path:
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise ContractError(f"unsafe repository path: {relpath}")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"path escapes repository: {relpath}") from exc
    return candidate


def _pytest_targets(manifest: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for capability in manifest["capabilities"]:
        for target in capability["pytest_targets"]:
            if target not in seen:
                seen.add(target)
                targets.append(target)
    return targets


def check_manifest(manifest: dict[str, Any], root: Path) -> list[str]:
    failures: list[str] = []
    for capability in manifest["capabilities"]:
        capability_id = capability["id"]
        capability_failures: list[str] = []
        for relpath in capability["required_files"]:
            if not _safe_path(root, relpath).is_file():
                capability_failures.append(f"missing required file: {relpath}")
        for target in capability["pytest_targets"]:
            if not _safe_path(root, target).is_file():
                capability_failures.append(f"missing executable regression target: {target}")
        if capability_failures:
            failures.extend(f"{capability_id}: {failure}" for failure in capability_failures)
            print(f"FAIL {capability_id} ({len(capability_failures)} contract violation(s))")
        else:
            print(f"PASS {capability_id}: executable regression contract present")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--print-pytest-targets",
        action="store_true",
        help="Print deduplicated regression test paths and exit after manifest validation.",
    )
    args = parser.parse_args()
    try:
        manifest = _load_manifest(args.manifest)
        if args.print_pytest_targets:
            for target in _pytest_targets(manifest):
                print(target)
            return 0
        failures = check_manifest(manifest, _repo_root())
    except ContractError as exc:
        print(f"LOCAL CAPABILITY GATE ERROR: {exc}", file=sys.stderr)
        return 2
    if failures:
        print("\nLocal capability preservation gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        f"Local capability preservation gate PASS: "
        f"{len(manifest['capabilities'])} executable capability contract(s) verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
