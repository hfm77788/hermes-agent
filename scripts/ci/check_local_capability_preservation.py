#!/usr/bin/env python3
"""Validate fork-local Hermes capabilities before merge/deploy.

This gate intentionally checks capability contracts, not historical blob SHAs.
Upstream code may evolve, but a sync must either preserve these symbols/wiring
and regressions or explicitly update this manifest under review.
"""

from __future__ import annotations

import argparse
import ast
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
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                raise ContractError(f"{capability_id}.{key} must be a non-empty string list")
            if not value:
                raise ContractError(f"{capability_id}.{key} must not be empty")
        for key in ("python_symbols", "text_markers"):
            value = capability.get(key, {})
            if not isinstance(value, dict):
                raise ContractError(f"{capability_id}.{key} must be an object")
            for relpath, items in value.items():
                if not isinstance(relpath, str) or not isinstance(items, list) or not items:
                    raise ContractError(f"{capability_id}.{key} contains an invalid entry")
                if not all(isinstance(item, str) and item for item in items):
                    raise ContractError(f"{capability_id}.{key}[{relpath!r}] must contain strings")
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


def _top_level_python_symbols(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ContractError(f"cannot parse {path}: {exc}") from exc
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


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
            path = _safe_path(root, relpath)
            if not path.is_file():
                capability_failures.append(f"missing required file: {relpath}")

        for relpath, expected_symbols in capability.get("python_symbols", {}).items():
            path = _safe_path(root, relpath)
            if not path.is_file():
                capability_failures.append(f"cannot inspect symbols; missing file: {relpath}")
                continue
            try:
                symbols = _top_level_python_symbols(path)
            except ContractError as exc:
                capability_failures.append(str(exc))
                continue
            for symbol in expected_symbols:
                if symbol not in symbols:
                    capability_failures.append(f"missing Python symbol {symbol!r} in {relpath}")

        for relpath, markers in capability.get("text_markers", {}).items():
            path = _safe_path(root, relpath)
            if not path.is_file():
                capability_failures.append(f"cannot inspect wiring; missing file: {relpath}")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                capability_failures.append(f"cannot read {relpath}: {exc}")
                continue
            for marker in markers:
                if marker not in text:
                    capability_failures.append(f"missing contract marker {marker!r} in {relpath}")

        for target in capability["pytest_targets"]:
            path = _safe_path(root, target)
            if not path.is_file():
                capability_failures.append(f"missing regression target: {target}")

        if capability_failures:
            failures.extend(f"{capability_id}: {failure}" for failure in capability_failures)
            print(f"FAIL {capability_id} ({len(capability_failures)} contract violation(s))")
        else:
            print(f"PASS {capability_id}")

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
        f"{len(manifest['capabilities'])} capability contract(s) verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
