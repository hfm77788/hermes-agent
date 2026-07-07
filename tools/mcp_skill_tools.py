"""
Hermes MCP Skill Tools -- read-only skill access for ChatGPT.

Phase 1 (read-only):
  hermes_health_check    -- MCP/Skills health status
  resolve_skill_uri      -- resolve skill name to canonical URI
  read_skill_bundle      -- read SKILL.md summary + manifest
  read_skill_file_chunked -- chunked file reading
  smoke_skill_access     -- accessibility check (PASS/WARN/FAIL)

Security:
  - skill_name must not be absolute or contain ..
  - All resolved paths verified relative_to(skills_root)
  - Forbidden paths checked on resolved target
  - Symlinks validated before read
  - Truncation signal preserved
  - Structured errors with error codes
  - Read-only: no writes, no mutations
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Hermes home resolver -- use the canonical resolver
# ---------------------------------------------------------------------------

try:
    from hermes_constants import get_hermes_home as _canonical_get_hermes_home
except ImportError:
    _canonical_get_hermes_home = None


def _get_hermes_home() -> Optional[Path]:
    """Resolve Hermes home directory using the canonical resolver."""
    if _canonical_get_hermes_home:
        try:
            return _canonical_get_hermes_home()
        except Exception:
            pass
    # Fallback
    for path_str in [
        os.environ.get("HERMES_HOME"),
        os.path.expanduser("~/.hermes"),
    ]:
        if path_str:
            p = Path(path_str)
            if p.exists():
                return p
    return None


def _get_skills_root() -> Optional[Path]:
    """Resolve skills root directory."""
    home = _get_hermes_home()
    if not home:
        return None
    skills = home / "skills"
    if skills.exists() and skills.is_dir():
        return skills
    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORBIDDEN_PATTERNS = [
    "**/.env",
    "**/*token*",
    "**/*secret*",
    "**/*cookie*",
    "**/*session*",
    "**/.ssh/**",
    "**/.config/gh/**",
    "**/settings/**",
    "**/secrets/**",
]

FORBIDDEN_PATH_NAMES = {
    ".env", ".ssh", ".config/gh", "token", "secrets", "settings",
    "cookies", "sessions", "credentials",
}

MAX_OUTPUT_CHARS = 20000
MAX_READ_LINES = 5000

# ---------------------------------------------------------------------------
# Error codes (structured)
# ---------------------------------------------------------------------------

ERROR_CODES = {
    "hermes_namespace_unavailable": "Hermes home directory not found",
    "htp_connection_error": "HTP connection failed",
    "skill_uri_not_found": "Skill name could not be resolved",
    "skill_read_stream_interrupted": "Skill read interrupted",
    "platform_safety_blocked": "Access blocked by platform safety rules",
    "forbidden_path_denied": "Path matches forbidden pattern",
    "skill_not_found": "Skill not found on filesystem",
    "chunk_required": "File too large, use chunked read",
}


def _is_forbidden_path(path: Path) -> Tuple[bool, str]:
    """Check if a resolved path matches any forbidden pattern.

    Args:
        path: Already-resolved Path (caller must resolve first).

    Returns (is_forbidden, reason).
    """
    path_str = str(path)
    path_lower = path_str.lower()

    parts = path.parts
    for part in parts:
        if part in FORBIDDEN_PATH_NAMES:
            return True, f"forbidden_path_denied: '{part}' in path"

    for pattern in FORBIDDEN_PATTERNS:
        if path.match(pattern):
            return True, f"forbidden_path_denied: matches '{pattern}'"

    sensitive = [".env", "token", "secret", "cookie", "session"]
    for s in sensitive:
        for part in parts:
            if part == s:
                return True, f"forbidden_path_denied: '{s}' in path"

    return False, ""


def _validate_symlink_target(link_path: Path, allowed_root: Path) -> Tuple[bool, str]:
    """Validate that a symlink target is within an allowed root.

    Returns (allowed, reason).
    """
    if not link_path.is_symlink():
        return True, ""
    resolved_target = link_path.resolve()
    try:
        resolved_target.relative_to(allowed_root.resolve())
    except ValueError:
        return False, (
            f"forbidden_path_denied: symlink {link_path} "
            f"points outside allowed root {allowed_root}"
        )
    # Also check the resolved target for forbidden patterns
    is_forbidden, reason = _is_forbidden_path(resolved_target)
    if is_forbidden:
        return False, reason
    return True, ""


def _resolve_skill_path(skill_name: str) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Resolve a skill name to its filesystem path.

    Security:
      - skill_name must not be absolute
      - skill_name must not contain '..'
      - All resolved paths must be relative_to(skills_root.resolve())
      - Forbidden path check on resolved target

    Returns (path, metadata).
    """
    skills_root = _get_skills_root()
    meta: Dict[str, Any] = {
        "skill_name": skill_name,
        "canonical_uri": None,
        "local_path": None,
        "category": None,
        "exists": False,
        "confidence": "none",
        "candidates": [],
    }

    if not skills_root:
        meta["error_code"] = "hermes_namespace_unavailable"
        return None, meta

    # === Security: deny absolute paths and traversal ===
    if os.path.isabs(skill_name):
        meta["error_code"] = "forbidden_path_denied: skill_name must not be absolute"
        return None, meta
    if ".." in Path(skill_name).parts:
        meta["error_code"] = "forbidden_path_denied: path traversal denied"
        return None, meta

    skills_root_resolved = skills_root.resolve()

    # Direct match: skills_root/<skill_name>/SKILL.md
    direct = skills_root / skill_name / "SKILL.md"
    direct_resolved = direct.resolve()
    try:
        direct_resolved.relative_to(skills_root_resolved)
    except ValueError:
        meta["error_code"] = "forbidden_path_denied: resolved path outside skills root"
        return None, meta

    if direct_resolved.exists() and direct_resolved.is_file():
        is_forbidden, reason = _is_forbidden_path(direct_resolved)
        if is_forbidden:
            meta["error_code"] = reason
            return None, meta
        meta["canonical_uri"] = f"skill:{skill_name}"
        meta["local_path"] = str(direct_resolved)
        meta["exists"] = True
        meta["confidence"] = "exact"
        return direct_resolved, meta

    # Recursive search in subdirectories
    candidates = []
    for root, dirs, files in os.walk(str(skills_root)):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f == "SKILL.md":
                parent = Path(root).name
                if parent == skill_name or skill_name in parent:
                    full_path = (Path(root) / f).resolve()
                    try:
                        full_path.relative_to(skills_root_resolved)
                    except ValueError:
                        continue
                    is_forbidden, reason = _is_forbidden_path(full_path)
                    if is_forbidden:
                        continue
                    rel = full_path.relative_to(skills_root_resolved)
                    category = rel.parts[0] if len(rel.parts) > 2 else None
                    candidates.append({
                        "canonical_uri": f"skill:{rel.parent}",
                        "local_path": str(full_path),
                        "category": str(category) if category else None,
                    })

    if candidates:
        meta["candidates"] = candidates
        if len(candidates) == 1:
            c = candidates[0]
            meta["canonical_uri"] = c["canonical_uri"]
            meta["local_path"] = c["local_path"]
            meta["category"] = c["category"]
            meta["exists"] = True
            meta["confidence"] = "single_match"
            return Path(c["local_path"]), meta
        else:
            meta["exists"] = False
            meta["confidence"] = "multiple_matches"
            return None, meta

    meta["error_code"] = "skill_uri_not_found"
    return None, meta


def _read_frontmatter(content: str) -> Dict[str, Any]:
    """Extract YAML frontmatter from SKILL.md content."""
    if not content.startswith("---"):
        return {}
    try:
        end = content.index("---", 3)
        frontmatter_str = content[3:end].strip()
        result = {}
        for line in frontmatter_str.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, _, val = line.partition(":")
                result[key.strip()] = val.strip().strip('"').strip("'")
        return result
    except ValueError:
        return {}


def _get_skill_dir(skill_path: Path) -> Path:
    """Get the skill directory from SKILL.md path."""
    return skill_path.parent


def _safe_read_index(skill_dir: Path, skills_root: Path) -> Tuple[Optional[str], Optional[str]]:
    """Safely read references/_index.md with symlink validation.

    Returns (content, error_code) -- one will be None.
    """
    refs_dir = skill_dir / "references"
    index_path = refs_dir / "_index.md"
    if not index_path.exists():
        return None, None

    # Validate symlink
    allowed, reason = _validate_symlink_target(index_path, skills_root)
    if not allowed:
        return None, reason

    try:
        # Resolve first, then read
        resolved = index_path.resolve()
        # Re-verify resolved path is within skills_root
        try:
            resolved.relative_to(skills_root.resolve())
        except ValueError:
            return None, "forbidden_path_denied: _index.md resolves outside skills root"
        content = resolved.read_text(encoding="utf-8")
        return content[:1000], None
    except Exception:
        return None, None


def _safe_list_dir(dir_path: Path, skills_root: Path) -> Tuple[List[str], Optional[str]]:
    """List files in a directory, validating symlinks.

    Validates the directory itself first, then each child.
    Returns (files, error_code) -- if error_code is set, files is empty.
    """
    if not dir_path.exists():
        return [], None

    # Validate the directory itself (if it's a symlink)
    allowed, reason = _validate_symlink_target(dir_path, skills_root)
    if not allowed:
        return [], reason

    resolved_dir = dir_path.resolve()
    try:
        resolved_dir.relative_to(skills_root.resolve())
    except ValueError:
        return [], "forbidden_path_denied: references/scripts dir resolves outside skills root"

    result = []
    for f in dir_path.iterdir():
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.is_symlink():
            allowed, _ = _validate_symlink_target(f, skills_root)
            if not allowed:
                continue
        # Also verify resolved child is within skills_root
        try:
            f.resolve().relative_to(skills_root.resolve())
        except ValueError:
            continue
        result.append(f.name)
    return sorted(result), None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hermes_health_check() -> Dict[str, Any]:
    """Check Hermes MCP health status.

    Does NOT assume a specific skill exists. Uses generic filesystem checks:
      - skills_root exists / is_dir / readable (os.access)
      - iterates to find any SKILL.md for readability test
    """
    result = {
        "mcp_ok": True,
        "htp_ok": True,
        "skills_root_exists": False,
        "skills_root_readable": False,
        "version": "0.1.1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    skills_root = _get_skills_root()
    if not skills_root:
        return result

    result["skills_root_exists"] = True
    result["skills_root"] = str(skills_root)

    # Generic readability: check exists + is_dir + os.access
    if skills_root.exists() and skills_root.is_dir():
        if os.access(str(skills_root), os.R_OK):
            # Try to find any SKILL.md for a deeper readability test
            for root, dirs, files in os.walk(str(skills_root)):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    if f == "SKILL.md":
                        try:
                            test_path = Path(root) / f
                            with open(test_path, "r") as fh:
                                fh.read(100)
                            result["skills_root_readable"] = True
                        except Exception:
                            pass
                        break
                if result["skills_root_readable"]:
                    break
            # If no SKILL.md found, check root dir readability
            if not result["skills_root_readable"]:
                result["skills_root_readable"] = os.access(str(skills_root), os.R_OK)

    return result


def resolve_skill_uri(skill_name: str) -> Dict[str, Any]:
    """Resolve a skill name to its canonical URI.

    Args:
        skill_name: Skill name, e.g. "reference-writing"

    Returns:
        canonical_uri, local_path, category, exists, confidence,
        candidates (if multiple matches)
    """
    path, meta = _resolve_skill_path(skill_name)
    return meta


def read_skill_bundle(skill_name: str) -> Dict[str, Any]:
    """Read a skill's SKILL.md summary, manifest, and references list.

    For instructional tools: if the SKILL.md is too large, returns
    explicit signals to prevent the client from operating on partial content.

    Args:
        skill_name: Skill name or canonical URI (e.g. "reference-writing"
                    or "skill:productivity/reference-writing")

    Returns:
        SKILL.md summary, line_count, frontmatter, references list,
        scripts list, _index.md summary (if exists).
        For large files: full_content_available=false, chunk_required=true,
        complete_instruction_loaded=false, required_next_chunks.
    """
    if skill_name.startswith("skill:"):
        skill_name = skill_name.split(":", 1)[1]

    path, meta = _resolve_skill_path(skill_name)
    if not path or not meta["exists"]:
        result = dict(meta)
        result["skill_name"] = skill_name
        return result

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return {
            "skill_name": skill_name,
            "error_code": "skill_read_stream_interrupted",
            "error": str(e),
        }

    lines = content.split("\n")
    line_count = len(lines)
    frontmatter = _read_frontmatter(content)
    content_size = len(content)

    skills_root = _get_skills_root()
    skill_dir = _get_skill_dir(path)

    if content_size > MAX_OUTPUT_CHARS:
        # Large file: return explicit partial-content signals
        summary_lines_count = 100
        summary_lines = lines[:summary_lines_count]
        # Calculate suggested ranges for chunked reading
        chunk_size = 500  # lines per suggested chunk
        suggested_ranges = []
        for i in range(summary_lines_count + 1, line_count + 1, chunk_size):
            suggested_ranges.append({
                "start_line": i,
                "end_line": min(i + chunk_size - 1, line_count),
            })
        result = {
            "skill_name": skill_name,
            "canonical_uri": meta.get("canonical_uri"),
            "line_count": line_count,
            "content_size_bytes": content_size,
            "frontmatter": frontmatter,
            "summary": "\n".join(summary_lines),
            "summary_lines": summary_lines_count,
            "full_content_available": False,
            "chunk_required": True,
            "complete_instruction_loaded": False,
            "required_next_chunks": len(suggested_ranges),
            "suggested_ranges": suggested_ranges[:5],
            "chunk_hint": (
                f"File is {content_size} bytes ({line_count} lines). "
                f"Use read_skill_file_chunked to read remaining sections."
            ),
        }
    else:
        result = {
            "skill_name": skill_name,
            "canonical_uri": meta.get("canonical_uri"),
            "line_count": line_count,
            "content_size_bytes": content_size,
            "frontmatter": frontmatter,
            "content": content,
            "full_content_available": True,
            "complete_instruction_loaded": True,
        }

    # List references (with symlink validation)
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        if skills_root:
            refs, refs_err = _safe_list_dir(refs_dir, skills_root)
            if refs_err:
                result["references_error"] = refs_err
                result["references"] = []
            else:
                result["references"] = refs
        else:
            result["references"] = sorted([
                f.name for f in refs_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ])

    # List scripts (with symlink validation)
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.exists():
        if skills_root:
            scripts, scripts_err = _safe_list_dir(scripts_dir, skills_root)
            if scripts_err:
                result["scripts_error"] = scripts_err
                result["scripts"] = []
            else:
                result["scripts"] = scripts
        else:
            result["scripts"] = sorted([
                f.name for f in scripts_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ])

    # _index.md summary (with symlink validation)
    if skills_root and refs_dir.exists():
        idx_content, idx_error = _safe_read_index(skill_dir, skills_root)
        if idx_error:
            result["_index_error"] = idx_error
        elif idx_content:
            result["_index_summary"] = idx_content

    return result


def read_skill_file_chunked(
    canonical_uri: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
    relative_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a chunk of a skill file by line range.

    Args:
        canonical_uri: Canonical URI, e.g. "skill:productivity/reference-writing"
        start_line: 1-indexed start line
        end_line: 1-indexed end line (inclusive). If None, reads to end.
        relative_path: Optional relative path within skill_dir for non-SKILL.md
                      files, e.g. "references/_index.md" or "references/styles/a.md".
                      Must be relative, no '..', resolve within skill_dir.

    Returns:
        chunk content, line numbers, and truncation signals.
    """
    if canonical_uri.startswith("skill:"):
        skill_name = canonical_uri.split(":", 1)[1]
    else:
        skill_name = canonical_uri

    path, meta = _resolve_skill_path(skill_name)
    if not path or not meta["exists"]:
        result = dict(meta)
        result["canonical_uri"] = canonical_uri
        return result

    # Determine the target file
    if relative_path:
        rp = Path(relative_path)
        if rp.is_absolute():
            return {"error_code": "forbidden_path_denied: relative_path must not be absolute",
                    "canonical_uri": canonical_uri}
        if ".." in rp.parts:
            return {"error_code": "forbidden_path_denied: path traversal denied",
                    "canonical_uri": canonical_uri}
        if any(part.startswith(".") for part in rp.parts):
            return {"error_code": "forbidden_path_denied: hidden path segment denied",
                    "canonical_uri": canonical_uri}
        skill_dir = _get_skill_dir(path).resolve()
        target = (skill_dir / rp).resolve()
        try:
            target.relative_to(skill_dir)
        except ValueError:
            return {"error_code": "forbidden_path_denied: relative_path outside skill_dir",
                    "canonical_uri": canonical_uri}
        # Also verify within skills_root
        skills_root = _get_skills_root()
        if skills_root:
            try:
                target.relative_to(skills_root.resolve())
            except ValueError:
                return {"error_code": "forbidden_path_denied: target outside skills_root",
                        "canonical_uri": canonical_uri}
            # Symlink validation
            if target.is_symlink():
                allowed, reason = _validate_symlink_target(target, skills_root)
                if not allowed:
                    return {"error_code": reason, "canonical_uri": canonical_uri}
        if not target.exists():
            return {"error_code": "skill_uri_not_found",
                    "error": f"File not found: {relative_path}",
                    "canonical_uri": canonical_uri}
        target_path = target
    else:
        target_path = path

    is_forbidden, reason = _is_forbidden_path(target_path)
    if is_forbidden:
        return {"error_code": reason, "canonical_uri": canonical_uri}

    try:
        content = target_path.read_text(encoding="utf-8")
    except Exception as e:
        return {
            "error_code": "skill_read_stream_interrupted",
            "error": str(e),
            "canonical_uri": canonical_uri,
        }

    lines = content.split("\n")
    total_lines = len(lines)

    if end_line is None:
        end_line = total_lines
    if end_line > total_lines:
        end_line = total_lines
    if start_line < 1:
        start_line = 1
    if start_line > total_lines:
        return {
            "error_code": "chunk_required",
            "message": f"start_line ({start_line}) exceeds total lines ({total_lines})",
            "canonical_uri": canonical_uri,
            "total_lines": total_lines,
        }

    requested_lines = end_line - start_line + 1
    chunk = lines[start_line - 1:end_line]
    chunk_size = len("\n".join(chunk))

    truncated = False
    next_start_line = None
    if chunk_size > MAX_OUTPUT_CHARS:
        truncated = True
        chunk = chunk[:100]
        chunk_size = len("\n".join(chunk))
        next_start_line = start_line + 100

    return {
        "canonical_uri": canonical_uri,
        "relative_path": relative_path,
        "start_line": start_line,
        "end_line": end_line if not truncated else start_line + 99,
        "total_lines": total_lines,
        "requested_lines": requested_lines,
        "returned_lines": len(chunk),
        "chunk": "\n".join(chunk),
        "chunk_size_bytes": chunk_size,
        "truncated": truncated,
        "chunk_required": truncated,
        "next_start_line": next_start_line,
    }


def smoke_skill_access(skill_name: str) -> Dict[str, Any]:
    """Check accessibility of a skill.

    Args:
        skill_name: Skill name, e.g. "reference-writing"

    Returns:
        checks dict with PASS/WARN/FAIL status for each check,
        overall status.
    """
    path, meta = _resolve_skill_path(skill_name)
    checks = {
        "skill_md": "FAIL",
        "references_index": "FAIL",
        "references_dir": "FAIL",
        "scripts_dir": "FAIL",
    }

    if not path:
        checks["resolved"] = "FAIL"
        checks["error"] = meta.get("error_code", "skill_uri_not_found")
        return {"skill_name": skill_name, "checks": checks, "overall": "FAIL"}

    checks["resolved"] = "PASS"
    checks["local_path"] = str(path)

    if path.exists():
        checks["skill_md"] = "PASS"
    else:
        checks["skill_md"] = "FAIL"

    skill_dir = _get_skill_dir(path)
    skills_root = _get_skills_root()

    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        checks["references_dir"] = "PASS"
        index_path = refs_dir / "_index.md"
        if index_path.exists():
            if skills_root:
                allowed, _ = _validate_symlink_target(index_path, skills_root)
                checks["references_index"] = "PASS" if allowed else "FAIL"
            else:
                checks["references_index"] = "PASS"
        else:
            checks["references_index"] = "WARN"
    else:
        checks["references_dir"] = "WARN"
        checks["references_index"] = "WARN"

    scripts_dir = skill_dir / "scripts"
    if scripts_dir.exists():
        checks["scripts_dir"] = "PASS"
    else:
        checks["scripts_dir"] = "WARN"

    if "FAIL" in checks.values():
        overall = "FAIL"
    elif "WARN" in checks.values():
        overall = "WARN"
    else:
        overall = "PASS"

    return {
        "skill_name": skill_name,
        "canonical_uri": meta.get("canonical_uri"),
        "checks": checks,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Phase 2: Preauthorized Write API
# ---------------------------------------------------------------------------

import datetime
import difflib
import re
import shutil
import uuid

ALLOWED_PATHS = [
    "/home/ubuntu/.hermes/skills",
    "/tmp/hermes-artifacts",
    "/tmp/hermes-skill-backups",
]

ALLOWED_SCOPES = ["P0_READ", "P1_ARTIFACT", "P2_SKILL_PATCH", "P3_PR_CREATE"]
FORBIDDEN_SCOPES = ["P4_RESTRICTED"]

SENSITIVE_CONTENT_PATTERNS = [
    ("api_key_pattern", re.compile(r'(?:api[_-]?key|apikey)\s*[:=]\s*[\'"\w]+', re.IGNORECASE)),
    ("token_pattern", re.compile(r'(?:access[_-]?token|refresh[_-]?token)\s*[:=]\s*[\'"\w\.\-]+', re.IGNORECASE)),
    ("private_key_pattern", re.compile(r'(?:private[_-]?key|secret[_-]?key)\s*[:=]', re.IGNORECASE)),
    ("password_pattern", re.compile(r'(?:password|passwd)\s*[:=]\s*[\'"]\S+[\'"]', re.IGNORECASE)),
    ("github_token_pattern", re.compile(r'(?:ghp_|gho_|github_pat_|sk-)[\w]{20,}', re.IGNORECASE)),
    ("pem_key_pattern", re.compile(r'(?:-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----)', re.IGNORECASE)),
]


def _is_in_allowed_paths(target_path: Path) -> bool:
    resolved = target_path.resolve()
    for allowed in ALLOWED_PATHS:
        try:
            resolved.relative_to(Path(allowed).resolve())
            return True
        except ValueError:
            continue
    skills_root = _get_skills_root()
    if skills_root:
        try:
            resolved.relative_to(skills_root.resolve())
            return True
        except ValueError:
            pass
    return False


def _scan_new_content(content: str) -> Tuple[bool, str]:
    for pattern_id, pattern in SENSITIVE_CONTENT_PATTERNS:
        if pattern.search(content):
            return True, f"forbidden_content_denied: {pattern_id}"
    return False, ""


def _redact_diff(diff_text: str) -> str:
    for _, pattern in SENSITIVE_CONTENT_PATTERNS:
        diff_text = pattern.sub("***REDACTED***", diff_text)
    return diff_text


def get_preauthorization_profile() -> Dict[str, Any]:
    return {
        "profile": "chatgpt-hermes-high-trust-v1",
        "allowed_scopes": ALLOWED_SCOPES,
        "forbidden_scopes": FORBIDDEN_SCOPES,
        "allowed_paths": [f"{p}/**" for p in ALLOWED_PATHS],
        "max_output_chars": MAX_OUTPUT_CHARS,
        "max_read_lines": MAX_READ_LINES,
        "version": "0.2.0",
        "p3_pr_create": "policy_declared_only",
        "p3_pr_create_note": "PR creation is handled by the GitHub connector.",
    }


def _create_backup_metadata(
    target_path: Path, skill_name: str, actual_file_path: Path,
    existed_before: bool
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Create a unique backup and metadata for the actual file.

    Always creates metadata even if the file doesn't exist (existed_before=False).
    """
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = str(uuid.uuid4())[:8]
    backup_base = Path("/tmp/hermes-skill-backups")
    backup_dir = backup_base / f"{ts}-{suffix}" / skill_name
    backup_dir.mkdir(parents=True, exist_ok=True)

    skill_dir = _get_skill_dir(target_path)
    relative = str(actual_file_path.relative_to(skill_dir)) if actual_file_path.is_relative_to(skill_dir) else actual_file_path.name

    backup_file_path = None
    try:
        if existed_before:
            # Preserve nested directory structure
            dest = backup_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(actual_file_path), str(dest))
            backup_file_path = str(dest)

        # Always back up SKILL.md if it exists
        if target_path.exists():
            shutil.copy2(str(target_path), str(backup_dir / "SKILL.md"))

        refs_src = skill_dir / "references"
        if refs_src.exists():
            refs_dest = backup_dir / "references"
            shutil.copytree(str(refs_src), str(refs_dest), dirs_exist_ok=True)

        scripts_src = skill_dir / "scripts"
        if scripts_src.exists():
            scripts_dest = backup_dir / "scripts"
            shutil.copytree(str(scripts_src), str(scripts_dest), dirs_exist_ok=True)

    except Exception as e:
        return None, {"error_code": "backup_failed", "error": str(e)}

    return backup_dir, {
        "backup_path": str(backup_dir),
        "timestamp": ts,
        "original_target_path": str(target_path),
        "actual_file_path": str(actual_file_path),
        "relative_file_path": relative,
        "skill_dir": str(skill_dir),
        "backup_file_path": backup_file_path,
        "skill_name": skill_name,
        "existed_before": existed_before,
    }


def _compute_diff(file_path: str, old_content: str, new_content: str) -> str:
    """Compute a redacted unified diff between old and new content."""
    return _redact_diff(
        "\n".join(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        ))
    )


def run_preauthorized_skill_patch(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a preauthorized skill file patch.

    Write transaction model:
    1. Validate file_path (relative, no .., no hidden segments, no forbidden)
    2. Determine if target exists (existed_before)
    3. Create backup metadata regardless
    4. Write new content (skip if dry_run=true)
    5. Return redacted diff + smoke check
    6. Dry-run mode: validate + diff + smoke, no write, no backup
    """
    skill_name = manifest.get("skill_name", "")
    file_path = manifest.get("file_path", "SKILL.md")
    new_content = manifest.get("new_content", "")
    action = manifest.get("action", "replace")
    dry_run = manifest.get("dry_run", False)

    if not skill_name:
        return {"error_code": "skill_uri_not_found", "error": "skill_name is required"}

    # === Validate file_path ===
    path_obj = Path(file_path)
    if path_obj.is_absolute():
        return {"error_code": "forbidden_path_denied", "error": "file_path must be relative"}
    if ".." in path_obj.parts:
        return {"error_code": "forbidden_path_denied", "error": "path traversal denied"}
    if any(part.startswith(".") for part in path_obj.parts):
        return {"error_code": "forbidden_path_denied", "error": "hidden path segment denied"}

    path, meta = _resolve_skill_path(skill_name)
    if not path or not meta["exists"]:
        result = dict(meta)
        result["skill_name"] = skill_name
        return result

    skill_dir = _get_skill_dir(path).resolve()
    actual_target = (skill_dir / path_obj).resolve()

    try:
        actual_target.relative_to(skill_dir)
    except ValueError:
        return {"error_code": "patch_target_not_in_allowed_paths", "error": "target outside skill_dir"}

    if not _is_in_allowed_paths(actual_target):
        return {"error_code": "patch_target_not_in_allowed_paths", "error": "not in allowed_paths"}

    is_forbidden, reason = _is_forbidden_path(actual_target)
    if is_forbidden:
        return {"error_code": reason}

    # Symlink check
    if actual_target.exists() and actual_target.is_symlink():
        skills_root = _get_skills_root()
        if skills_root:
            allowed, reason = _validate_symlink_target(actual_target, skills_root)
            if not allowed:
                return {"error_code": reason}

    if action == "delete" and file_path == ".":
        return {"error_code": "delete_skill_dir_denied"}

    # === Content scan ===
    found_sensitive, scan_reason = _scan_new_content(new_content)
    if found_sensitive:
        return {"error_code": scan_reason}

    # === Determine existed_before ===
    existed_before = actual_target.exists()

    # === Read old content (if exists) ===
    old_content = ""
    if existed_before:
        try:
            old_content = actual_target.read_text(encoding="utf-8")
        except Exception as e:
            return {"error_code": "skill_read_stream_interrupted", "error": str(e)}

    # === Compute diff (always, even for dry-run) ===
    diff = _compute_diff(file_path, old_content, new_content)
    would_change = old_content != new_content

    # === Dry-run: return preview without writing ===
    if dry_run:
        return {
            "dry_run": True,
            "would_change": would_change,
            "skill_name": skill_name,
            "file_path": str(actual_target),
            "action": action,
            "diff": diff,
            "diff_lines": len(diff.split("\n")) - 1 if diff else 0,
            "old_size_bytes": len(old_content),
            "new_size_bytes": len(new_content),
            "existed_before": existed_before,
            "planned_files": [str(actual_target)],
            "backup_required": would_change and existed_before,
            "status": "dry_run_ok",
        }

    # === Create backup metadata ===
    backup_path, backup_meta = _create_backup_metadata(path, skill_name, actual_target, existed_before)
    if not backup_path:
        return backup_meta

    # === Write new content ===
    try:
        actual_target.parent.mkdir(parents=True, exist_ok=True)
        actual_target.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return {"error_code": "skill_patch_validation_failed", "error": str(e)}

    smoke_result = smoke_skill_access(skill_name)

    return {
        "skill_name": skill_name,
        "file_path": str(actual_target),
        "action": action,
        "backup_path": backup_meta.get("backup_path"),
        "backup_metadata": backup_meta,
        "diff": diff,
        "diff_lines": len(diff.split("\n")) - 1 if diff else 0,
        "files_written": [str(actual_target)],
        "old_size_bytes": len(old_content),
        "new_size_bytes": len(new_content),
        "existed_before": existed_before,
        "validation": {
            "smoke_skill_access": smoke_result["overall"],
            "smoke_checks": smoke_result["checks"],
        },
        "status": "ok" if smoke_result["overall"] in ("PASS", "WARN") else "validation_failed",
    }


def rollback_skill_patch(backup_path: str) -> Dict[str, Any]:
    """Rollback a skill patch using backup metadata.

    For each backed-up file:
      - existed_before=true: restore original content
      - existed_before=false: delete the newly created file
    Supports all target types: SKILL.md, README.md, references/*, scripts/*, nested paths.
    """
    backup_dir = Path(backup_path).resolve()
    backup_root = Path("/tmp/hermes-skill-backups").resolve()

    try:
        backup_dir.relative_to(backup_root)
    except ValueError:
        return {"error_code": "forbidden_path_denied", "error": "backup not under /tmp/hermes-skill-backups/"}

    if not backup_dir.exists():
        return {"error_code": "backup_path_not_found"}

    skill_name = backup_dir.name
    skill_md_path = backup_dir / "SKILL.md"
    if not skill_md_path.exists():
        return {"error_code": "backup_path_not_found", "error": "SKILL.md not found in backup"}

    path, meta = _resolve_skill_path(skill_name)
    if not path:
        return {"error_code": "skill_uri_not_found"}

    skill_dir = _get_skill_dir(path).resolve()
    skills_root = _get_skills_root()
    if not skills_root:
        return {"error_code": "hermes_namespace_unavailable"}
    try:
        skill_dir.relative_to(skills_root.resolve())
    except ValueError:
        return {"error_code": "forbidden_path_denied", "error": "target outside skills root"}

    restored = []
    deleted = []
    try:
        # Restore root-level backup files (non-SKILL.md, non-dir)
        for f in backup_dir.iterdir():
            if f.is_file() and f.name != "SKILL.md":
                dest = skill_dir / f.name
                dest.write_text(f.read_text(encoding="utf-8"))
                restored.append(str(dest))

        # Restore SKILL.md
        original_content = skill_md_path.read_text(encoding="utf-8")
        path.write_text(original_content, encoding="utf-8")
        restored.append(str(path))

        # Restore references (preserving nested structure)
        refs_backup = backup_dir / "references"
        if refs_backup.exists():
            refs_original = skill_dir / "references"
            refs_original.mkdir(exist_ok=True)
            for f in refs_backup.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(refs_backup)
                    dest = refs_original / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))
                    restored.append(str(dest))

        # Restore scripts (preserving nested structure)
        scripts_backup = backup_dir / "scripts"
        if scripts_backup.exists():
            scripts_original = skill_dir / "scripts"
            scripts_original.mkdir(exist_ok=True)
            for f in scripts_backup.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(scripts_backup)
                    dest = scripts_original / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))
                    restored.append(str(dest))

        # Remove files that were newly created by the patch (existed_before=false)
        for f in skill_dir.rglob("*"):
            if f.is_file() and str(f) not in restored:
                # Check if this file was created by the patch (not in backup)
                if not any(str(f).startswith(str(r)) for r in restored):
                    f.unlink()
                    deleted.append(str(f))

    except Exception as e:
        return {"error_code": "rollback_failed", "error": str(e)}

    smoke_result = smoke_skill_access(skill_name)
    return {
        "rollback": "ok",
        "backup_path": str(backup_dir),
        "skill_name": skill_name,
        "restored_files": restored,
        "deleted_files": deleted,
        "count_restored": len(restored),
        "count_deleted": len(deleted),
        "validation": {
            "smoke_skill_access": smoke_result["overall"],
            "smoke_checks": smoke_result["checks"],
        },
    }