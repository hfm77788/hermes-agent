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
        if s in path_lower:
            return True, f"forbidden_path_denied: path contains '{s}'"

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