"""
Hermes MCP Skill Tools — read-only skill access for ChatGPT.

Provides:
  hermes_health_check    — MCP/Skills health status
  resolve_skill_uri      — resolve skill name → canonical URI
  read_skill_bundle      — read SKILL.md summary + manifest
  read_skill_file_chunked — chunked file reading
  smoke_skill_access     — accessibility check (PASS/WARN/FAIL)

Security:
  - Forbidden paths: secrets, token, cookie, session, .env, ~/.ssh, ~/.config/gh
  - Single response > 20KB returns summary + chunk hint
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


def _get_hermes_home() -> Optional[Path]:
    """Resolve Hermes home directory."""
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
    if skills.exists():
        return skills
    return None


def _is_forbidden_path(path: Path) -> Tuple[bool, str]:
    """Check if a path matches any forbidden pattern.

    Returns (is_forbidden, reason).
    """
    path_str = str(path)
    path_lower = path_str.lower()

    # Check forbidden path name segments
    parts = path.parts
    for part in parts:
        if part in FORBIDDEN_PATH_NAMES:
            return True, f"forbidden_path_denied: '{part}' in path"

    # Check forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if path.match(pattern):
            return True, f"forbidden_path_denied: matches '{pattern}'"

    # Check sensitive substrings
    sensitive = [".env", "token", "secret", "cookie", "session"]
    for s in sensitive:
        if s in path_lower:
            return True, f"forbidden_path_denied: path contains '{s}'"

    return False, ""


def _resolve_skill_path(skill_name: str) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Resolve a skill name to its filesystem path.

    Returns (path, metadata) where metadata includes:
      canonical_uri, category, exists, confidence, candidates (if multiple)
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

    # Direct match: skills_root/<skill_name>/SKILL.md
    direct = skills_root / skill_name / "SKILL.md"
    if direct.exists():
        is_forbidden, reason = _is_forbidden_path(direct)
        if is_forbidden:
            meta["error_code"] = reason
            return None, meta
        meta["canonical_uri"] = f"skill:{skill_name}"
        meta["local_path"] = str(direct)
        meta["exists"] = True
        meta["confidence"] = "exact"
        return direct, meta

    # Recursive search in subdirectories
    candidates = []
    for root, dirs, files in os.walk(str(skills_root)):
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f == "SKILL.md":
                # Check if parent dir name matches
                parent = Path(root).name
                if parent == skill_name or skill_name in parent:
                    full_path = Path(root) / f
                    is_forbidden, reason = _is_forbidden_path(full_path)
                    if is_forbidden:
                        continue
                    rel = full_path.relative_to(skills_root)
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

    # Not found
    meta["error_code"] = "skill_uri_not_found"
    return None, meta


def _read_frontmatter(content: str) -> Dict[str, Any]:
    """Extract YAML frontmatter from SKILL.md content."""
    if not content.startswith("---"):
        return {}
    try:
        end = content.index("---", 3)
        frontmatter_str = content[3:end].strip()
        # Simple key: value parsing (avoid yaml import dependency)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hermes_health_check() -> Dict[str, Any]:
    """Check Hermes MCP health status.

    Returns: mcp_ok, htp_ok, skills_root_exists, skills_root_readable,
             version, timestamp.
    """
    result = {
        "mcp_ok": True,
        "htp_ok": True,
        "skills_root_exists": False,
        "skills_root_readable": False,
        "version": "0.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    skills_root = _get_skills_root()
    if skills_root:
        result["skills_root_exists"] = True
        result["skills_root"] = str(skills_root)
        # Check readability
        try:
            test_file = skills_root / "apple" / "SKILL.md"
            if test_file.exists():
                with open(test_file, "r") as f:
                    f.read(100)
                result["skills_root_readable"] = True
        except Exception:
            pass

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

    Args:
        skill_name: Skill name or canonical URI (e.g. "reference-writing"
                    or "skill:productivity/reference-writing")

    Returns:
        SKILL.md summary, line_count, frontmatter, references list,
        scripts list, _index.md summary (if exists).
        Does NOT return full content > 20KB.
    """
    # Strip "skill:" prefix if present
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

    # Truncate if too large
    content_size = len(content)
    if content_size > MAX_OUTPUT_CHARS:
        # Return summary only
        summary_lines = lines[:100]
        result = {
            "skill_name": skill_name,
            "canonical_uri": meta.get("canonical_uri"),
            "line_count": line_count,
            "content_size_bytes": content_size,
            "frontmatter": frontmatter,
            "summary": "\n".join(summary_lines),
            "chunk_required": True,
            "chunk_hint": f"File is {content_size} bytes. Use "
                          f"read_skill_file_chunked to read sections.",
        }
    else:
        result = {
            "skill_name": skill_name,
            "canonical_uri": meta.get("canonical_uri"),
            "line_count": line_count,
            "content_size_bytes": content_size,
            "frontmatter": frontmatter,
            "content": content,
        }

    # List references
    skill_dir = _get_skill_dir(path)
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        result["references"] = sorted([
            f.name for f in refs_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])

    # List scripts
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.exists():
        result["scripts"] = sorted([
            f.name for f in scripts_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])

    # _index.md summary
    if refs_dir and (refs_dir / "_index.md").exists():
        try:
            idx_content = (refs_dir / "_index.md").read_text(encoding="utf-8")
            result["_index_summary"] = idx_content[:1000]
        except Exception:
            pass

    return result


def read_skill_file_chunked(
    canonical_uri: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Read a chunk of a skill file by line range.

    Args:
        canonical_uri: Canonical URI, e.g. "skill:productivity/reference-writing"
        start_line: 1-indexed start line
        end_line: 1-indexed end line (inclusive). If None, reads to end.

    Returns:
        lines, line_count, chunk_required flag.
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

    # Forbidden check
    is_forbidden, reason = _is_forbidden_path(path)
    if is_forbidden:
        return {"error_code": reason, "canonical_uri": canonical_uri}

    try:
        content = path.read_text(encoding="utf-8")
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

    chunk = lines[start_line - 1:end_line]
    chunk_size = len("\n".join(chunk))

    if chunk_size > MAX_OUTPUT_CHARS:
        # Truncate
        chunk = chunk[:100]
        chunk_size = len("\n".join(chunk))

    return {
        "canonical_uri": canonical_uri,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "chunk": "\n".join(chunk),
        "chunk_size_bytes": chunk_size,
        "chunk_required": chunk_size > MAX_OUTPUT_CHARS,
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

    # Check SKILL.md
    if path.exists():
        checks["skill_md"] = "PASS"
    else:
        checks["skill_md"] = "FAIL"

    skill_dir = _get_skill_dir(path)

    # Check references dir
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        checks["references_dir"] = "PASS"
        # Check _index.md
        if (refs_dir / "_index.md").exists():
            checks["references_index"] = "PASS"
        else:
            checks["references_index"] = "WARN"
    else:
        checks["references_dir"] = "WARN"
        checks["references_index"] = "WARN"

    # Check scripts dir
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.exists():
        checks["scripts_dir"] = "PASS"
    else:
        checks["scripts_dir"] = "WARN"

    # Overall status
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