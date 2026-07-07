"""
Tests for mcp_skill_tools — Phase 1 read-only skill access.

Tests run in an isolated HERMES_HOME (set by conftest.py).
Tests that require actual skills must create them in the temp dir.
"""

import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mcp_skill_tools import (
    hermes_health_check,
    resolve_skill_uri,
    read_skill_bundle,
    read_skill_file_chunked,
    smoke_skill_access,
    _is_forbidden_path,
    _get_skills_root,
    _get_hermes_home,
)


class TestHermesHealthCheck:
    def test_health_check_returns_mcp_ok(self):
        result = hermes_health_check()
        assert result["mcp_ok"] is True
        assert "version" in result
        assert "timestamp" in result

    def test_health_check_skills_root(self):
        result = hermes_health_check()
        assert "skills_root_exists" in result
        assert "skills_root_readable" in result


class TestResolveSkillUri:
    def test_nonexistent_skill(self):
        result = resolve_skill_uri("nonexistent-skill-xyz-123")
        assert result["exists"] is False
        assert result.get("error_code") == "skill_uri_not_found" or result["confidence"] == "none"

    def test_returns_candidates_field(self):
        result = resolve_skill_uri("nonexistent-skill-xyz-123")
        assert "candidates" in result
        assert isinstance(result["candidates"], list)

    def test_skill_in_isolated_env(self):
        """Create a skill in the isolated HERMES_HOME and resolve it."""
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "test-skill"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            (test_skill_dir / "SKILL.md").write_text("---\nname: test-skill\n---\n\n# Test Skill\n")
            result = resolve_skill_uri("test-skill")
            assert result["exists"] is True
            assert result["confidence"] in ("exact", "single_match")


class TestReadSkillBundle:
    def test_nonexistent_skill(self):
        result = read_skill_bundle("nonexistent-skill-xyz-123")
        assert result.get("error_code") is not None or result.get("exists") is False

    def test_skill_in_isolated_env(self):
        """Create a skill in isolated env and read its bundle."""
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "bundle-test"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            (test_skill_dir / "SKILL.md").write_text(
                "---\nname: bundle-test\nversion: 1.0.0\n---\n\n# Bundle Test\n\nReference content.\n"
            )
            result = read_skill_bundle("bundle-test")
            assert result.get("line_count", 0) > 0
            assert "frontmatter" in result
            assert result["frontmatter"].get("name") == "bundle-test"


class TestReadSkillFileChunked:
    def test_chunked_read_isolated(self):
        """Create a skill and read it chunked."""
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "chunked-test"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            content = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
            (test_skill_dir / "SKILL.md").write_text(content)
            result = read_skill_file_chunked(
                "skill:chunked-test", start_line=1, end_line=5
            )
            assert result.get("start_line") == 1
            assert result.get("end_line") == 5
            assert len(result.get("chunk", "")) > 0

    def test_chunked_read_default_end(self):
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "chunked-test2"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            (test_skill_dir / "SKILL.md").write_text("line1\nline2\n")
            result = read_skill_file_chunked(
                "skill:chunked-test2", start_line=1
            )
            assert result.get("start_line") == 1
            assert len(result.get("chunk", "")) > 0


class TestForbiddenPath:
    def test_env_is_forbidden(self):
        is_forbidden, reason = _is_forbidden_path(Path("/home/ubuntu/.env"))
        assert is_forbidden

    def test_ssh_is_forbidden(self):
        is_forbidden, reason = _is_forbidden_path(Path("/home/ubuntu/.ssh/id_rsa"))
        assert is_forbidden

    def test_token_pattern_is_forbidden(self):
        is_forbidden, reason = _is_forbidden_path(Path("/tmp/some_token_file"))
        assert is_forbidden

    def test_secret_in_path_denied(self):
        is_forbidden, reason = _is_forbidden_path(Path("/etc/secrets/mysecret"))
        assert is_forbidden

    def test_cookie_in_path_denied(self):
        is_forbidden, reason = _is_forbidden_path(Path("/tmp/cookies.txt"))
        assert is_forbidden

    def test_allowed_path_is_not_forbidden(self):
        is_forbidden, reason = _is_forbidden_path(
            Path("/home/ubuntu/.hermes/skills/test/SKILL.md")
        )
        assert not is_forbidden


class TestSmokeSkillAccess:
    def test_nonexistent_skill_fails(self):
        result = smoke_skill_access("nonexistent-skill-xyz-123")
        assert result["overall"] == "FAIL"

    def test_skill_in_isolated_env(self):
        """Create a skill and smoke-check it."""
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "smoke-test"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            (test_skill_dir / "SKILL.md").write_text("---\nname: smoke-test\n---\n\n# Smoke\n")
            # Create references dir
            (test_skill_dir / "references").mkdir(exist_ok=True)
            (test_skill_dir / "references" / "_index.md").write_text("# Index\n")
            result = smoke_skill_access("smoke-test")
            assert result["overall"] in ("PASS", "WARN")
            assert result["checks"]["skill_md"] == "PASS"