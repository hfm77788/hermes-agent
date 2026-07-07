"""
Tests for mcp_skill_tools -- Phase 1 read-only skill access.

Tests run in an isolated HERMES_HOME (set by conftest.py).
Tests that require actual skills must create them in the temp dir.
"""

import os
import sys
from pathlib import Path

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
    _validate_symlink_target,
    _safe_read_index,
    MAX_OUTPUT_CHARS,
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

    def test_health_check_without_apple_skill(self):
        """health_check must work without skills/apple/SKILL.md."""
        result = hermes_health_check()
        # Should not crash or raise
        assert result["mcp_ok"] is True
        # skills_root_exists should be accurate
        skills_root = _get_skills_root()
        if skills_root and skills_root.exists():
            assert result["skills_root_exists"] is True


class TestResolveSkillUri:
    def test_nonexistent_skill(self):
        result = resolve_skill_uri("nonexistent-skill-xyz-123")
        assert result["exists"] is False

    def test_returns_candidates_field(self):
        result = resolve_skill_uri("nonexistent-skill-xyz-123")
        assert "candidates" in result
        assert isinstance(result["candidates"], list)

    def test_skill_in_isolated_env(self):
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "test-skill"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            (test_skill_dir / "SKILL.md").write_text("---\nname: test-skill\n---\n\n# Test Skill\n")
            result = resolve_skill_uri("test-skill")
            assert result["exists"] is True
            assert result["confidence"] in ("exact", "single_match")

    def test_absolute_skill_name_denied(self):
        result = resolve_skill_uri("/etc/passwd")
        assert result.get("error_code", "").startswith("forbidden_path_denied")

    def test_traversal_skill_name_denied(self):
        result = resolve_skill_uri("../etc/passwd")
        assert result.get("error_code", "").startswith("forbidden_path_denied")


class TestReadSkillBundle:
    def test_nonexistent_skill(self):
        result = read_skill_bundle("nonexistent-skill-xyz-123")
        assert result.get("error_code") is not None or result.get("exists") is False

    def test_skill_in_isolated_env(self):
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

    def test_large_bundle_partial_signal(self):
        """Large SKILL.md must return full_content_available=false."""
        skills_root = _get_skills_root()
        if skills_root:
            test_dir = skills_root / "large-bundle-test"
            test_dir.mkdir(parents=True, exist_ok=True)
            # Create a file larger than MAX_OUTPUT_CHARS
            large_content = "---\nname: large-test\n---\n\n" + ("x" * (MAX_OUTPUT_CHARS + 100))
            (test_dir / "SKILL.md").write_text(large_content)
            result = read_skill_bundle("large-bundle-test")
            assert result.get("full_content_available") is False
            assert result.get("chunk_required") is True
            assert result.get("complete_instruction_loaded") is False
            assert result.get("suggested_ranges") is not None


class TestReadSkillFileChunked:
    def test_chunked_read_isolated(self):
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "chunked-test"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            content = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
            (test_skill_dir / "SKILL.md").write_text(content)
            result = read_skill_file_chunked("skill:chunked-test", start_line=1, end_line=5)
            assert result.get("start_line") == 1
            assert result.get("end_line") == 5
            assert len(result.get("chunk", "")) > 0

    def test_chunked_read_default_end(self):
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "chunked-test2"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            (test_skill_dir / "SKILL.md").write_text("line1\nline2\n")
            result = read_skill_file_chunked("skill:chunked-test2", start_line=1)
            assert result.get("start_line") == 1
            assert len(result.get("chunk", "")) > 0

    def test_truncation_signal_preserved(self):
        """When chunk exceeds MAX_OUTPUT_CHARS, truncated=true and chunk_required=true."""
        skills_root = _get_skills_root()
        if skills_root:
            test_dir = skills_root / "truncation-test"
            test_dir.mkdir(parents=True, exist_ok=True)
            # Create a file with many long lines
            huge = "\n".join(["line" + str(i) + "x" * 500 for i in range(100)])
            (test_dir / "SKILL.md").write_text(huge)
            result = read_skill_file_chunked("skill:truncation-test", start_line=1, end_line=100)
            if result.get("truncated"):
                assert result["chunk_required"] is True
                assert result["truncated"] is True
                assert "returned_lines" in result
                assert "requested_lines" in result
                assert "next_start_line" in result


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


class TestSymlinkValidation:
    def test_symlink_to_forbidden_denied(self):
        """Symlink pointing outside skills_root must be denied."""
        skills_root = _get_skills_root()
        if skills_root:
            test_dir = skills_root / "symlink-test"
            test_dir.mkdir(parents=True, exist_ok=True)
            (test_dir / "SKILL.md").write_text("---\nname: symlink-test\n---\n\n# Test\n")
            refs_dir = test_dir / "references"
            refs_dir.mkdir(exist_ok=True)
            # Create a symlink to /etc/passwd
            symlink_path = refs_dir / "_index.md"
            if not symlink_path.exists():
                try:
                    os.symlink("/etc/passwd", str(symlink_path))
                except OSError:
                    pass  # May not have permission
            if symlink_path.exists() and symlink_path.is_symlink():
                allowed, reason = _validate_symlink_target(symlink_path, skills_root)
                assert not allowed, f"Symlink to /etc/passwd should be denied, got: {reason}"
            # Cleanup
            if symlink_path.exists():
                symlink_path.unlink()


class TestSmokeSkillAccess:
    def test_nonexistent_skill_fails(self):
        result = smoke_skill_access("nonexistent-skill-xyz-123")
        assert result["overall"] == "FAIL"

    def test_skill_in_isolated_env(self):
        skills_root = _get_skills_root()
        if skills_root:
            test_skill_dir = skills_root / "smoke-test"
            test_skill_dir.mkdir(parents=True, exist_ok=True)
            (test_skill_dir / "SKILL.md").write_text("---\nname: smoke-test\n---\n\n# Smoke\n")
            (test_skill_dir / "references").mkdir(exist_ok=True)
            (test_skill_dir / "references" / "_index.md").write_text("# Index\n")
            result = smoke_skill_access("smoke-test")
            assert result["overall"] in ("PASS", "WARN")
            assert result["checks"]["skill_md"] == "PASS"


class TestHermesHomeResolver:
    def test_resolver_uses_hermes_home_env(self):
        """get_hermes_home should respect HERMES_HOME env var."""
        home = _get_hermes_home()
        assert home is not None

    def test_skills_root_from_hermes_home(self):
        skills_root = _get_skills_root()
        home = _get_hermes_home()
        if home and skills_root:
            assert str(skills_root).startswith(str(home))