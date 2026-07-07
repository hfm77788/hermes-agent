"""
Tests for mcp_skill_tools with Phase 2 write transaction model.

Phase 1 (25): health, resolve, bundle, chunked, forbidden, symlinks, smoke
Phase 2 (13+): unique backup, nested paths, add files, hidden deny,
               root-level restore, backup metadata
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mcp_skill_tools import (
    hermes_health_check, resolve_skill_uri, read_skill_bundle,
    read_skill_file_chunked, smoke_skill_access,
    _is_forbidden_path, _get_skills_root, _get_hermes_home,
    _validate_symlink_target, _safe_read_index,
    get_preauthorization_profile, run_preauthorized_skill_patch,
    rollback_skill_patch, _is_in_allowed_paths,
    _scan_new_content, _redact_diff,
)


class TestHermesHealthCheck:
    def test_returns_mcp_ok(self):
        r = hermes_health_check()
        assert r["mcp_ok"]
        assert "skills_root_exists" in r

    def test_without_apple_skill(self):
        r = hermes_health_check()
        assert r["mcp_ok"]


class TestResolveSkillUri:
    def test_nonexistent(self):
        assert resolve_skill_uri("nonexistent-xyz")["exists"] is False

    def test_absolute_denied(self):
        r = resolve_skill_uri("/etc/passwd")
        assert r.get("error_code", "").startswith("forbidden_path_denied")

    def test_traversal_denied(self):
        r = resolve_skill_uri("../etc/passwd")
        assert r.get("error_code", "").startswith("forbidden_path_denied")

    def test_isolated_env(self):
        sr = _get_skills_root()
        if sr:
            (sr / "t1" / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
            (sr / "t1" / "SKILL.md").write_text("---\nname: t1\n---\n")
            assert resolve_skill_uri("t1")["exists"]


class TestReadSkillBundle:
    def test_nonexistent(self):
        assert "error_code" in read_skill_bundle("nonexistent-xyz")

    def test_isolated_env(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "b1"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: b1\n---\n# Bundle\n")
            r = read_skill_bundle("b1")
            assert r.get("line_count", 0) > 0


class TestReadSkillFileChunked:
    def test_default_end(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "c1"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("a\nb\n")
            r = read_skill_file_chunked("skill:c1", 1)
            assert r.get("start_line") == 1


class TestForbiddenPath:
    def test_env(self): assert _is_forbidden_path(Path("/u/.env"))[0]
    def test_ssh(self): assert _is_forbidden_path(Path("/u/.ssh/x"))[0]
    def test_token(self): assert _is_forbidden_path(Path("/t/token_file"))[0]
    def test_secret(self): assert _is_forbidden_path(Path("/t/secret"))[0]
    def test_allowed(self): assert not _is_forbidden_path(Path("/u/.hermes/skills/x/SKILL.md"))[0]


class TestSymlinkValidation:
    def test_outside_denied(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "symt"
            d.mkdir(parents=True, exist_ok=True)
            refs = d / "references"
            refs.mkdir(exist_ok=True)
            sl = refs / "_index.md"
            if not sl.exists():
                try:
                    os.symlink("/etc/passwd", str(sl))
                except OSError:
                    return
            allowed, _ = _validate_symlink_target(sl, sr)
            assert not allowed
            sl.unlink()


class TestSmokeSkillAccess:
    def test_nonexistent_fails(self):
        assert smoke_skill_access("nonexistent-xyz")["overall"] == "FAIL"

    def test_isolated_env(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "sm1"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: sm1\n---\n")
            (d / "references").mkdir(exist_ok=True)
            (d / "references" / "_index.md").write_text("# Index\n")
            assert smoke_skill_access("sm1")["overall"] in ("PASS", "WARN")


# ---------------------------------------------------------------------------
# Phase 2: Write Transaction Model Tests
# ---------------------------------------------------------------------------

class TestAddNewFiles:
    """Thread 1: Allow patches to add new skill files."""

    def _setup_skill(self, name):
        sr = _get_skills_root()
        if not sr:
            return None
        d = sr / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n# Test\n")
        (d / "references").mkdir(exist_ok=True)
        return d

    def test_add_reference_file_and_rollback_removes_it(self):
        d = self._setup_skill("add-ref")
        if not d:
            return
        r = run_preauthorized_skill_patch({
            "skill_name": "add-ref", "file_path": "references/new.md",
            "new_content": "# New ref\n",
        })
        assert r.get("status") == "ok"
        assert r["backup_metadata"]["existed_before"] is False
        rb = rollback_skill_patch(r["backup_path"])
        assert not (d / "references" / "new.md").exists()

    def test_add_nested_reference_file_and_rollback_removes_it(self):
        d = self._setup_skill("add-nest-ref")
        if not d:
            return
        r = run_preauthorized_skill_patch({
            "skill_name": "add-nest-ref", "file_path": "references/styles/t.md",
            "new_content": "# Nested\n",
        })
        assert r.get("status") == "ok"
        assert r["backup_metadata"]["existed_before"] is False
        rb = rollback_skill_patch(r["backup_path"])
        assert not (d / "references" / "styles" / "t.md").exists()

    def test_add_script_file_and_rollback_removes_it(self):
        d = self._setup_skill("add-script")
        if not d:
            return
        r = run_preauthorized_skill_patch({
            "skill_name": "add-script", "file_path": "scripts/new.py",
            "new_content": "print('new')\n",
        })
        assert r.get("status") == "ok"
        rb = rollback_skill_patch(r["backup_path"])
        assert not (d / "scripts" / "new.py").exists()


class TestHiddenPathDeny:
    """Thread 2: Reject hidden patch path segments."""

    def test_git_config(self):
        r = run_preauthorized_skill_patch({"skill_name": "x", "file_path": ".git/config", "new_content": "x"})
        assert "hidden" in r.get("error", "").lower()

    def test_hidden_reference(self):
        r = run_preauthorized_skill_patch({"skill_name": "x", "file_path": "references/.internal.md", "new_content": "x"})
        assert "hidden" in r.get("error", "").lower()

    def test_hidden_script(self):
        r = run_preauthorized_skill_patch({"skill_name": "x", "file_path": "scripts/.hidden.py", "new_content": "x"})
        assert "hidden" in r.get("error", "").lower()

    def test_normal_reference_allowed(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "hidden-normal"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: hn\n---\n")
            (d / "references").mkdir(exist_ok=True)
            r = run_preauthorized_skill_patch({"skill_name": "hidden-normal", "file_path": "references/internal.md", "new_content": "# OK\n"})
            assert r.get("status") == "ok"


class TestRootLevelRestore:
    """Thread 3: Restore root-level backup files."""

    def test_skilL_md_restored(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "root-sm"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: rsm\n---\n# V1\n")
            (d / "references").mkdir(exist_ok=True)
            r = run_preauthorized_skill_patch({"skill_name": "root-sm", "file_path": "SKILL.md", "new_content": "---\nname: rsm\n---\n# V2\n"})
            rb = rollback_skill_patch(r["backup_path"])
            assert rb["rollback"] == "ok"
            assert "V1" in (d / "SKILL.md").read_text()

    def test_readme_md_restored(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "root-readme"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: rre\n---\n")
            (d / "references").mkdir(exist_ok=True)
            (d / "README.md").write_text("# README v1\n")
            r = run_preauthorized_skill_patch({"skill_name": "root-readme", "file_path": "README.md", "new_content": "# README v2\n"})
            assert r.get("status") == "ok"
            rb = rollback_skill_patch(r["backup_path"])
            assert rb["rollback"] == "ok"
            assert "# README v1" in (d / "README.md").read_text()

    def test_nested_reference_preserved(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "root-nest-ref"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: rnr\n---\n")
            styles = d / "references" / "styles"
            styles.mkdir(parents=True, exist_ok=True)
            (styles / "demo.md").write_text("# Original\nOld content.\n")
            r = run_preauthorized_skill_patch({"skill_name": "root-nest-ref", "file_path": "references/styles/demo.md", "new_content": "# Modified\nNew.\n"})
            rb = rollback_skill_patch(r["backup_path"])
            assert rb["rollback"] == "ok"
            assert "Old content" in (styles / "demo.md").read_text()

    def test_nested_script_preserved(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "root-nest-script"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: rns\n---\n")
            office = d / "scripts" / "office"
            office.mkdir(parents=True, exist_ok=True)
            (office / "run.py").write_text("print('old')\n")
            r = run_preauthorized_skill_patch({"skill_name": "root-nest-script", "file_path": "scripts/office/run.py", "new_content": "print('new')\n"})
            rb = rollback_skill_patch(r["backup_path"])
            assert rb["rollback"] == "ok"
            assert "old" in (office / "run.py").read_text()

    def test_no_extra_root_level_files(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "no-extra"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: ne\n---\n")
            (d / "references").mkdir(exist_ok=True)
            styles = d / "references" / "styles"
            styles.mkdir(exist_ok=True)
            (styles / "demo.md").write_text("# Original\n")
            r = run_preauthorized_skill_patch({"skill_name": "no-extra", "file_path": "references/styles/demo.md", "new_content": "# Modified\n"})
            rb = rollback_skill_patch(r["backup_path"])
            assert not (d / "demo.md").exists()


class TestBackupMetadata:
    def test_records_existed_before_true(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "meta-ebt"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: mebt\n---\n")
            (d / "references").mkdir(exist_ok=True)
            r = run_preauthorized_skill_patch({"skill_name": "meta-ebt", "file_path": "SKILL.md", "new_content": "---\nname: mebt\n---\n# V2\n"})
            assert r["backup_metadata"]["existed_before"] is True

    def test_records_existed_before_false(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "meta-ebf"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: mebf\n---\n")
            (d / "references").mkdir(exist_ok=True)
            r = run_preauthorized_skill_patch({"skill_name": "meta-ebf", "file_path": "references/new.md", "new_content": "# New\n"})
            assert r["backup_metadata"]["existed_before"] is False

    def test_records_relative_path(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "meta-rp"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: mrp\n---\n")
            (d / "references").mkdir(exist_ok=True)
            for fp, expected in [
                ("SKILL.md", "SKILL.md"),
                ("README.md", "README.md"),
                ("references/demo.md", "references/demo.md"),
            ]:
                r = run_preauthorized_skill_patch({"skill_name": "meta-rp", "file_path": fp, "new_content": "# x\n"})
                assert r["backup_metadata"]["relative_file_path"] == expected, f"{fp} -> {r['backup_metadata']['relative_file_path']}"


class TestUniqueBackup:
    def test_two_patches_different_backups(self):
        sr = _get_skills_root()
        if sr:
            d = sr / "ubt"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text("---\nname: ubt\n---\n# V1\n")
            (d / "references").mkdir(exist_ok=True)
            r1 = run_preauthorized_skill_patch({"skill_name": "ubt", "file_path": "SKILL.md", "new_content": "---\nname: ubt\n---\n# V2\n"})
            r2 = run_preauthorized_skill_patch({"skill_name": "ubt", "file_path": "SKILL.md", "new_content": "---\nname: ubt\n---\n# V3\n"})
            assert r1["backup_path"] != r2["backup_path"]
            rb = rollback_skill_patch(r1["backup_path"])
            assert "V1" in (d / "SKILL.md").read_text()


class TestContentScanning:
    def test_pattern_id_no_leak(self):
        found, reason = _scan_new_content('api_key = "sk-abc"')
        assert found and "sk-abc" not in reason

    def test_redact_diff(self):
        redacted = _redact_diff("+\n+api_key = 'secret'\n+ok")
        assert "secret" not in redacted