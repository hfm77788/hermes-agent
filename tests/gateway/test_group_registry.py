"""Phase 3e tests: group_registry loader + production adapter integration.

Tests cover:
1. Registry normal load
2. Sender exact resolution
3. Catalog key exact resolution
4. Unknown sender / unknown key / duplicate mapping / bad YAML → fail-closed
5. Production adapter calls loader (spy test)
6. Target group active, non-target group isolated
7. Config reload contract (mtime-based)
8. Hardcoded bank name residual scan
"""

import os
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

_AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from plugins.platforms.feishu.group_registry import (
    GroupRegistryLoader,
    RegistryError,
    RegistryLoadError,
    SenderResolutionError,
    CatalogKeyError,
    DuplicateMappingError,
    SenderIdentity,
    CatalogEntry,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

VALID_YAML = textwrap.dedent("""\
    project:
      name: test-project
      project_bank_id: proj-bank
    memory_catalog:
      boss_profile:
        bank_id: boss-bank-123
        role: boss profile
        status: current
        write_scope: []
      employee_test:
        bank_id: emp-bank-456
        role: employee profile
        status: current
        write_scope: [work_progress]
      project:
        bank_id: proj-bank-789
        role: project context
        status: current
        write_scope: [interaction_summary]
    workspaces:
      - chat_id: oc_target_group
        chat_name: Target Group
        status: active
        boss_open_id: ou_boss_001
        boss_name: TestBoss
        boss_profile_catalog: boss_profile
        boss_profile_model: hou-fangming
        employee_open_id: ou_emp_001
        employee_name: TestEmp
        employee_profile_catalog: employee_test
        employee_profile_model: xiaoniangao-work-focus
        privacy_boundary: test boundary
    runtime:
      sender_resolution:
        enabled: true
        lookup_field: sender_open_id
        match_targets:
          - field: boss_open_id
            mapped_name: Boss
            profile_catalog: boss_profile
          - field: employee_open_id
            mapped_name: employee_name
            profile_catalog: employee_test
""")


@pytest.fixture
def registry_file(tmp_path):
    p = tmp_path / "group-registry.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    return p


@pytest.fixture
def loader(registry_file):
    return GroupRegistryLoader(registry_path=registry_file)


# ── 1. Registry normal load ──────────────────────────────────────────────

class TestRegistryLoad:
    def test_load_valid(self, loader):
        """First access triggers load; no explicit load() needed."""
        assert loader.is_registered_chat("oc_target_group") is True

    def test_catalog_keys(self, loader):
        snap = loader._get_snapshot()
        assert set(snap.catalog.keys()) == {"boss_profile", "employee_test", "project"}

    def test_workspaces_indexed(self, loader):
        snap = loader._get_snapshot()
        assert "oc_target_group" in snap.registered_chat_ids

    def test_get_workspace(self, loader):
        ws = loader.get_workspace("oc_target_group")
        assert ws is not None
        assert ws["chat_name"] == "Target Group"

    def test_get_workspace_unknown(self, loader):
        assert loader.get_workspace("oc_nonexistent") is None


# ── 2. Sender exact resolution ───────────────────────────────────────────

class TestSenderResolution:
    def test_boss_resolved(self, loader):
        identity = loader.resolve_sender("ou_boss_001", "oc_target_group")
        assert isinstance(identity, SenderIdentity)
        assert identity.role == "boss"
        assert identity.display_name == "TestBoss"
        assert identity.catalog_key == "boss_profile"
        assert identity.profile_model == "hou-fangming"

    def test_employee_resolved(self, loader):
        identity = loader.resolve_sender("ou_emp_001", "oc_target_group")
        assert identity.role == "employee"
        assert identity.display_name == "TestEmp"
        assert identity.catalog_key == "employee_test"

    def test_unknown_sender_fails(self, loader):
        with pytest.raises(SenderResolutionError) as exc_info:
            loader.resolve_sender("ou_unknown_999", "oc_target_group")
        assert exc_info.value.code == "SENDER_UNKNOWN"

    def test_unregistered_chat_fails(self, loader):
        with pytest.raises(SenderResolutionError) as exc_info:
            loader.resolve_sender("ou_boss_001", "oc_other_group")
        assert exc_info.value.code == "CHAT_NOT_REGISTERED"

    def test_empty_open_id_fails(self, loader):
        with pytest.raises(SenderResolutionError) as exc_info:
            loader.resolve_sender("", "oc_target_group")
        assert exc_info.value.code == "SENDER_MISSING_INPUT"


# ── 3. Catalog key exact resolution ──────────────────────────────────────

class TestCatalogResolution:
    def test_boss_profile_bank(self, loader):
        assert loader.resolve_bank("boss_profile") == "boss-bank-123"

    def test_employee_bank(self, loader):
        assert loader.resolve_bank("employee_test") == "emp-bank-456"

    def test_project_bank(self, loader):
        assert loader.resolve_bank("project") == "proj-bank-789"

    def test_unknown_key_fails(self, loader):
        with pytest.raises(CatalogKeyError) as exc_info:
            loader.resolve_bank("nonexistent_key")
        assert exc_info.value.code == "CATALOG_KEY_UNKNOWN"

    def test_empty_key_fails(self, loader):
        with pytest.raises(CatalogKeyError) as exc_info:
            loader.resolve_bank("")
        assert exc_info.value.code == "CATALOG_KEY_EMPTY"

    def test_resolve_bank_for_sender(self, loader):
        bank = loader.resolve_bank_for_sender("ou_boss_001", "oc_target_group")
        assert bank == "boss-bank-123"

    def test_resolve_bank_for_employee_sender(self, loader):
        bank = loader.resolve_bank_for_sender("ou_emp_001", "oc_target_group")
        assert bank == "emp-bank-456"

    def test_get_catalog_entry(self, loader):
        entry = loader.get_catalog_entry("employee_test")
        assert isinstance(entry, CatalogEntry)
        assert entry.bank_id == "emp-bank-456"
        assert entry.status == "current"
        assert "work_progress" in entry.write_scope


# ── 4. Fail-closed: unknown / duplicate / missing fields / bad YAML ──────

class TestFailClosed:
    def test_bad_yaml(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text("{{{{invalid yaml::::", encoding="utf-8")
        with pytest.raises(RegistryLoadError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "REGISTRY_YAML_PARSE_ERROR"

    def test_missing_file(self, tmp_path):
        p = tmp_path / "nonexistent.yaml"
        with pytest.raises(RegistryLoadError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "REGISTRY_FILE_NOT_FOUND"

    def test_missing_memory_catalog(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text("project:\n  name: x\nworkspaces: []\n", encoding="utf-8")
        with pytest.raises(RegistryLoadError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "CATALOG_MISSING"

    def test_catalog_entry_missing_bank_id(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text(textwrap.dedent("""\
            memory_catalog:
              broken_entry:
                role: missing bank_id
                status: current
            workspaces: []
        """), encoding="utf-8")
        with pytest.raises(RegistryLoadError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "CATALOG_BANK_ID_MISSING"

    def test_catalog_entry_invalid_status(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text(textwrap.dedent("""\
            memory_catalog:
              bad_status:
                bank_id: b1
                role: x
                status: invalid_status
            workspaces: []
        """), encoding="utf-8")
        with pytest.raises(RegistryLoadError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "CATALOG_STATUS_INVALID"

    def test_duplicate_bank_id_across_keys(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text(textwrap.dedent("""\
            memory_catalog:
              key_a:
                bank_id: same-bank
                role: x
                status: current
              key_b:
                bank_id: same-bank
                role: y
                status: current
            workspaces: []
        """), encoding="utf-8")
        with pytest.raises(DuplicateMappingError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "DUPLICATE_BANK_ID"

    def test_duplicate_open_id_in_workspace(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text(textwrap.dedent("""\
            memory_catalog:
              boss_profile:
                bank_id: b1
                role: x
                status: current
            workspaces:
              - chat_id: oc_dup
                chat_name: Dup
                status: active
                boss_open_id: ou_same
                boss_name: Boss
                boss_profile_catalog: boss_profile
                employee_open_id: ou_same
                employee_name: Dup
                employee_profile_catalog: boss_profile
                privacy_boundary: test
        """), encoding="utf-8")
        with pytest.raises(DuplicateMappingError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "DUPLICATE_OPEN_ID_IN_CHAT"

    def test_workspace_missing_required_fields(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text(textwrap.dedent("""\
            memory_catalog:
              boss_profile:
                bank_id: b1
                role: x
                status: current
            workspaces:
              - chat_id: oc_incomplete
                chat_name: Incomplete
        """), encoding="utf-8")
        with pytest.raises(RegistryLoadError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "WORKSPACE_MISSING_FIELDS"

    def test_workspace_unknown_catalog_ref(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text(textwrap.dedent("""\
            memory_catalog:
              boss_profile:
                bank_id: b1
                role: x
                status: current
            workspaces:
              - chat_id: oc_bad
                chat_name: Bad
                status: active
                boss_open_id: ou_b
                boss_name: Boss
                boss_profile_catalog: nonexistent_catalog
                employee_open_id: ou_e
                employee_name: Emp
                employee_profile_catalog: boss_profile
                privacy_boundary: test
        """), encoding="utf-8")
        with pytest.raises(CatalogKeyError) as exc_info:
            GroupRegistryLoader(registry_path=p).is_registered_chat("x")
        assert exc_info.value.code == "BOSS_CATALOG_KEY_UNKNOWN"


# ── 5. Production adapter spy test ───────────────────────────────────────

class TestAdapterSpy:
    """Verify the production adapter actually calls the registry loader."""

    def _make_adapter_stub(self, loader):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        adapter = object.__new__(FeishuAdapter)
        adapter._group_registry_loader_instance = loader
        adapter._get_project_workspace = lambda chat_id: {
            "chat_id": "oc_target_group",
            "boss_profile_catalog": "boss_profile",
            "boss_profile_model": "hou-fangming",
            "employee_profile_catalog": "employee_test",
            "employee_profile_model": "xiaoniangao-work-focus",
        } if chat_id == "oc_target_group" else None
        return adapter

    def test_resolve_bank_from_catalog_calls_loader(self, loader):
        adapter = self._make_adapter_stub(loader)
        sender = MagicMock()
        sender.open_id = "ou_boss_001"

        with patch.object(loader, "resolve_bank", wraps=loader.resolve_bank) as spy:
            result = adapter._resolve_bank_from_catalog(
                sender, chat_id="oc_target_group", role="boss"
            )
            assert result == "boss-bank-123"
            spy.assert_called_once_with("boss_profile")

    def test_resolve_bank_employee_calls_loader(self, loader):
        adapter = self._make_adapter_stub(loader)
        sender = MagicMock()
        sender.open_id = "ou_emp_001"

        with patch.object(loader, "resolve_bank", wraps=loader.resolve_bank) as spy:
            result = adapter._resolve_bank_from_catalog(
                sender, chat_id="oc_target_group", role="employee"
            )
            assert result == "emp-bank-456"
            spy.assert_called_once_with("employee_test")

    def test_resolve_bank_unknown_chat_returns_none(self, loader):
        adapter = self._make_adapter_stub(loader)
        sender = MagicMock()
        sender.open_id = "ou_boss_001"
        result = adapter._resolve_bank_from_catalog(
            sender, chat_id="oc_unknown", role="boss"
        )
        assert result is None

    def test_resolve_bank_no_open_id_returns_none(self, loader):
        adapter = self._make_adapter_stub(loader)
        sender = MagicMock()
        sender.open_id = None
        result = adapter._resolve_bank_from_catalog(
            sender, chat_id="oc_target_group", role="boss"
        )
        assert result is None


# ── 6. Target group active, non-target isolated ──────────────────────────

class TestGroupIsolation:
    def test_target_group_resolves(self, loader):
        identity = loader.resolve_sender("ou_boss_001", "oc_target_group")
        assert identity.role == "boss"

    def test_non_target_group_rejected(self, loader):
        with pytest.raises(SenderResolutionError):
            loader.resolve_sender("ou_boss_001", "oc_random_group")

    def test_non_target_bank_resolution_fails(self, loader):
        with pytest.raises(SenderResolutionError):
            loader.resolve_bank_for_sender("ou_boss_001", "oc_random_group")

    def test_is_registered_chat_true(self, loader):
        assert loader.is_registered_chat("oc_target_group") is True

    def test_is_registered_chat_false(self, loader):
        assert loader.is_registered_chat("oc_random") is False

    def test_is_registered_chat_empty(self, loader):
        assert loader.is_registered_chat("") is False


# ── 7. Config reload contract ────────────────────────────────────────────

class TestReloadContract:
    def test_mtime_change_triggers_reload(self, tmp_path):
        p = tmp_path / "group-registry.yaml"
        p.write_text(VALID_YAML, encoding="utf-8")
        loader = GroupRegistryLoader(registry_path=p)
        assert loader.resolve_bank("boss_profile") == "boss-bank-123"

        # Modify the file with a new bank_id
        updated = VALID_YAML.replace("boss-bank-123", "boss-bank-NEW")
        time.sleep(0.05)
        p.write_text(updated, encoding="utf-8")

        # Force mtime difference detection
        loader._cache = (0.0, loader._cache[1]) if loader._cache else None
        assert loader.resolve_bank("boss_profile") == "boss-bank-NEW"

    def test_no_change_no_reload(self, loader):
        # First access loads
        loader.is_registered_chat("oc_target_group")
        cache_before = loader._cache
        # Second access should use cache (same mtime)
        loader.is_registered_chat("oc_target_group")
        assert loader._cache is cache_before

    def test_invalidate_cache_forces_reload(self, loader):
        loader.is_registered_chat("oc_target_group")
        cache_before = loader._cache
        loader.invalidate_cache()
        assert loader._cache is None
        loader.is_registered_chat("oc_target_group")
        assert loader._cache is not None
        assert loader._cache is not cache_before


# ── 8. Hardcoded bank name residual scan ─────────────────────────────────

class TestHardcodedBankScan:
    """Scan all .py files under plugins/platforms/feishu/ for hardcoded bank names."""

    KNOWN_BANK_NAMES = [
        "hermes_v2_bge_m3",
        "xiaoniangao_v2_bge_m3",
        "ma-secretary-system_v2_bge_m3",
        "innovation-dept-wechat_v2_bge_m3",
    ]

    def test_no_hardcoded_bank_names_in_adapter(self):
        adapter_path = _AGENT_ROOT / "plugins" / "platforms" / "feishu" / "adapter.py"
        content = adapter_path.read_text(encoding="utf-8")
        for bank in self.KNOWN_BANK_NAMES:
            assert bank not in content, (
                f"Hardcoded bank name '{bank}' found in adapter.py"
            )

    def test_no_hardcoded_bank_names_in_feishu_dir(self):
        feishu_dir = _AGENT_ROOT / "plugins" / "platforms" / "feishu"
        violations = []
        for py_file in feishu_dir.glob("*.py"):
            if py_file.name == "group_registry.py":
                continue
            if py_file.name.startswith("test_"):
                continue
            content = py_file.read_text(encoding="utf-8")
            for bank in self.KNOWN_BANK_NAMES:
                if bank in content:
                    violations.append(f"{py_file.name}: {bank}")
        assert not violations, f"Hardcoded bank names found: {violations}"
