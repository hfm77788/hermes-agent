"""Regression tests for profile-aware Feishu group state paths."""

from plugins.platforms.feishu.adapter import FeishuAdapter
from plugins.platforms.feishu.group_registry import GroupRegistryLoader


def test_group_registry_default_path_uses_active_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    loader = GroupRegistryLoader()

    assert loader._path == (
        tmp_path
        / "projects"
        / "ma-secretary-interaction-system"
        / "group-registry.yaml"
    )


def test_group_buffer_uses_active_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_dir = tmp_path / "projects" / "ma-secretary-interaction-system"
    project_dir.mkdir(parents=True)
    (project_dir / "group-registry.yaml").write_text(
        """
workspaces:
  - chat_id: oc_profile_test
    context_buffer: true
""".lstrip(),
        encoding="utf-8",
    )

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._group_buffer_chat_ids = set()
    adapter._group_buffer_db_path = None
    adapter._init_group_context_buffer()

    expected_db = project_dir / "group_buffer.db"
    assert adapter._group_buffer_db_path == expected_db
    assert expected_db.exists()
    assert adapter._group_buffer_chat_ids == {"oc_profile_test"}
