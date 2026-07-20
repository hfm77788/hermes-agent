"""Regression coverage for profile-aware shared-group outbox state."""

from gateway.platforms.group_outbox import GroupOutboxMixin


class _OutboxHost(GroupOutboxMixin):
    pass


def test_group_outbox_uses_active_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    host = _OutboxHost()
    host._init_group_outbox()

    db_path = host._ensure_outbox_db()

    assert db_path == tmp_path / "data" / "group_session_outbox.db"
    assert db_path.exists()
