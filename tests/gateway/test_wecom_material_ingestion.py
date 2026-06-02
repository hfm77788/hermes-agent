"""Tests for WeCom material ingestion PR creation."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.we_com_material_ingestion import (
    WIKI_ROOT,
    batch_id_from_message,
    create_material_pr,
)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


class TestCreateMaterialPr:
    @patch("gateway.platforms.we_com_material_ingestion.subprocess.run")
    def test_git_checkout_failure(self, mock_run):
        mock_run.return_value = _completed(returncode=1, stderr="branch exists")

        result = create_material_pr(
            batch_id="abc123",
            topic_key="daily_work",
            topic_name="日常工作",
            sender="test_user",
            message_id="msg_001",
            file_count=1,
            confidence="HIGH",
            staging_path=Path("/tmp/fake/staging"),
        )

        assert result["success"] is False
        assert result["stage"] == "git_checkout"
        assert "git checkout failed" in result["error"]

    @patch("gateway.platforms.we_com_material_ingestion.subprocess.run")
    def test_gh_pr_create_success(self, mock_run):
        staging_path = Path("/tmp/fake/staging")

        def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
            assert cwd == WIKI_ROOT
            assert capture_output is True
            assert text is True
            assert timeout in {120, 180}

            if cmd[:3] == ["git", "checkout", "-b"]:
                return _completed()
            if cmd[:2] == ["git", "checkout"]:
                return _completed()
            if cmd == ["git", "add", "--", str(staging_path)]:
                return _completed()
            if cmd == ["git", "status", "--porcelain"]:
                return _completed(stdout="M  somefile.md\n")
            if cmd[:2] == ["git", "commit"]:
                return _completed(stdout="[branch abc123] commit")
            if cmd == ["git", "rev-parse", "HEAD"]:
                return _completed(stdout="abc123def456\n")
            if cmd == ["git", "push", "-u", "fork", "staging/materials/daily_work/abc123"]:
                return _completed()
            if cmd[:3] == ["gh", "pr", "create"]:
                return _completed()
            if cmd == ["gh", "pr", "view", "--json", "number,url,headRefOid"]:
                return _completed(
                    stdout='{"number": 42, "url": "https://github.com/hfm77788/raymond-wiki/pull/42", "headRefOid": "abc123def456"}'
                )
            if cmd == ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]:
                return _completed(stdout="somefile.md\n")
            raise AssertionError(f"unexpected command: {cmd}")

        mock_run.side_effect = fake_run

        result = create_material_pr(
            batch_id="abc123",
            topic_key="daily_work",
            topic_name="日常工作",
            sender="test_user",
            message_id="msg_001",
            file_count=1,
            confidence="HIGH",
            staging_path=staging_path,
        )

        assert result["success"] is True
        assert result["pr_number"] == 42
        assert result["pr_url"] == "https://github.com/hfm77788/raymond-wiki/pull/42"
        assert result["head_sha"] == "abc123def456"
        assert result["changed_files"] == ["somefile.md"]
        assert result["stage"] == "complete"

    @patch("gateway.platforms.we_com_material_ingestion.subprocess.run")
    def test_gh_pr_create_failure(self, mock_run):
        staging_path = Path("/tmp/fake/staging")

        def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
            if cmd[:3] == ["git", "checkout", "-b"]:
                return _completed()
            if cmd[:2] == ["git", "checkout"]:
                return _completed()
            if cmd == ["git", "add", "--", str(staging_path)]:
                return _completed()
            if cmd == ["git", "status", "--porcelain"]:
                return _completed(stdout="M  somefile.md\n")
            if cmd[:2] == ["git", "commit"]:
                return _completed(stdout="[branch abc123] commit")
            if cmd == ["git", "rev-parse", "HEAD"]:
                return _completed(stdout="abc123def456\n")
            if cmd == ["git", "push", "-u", "fork", "staging/materials/daily_work/abc123"]:
                return _completed()
            if cmd[:3] == ["gh", "pr", "create"]:
                return _completed(returncode=1, stderr="gh: pull request create failed")
            raise AssertionError(f"unexpected command: {cmd}")

        mock_run.side_effect = fake_run

        result = create_material_pr(
            batch_id="abc123",
            topic_key="daily_work",
            topic_name="日常工作",
            sender="test_user",
            message_id="msg_001",
            file_count=1,
            confidence="HIGH",
            staging_path=staging_path,
        )

        assert result["success"] is False
        assert result["stage"] == "gh_pr_create"
        assert "gh pr create failed" in result["error"]


class TestBatchIdFromMessage:
    def test_deterministic(self):
        assert batch_id_from_message("msg_001", "group_abc") == batch_id_from_message(
            "msg_001", "group_abc"
        )

    def test_changes_when_inputs_change(self):
        assert batch_id_from_message("msg_001", "group_abc") != batch_id_from_message(
            "msg_002", "group_abc"
        )


class TestHandleMaterialIngestionBatchNotFound:
    """Test that batch directory not found triggers early return without PR creation."""

    @pytest.mark.asyncio
    async def test_batch_directory_not_found(self):
        """找不到 batch_id 时不会创建 PR。"""
        from gateway.platforms.wecom import WeComAdapter

        # Build a minimal gateway instance
        gw = WeComAdapter.__new__(WeComAdapter)
        gw._name = "test-wecom"
        gw._text_batch_delay_seconds = 0
        gw._ingestion_group_id = "test_group"
        gw._pending_text_batches = {}

        # Mock send to capture the early-return reply
        gw.send = AsyncMock()

        # Mock MessageEvent
        mock_event = MagicMock()
        mock_event.source = MagicMock()
        mock_event.source.chat_id = "test_chat"
        mock_event.source.user_id = "test_user"
        mock_event.message_id = "msg_001"
        mock_event.media_urls = ["file1.pdf"]

        # Patch Path.rglob so batch_dirs is always empty
        with patch("gateway.platforms.wecom.Path") as mock_path_cls:
            mock_path_cls.return_value.rglob.return_value = []

            # Also patch create_material_pr to ensure it's never called
            with patch(
                "gateway.platforms.wecom.create_material_pr"
            ) as mock_create:
                await gw._handle_material_ingestion(mock_event, batch_id="nonexistent_batch")

                # Verify create_material_pr was NOT called
                mock_create.assert_not_called()

                # Verify send was called with failure message
                gw.send.assert_called_once()
                args, kwargs = gw.send.call_args
                assert "资料录入失败" in kwargs["content"]
                assert "batch directory not found" in kwargs["content"]

    @pytest.mark.asyncio
    async def test_batch_directory_found_creates_pr(self):
        """batch_id 存在时正常创建 PR。"""
        from gateway.platforms.wecom import WeComAdapter

        gw = WeComAdapter.__new__(WeComAdapter)
        gw._name = "test-wecom"
        gw._text_batch_delay_seconds = 0
        gw._ingestion_group_id = "test_group"
        gw._pending_text_batches = {}

        gw.send = AsyncMock()

        mock_event = MagicMock()
        mock_event.source = MagicMock()
        mock_event.source.chat_id = "test_chat"
        mock_event.source.user_id = "test_user"
        mock_event.message_id = "msg_001"
        mock_event.media_urls = ["file1.pdf"]

        fake_batch_dir = MagicMock()
        fake_batch_dir.__str__ = lambda self: "/tmp/fake/batch_dir"
        fake_batch_dir.__fspath__ = lambda self: "/tmp/fake/batch_dir"

        with patch("gateway.platforms.wecom.Path") as mock_path_cls:
            mock_path_cls.return_value.rglob.return_value = [fake_batch_dir]

            with patch(
                "gateway.platforms.wecom.create_material_pr"
            ) as mock_create:
                mock_create.return_value = {
                    "success": True,
                    "pr_number": 42,
                    "pr_url": "https://github.com/hfm77788/raymond-wiki/pull/42",
                    "head_sha": "abc123def456",
                    "changed_files": ["file1.pdf"],
                    "error": None,
                    "stage": "complete",
                }

                await gw._handle_material_ingestion(mock_event, batch_id="existing_batch")

                mock_create.assert_called_once()
                gw.send.assert_called_once()
