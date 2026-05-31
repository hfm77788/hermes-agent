"""Smoke tests for inbound material-handling detection — plain text / GitHub URLs must not
trigger the material-ingestion path; only real attachments or explicit ingest intent do."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.wecom import WeComAdapter
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._session_key_for_source = lambda source: "session-1"
    runner._consume_pending_native_image_paths = lambda session_key: None
    return runner


def _make_source():
    return SessionSource(
        platform=Platform.WECOM,
        chat_id="group-1",
        chat_type="group",
        user_id="user-1",
    )


class TestShouldAddMaterialContext:
    def test_captionless_document_gets_material_context(self):
        """Captionless document attachment -> True, even when event.text is empty."""
        event = MessageEvent(
            text="",
            message_type=MessageType.DOCUMENT,
            media_urls=["https://example.com/file.pdf"],
        )

        assert gateway_run._should_add_material_context(event, "") is True

    def test_document_with_github_url_caption_still_gets_material_context(self):
        """Document + GitHub caption -> True because document takes priority."""
        event = MessageEvent(
            text="Please review https://github.com/user/repo/pull/123",
            message_type=MessageType.DOCUMENT,
            media_urls=["https://example.com/doc.md"],
        )

        assert gateway_run._should_add_material_context(event, event.text) is True

    def test_plain_github_url_no_attachment_returns_false(self):
        """Plain GitHub URL with no attachment -> False (exemption applies)."""
        event = MessageEvent(
            text="see https://github.com/user/repo/pull/123",
            message_type=MessageType.TEXT,
            media_urls=[],
        )

        assert gateway_run._should_add_material_context(event, event.text) is False

    def test_empty_text_no_media_returns_false(self):
        """Empty text, no media -> False."""
        event = MessageEvent(
            text="",
            message_type=MessageType.TEXT,
            media_urls=[],
        )

        assert gateway_run._should_add_material_context(event, "") is False


# ----------------------------------------------------------------------
# Helper: assert plain text passes through unchanged (no material context)
# ----------------------------------------------------------------------
async def _assert_pure_text(text: str) -> None:
    runner = _make_runner()
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m-generic",
    )
    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )
    assert result == text, f"Expected raw text back, got material context prepended"


# ----------------------------------------------------------------------
# Plain-text commands must NOT trigger material-ingestion context
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plain_text_hostname_does_not_get_material_context():
    await _assert_pure_text("hostname")


@pytest.mark.asyncio
async def test_plain_text_arbitrary_command_does_not_get_material_context():
    await _assert_pure_text("请帮我查一下今天的日期")


@pytest.mark.asyncio
async def test_plain_text_single_char_does_not_get_material_context():
    await _assert_pure_text("?")


# ----------------------------------------------------------------------
# GitHub resource URLs must NOT trigger material-ingestion context
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_github_actions_jobs_url_does_not_get_material_context():
    await _assert_pure_text(
        "https://github.com/example/repo/actions/runs/123456789/jobs/987654321"
    )


@pytest.mark.asyncio
async def test_github_pr_url_does_not_get_material_context():
    await _assert_pure_text("https://github.com/example/repo/pull/42")


@pytest.mark.asyncio
async def test_github_issue_url_does_not_get_material_context():
    await _assert_pure_text("https://github.com/example/repo/issues/99")


@pytest.mark.asyncio
async def test_github_commit_url_does_not_get_material_context():
    await _assert_pure_text(
        "https://github.com/example/repo/commit/abc123def456"
    )


@pytest.mark.asyncio
async def test_github_tree_url_does_not_get_material_context():
    await _assert_pure_text(
        "https://github.com/example/repo/tree/main/src"
    )


@pytest.mark.asyncio
async def test_github_blob_url_does_not_get_material_context():
    await _assert_pure_text(
        "https://github.com/example/repo/blob/main/README.md"
    )


# ----------------------------------------------------------------------
# File attachments must STILL trigger material-ingestion context
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_file_attachment_still_gets_material_context(tmp_path):
    runner = _make_runner()
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    event = MessageEvent(
        text="please review",
        message_type=MessageType.DOCUMENT,
        source=_make_source(),
        message_id="m-attachment",
        media_urls=[str(file_path)],
        media_types=["application/pdf"],
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert "The user sent a document" in result
    assert "please review" in result


@pytest.mark.asyncio
async def test_text_file_attachment_gets_document_context(tmp_path):
    runner = _make_runner()
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes\n\nSome content here.")
    event = MessageEvent(
        text="here are my notes",
        message_type=MessageType.DOCUMENT,
        source=_make_source(),
        message_id="m-text-file",
        media_urls=[str(file_path)],
        media_types=["text/markdown"],
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert "The user sent a text document" in result


# ----------------------------------------------------------------------
# Explicit ingest intent phrase + GitHub URL => should trigger
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_explicit_archive_request_gets_material_context():
    runner = _make_runner()
    text = "把这个链接资料入库：https://github.com/example/repo/pull/42"
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m-explicit",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert "explicitly asked" in result
    assert text in result


@pytest.mark.asyncio
async def test_explicit_archive_word_gets_material_context():
    runner = _make_runner()
    text = "归档这份资料到知识库"
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m-archive",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert "explicitly asked" in result or "material" in result.lower()


# ----------------------------------------------------------------------
# Unit-level helpers
# ----------------------------------------------------------------------
def test_looks_like_material_ingest_intent():
    positives = [
        "把这个链接资料入库",
        "归档这份报告",
        "保存到知识库",
        "整理这个文件",
        "处理这份资料",
        "转 Markdown",
        "source 包",
    ]
    for text in positives:
        assert gateway_run._looks_like_material_ingest_intent(text), f"Expected True for: {text}"

    negatives = [
        "请执行 hostname",
        "帮我看看这个 PR",
        "https://github.com/example/repo/pull/42",
        "hello world",
        "",
    ]
    for text in negatives:
        assert not gateway_run._looks_like_material_ingest_intent(text), f"Expected False for: {text}"


def test_contains_github_resource_url():
    positives = [
        "https://github.com/example/repo/actions/runs/123/jobs/456",
        "https://github.com/example/repo/pull/42",
        "https://github.com/example/repo/issues/99",
        "https://github.com/example/repo/commit/abc123",
        "https://github.com/example/repo/tree/main/src",
        "https://github.com/example/repo/blob/main/README.md",
        "https://www.github.com/example/repo/jobs",
    ]
    for text in positives:
        assert gateway_run._contains_github_resource_url(text), f"Expected True for: {text}"

    negatives = [
        "https://github.com/example/repo",
        "https://github.com/example/repo/releases",
        "请执行 hostname",
        "归档这份资料",
        "",
    ]
    for text in negatives:
        assert not gateway_run._contains_github_resource_url(text), f"Expected False for: {text}"


def test_wecom_text_plain_does_not_derive_document():
    body = {"msgtype": "text"}
    assert WeComAdapter._derive_message_type(body, "hello", ["text/plain"]) == MessageType.TEXT
    assert WeComAdapter._derive_message_type(body, "hello", ["text/plain; charset=utf-8"]) == MessageType.TEXT


def test_wecom_real_document_mime_types_still_derive_document():
    body = {"msgtype": "text"}
    assert WeComAdapter._derive_message_type(body, "notes", ["text/markdown"]) == MessageType.DOCUMENT
    assert WeComAdapter._derive_message_type(body, "notes", ["application/pdf"]) == MessageType.DOCUMENT
