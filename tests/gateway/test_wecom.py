"""Tests for the WeCom platform adapter."""

import asyncio
import base64
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


class TestWeComRequirements:
    def test_returns_false_without_aiohttp(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wecom.AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr("gateway.platforms.wecom.HTTPX_AVAILABLE", True)
        from gateway.platforms.wecom import check_wecom_requirements

        assert check_wecom_requirements() is False

    def test_returns_false_without_httpx(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wecom.AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.wecom.HTTPX_AVAILABLE", False)
        from gateway.platforms.wecom import check_wecom_requirements

        assert check_wecom_requirements() is False

    def test_returns_true_when_available(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wecom.AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.wecom.HTTPX_AVAILABLE", True)
        from gateway.platforms.wecom import check_wecom_requirements

        assert check_wecom_requirements() is True


class TestWeComAdapterInit:
    def test_declares_non_editable_message_capability(self):
        from gateway.platforms.wecom import WeComAdapter

        assert WeComAdapter.SUPPORTS_MESSAGE_EDITING is False

    def test_reads_config_from_extra(self):
        from gateway.platforms.wecom import WeComAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "bot_id": "cfg-bot",
                "secret": "cfg-secret",
                "websocket_url": "wss://custom.wecom.example/ws",
                "group_policy": "allowlist",
                "group_allow_from": ["group-1"],
            },
        )
        adapter = WeComAdapter(config)

        assert adapter._bot_id == "cfg-bot"
        assert adapter._secret == "cfg-secret"
        assert adapter._ws_url == "wss://custom.wecom.example/ws"
        assert adapter._group_policy == "allowlist"
        assert adapter._group_allow_from == ["group-1"]

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("WECOM_BOT_ID", "env-bot")
        monkeypatch.setenv("WECOM_SECRET", "env-secret")
        monkeypatch.setenv("WECOM_WEBSOCKET_URL", "wss://env.example/ws")
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter._bot_id == "env-bot"
        assert adapter._secret == "env-secret"
        assert adapter._ws_url == "wss://env.example/ws"


class TestWeComConnect:
    @pytest.mark.asyncio
    async def test_connect_records_missing_credentials(self, monkeypatch):
        import gateway.platforms.wecom as wecom_module
        from gateway.platforms.wecom import WeComAdapter

        monkeypatch.setattr(wecom_module, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(wecom_module, "HTTPX_AVAILABLE", True)

        adapter = WeComAdapter(PlatformConfig(enabled=True))

        success = await adapter.connect()

        assert success is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "wecom_missing_credentials"
        assert "WECOM_BOT_ID" in (adapter.fatal_error_message or "")

    @pytest.mark.asyncio
    async def test_connect_records_handshake_failure_details(self, monkeypatch):
        import gateway.platforms.wecom as wecom_module
        from gateway.platforms.wecom import WeComAdapter

        class DummyClient:
            async def aclose(self):
                return None

        monkeypatch.setattr(wecom_module, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(wecom_module, "HTTPX_AVAILABLE", True)
        monkeypatch.setattr(
            wecom_module,
            "httpx",
            SimpleNamespace(AsyncClient=lambda **kwargs: DummyClient()),
        )

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"bot_id": "bot-1", "secret": "secret-1"})
        )
        adapter._open_connection = AsyncMock(side_effect=RuntimeError("invalid secret (errcode=40013)"))

        success = await adapter.connect()

        assert success is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "wecom_connect_error"
        assert "invalid secret" in (adapter.fatal_error_message or "")


class TestWeComQrScan:
    @patch("gateway.platforms.wecom.time")
    @patch("gateway.platforms.wecom.json.loads")
    @patch("gateway.platforms.wecom.logger")
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_qr_scan_timeout_uses_monotonic_clock(
        self,
        mock_request,
        mock_urlopen,
        _mock_logger,
        mock_json_loads,
        mock_time,
    ):
        from gateway.platforms.wecom import qr_scan_for_bot_info

        generate_resp = MagicMock()
        generate_resp.read.return_value = b'{"data":{"scode":"abc","auth_url":"https://example.com/qr"}}'
        generate_resp.__enter__.return_value = generate_resp
        generate_resp.__exit__.return_value = False

        poll_resp = MagicMock()
        poll_resp.read.return_value = b'{"data":{"status":"pending"}}'
        poll_resp.__enter__.return_value = poll_resp
        poll_resp.__exit__.return_value = False

        mock_urlopen.side_effect = [generate_resp, poll_resp]
        mock_json_loads.side_effect = [
            {"data": {"scode": "abc", "auth_url": "https://example.com/qr"}},
            {"data": {"status": "pending"}},
        ]
        mock_time.monotonic.side_effect = [1000, 1000.2, 1001.1]
        mock_time.time.side_effect = [1000, 900, 901, 902]
        mock_time.sleep = MagicMock()

        with patch("builtins.print"), patch.dict("sys.modules", {"qrcode": None}):
            result = qr_scan_for_bot_info(timeout_seconds=1)

        assert result is None
        assert mock_urlopen.call_count == 2


class TestWeComReplyMode:
    @pytest.mark.asyncio
    async def test_send_uses_passive_reply_markdown_when_reply_context_exists(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._reply_req_ids["msg-1"] = "req-1"
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-1"}, "errcode": 0}
        )

        result = await adapter.send("chat-123", "hello from reply", reply_to="msg-1")

        assert result.success is True
        adapter._send_reply_request.assert_awaited_once()
        args = adapter._send_reply_request.await_args.args
        assert args[0] == "req-1"
        # msgtype: stream triggers WeCom errcode 600039 on many mobile clients
        # (unsupported type). Markdown renders everywhere.
        assert args[1]["msgtype"] == "markdown"
        assert args[1]["markdown"]["content"] == "hello from reply"

    @pytest.mark.asyncio
    async def test_send_image_file_uses_passive_reply_media_when_reply_context_exists(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._reply_req_ids["msg-1"] = "req-1"
        adapter._prepare_outbound_media = AsyncMock(
            return_value={
                "data": b"image-bytes",
                "content_type": "image/png",
                "file_name": "demo.png",
                "detected_type": "image",
                "final_type": "image",
                "rejected": False,
                "reject_reason": None,
                "downgraded": False,
                "downgrade_note": None,
            }
        )
        adapter._upload_media_bytes = AsyncMock(return_value={"media_id": "media-1", "type": "image"})
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-1"}, "errcode": 0}
        )

        result = await adapter.send_image_file("chat-123", "/tmp/demo.png", reply_to="msg-1")

        assert result.success is True
        adapter._send_reply_request.assert_awaited_once()
        args = adapter._send_reply_request.await_args.args
        assert args[0] == "req-1"
        assert args[1] == {"msgtype": "image", "image": {"media_id": "media-1"}}


class TestExtractText:
    def test_extracts_plain_text(self):
        from gateway.platforms.wecom import WeComAdapter

        body = {
            "msgtype": "text",
            "text": {"content": "  hello world  "},
        }
        text, reply_text = WeComAdapter._extract_text(body)
        assert text == "hello world"
        assert reply_text is None

    def test_extracts_mixed_text(self):
        from gateway.platforms.wecom import WeComAdapter

        body = {
            "msgtype": "mixed",
            "mixed": {
                "msg_item": [
                    {"msgtype": "text", "text": {"content": "part1"}},
                    {"msgtype": "image", "image": {"url": "https://example.com/x.png"}},
                    {"msgtype": "text", "text": {"content": "part2"}},
                ]
            },
        }
        text, _reply_text = WeComAdapter._extract_text(body)
        assert text == "part1\npart2"

    def test_extracts_voice_and_quote(self):
        from gateway.platforms.wecom import WeComAdapter

        body = {
            "msgtype": "voice",
            "voice": {"content": "spoken text"},
            "quote": {"msgtype": "text", "text": {"content": "quoted"}},
        }
        text, reply_text = WeComAdapter._extract_text(body)
        assert text == "spoken text"
        assert reply_text == "quoted"


class TestCallbackDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cmd", ["aibot_msg_callback", "aibot_callback"])
    async def test_dispatch_accepts_new_and_legacy_callback_cmds(self, cmd):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._on_message = AsyncMock()

        await adapter._dispatch_payload({"cmd": cmd, "headers": {"req_id": "req-1"}, "body": {}})

        adapter._on_message.assert_awaited_once()


class TestPolicyHelpers:
    def test_dm_allowlist(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"dm_policy": "allowlist", "allow_from": ["user-1"]})
        )
        assert adapter._is_dm_allowed("user-1") is True
        assert adapter._is_dm_allowed("user-2") is False

    def test_group_allowlist_and_per_group_sender_allowlist(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "group_policy": "allowlist",
                    "group_allow_from": ["group-1"],
                    "groups": {"group-1": {"allow_from": ["user-1"]}},
                },
            )
        )

        assert adapter._is_group_allowed("group-1", "user-1") is True
        assert adapter._is_group_allowed("group-1", "user-2") is False
        assert adapter._is_group_allowed("group-2", "user-1") is False


class TestMediaHelpers:
    def test_detect_wecom_media_type(self):
        from gateway.platforms.wecom import WeComAdapter

        assert WeComAdapter._detect_wecom_media_type("image/png") == "image"
        assert WeComAdapter._detect_wecom_media_type("video/mp4") == "video"
        assert WeComAdapter._detect_wecom_media_type("audio/amr") == "voice"
        assert WeComAdapter._detect_wecom_media_type("application/pdf") == "file"

    def test_voice_non_amr_downgrades_to_file(self):
        from gateway.platforms.wecom import WeComAdapter

        result = WeComAdapter._apply_file_size_limits(128, "voice", "audio/mpeg")

        assert result["final_type"] == "file"
        assert result["downgraded"] is True
        assert "AMR" in (result["downgrade_note"] or "")

    def test_oversized_file_is_rejected(self):
        from gateway.platforms.wecom import ABSOLUTE_MAX_BYTES, WeComAdapter

        result = WeComAdapter._apply_file_size_limits(ABSOLUTE_MAX_BYTES + 1, "file", "application/pdf")

        assert result["rejected"] is True
        assert "20MB" in (result["reject_reason"] or "")

    def test_decrypt_file_bytes_round_trip(self):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from gateway.platforms.wecom import WeComAdapter

        plaintext = b"wecom-secret"
        key = os.urandom(32)
        pad_len = 32 - (len(plaintext) % 32)
        padded = plaintext + bytes([pad_len]) * pad_len
        encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()

        decrypted = WeComAdapter._decrypt_file_bytes(encrypted, base64.b64encode(key).decode("ascii"))

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_load_outbound_media_rejects_placeholder_path(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))

        with pytest.raises(ValueError, match="placeholder was not replaced"):
            await adapter._load_outbound_media("<path>")


class TestMediaUpload:
    @pytest.mark.asyncio
    async def test_upload_media_bytes_uses_sdk_sequence(self, monkeypatch):
        import gateway.platforms.wecom as wecom_module
        from gateway.platforms.wecom import (
            APP_CMD_UPLOAD_MEDIA_CHUNK,
            APP_CMD_UPLOAD_MEDIA_FINISH,
            APP_CMD_UPLOAD_MEDIA_INIT,
            WeComAdapter,
        )

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        calls = []

        async def fake_send_request(cmd, body, timeout=0):
            calls.append((cmd, body))
            if cmd == APP_CMD_UPLOAD_MEDIA_INIT:
                return {"errcode": 0, "body": {"upload_id": "upload-1"}}
            if cmd == APP_CMD_UPLOAD_MEDIA_CHUNK:
                return {"errcode": 0}
            if cmd == APP_CMD_UPLOAD_MEDIA_FINISH:
                return {
                    "errcode": 0,
                    "body": {
                        "media_id": "media-1",
                        "type": "file",
                        "created_at": "2026-03-18T00:00:00Z",
                    },
                }
            raise AssertionError(f"unexpected cmd {cmd}")

        monkeypatch.setattr(wecom_module, "UPLOAD_CHUNK_SIZE", 4)
        adapter._send_request = fake_send_request

        result = await adapter._upload_media_bytes(b"abcdefghij", "file", "demo.bin")

        assert result["media_id"] == "media-1"
        assert [cmd for cmd, _body in calls] == [
            APP_CMD_UPLOAD_MEDIA_INIT,
            APP_CMD_UPLOAD_MEDIA_CHUNK,
            APP_CMD_UPLOAD_MEDIA_CHUNK,
            APP_CMD_UPLOAD_MEDIA_CHUNK,
            APP_CMD_UPLOAD_MEDIA_FINISH,
        ]
        assert calls[1][1]["chunk_index"] == 0
        assert calls[2][1]["chunk_index"] == 1
        assert calls[3][1]["chunk_index"] == 2

    @pytest.mark.asyncio
    @patch("tools.url_safety.is_safe_url", return_value=True)
    async def test_download_remote_bytes_rejects_large_content_length(self, _mock_safe):
        from gateway.platforms.wecom import WeComAdapter

        class FakeResponse:
            headers = {"content-length": "10"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield b"abc"

        class FakeClient:
            def stream(self, method, url, headers=None):
                return FakeResponse()

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._http_client = FakeClient()

        with pytest.raises(ValueError, match="exceeds WeCom limit"):
            await adapter._download_remote_bytes("https://example.com/file.bin", max_bytes=4)

    @pytest.mark.asyncio
    async def test_cache_media_decrypts_url_payload_before_writing(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        plaintext = b"secret document bytes"
        key = os.urandom(32)
        pad_len = 32 - (len(plaintext) % 32)
        padded = plaintext + bytes([pad_len]) * pad_len

        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()
        adapter._download_remote_bytes = AsyncMock(
            return_value=(
                encrypted,
                {
                    "content-type": "application/octet-stream",
                    "content-disposition": 'attachment; filename="secret.bin"',
                },
            )
        )

        cached = await adapter._cache_media(
            "file",
            {
                "url": "https://example.com/secret.bin",
                "aeskey": base64.b64encode(key).decode("ascii"),
            },
        )

        assert cached is not None
        cached_path, content_type = cached
        assert Path(cached_path).read_bytes() == plaintext
        assert content_type == "application/octet-stream"


class TestSend:
    @pytest.mark.asyncio
    async def test_send_uses_proactive_payload(self):
        from gateway.platforms.wecom import APP_CMD_SEND, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_request = AsyncMock(return_value={"headers": {"req_id": "req-1"}, "errcode": 0})

        result = await adapter.send("chat-123", "Hello WeCom")

        assert result.success is True
        adapter._send_request.assert_awaited_once_with(
            APP_CMD_SEND,
            {
                "chatid": "chat-123",
                "msgtype": "markdown",
                "markdown": {"content": "Hello WeCom"},
            },
        )

    @pytest.mark.asyncio
    async def test_send_reports_wecom_errors(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_request = AsyncMock(return_value={"errcode": 40001, "errmsg": "bad request"})

        result = await adapter.send("chat-123", "Hello WeCom")

        assert result.success is False
        assert "40001" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_image_falls_back_to_text_for_remote_url(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_media_source = AsyncMock(return_value=SendResult(success=False, error="upload failed"))
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))

        result = await adapter.send_image("chat-123", "https://example.com/demo.png", caption="demo")

        assert result.success is True
        adapter.send.assert_awaited_once_with(chat_id="chat-123", content="demo\nhttps://example.com/demo.png", reply_to=None)

    @pytest.mark.asyncio
    async def test_send_voice_sends_caption_and_downgrade_note(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._prepare_outbound_media = AsyncMock(
            return_value={
                "data": b"voice-bytes",
                "content_type": "audio/mpeg",
                "file_name": "voice.mp3",
                "detected_type": "voice",
                "final_type": "file",
                "rejected": False,
                "reject_reason": None,
                "downgraded": True,
                "downgrade_note": "语音格式 audio/mpeg 不支持，企微仅支持 AMR 格式，已转为文件格式发送",
            }
        )
        adapter._upload_media_bytes = AsyncMock(return_value={"media_id": "media-1", "type": "file"})
        adapter._send_media_message = AsyncMock(return_value={"headers": {"req_id": "req-media"}, "errcode": 0})
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))

        result = await adapter.send_voice("chat-123", "/tmp/voice.mp3", caption="listen")

        assert result.success is True
        adapter._send_media_message.assert_awaited_once_with("chat-123", "file", "media-1")
        assert adapter.send.await_count == 2
        adapter.send.assert_any_await(chat_id="chat-123", content="listen", reply_to=None)
        adapter.send.assert_any_await(
            chat_id="chat-123",
            content="ℹ️ 语音格式 audio/mpeg 不支持，企微仅支持 AMR 格式，已转为文件格式发送",
            reply_to=None,
        )


class TestInboundMessages:
    @pytest.mark.asyncio
    async def test_on_message_builds_event(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0  # disable batching for tests
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-1"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-1",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hello"},
            },
        }

        await adapter._on_message(payload)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello"
        assert event.source.chat_id == "group-1"
        assert event.source.user_id == "user-1"
        assert event.media_urls == []
        assert event.media_types == []

    @pytest.mark.asyncio
    async def test_on_message_preserves_quote_context(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0  # disable batching for tests
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-1"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-1",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "follow up"},
                "quote": {"msgtype": "text", "text": {"content": "quoted message"}},
            },
        }

        await adapter._on_message(payload)

        event = adapter.handle_message.await_args.args[0]
        assert event.reply_to_text == "quoted message"
        assert event.reply_to_message_id == "quote:msg-1"

    @pytest.mark.asyncio
    async def test_on_message_respects_group_policy(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-allowed"]},
            )
        )
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_callback",
            "headers": {"req_id": "req-1"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-blocked",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hello"},
            },
        }

        await adapter._on_message(payload)
        adapter.handle_message.assert_not_awaited()


class TestWeComTwoPhaseIngestion:
    def _adapter(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0
        adapter.handle_message = AsyncMock()
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reply-1"))
        return adapter

    def _payload(self, body):
        merged = {
            "msgid": "msg-1",
            "chatid": "group-1",
            "chattype": "group",
            "from": {"userid": "user-1"},
        }
        merged.update(body)
        return {"cmd": "aibot_msg_callback", "headers": {"req_id": "req-1"}, "body": merged}

    def _pending(self, adapter, chat_id="group-1"):
        return adapter.pending_wecom_ingestion[chat_id]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            "项目资料.pdf",
            "合作方案.docx",
            "预算表.xlsx",
            "路演材料.pptx",
            "归档材料.zip",
            "readme.txt",
        ],
    )
    async def test_file_material_starts_pending_confirmation(self, filename):
        from gateway.platforms.wecom import WECOM_INGESTION_CONFIRMATION_TEMPLATE

        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(self._payload({"msgtype": "file", "file": {"filename": filename}}))

        adapter.handle_message.assert_not_awaited()
        pending = self._pending(adapter)
        assert pending["source_type"] == "file"
        assert pending["chat_id"] == "group-1"
        assert pending["message_id"] == "msg-1"
        assert pending["confidence"] in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
        assert pending["suggested_path"].startswith(
            "projects/_staging/materials/uploads/confirmed/"
        )
        assert pending["created_at"].endswith("+08:00")
        sent = adapter.send.await_args.kwargs["content"]
        assert "1. 进入知识库自动处理工作流" in sent
        assert "2. 暂不处理" in sent
        assert "3. 仅保存原始资料，稍后人工判断" in sent
        assert "初步分析" in sent
        assert "可能主体" in sent
        assert "可能类目" in sent
        assert "推荐暂存位置" in sent

    @pytest.mark.asyncio
    async def test_image_material_starts_pending_confirmation(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(self._payload({"msgtype": "image", "image": {"filename": "现场照片.png"}}))

        adapter.handle_message.assert_not_awaited()
        pending = self._pending(adapter)
        assert pending["source_type"] == "image"
        assert pending["predicted_topic"] in ("X", "undetermined")
        assert pending["confidence"] in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
        assert pending["suggested_path"].startswith("projects/_staging/materials/uploads/confirmed/")
        assert "analysis" in pending
        assert "future_location" in pending
        assert "资料类型" in adapter.send.await_args.kwargs["content"]
        assert "初步分析" in adapter.send.await_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_url_material_starts_pending_confirmation(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload(
                {
                    "msgtype": "text",
                    "text": {"content": "请存一下 https://example.com/reports/q1"},
                }
            )
        )

        adapter.handle_message.assert_not_awaited()
        assert self._pending(adapter)["source_type"] == "url"
        assert "资料类型：链接" in adapter.send.await_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_obvious_long_material_text_does_not_trigger_pending(self):
        """Long text without attachment/URL/intent should NOT start ingestion candidate."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))
        long_text = "季度经营分析报告\n" + ("这是本季度项目进展、风险、预算和下一步计划。" * 60)

        await adapter._on_message(self._payload({"msgtype": "text", "text": {"content": long_text}}))

        # Should go through normal handle_message, not pending
        adapter.handle_message.assert_awaited_once()
        assert "group-1" not in adapter.pending_wecom_ingestion

    @pytest.mark.asyncio
    async def test_notice_text_without_intent_does_not_trigger_pending(self):
        """Notice text without attachment/URL/intent should NOT start ingestion candidate."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))
        notice = "项目申报通知：" + ("请各部门按要求提交资料，逾期不再受理。" * 5)

        await adapter._on_message(self._payload({"msgtype": "text", "text": {"content": notice}}))

        adapter.handle_message.assert_awaited_once()
        assert "group-1" not in adapter.pending_wecom_ingestion

    @pytest.mark.asyncio
    async def test_pending_ingestion_is_isolated_per_chat_and_same_chat_overwrites(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload(
                {
                    "chatid": "chat-a",
                    "msgid": "a-1",
                    "msgtype": "file",
                    "file": {"filename": "A资料.pdf"},
                }
            )
        )
        await adapter._on_message(
            self._payload(
                {
                    "chatid": "chat-b",
                    "msgid": "b-1",
                    "msgtype": "file",
                    "file": {"filename": "B资料.pdf"},
                }
            )
        )
        await adapter._on_message(
            self._payload(
                {
                    "chatid": "chat-a",
                    "msgid": "a-2",
                    "msgtype": "file",
                    "file": {"filename": "A新资料.pdf"},
                }
            )
        )

        assert set(adapter.pending_wecom_ingestion) == {"chat-a", "chat-b"}
        assert adapter.pending_wecom_ingestion["chat-a"]["message_id"] == "a-2"
        assert adapter.pending_wecom_ingestion["chat-b"]["message_id"] == "b-1"

    @pytest.mark.asyncio
    async def test_chat_a_reply_one_consumes_only_chat_a_pending(self):
        from gateway.platforms.wecom import WECOM_INGESTION_QUEUED_TEXT
        from unittest.mock import patch

        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))
        pending_a = {
            "chat_id": "chat-a",
            "message_id": "a-source",
            "source_type": "file",
            "predicted_topic": "competition_aild",
            "confidence": "HIGH",
            "suggested_path": "projects/_staging/materials/competition_aild/a-source",
            "analysis": {
                "title": "AILD 智能设计大赛资料通知",
                "keywords": [],
                "subject_code": "X",
                "subject_label": "待判断",
                "category_code": "YEV",
                "category_label": "青少年赛事活动",
                "confidence": "HIGH",
                "basis": ["text_excerpt"],
            },
            "future_location": "projects/_staging/materials/uploads/review_required/（待人工判断）",
            "created_at": "2026-05-30T12:00:00+08:00",
        }
        pending_b = {
            "chat_id": "chat-b",
            "message_id": "b-source",
            "source_type": "url",
            "predicted_topic": "undetermined",
            "confidence": "UNKNOWN",
            "suggested_path": "projects/_staging/materials/undetermined/b-source",
            "analysis": {
                "title": "未命名资料",
                "keywords": [],
                "subject_code": "X",
                "subject_label": "待判断",
                "category_code": "TMP",
                "category_label": "临时资料",
                "confidence": "LOW",
                "basis": [],
            },
            "future_location": "projects/_staging/materials/uploads/review_required/（待人工判断）",
            "created_at": "2026-05-30T12:01:00+08:00",
        }
        adapter.pending_wecom_ingestion = {"chat-a": pending_a, "chat-b": pending_b}

        with patch(
            "gateway.platforms.wecom._write_wecom_ingestion_staging",
            return_value={"staging_write_status": "skipped_no_wiki_root"},
        ):
            await adapter._on_message(
                self._payload(
                    {
                        "chatid": "chat-a",
                        "msgid": "a-reply",
                        "msgtype": "text",
                        "text": {"content": "1"},
                    }
                )
            )

        adapter.handle_message.assert_not_awaited()
        assert adapter.pending_wecom_ingestion == {"chat-b": pending_b}
        assert adapter.confirmed_ingestion["queue_manifest"]["pending"] == pending_a
        adapter.send.assert_awaited_once_with(
            chat_id="chat-a",
            content=WECOM_INGESTION_QUEUED_TEXT,
            reply_to="a-reply",
        )

    @pytest.mark.asyncio
    async def test_aild_file_triggers_ingestion_candidate(self):
        """File with AILD keyword triggers ingestion; backward compat uses old topic + confidence."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload(
                {
                    "msgtype": "file",
                    "file": {"filename": "AILD智能设计大赛参赛指南.pdf"},
                    "text": {"content": "AILD 智能设计大赛资料通知"},
                }
            )
        )

        pending = self._pending(adapter)
        assert pending["source_type"] == "file"
        assert pending["predicted_topic"] == "competition_aild"
        assert pending["confidence"] == "HIGH"
        assert "analysis" in pending
        assert pending["analysis"]["subject_code"] in ("X", "FDN", "YDC")
        assert pending["analysis"]["category_code"] in ("YEV", "DOC", "TMP")

    @pytest.mark.asyncio
    async def test_pure_legal_ai_text_does_not_trigger_pending(self):
        """Pure text about legal AI (no attachment/intent) should NOT start ingestion candidate."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))
        text = "人工智能法律资料通知：" + ("请整理法规、案例和问答材料。" * 5)

        await adapter._on_message(self._payload({"msgtype": "text", "text": {"content": text}}))

        adapter.handle_message.assert_awaited_once()
        assert "group-1" not in adapter.pending_wecom_ingestion

    @pytest.mark.asyncio
    async def test_emergency_safety_file_triggers_ingestion_candidate(self):
        """File with emergency-safety keyword triggers ingestion; backward compat uses old topic."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload(
                {
                    "msgtype": "file",
                    "file": {"filename": "应急安全预案.docx"},
                    "text": {"content": "应急安全竞赛资料报告"},
                }
            )
        )

        pending = self._pending(adapter)
        assert pending["source_type"] == "file"
        assert pending["predicted_topic"] == "competition_emergency_safety"
        assert pending["confidence"] == "HIGH"
        assert "analysis" in pending

    @pytest.mark.asyncio
    async def test_chuangqingchun_file_triggers_ingestion_candidate(self):
        """File with 创青春 keyword triggers ingestion; backward compat uses old topic."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload(
                {
                    "msgtype": "file",
                    "file": {"filename": "创青春项目申报通知.docx"},
                    "text": {"content": "创青春 项目申报通知：请保存报名材料、商业计划书和答辩安排。"},
                }
            )
        )

        pending = self._pending(adapter)
        assert pending["source_type"] == "file"
        assert pending["predicted_topic"] == "chuangqingchun"
        assert pending["confidence"] == "MEDIUM"
        assert "analysis" in pending

    @pytest.mark.asyncio
    async def test_normal_chat_does_not_trigger_ingestion(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(self._payload({"msgtype": "text", "text": {"content": "你好"}}))

        adapter.handle_message.assert_awaited_once()
        adapter.send.assert_not_awaited()
        assert adapter.pending_wecom_ingestion == {}

    @pytest.mark.asyncio
    async def test_aild_question_does_not_trigger_pending(self):
        """AILD question without attachment / URL / intent should NOT trigger pending."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload({"msgtype": "text", "text": {"content": "AILD 智能设计大赛什么时候比赛？"}})
        )

        adapter.handle_message.assert_awaited_once()
        assert "group-1" not in adapter.pending_wecom_ingestion

    @pytest.mark.asyncio
    async def test_notice_without_intent_does_not_trigger_pending(self):
        """Notice text without attachment / URL / intent should NOT trigger pending."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload({"msgtype": "text", "text": {"content": "项目申报通知：请各部门提交材料"}})
        )

        adapter.handle_message.assert_awaited_once()
        assert "group-1" not in adapter.pending_wecom_ingestion

    @pytest.mark.asyncio
    async def test_chinese_intent_phrase_triggers_pending(self):
        """Chinese intent phrase '这份资料存一下' should trigger pending with source_type=text."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload({"msgtype": "text", "text": {"content": "这份资料存一下"}})
        )

        adapter.handle_message.assert_not_awaited()
        pending = self._pending(adapter)
        assert pending["source_type"] == "text"
        assert "analysis" in pending

    @pytest.mark.asyncio
    async def test_english_intent_phrase_triggers_pending(self):
        """English intent phrase 'Save to Raymond Wiki' should trigger pending with source_type=text."""
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(
            self._payload({"msgtype": "text", "text": {"content": "存到 Raymond Wiki"}})
        )

        adapter.handle_message.assert_not_awaited()
        pending = self._pending(adapter)
        assert pending["source_type"] == "text"
        assert "analysis" in pending

    @pytest.mark.asyncio
    async def test_reply_one_consumes_pending_and_marks_confirmed(self):
        from gateway.platforms.wecom import WECOM_INGESTION_QUEUED_TEXT
        from unittest.mock import patch

        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))
        pending = {
            "chat_id": "group-1",
            "message_id": "source-msg",
            "source_type": "file",
            "predicted_topic": "competition_aild",
            "confidence": "HIGH",
            "suggested_path": "projects/_staging/materials/competition_aild/source-msg",
            "created_at": "2026-05-30T12:00:00+08:00",
        }
        adapter.pending_wecom_ingestion = {"group-1": pending}

        with patch(
            "gateway.platforms.wecom._write_wecom_ingestion_staging",
            return_value={"staging_write_status": "skipped_no_wiki_root"},
        ):
            await adapter._on_message(self._payload({"msgid": "reply-msg", "msgtype": "text", "text": {"content": "1"}}))

        adapter.handle_message.assert_not_awaited()
        assert adapter.pending_wecom_ingestion == {}
        assert adapter.confirmed_ingestion["message_id"] == "source-msg"
        assert adapter.confirmed_ingestion["confirmed_message_id"] == "reply-msg"
        assert adapter.confirmed_ingestion["queue_manifest"]["pending"] == pending
        adapter.send.assert_awaited_once_with(
            chat_id="group-1",
            content=WECOM_INGESTION_QUEUED_TEXT,
            reply_to="reply-msg",
        )

    @pytest.mark.asyncio
    async def test_reply_one_without_pending_is_normal_chat(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(self._payload({"msgtype": "text", "text": {"content": "1"}}))

        adapter.handle_message.assert_awaited_once()
        adapter.send.assert_not_awaited()
        assert adapter.pending_wecom_ingestion == {}
        assert adapter.confirmed_ingestion is None

    @pytest.mark.asyncio
    async def test_reply_two_cancels_pending_with_exact_text(self):
        from gateway.platforms.wecom import WECOM_INGESTION_CANCELLED_TEXT

        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))
        adapter.pending_wecom_ingestion = {"group-1": {
            "chat_id": "group-1",
            "message_id": "source-msg",
            "source_type": "url",
            "predicted_topic": "undetermined",
            "confidence": "UNKNOWN",
            "suggested_path": "projects/_staging/materials/undetermined/source-msg",
            "created_at": "2026-05-30T12:00:00+08:00",
        }}

        await adapter._on_message(self._payload({"msgid": "reply-msg", "msgtype": "text", "text": {"content": "2"}}))

        assert adapter.pending_wecom_ingestion == {}
        adapter.send.assert_awaited_once_with(
            chat_id="group-1",
            content=WECOM_INGESTION_CANCELLED_TEXT,
            reply_to="reply-msg",
        )

    @pytest.mark.asyncio
    async def test_reply_three_records_raw_candidate_with_exact_text(self):
        from gateway.platforms.wecom import WECOM_INGESTION_RAW_RECORDED_TEXT

        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))
        pending = {
            "chat_id": "group-1",
            "message_id": "source-msg",
            "source_type": "text",
            "predicted_topic": "undetermined",
            "confidence": "UNKNOWN",
            "suggested_path": "projects/_staging/materials/undetermined/source-msg",
            "created_at": "2026-05-30T12:00:00+08:00",
        }
        adapter.pending_wecom_ingestion = {"group-1": pending}

        await adapter._on_message(self._payload({"msgid": "reply-msg", "msgtype": "text", "text": {"content": "3"}}))

        assert adapter.pending_wecom_ingestion == {}
        assert adapter.raw_candidate_ingestion["message_id"] == "source-msg"
        assert adapter.raw_candidate_ingestion["recorded_message_id"] == "reply-msg"
        assert adapter.raw_candidate_ingestion["queue_manifest"]["pending"] == pending
        adapter.send.assert_awaited_once_with(
            chat_id="group-1",
            content=WECOM_INGESTION_RAW_RECORDED_TEXT,
            reply_to="reply-msg",
        )

    @pytest.mark.asyncio
    async def test_ingestion_logs_are_privacy_safe(self, caplog):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))
        secret_url = "https://example.com/private/full/path"
        full_text = f"请保存 {secret_url}"

        with caplog.at_level("INFO", logger="gateway.platforms.wecom"):
            await adapter._on_message(
                self._payload({"msgtype": "text", "text": {"content": full_text}})
            )

        log_output = "\n".join(record.getMessage() for record in caplog.records)
        assert "platform=wecom" in log_output
        assert "chat_id=group-1" in log_output
        assert "message_id=msg-1" in log_output
        assert "source_type=url" in log_output
        assert "action=pending" in log_output
        assert "pending_state=created" in log_output
        assert secret_url not in log_output
        assert full_text not in log_output


class TestWeComZombieSessionFix:
    """Tests for PR #11572 — device_id, markdown reply, group req_id fallback."""

    def test_adapter_generates_stable_device_id_per_instance(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert isinstance(adapter._device_id, str)
        assert len(adapter._device_id) > 0
        # Second snapshot on the same adapter must be identical — only a fresh
        # adapter instance should get a new device_id (one-per-reconnect is the
        # zombie-session footgun we're fixing).
        assert adapter._device_id == adapter._device_id

    def test_different_adapter_instances_get_distinct_device_ids(self):
        from gateway.platforms.wecom import WeComAdapter

        a = WeComAdapter(PlatformConfig(enabled=True))
        b = WeComAdapter(PlatformConfig(enabled=True))
        assert a._device_id != b._device_id

    @pytest.mark.asyncio
    async def test_open_connection_includes_device_id_in_subscribe(self):
        from gateway.platforms.wecom import APP_CMD_SUBSCRIBE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._bot_id = "test-bot"
        adapter._secret = "test-secret"

        sent_payloads = []

        class _FakeWS:
            closed = False

            async def send_json(self, payload):
                sent_payloads.append(payload)

            async def close(self):
                return None

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                pass

            async def ws_connect(self, *args, **kwargs):
                return _FakeWS()

            async def close(self):
                return None

        async def _fake_cleanup():
            return None

        async def _fake_handshake(req_id):
            return {"errcode": 0, "headers": {"req_id": req_id}}

        adapter._cleanup_ws = _fake_cleanup
        adapter._wait_for_handshake = _fake_handshake

        with patch("gateway.platforms.wecom.aiohttp", SimpleNamespace(ClientSession=_FakeSession)):
            await adapter._open_connection()

        assert len(sent_payloads) == 1
        subscribe = sent_payloads[0]
        assert subscribe["cmd"] == APP_CMD_SUBSCRIBE
        assert subscribe["body"]["bot_id"] == "test-bot"
        assert subscribe["body"]["secret"] == "test-secret"
        assert subscribe["body"]["device_id"] == adapter._device_id

    @pytest.mark.asyncio
    async def test_on_message_caches_last_req_id_per_chat(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-abc"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-1",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hi"},
            },
        }

        await adapter._on_message(payload)
        assert adapter._last_chat_req_ids["group-1"] == "req-abc"

    @pytest.mark.asyncio
    async def test_on_message_does_not_cache_blocked_sender_req_id(self):
        """Blocked chats shouldn't populate the proactive-send fallback cache."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-ok"]},
            )
        )
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-abc"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-blocked",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hi"},
            },
        }

        await adapter._on_message(payload)
        adapter.handle_message.assert_not_awaited()
        assert "group-blocked" not in adapter._last_chat_req_ids

    def test_remember_chat_req_id_is_bounded(self):
        from gateway.platforms.wecom import DEDUP_MAX_SIZE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        for i in range(DEDUP_MAX_SIZE + 50):
            adapter._remember_chat_req_id(f"chat-{i}", f"req-{i}")
        assert len(adapter._last_chat_req_ids) <= DEDUP_MAX_SIZE
        # The most recently remembered chat must still be present.
        latest = f"chat-{DEDUP_MAX_SIZE + 49}"
        assert adapter._last_chat_req_ids[latest] == f"req-{DEDUP_MAX_SIZE + 49}"

    def test_remember_chat_req_id_ignores_empty_values(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._remember_chat_req_id("", "req-1")
        adapter._remember_chat_req_id("chat-1", "")
        adapter._remember_chat_req_id("   ", "   ")
        assert adapter._last_chat_req_ids == {}

    @pytest.mark.asyncio
    async def test_proactive_group_send_falls_back_to_cached_req_id(self):
        """Sending into a group without reply_to should use the last cached
        req_id via APP_CMD_RESPONSE — WeCom AI Bots cannot initiate APP_CMD_SEND
        in group chats (errcode 600039)."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["group-1"] = "inbound-req-42"
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "inbound-req-42"}, "errcode": 0}
        )
        adapter._send_request = AsyncMock(
            return_value={"headers": {"req_id": "new"}, "errcode": 0}
        )

        result = await adapter.send("group-1", "ping", reply_to=None)

        assert result.success is True
        # Must route through reply (APP_CMD_RESPONSE), not proactive send.
        adapter._send_reply_request.assert_awaited_once()
        adapter._send_request.assert_not_awaited()
        args = adapter._send_reply_request.await_args.args
        assert args[0] == "inbound-req-42"
        assert args[1]["msgtype"] == "markdown"
        assert args[1]["markdown"]["content"] == "ping"

    @pytest.mark.asyncio
    async def test_proactive_send_without_cached_req_id_uses_app_cmd_send(self):
        """When we have no prior req_id (fresh DM target), APP_CMD_SEND is used."""
        from gateway.platforms.wecom import APP_CMD_SEND, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_request = AsyncMock(
            return_value={"headers": {"req_id": "new"}, "errcode": 0}
        )

        result = await adapter.send("fresh-dm-chat", "ping", reply_to=None)

        assert result.success is True
        adapter._send_request.assert_awaited_once()
        cmd = adapter._send_request.await_args.args[0]
        assert cmd == APP_CMD_SEND



class TestTextBatchFlushRace:
    """Regression tests for the cancel-delivery race in _flush_text_batch.

    When asyncio.sleep() fires and Task.cancel() is called before the task
    runs, CPython sets _must_cancel but cannot cancel the already-done sleep
    future.  CancelledError is then delivered at the *next* await
    (handle_message), after the task has already popped the event — the
    superseding task sees an empty batch and silently drops the message.
    The fix adds a synchronous task-registry check between the sleep and
    the pop so a superseded task returns before touching the event.
    """

    @pytest.mark.asyncio
    async def test_superseded_task_does_not_pop_or_process_event(self):
        """A flush task that has been superseded must leave the event in the
        batch dict for the new task to handle."""
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0

        key = "test-session"
        event = MessageEvent(text="hello", message_type=MessageType.TEXT)
        adapter._pending_text_batches[key] = event

        handle_calls = []

        async def fake_handle(evt):
            handle_calls.append(evt)

        adapter.handle_message = fake_handle

        # Create T1 and register it.
        t1 = asyncio.create_task(adapter._flush_text_batch(key))
        adapter._pending_text_batch_tasks[key] = t1

        # Simulate T2 superseding T1 before T1 wakes from sleep.
        t2 = asyncio.create_task(asyncio.sleep(9999))
        adapter._pending_text_batch_tasks[key] = t2

        # Yield long enough for T1's sleep(0) to complete and T1 to run.
        await asyncio.sleep(0.05)

        t2.cancel()
        try:
            await t2
        except asyncio.CancelledError:
            pass

        # T1 must have returned without processing or removing the event.
        assert handle_calls == [], "superseded task must not call handle_message"
        assert adapter._pending_text_batches.get(key) is event, (
            "superseded task must not pop the event"
        )

    @pytest.mark.asyncio
    async def test_active_task_processes_event_normally(self):
        """When the task is not superseded it must still process the event."""
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0

        key = "test-session"
        event = MessageEvent(text="world", message_type=MessageType.TEXT)
        adapter._pending_text_batches[key] = event

        handle_calls = []

        async def fake_handle(evt):
            handle_calls.append(evt)

        adapter.handle_message = fake_handle

        t1 = asyncio.create_task(adapter._flush_text_batch(key))
        adapter._pending_text_batch_tasks[key] = t1

        # No superseding task — T1 should process normally.
        await asyncio.sleep(0.05)

        assert handle_calls == [event], "active task must call handle_message"
        assert adapter._pending_text_batches.get(key) is None, (
            "active task must pop the event after processing"
        )


# ─── Tests for WeCom file metadata extraction ─────────────────────────────────

from unittest.mock import AsyncMock, patch, MagicMock
import json
import os
import tempfile
from pathlib import Path


class TestWeComFileMetadataExtraction:
    """Verify that filenames are extracted from all WeCom payload blocks and drive pre-analysis."""

    def _adapter(self):
        from gateway.platforms.wecom import WeComAdapter
        from gateway.config import PlatformConfig
        return WeComAdapter(PlatformConfig(enabled=True))

    # ── Test 1: filename from body.file ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_docx_filename_from_file_block_drives_fdn_pub_analysis(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        body = {
            "msgtype": "file",
            "file": {
                "filename": "天津市青年创业就业基金会简介（2026.5）.docx",
                "media_id": "media-file-001",
            },
        }

        result = adapter._detect_wecom_ingestion_candidate(
            body=body,
            text="",
            media_urls=[],
            media_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            message_id="test-file-block",
        )
        assert result is not None, "DOCX file should trigger ingestion candidate"
        # filename must appear in candidate
        assert "天津市青年创业就业基金会简介（2026.5）.docx" in result.get("file_names", [])
        # pre-analysis must recognise FDN + PUB from filename
        assert result["analysis"]["subject_code"] == "FDN"
        assert result["analysis"]["category_code"] == "PUB"
        assert result["analysis"]["confidence"] == "HIGH"
        # file_metadata must be populated
        assert result["file_metadata"].get("media_ids") == ["media-file-001"]

    # ── Test 2: filename from body.appmsg.title ───────────────────────────────

    @pytest.mark.asyncio
    async def test_docx_filename_from_appmsg_title_drives_fdn_pub_analysis(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        body = {
            "msgtype": "appmsg",
            "appmsg": {
                "title": "天津市青年创业就业基金会简介（2026.5）.docx",
                "des": "这是一个docx文件",
            },
        }

        result = adapter._detect_wecom_ingestion_candidate(
            body=body,
            text="",
            media_urls=[],
            media_types=["application/msword"],
            message_id="test-appmsg-title",
        )
        assert result is not None, "appmsg with docx title should trigger candidate"
        assert "天津市青年创业就业基金会简介（2026.5）.docx" in result.get("file_names", [])
        assert result["analysis"]["subject_code"] == "FDN"
        assert result["analysis"]["category_code"] == "PUB"
        assert result["analysis"]["confidence"] == "HIGH"

    # ── Test 3: filename from mixed.msg_item ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_filename_from_mixed_msg_item_drives_analysis(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        body = {
            "msgtype": "mixed",
            "mixed": {
                "msg_item": [
                    {
                        "msgtype": "file",
                        "file": {
                            "filename": "2026年项目申报指南.pdf",
                            "media_id": "media-pdf-002",
                        },
                    },
                    {"msgtype": "text", "content": "请查收"},
                ]
            },
        }

        result = adapter._detect_wecom_ingestion_candidate(
            body=body,
            text="请查收",
            media_urls=[],
            media_types=["application/pdf"],
            message_id="test-mixed",
        )
        assert result is not None, "mixed msg_item with file should trigger candidate"
        assert "2026年项目申报指南.pdf" in result.get("file_names", [])
        # 2026年项目申报指南.pdf → subject=X (no FDN/YDC/C keyword), category=ENT from 项目申报
        assert result["analysis"]["subject_code"] == "X"
        assert result["analysis"]["category_code"] in ("YEV", "ENT", "TMP")

    # ── Test 4: filename from cached media path ───────────────────────────────

    @pytest.mark.asyncio
    async def test_cached_media_path_filename_drives_analysis(self):
        adapter = self._adapter()
        # Simulate _extract_media returning a local cache path with Chinese filename
        adapter._extract_media = AsyncMock(
            return_value=(["/tmp/天津市青年创业就业基金会简介（2026.5）.docx"], ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"])
        )

        body = {"msgtype": "file"}

        result = adapter._detect_wecom_ingestion_candidate(
            body=body,
            text="",
            media_urls=["/tmp/天津市青年创业就业基金会简介（2026.5）.docx"],
            media_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            message_id="test-metadata",
        )
        assert result is not None, "cached media path with filename should trigger candidate"
        assert "天津市青年创业就业基金会简介（2026.5）.docx" in result.get("file_names", [])
        assert result["analysis"]["subject_code"] == "FDN"
        assert result["analysis"]["category_code"] == "PUB"

    # ── Test 5: manifest preserves file_names + file_metadata after reply=1 ─────

    @pytest.mark.asyncio
    async def test_pending_manifest_preserves_file_names_and_metadata(self):
        from gateway.platforms.wecom import WeComAdapter
        from gateway.config import PlatformConfig
        import importlib
        import gateway.platforms.wecom as _wecom
        importlib.reload(_wecom)
        from gateway.platforms.wecom import _write_wecom_ingestion_staging

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._extract_media = AsyncMock(return_value=([], []))

        body = {
            "msgtype": "file",
            "file": {
                "filename": "天津市青年创业就业基金会简介（2026.5）.docx",
                "media_id": "media-file-001",
            },
        }

        result = adapter._detect_wecom_ingestion_candidate(
            body=body,
            text="",
            media_urls=[],
            media_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            message_id="test-file-block",
        )
        assert result is not None

        # Simulate pending → reply=1 → manifest
        pending = {
            "chat_id": "chat-test",
            "message_id": "msg-metadata-test",
            "source_type": result["source_type"],
            "predicted_topic": result["predicted_topic"],
            "confidence": result["confidence"],
            "suggested_path": result["suggested_path"],
            "file_names": result.get("file_names", []),
            "file_metadata": result.get("file_metadata", {}),
            "analysis": result.get("analysis", {}),
            "future_location": result.get("future_location", ""),
            "created_at": "2026-05-30T12:00:00+08:00",
        }

        manifest = adapter._build_wecom_ingestion_manifest(
            pending=pending,
            action_message_id="confirmed-metadata-test",
            action_at_key="confirmed_at",
            action_message_id_key="confirmed_message_id",
            queue_name="wecom_ingestion",
            status="confirmed",
        )

        # file_names must survive the manifest build
        assert "file_names" in manifest, "manifest must contain file_names"
        assert "天津市青年创业就业基金会简介（2026.5）.docx" in manifest["file_names"]
        # file_metadata must survive
        assert "file_metadata" in manifest
        assert "media_ids" in manifest["file_metadata"]
        # analysis must survive
        assert manifest.get("analysis", {}).get("subject_code") == "FDN"
        assert manifest.get("analysis", {}).get("category_code") == "PUB"

    # ── Test 6: display_name / media_name also usable ─────────────────────────

    @pytest.mark.asyncio
    async def test_display_name_extracted_as_filename(self):
        adapter = self._adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        body = {
            "msgtype": "appmsg",
            "appmsg": {
                "display_name": "天津市青年创业就业基金会简介（2026.5）.docx",
                "des": "简介文件",
            },
        }

        result = adapter._detect_wecom_ingestion_candidate(
            body=body,
            text="",
            media_urls=[],
            media_types=[],
            message_id="test-display-name",
        )
        assert result is not None
        assert "天津市青年创业就业基金会简介（2026.5）.docx" in result.get("file_names", [])
        assert result["analysis"]["subject_code"] == "FDN"
        assert result["analysis"]["category_code"] == "PUB"


class TestWeComStagingWrite:
    """Test the raymond-wiki staging write path triggered on reply=1."""

    def _make_manifest(self, msg_id="msg-123", topic="competition_aild"):
        return {
            "message_id": msg_id,
            "topic": topic,
            "topic_label": "AILD",
            "confirmed_message_id": f"confirmed-{msg_id}",
            "confirmed_at": "2026-05-30T10:00:00+08:00",
            "action": "1",
            "file_names": ["test.pdf"],
        }

    def test_safe_message_id_strips_dots_and_slashes(self):
        from gateway.platforms.wecom import _safe_message_id
        assert _safe_message_id("a/b/c.txt") == "a-b-c-txt"
        # dots → hyphens, then consecutive runs collapsed to one
        assert _safe_message_id("a..b") == "a-b"
        assert _safe_message_id("a\\b") == "a-b"
        assert _safe_message_id("normal-id-123") == "normal-id-123"

    def test_safe_message_id_prevents_path_traversal(self):
        from gateway.platforms.wecom import _safe_message_id
        result = _safe_message_id("../../../etc/passwd")
        assert ".." not in result
        assert result == "etc-passwd"

    def test_resolve_raymond_wiki_root_from_env(self, tmp_path):
        # Patch env BEFORE import so the function sees the patched value
        with patch.dict(os.environ, {"RAYMOND_WIKI_ROOT": str(tmp_path)}):
            # Re-import to pick up fresh module state with patched env
            import importlib
            import gateway.platforms.wecom as _wecom
            importlib.reload(_wecom)
            from gateway.platforms.wecom import _resolve_raymond_wiki_root
            result = _resolve_raymond_wiki_root()
            assert result == tmp_path.resolve()

    def test_resolve_raymond_wiki_root_fallback_skips_nonexistent(self, tmp_path):
        nonexistent = tmp_path / "nonexistent-wiki"
        with patch.dict(os.environ, {"RAYMOND_WIKI_ROOT": str(nonexistent)}):
            import importlib
            import gateway.platforms.wecom as _wecom
            importlib.reload(_wecom)
            from gateway.platforms.wecom import _resolve_raymond_wiki_root
            result = _resolve_raymond_wiki_root()
            assert result is None

    def test_resolve_raymond_wiki_root_returns_none_for_nonexistent(self, tmp_path):
        fake = tmp_path / "nonexistent-wiki"
        with patch.dict(os.environ, {"RAYMOND_WIKI_ROOT": str(fake)}):
            import importlib
            import gateway.platforms.wecom as _wecom
            importlib.reload(_wecom)
            from gateway.platforms.wecom import _resolve_raymond_wiki_root
            assert _resolve_raymond_wiki_root() is None

    def test_write_wecom_ingestion_staging_writes_files(self, tmp_path):
        manifest = self._make_manifest("abc-123", "competition_aild")
        with patch.dict(os.environ, {"RAYMOND_WIKI_ROOT": str(tmp_path)}):
            import importlib
            import gateway.platforms.wecom as _wecom
            importlib.reload(_wecom)
            from gateway.platforms.wecom import _write_wecom_ingestion_staging
            result = _write_wecom_ingestion_staging(manifest)
        # safe_msg_id = _safe_message_id(confirmed_message_id) = "confirmed-abc-123"
        confirmed = tmp_path / "projects" / "_staging" / "materials" / "uploads" / "confirmed" / "2026" / "05" / "confirmed-abc-123"
        assert result["staging_write_status"] == "written"
        assert (confirmed / "INGESTION_MANIFEST.json").exists()
        assert (confirmed / "QUICK.md").exists()
        assert (tmp_path / "projects" / "_staging" / "materials" / "uploads" / "state" / "QUICK_INDEX.jsonl").exists()
        report_file = tmp_path / "projects" / "_staging" / "materials" / "uploads" / "reports" / "20260530_wecom_ingestion_report.jsonl"
        assert report_file.exists()
        with open(confirmed / "INGESTION_MANIFEST.json") as f:
            saved = json.load(f)
        assert saved["topic"] == "competition_aild"
        assert saved["confirmed_message_id"] == "confirmed-abc-123"
        assert saved["staging_write_status"] == "written"
        assert saved["staging_path"] == "projects/_staging/materials/uploads/confirmed/2026/05/confirmed-abc-123"

    def test_write_wecom_ingestion_staging_skipped_when_no_wiki_root(self, tmp_path):
        fake = tmp_path / "nonexistent"
        with patch.dict(os.environ, {"RAYMOND_WIKI_ROOT": str(fake)}):
            import importlib
            import gateway.platforms.wecom as _wecom
            importlib.reload(_wecom)
            from gateway.platforms.wecom import _write_wecom_ingestion_staging
            result = _write_wecom_ingestion_staging(self._make_manifest())
        assert result["staging_write_status"] == "skipped_no_wiki_root"

    def test_write_wecom_ingestion_staging_handles_write_error(self, tmp_path):
        manifest = self._make_manifest()
        readonly = tmp_path / "read-only-staging"
        readonly.mkdir()
        readonly.chmod(0o444)
        (tmp_path / "projects").mkdir()
        (tmp_path / "projects").chmod(0o444)
        try:
            with patch.dict(os.environ, {"RAYMOND_WIKI_ROOT": str(tmp_path)}):
                import importlib
                import gateway.platforms.wecom as _wecom
                importlib.reload(_wecom)
                from gateway.platforms.wecom import _write_wecom_ingestion_staging
                result = _write_wecom_ingestion_staging(manifest)
            assert result["staging_write_status"] == "skipped_write_error"
        finally:
            (tmp_path / "projects").chmod(0o755)
            readonly.chmod(0o755)
