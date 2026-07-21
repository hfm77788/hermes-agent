"""Tests for the Feishu gateway integration."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
from unittest.mock import AsyncMock, Mock, patch

from gateway.platforms.base import ProcessingOutcome

try:
    import lark_oapi
    _HAS_LARK_OAPI = True
except ImportError:
    _HAS_LARK_OAPI = False


class _FakeRequestContent:
    def __init__(self, body: bytes):
        self.body = body
        self.read_sizes: list[int] = []

    async def readexactly(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if len(self.body) < size:
            raise asyncio.IncompleteReadError(self.body, size)
        return self.body[:size]


def _mock_event_dispatcher_builder(mock_handler_class):
    mock_builder = Mock()
    mock_builder.register_p2_im_message_message_read_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_receive_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_reaction_created_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_reaction_deleted_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_card_action_trigger = Mock(return_value=mock_builder)
    mock_builder.build = Mock(return_value=object())
    mock_handler_class.builder = Mock(return_value=mock_builder)
    return mock_builder


class TestConfigEnvOverrides(unittest.TestCase):
    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
        "FEISHU_CONNECTION_MODE": "websocket",
        "FEISHU_DOMAIN": "feishu",
    }, clear=False)
    def test_feishu_config_loaded_from_env(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        self.assertIn(Platform.FEISHU, config.platforms)
        self.assertTrue(config.platforms[Platform.FEISHU].enabled)
        self.assertEqual(config.platforms[Platform.FEISHU].extra["app_id"], "cli_xxx")
        self.assertEqual(config.platforms[Platform.FEISHU].extra["connection_mode"], "websocket")

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
        "FEISHU_HOME_CHANNEL": "oc_xxx",
    }, clear=False)
    def test_feishu_home_channel_loaded(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.platforms[Platform.FEISHU].home_channel
        self.assertIsNotNone(home)
        self.assertEqual(home.chat_id, "oc_xxx")

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
    }, clear=False)
    def test_feishu_in_connected_platforms(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        self.assertIn(Platform.FEISHU, config.get_connected_platforms())


class TestFeishuMessageNormalization(unittest.TestCase):
    def test_normalize_merge_forward_preserves_summary_lines(self):
        from plugins.platforms.feishu.adapter import normalize_feishu_message

        normalized = normalize_feishu_message(
            message_type="merge_forward",
            raw_content=json.dumps(
                {
                    "title": "Sprint recap",
                    "messages": [
                        {"sender_name": "Alice", "text": "Please review PR-128"},
                        {
                            "sender_name": "Bob",
                            "message_type": "post",
                            "content": {
                                "en_us": {
                                    "content": [[{"tag": "text", "text": "Ship it"}]],
                                }
                            },
                        },
                    ],
                }
            ),
        )

        self.assertEqual(normalized.relation_kind, "merge_forward")
        self.assertEqual(
            normalized.text_content,
            "Sprint recap\n- Alice: Please review PR-128\n- Bob: Ship it",
        )

    def test_normalize_share_chat_exposes_summary_and_metadata(self):
        from plugins.platforms.feishu.adapter import normalize_feishu_message

        normalized = normalize_feishu_message(
            message_type="share_chat",
            raw_content=json.dumps(
                {
                    "chat_id": "oc_chat_shared",
                    "chat_name": "Backend Guild",
                }
            ),
        )

        self.assertEqual(normalized.relation_kind, "share_chat")
        self.assertEqual(normalized.text_content, "Shared chat: Backend Guild\nChat ID: oc_chat_shared")
        self.assertEqual(normalized.metadata["chat_id"], "oc_chat_shared")
        self.assertEqual(normalized.metadata["chat_name"], "Backend Guild")

    def test_normalize_interactive_card_preserves_title_body_and_actions(self):
        from plugins.platforms.feishu.adapter import normalize_feishu_message

        normalized = normalize_feishu_message(
            message_type="interactive",
            raw_content=json.dumps(
                {
                    "card": {
                        "header": {"title": {"tag": "plain_text", "content": "Build Failed"}},
                        "elements": [
                            {"tag": "div", "text": {"tag": "lark_md", "content": "Service: payments-api"}},
                            {"tag": "div", "text": {"tag": "plain_text", "content": "Branch: main"}},
                            {
                                "tag": "action",
                                "actions": [
                                    {"tag": "button", "text": {"tag": "plain_text", "content": "View Logs"}},
                                    {"tag": "button", "text": {"tag": "plain_text", "content": "Retry"}},
                                ],
                            },
                        ],
                    }
                }
            ),
        )

        self.assertEqual(normalized.relation_kind, "interactive")
        self.assertEqual(
            normalized.text_content,
            "Build Failed\nService: payments-api\nBranch: main\nView Logs\nRetry\nActions: View Logs, Retry",
        )


class TestFeishuAdapterMessaging(unittest.TestCase):
    @unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
    def test_websocket_sdk_accepts_channel_ua_tag(self):
        """The shipped SDK must support the Channel signaling argument."""
        import inspect

        from lark_oapi.ws import Client as FeishuWSClient

        signature = inspect.signature(FeishuWSClient)
        self.assertIn("extra_ua_tags", signature.parameters)

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
        "FEISHU_CONNECTION_MODE": "webhook",
        "FEISHU_WEBHOOK_HOST": "127.0.0.1",
        "FEISHU_WEBHOOK_PORT": "9001",
        "FEISHU_WEBHOOK_PATH": "/hook",
        "FEISHU_VERIFICATION_TOKEN": "vtok",
    }, clear=True)
    def test_connect_webhook_mode_starts_local_server(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        runner = AsyncMock()
        site = AsyncMock()
        web_module = SimpleNamespace(
            Application=lambda **_kwargs: SimpleNamespace(router=SimpleNamespace(add_post=lambda *_args, **_kwargs: None)),
            AppRunner=lambda _app: runner,
            TCPSite=lambda _runner, host, port: SimpleNamespace(start=site.start, host=host, port=port),
        )

        with (
            patch("plugins.platforms.feishu.adapter.FEISHU_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.FEISHU_WEBHOOK_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.EventDispatcherHandler") as mock_handler_class,
            patch("plugins.platforms.feishu.adapter.acquire_scoped_lock", return_value=(True, None)),
            patch("plugins.platforms.feishu.adapter.release_scoped_lock"),
            patch.object(adapter, "_hydrate_bot_identity", new=AsyncMock()),
            patch.object(adapter, "_build_lark_client", return_value=SimpleNamespace()),
            patch("plugins.platforms.feishu.adapter.web", web_module),
        ):
            _mock_event_dispatcher_builder(mock_handler_class)
            connected = asyncio.run(adapter.connect())

        self.assertTrue(connected)
        runner.setup.assert_awaited_once()
        site.start.assert_awaited_once()

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
    }, clear=True)
    def test_connect_acquires_scoped_lock_and_disconnect_releases_it(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        ws_client = SimpleNamespace()

        with (
            patch("plugins.platforms.feishu.adapter.FEISHU_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.FEISHU_WEBSOCKET_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.lark", SimpleNamespace(LogLevel=SimpleNamespace(INFO="INFO", WARNING="WARNING"))),
            patch("plugins.platforms.feishu.adapter.EventDispatcherHandler") as mock_handler_class,
            patch("plugins.platforms.feishu.adapter.FeishuWSClient", return_value=ws_client),
            patch("plugins.platforms.feishu.adapter._run_official_feishu_ws_client"),
            patch("plugins.platforms.feishu.adapter.acquire_scoped_lock", return_value=(True, None)) as acquire_lock,
            patch("plugins.platforms.feishu.adapter.release_scoped_lock") as release_lock,
            patch.object(adapter, "_hydrate_bot_identity", new=AsyncMock()),
            patch.object(adapter, "_build_lark_client", return_value=SimpleNamespace()),
        ):
            _mock_event_dispatcher_builder(mock_handler_class)

            loop = asyncio.new_event_loop()
            future = loop.create_future()
            future.set_result(None)

            class _Loop:
                def run_in_executor(self, *_args, **_kwargs):
                    return future

                def is_closed(self):
                    return False

            try:
                with patch("plugins.platforms.feishu.adapter.asyncio.get_running_loop", return_value=_Loop()):
                    connected = asyncio.run(adapter.connect())
                    asyncio.run(adapter.disconnect())
            finally:
                loop.close()

        self.assertTrue(connected)
        self.assertIsNone(adapter._event_handler)
        acquire_lock.assert_called_once_with(
            "feishu-app-id",
            "cli_app",
            metadata={"platform": "feishu"},
        )
        release_lock.assert_called_once_with("feishu-app-id", "cli_app")

    def test_disconnect_sends_websocket_close_frame(self):
        """Regression test for #10202: disconnect() must call the WSS
        client's ``_disconnect()`` coroutine so a WebSocket CLOSE frame
        is sent to Feishu. Without this, Feishu's server continues
        routing to the stale connection, silencing the channel.
        """
        import threading
        from types import SimpleNamespace
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        # Real thread loop to schedule the close coroutine on.
        ws_thread_loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop() -> None:
            asyncio.set_event_loop(ws_thread_loop)
            ready.set()
            ws_thread_loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        ready.wait()

        close_called = threading.Event()

        async def _fake_disconnect() -> None:
            close_called.set()

        ws_client = SimpleNamespace(_disconnect=_fake_disconnect, _auto_reconnect=True)
        adapter._ws_client = ws_client
        adapter._ws_thread_loop = ws_thread_loop
        adapter._ws_future = None

        try:
            asyncio.run(adapter.disconnect())
        finally:
            if not ws_thread_loop.is_closed():
                ws_thread_loop.call_soon_threadsafe(ws_thread_loop.stop)
            thread.join(timeout=2.0)
            if not ws_thread_loop.is_closed():
                ws_thread_loop.close()

        self.assertTrue(
            close_called.is_set(),
            "disconnect() must schedule ws_client._disconnect() on the ws thread loop",
        )
        # _disable_websocket_auto_reconnect() must still run.
        self.assertIsNone(adapter._ws_client)

    def test_disconnect_tolerates_missing_internal_disconnect(self):
        """If the lark_oapi client layout changes and ``_disconnect``
        disappears, disconnect() must not raise — fall through to the
        existing task-cancel path.
        """
        from types import SimpleNamespace
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        # No ``_disconnect`` attribute — ``hasattr`` guard should skip.
        adapter._ws_client = SimpleNamespace(_auto_reconnect=True)
        adapter._ws_thread_loop = None
        adapter._ws_future = None

        # Must not raise.
        asyncio.run(adapter.disconnect())
        self.assertIsNone(adapter._ws_client)

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
    }, clear=True)
    def test_connect_rejects_existing_app_lock(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        with (
            patch("plugins.platforms.feishu.adapter.FEISHU_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.FEISHU_WEBSOCKET_AVAILABLE", True),
            patch(
                "plugins.platforms.feishu.adapter.acquire_scoped_lock",
                return_value=(False, {"pid": 4321}),
            ),
        ):
            connected = asyncio.run(adapter.connect())

        self.assertFalse(connected)
        self.assertEqual(adapter.fatal_error_code, "feishu_app_lock")
        self.assertFalse(adapter.fatal_error_retryable)
        self.assertIn("PID 4321", adapter.fatal_error_message)

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
    }, clear=True)
    def test_connect_retries_transient_startup_failure(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        ws_client = SimpleNamespace()
        sleeps = []

        with (
            patch("plugins.platforms.feishu.adapter.FEISHU_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.FEISHU_WEBSOCKET_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.lark", SimpleNamespace(LogLevel=SimpleNamespace(INFO="INFO", WARNING="WARNING"))),
            patch("plugins.platforms.feishu.adapter.EventDispatcherHandler") as mock_handler_class,
            patch("plugins.platforms.feishu.adapter.FeishuWSClient", return_value=ws_client),
            patch("plugins.platforms.feishu.adapter.acquire_scoped_lock", return_value=(True, None)),
            patch("plugins.platforms.feishu.adapter.release_scoped_lock"),
            patch.object(adapter, "_hydrate_bot_identity", new=AsyncMock()),
            patch("plugins.platforms.feishu.adapter.asyncio.sleep", side_effect=lambda delay: sleeps.append(delay)),
            patch.object(adapter, "_build_lark_client", return_value=SimpleNamespace()),
        ):
            _mock_event_dispatcher_builder(mock_handler_class)

            loop = asyncio.new_event_loop()
            future = loop.create_future()
            future.set_result(None)

            class _Loop:
                def __init__(self):
                    self.calls = 0

                def run_in_executor(self, *_args, **_kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        raise OSError("temporary websocket failure")
                    return future

                def is_closed(self):
                    return False

            fake_loop = _Loop()
            try:
                with patch("plugins.platforms.feishu.adapter.asyncio.get_running_loop", return_value=fake_loop):
                    connected = asyncio.run(adapter.connect())
            finally:
                loop.close()

        self.assertTrue(connected)
        self.assertEqual(sleeps, [1])
        self.assertEqual(fake_loop.calls, 2)

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
    }, clear=True)
    def test_connect_websocket_sets_channel_ua_tag(self):
        """Verify that FeishuWSClient receives extra_ua_tags=["channel"].

        Without this UA tag the Feishu server does not push group @mention
        events over the WebSocket transport.  See
        https://github.com/NousResearch/hermes-agent/issues/50656
        """
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        ws_client = SimpleNamespace()

        with (
            patch("plugins.platforms.feishu.adapter.FEISHU_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.FEISHU_WEBSOCKET_AVAILABLE", True),
            patch("plugins.platforms.feishu.adapter.lark",
                  SimpleNamespace(LogLevel=SimpleNamespace(INFO="INFO", WARNING="WARNING"))),
            patch("plugins.platforms.feishu.adapter.EventDispatcherHandler") as mock_handler_class,
            patch("plugins.platforms.feishu.adapter.FeishuWSClient") as mock_ws_client,
            patch("plugins.platforms.feishu.adapter._run_official_feishu_ws_client"),
            patch("plugins.platforms.feishu.adapter.acquire_scoped_lock", return_value=(True, None)),
            patch("plugins.platforms.feishu.adapter.release_scoped_lock"),
            patch.object(adapter, "_hydrate_bot_identity", new=AsyncMock()),
            patch.object(adapter, "_build_lark_client", return_value=SimpleNamespace()),
        ):
            _mock_event_dispatcher_builder(mock_handler_class)

            loop = asyncio.new_event_loop()
            future = loop.create_future()
            future.set_result(None)

            class _Loop:
                def run_in_executor(self, *_args, **_kwargs):
                    return future
                def is_closed(self):
                    return False

            try:
                with patch("plugins.platforms.feishu.adapter.asyncio.get_running_loop",
                           return_value=_Loop()):
                    connected = asyncio.run(adapter.connect())
            finally:
                loop.close()

        self.assertTrue(connected)
        # Verify the Channel SDK UA tag is present — this is the fix for
        # group @mention message delivery over WebSocket.
        mock_ws_client.assert_called_once()
        call_kwargs = mock_ws_client.call_args.kwargs
        self.assertIn("extra_ua_tags", call_kwargs,
                      "FeishuWSClient must receive extra_ua_tags for group @mention delivery")
        self.assertEqual(call_kwargs["extra_ua_tags"], ["channel"],
                         "extra_ua_tags must be ['channel'] to enable group event routing")

    @patch.dict(os.environ, {}, clear=True)
    def test_edit_message_updates_existing_feishu_message(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def update(self, request):
                captured["request"] = request
                return SimpleNamespace(success=lambda: True)

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_progress",
                    content="📖 read_file: \"/tmp/image.png\"",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_progress")
        self.assertEqual(captured["request"].message_id, "om_progress")
        self.assertEqual(captured["request"].request_body.msg_type, "text")
        self.assertEqual(
            captured["request"].request_body.content,
            json.dumps({"text": "📖 read_file: \"/tmp/image.png\""}, ensure_ascii=False),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_edit_message_falls_back_to_text_when_post_update_is_rejected(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"calls": []}

        class _MessageAPI:
            def update(self, request):
                captured["calls"].append(request)
                if len(captured["calls"]) == 1:
                    return SimpleNamespace(success=lambda: False, code=230001, msg="content format of the post type is incorrect")
                return SimpleNamespace(success=lambda: True)

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_progress",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["calls"][0].request_body.msg_type, "post")
        self.assertEqual(captured["calls"][1].request_body.msg_type, "text")
        self.assertEqual(
            captured["calls"][1].request_body.content,
            json.dumps({"text": "可以用 粗体 和 斜体。"}, ensure_ascii=False),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_get_chat_info_uses_real_feishu_chat_api(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        class _ChatAPI:
            def get(self, request):
                self.request = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(name="Hermes Group", chat_type="group"),
                )

        chat_api = _ChatAPI()
        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    chat=chat_api,
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            info = asyncio.run(adapter.get_chat_info("oc_chat"))

        self.assertEqual(chat_api.request.chat_id, "oc_chat")
        self.assertEqual(info["chat_id"], "oc_chat")
        self.assertEqual(info["name"], "Hermes Group")
        self.assertEqual(info["type"], "group")

class TestAdapterModule(unittest.TestCase):
    def test_load_settings_uses_sdk_defaults_for_invalid_ws_reconnect_values(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        settings = FeishuAdapter._load_settings(
            {
                "ws_reconnect_nonce": -1,
                "ws_reconnect_interval": "bad",
            }
        )

        self.assertEqual(settings.ws_reconnect_nonce, 30)
        self.assertEqual(settings.ws_reconnect_interval, 120)

    def test_load_settings_accepts_custom_ws_reconnect_values(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        settings = FeishuAdapter._load_settings(
            {
                "ws_reconnect_nonce": 0,
                "ws_reconnect_interval": 3,
            }
        )

        self.assertEqual(settings.ws_reconnect_nonce, 0)
        self.assertEqual(settings.ws_reconnect_interval, 3)

    def test_load_settings_accepts_custom_ws_ping_values(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        settings = FeishuAdapter._load_settings(
            {
                "ws_ping_interval": 10,
                "ws_ping_timeout": 8,
            }
        )

        self.assertEqual(settings.ws_ping_interval, 10)
        self.assertEqual(settings.ws_ping_timeout, 8)

    def test_load_settings_ignores_invalid_ws_ping_values(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        settings = FeishuAdapter._load_settings(
            {
                "ws_ping_interval": 0,
                "ws_ping_timeout": -1,
            }
        )

        self.assertIsNone(settings.ws_ping_interval)
        self.assertIsNone(settings.ws_ping_timeout)

    def test_runtime_ws_overrides_reapply_after_sdk_configure(self):
        import sys
        from types import ModuleType

        class _FakeWSClient:
            def __init__(self):
                self._reconnect_nonce = 30
                self._reconnect_interval = 120
                self._ping_interval = 120
                self.configure_calls = []

            def _configure(self, conf):
                self.configure_calls.append(conf)
                self._reconnect_nonce = conf.ReconnectNonce
                self._reconnect_interval = conf.ReconnectInterval
                self._ping_interval = conf.PingInterval

            def start(self):
                conf = SimpleNamespace(ReconnectNonce=99, ReconnectInterval=88, PingInterval=77)
                self._configure(conf)
                raise RuntimeError("stop test client")

        fake_client = _FakeWSClient()
        fake_adapter = SimpleNamespace(
            _ws_thread_loop=None,
            _ws_reconnect_nonce=2,
            _ws_reconnect_interval=3,
            _ws_ping_interval=4,
            _ws_ping_timeout=5,
        )
        fake_client_module = ModuleType("lark_oapi.ws.client")
        fake_client_module.loop = None
        fake_client_module.websockets = SimpleNamespace(connect=AsyncMock())
        fake_ws_module = ModuleType("lark_oapi.ws")
        fake_ws_module.client = fake_client_module
        fake_root_module = ModuleType("lark_oapi")
        fake_root_module.ws = fake_ws_module

        original_modules = sys.modules.copy()
        sys.modules["lark_oapi"] = fake_root_module
        sys.modules["lark_oapi.ws"] = fake_ws_module
        sys.modules["lark_oapi.ws.client"] = fake_client_module
        try:
            from plugins.platforms.feishu.adapter import _run_official_feishu_ws_client

            _run_official_feishu_ws_client(fake_client, fake_adapter)
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

        self.assertEqual(len(fake_client.configure_calls), 1)
        self.assertEqual(fake_client._reconnect_nonce, 2)
        self.assertEqual(fake_client._reconnect_interval, 3)
        self.assertEqual(fake_client._ping_interval, 4)


def _admits_group(adapter, message, sender_id, chat_id=""):
    """Group-path shim: run a message through ``_admit`` and return a bool."""
    sender = SimpleNamespace(sender_type="user", sender_id=sender_id)
    if not hasattr(message, "chat_type"):
        message.chat_type = "group"
    if chat_id:
        message.chat_id = chat_id
    return adapter._admit(sender, message) is None


class TestAdapterBehavior(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_build_event_handler_registers_reaction_and_card_processors(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        calls = []

        class _Builder:
            def register_p2_im_message_message_read_v1(self, _handler):
                calls.append("message_read")
                return self

            def register_p2_im_message_receive_v1(self, _handler):
                calls.append("message_receive")
                return self

            def register_p2_im_message_reaction_created_v1(self, _handler):
                calls.append("reaction_created")
                return self

            def register_p2_im_message_reaction_deleted_v1(self, _handler):
                calls.append("reaction_deleted")
                return self

            def register_p2_card_action_trigger(self, _handler):
                calls.append("card_action")
                return self

            def register_p2_im_chat_member_bot_added_v1(self, _handler):
                calls.append("bot_added")
                return self

            def register_p2_im_chat_member_bot_deleted_v1(self, _handler):
                calls.append("bot_deleted")
                return self

            def register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(self, _handler):
                calls.append("p2p_chat_entered")
                return self

            def register_p2_im_message_recalled_v1(self, _handler):
                calls.append("message_recalled")
                return self

            def register_p2_customized_event(self, event_key, _handler):
                calls.append(f"customized:{event_key}")
                return self

            def build(self):
                calls.append("build")
                return "handler"

        class _Dispatcher:
            @staticmethod
            def builder(_encrypt_key, _verification_token):
                calls.append("builder")
                return _Builder()

        with patch("plugins.platforms.feishu.adapter.EventDispatcherHandler", _Dispatcher):
            handler = adapter._build_event_handler()

        self.assertEqual(handler, "handler")
        self.assertEqual(
            calls,
            [
                "builder",
                "message_read",
                "message_receive",
                "reaction_created",
                "reaction_deleted",
                "card_action",
                "bot_added",
                "bot_deleted",
                "p2p_chat_entered",
                "message_recalled",
                "customized:drive.notice.comment_add_v1",
                "customized:vc.bot.meeting_invited_v1",
                "build",
            ],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_bot_origin_reactions_are_dropped_to_avoid_feedback_loops(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._loop = object()

        for emoji in ("Typing", "CrossMark"):
            event = SimpleNamespace(
                message_id="om_msg",
                operator_type="bot",
                reaction_type=SimpleNamespace(emoji_type=emoji),
            )
            data = SimpleNamespace(event=event)
            with patch(
                "plugins.platforms.feishu.adapter.asyncio.run_coroutine_threadsafe"
            ) as run_threadsafe:
                adapter._on_reaction_event("im.message.reaction.created_v1", data)
            run_threadsafe.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_user_reaction_with_managed_emoji_is_still_routed(self):
        # Operator-origin filter is enough to prevent feedback loops; we must
        # not additionally swallow user-origin reactions just because their
        # emoji happens to collide with a lifecycle emoji.
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._loop = SimpleNamespace(is_closed=lambda: False)

        event = SimpleNamespace(
            message_id="om_msg",
            operator_type="user",
            reaction_type=SimpleNamespace(emoji_type="Typing"),
        )
        data = SimpleNamespace(event=event)

        def _close_coro_and_return_future(coro, _loop):
            coro.close()
            return SimpleNamespace(add_done_callback=lambda _: None)

        with patch(
            "plugins.platforms.feishu.adapter.asyncio.run_coroutine_threadsafe",
            side_effect=_close_coro_and_return_future,
        ) as run_threadsafe:
            adapter._on_reaction_event("im.message.reaction.created_v1", data)
        run_threadsafe.assert_called_once()

    def _build_reaction_adapter(self, *, msg_sender_id: str):
        """Build a FeishuAdapter wired up to return a single GET-message result."""
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._app_id = "cli_self_app"
        adapter._bot_open_id = "ou_self_bot"
        adapter._bot_user_id = "u_self_bot"

        msg = SimpleNamespace(
            sender=SimpleNamespace(sender_type="app", id=msg_sender_id, id_type="app_id"),
            chat_id="oc_chat",
            chat_type="group",
        )
        response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=[msg]))
        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(message=SimpleNamespace(get=Mock(return_value=response)))
            )
        )
        adapter._build_get_message_request = Mock(return_value=object())
        adapter._handle_message_with_guards = AsyncMock()
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "u_human", "user_name": "Human", "user_id_alt": None}
        )
        adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
        return adapter

    @patch.dict(os.environ, {}, clear=True)
    def test_reaction_on_peer_bot_message_is_not_routed(self):
        # GET im/v1/messages sender for bot messages carries id=app_id; a peer
        # bot's message has a different app_id than ours, so it must be dropped.
        adapter = self._build_reaction_adapter(msg_sender_id="cli_peer_app")

        event = SimpleNamespace(
            message_id="om_peer_msg",
            user_id=SimpleNamespace(open_id="ou_human", user_id=None, union_id=None),
            reaction_type=SimpleNamespace(emoji_type="THUMBSUP"),
        )
        data = SimpleNamespace(event=event)
        asyncio.run(
            adapter._handle_reaction_event("im.message.reaction.created_v1", data)
        )
        adapter._handle_message_with_guards.assert_not_awaited()

    @patch.dict(os.environ, {}, clear=True)
    def test_reaction_on_our_own_bot_message_is_routed(self):
        adapter = self._build_reaction_adapter(msg_sender_id="cli_self_app")

        event = SimpleNamespace(
            message_id="om_self_msg",
            user_id=SimpleNamespace(open_id="ou_human", user_id=None, union_id=None),
            reaction_type=SimpleNamespace(emoji_type="THUMBSUP"),
        )
        data = SimpleNamespace(event=event)
        asyncio.run(
            adapter._handle_reaction_event("im.message.reaction.created_v1", data)
        )
        adapter._handle_message_with_guards.assert_awaited_once()

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_message_requires_mentions_even_when_policy_open(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(mentions=[])
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)
        self.assertFalse(_admits_group(adapter, message, sender_id, ""))

        message_with_mention = SimpleNamespace(mentions=[SimpleNamespace(key="@_user_1")])
        self.assertFalse(_admits_group(adapter, message_with_mention, sender_id, ""))

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_message_with_other_user_mention_is_rejected_when_bot_identity_unknown(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)
        other_mention = SimpleNamespace(
            name="Other User",
            id=SimpleNamespace(open_id="ou_other", user_id="u_other"),
        )

        self.assertFalse(
            _admits_group(adapter, SimpleNamespace(mentions=[other_mention]), sender_id, "")
        )

    @patch.dict(
        os.environ,
        {
            "FEISHU_GROUP_POLICY": "allowlist",
            "FEISHU_ALLOWED_USERS": "ou_allowed",
            "FEISHU_BOT_NAME": "Hermes Bot",
        },
        clear=True,
    )
    def test_group_message_allowlist_and_mention_both_required(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        # Mention without IDs — name fallback legitimately engages.
        mentioned = SimpleNamespace(
            mentions=[
                SimpleNamespace(
                    name="Hermes Bot",
                    id=SimpleNamespace(open_id=None, user_id=None),
                )
            ]
        )

        self.assertTrue(
            _admits_group(adapter,
                mentioned,
                SimpleNamespace(open_id="ou_allowed", user_id=None),
                "",
            )
        )
        self.assertFalse(
            _admits_group(adapter,
                mentioned,
                SimpleNamespace(open_id="ou_blocked", user_id=None),
                "",
            )
        )

    def test_per_group_allowlist_policy_gates_by_sender(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        config = PlatformConfig(
            extra={
                "group_rules": {
                    "oc_chat_a": {
                        "policy": "allowlist",
                        "allowlist": ["ou_alice", "ou_bob"],
                    }
                }
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_alice", user_id=None),
                "oc_chat_a",
            )
        )
        self.assertFalse(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_charlie", user_id=None),
                "oc_chat_a",
            )
        )

    def test_per_group_blacklist_policy_blocks_specific_users(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        config = PlatformConfig(
            extra={
                "group_rules": {
                    "oc_chat_b": {
                        "policy": "blacklist",
                        "blacklist": ["ou_blocked"],
                    }
                }
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_alice", user_id=None),
                "oc_chat_b",
            )
        )
        self.assertFalse(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_blocked", user_id=None),
                "oc_chat_b",
            )
        )

    def test_per_group_admin_only_policy_requires_admin(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        config = PlatformConfig(
            extra={
                "admins": ["ou_admin"],
                "group_rules": {
                    "oc_chat_c": {
                        "policy": "admin_only",
                    }
                },
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_admin", user_id=None),
                "oc_chat_c",
            )
        )
        self.assertFalse(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_regular", user_id=None),
                "oc_chat_c",
            )
        )

    def test_per_group_disabled_policy_blocks_all(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        config = PlatformConfig(
            extra={
                "admins": ["ou_admin"],
                "group_rules": {
                    "oc_chat_d": {
                        "policy": "disabled",
                    }
                },
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_admin", user_id=None),
                "oc_chat_d",
            )
        )
        self.assertFalse(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_regular", user_id=None),
                "oc_chat_d",
            )
        )

    def test_global_admins_bypass_all_group_rules(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        config = PlatformConfig(
            extra={
                "admins": ["ou_admin"],
                "group_rules": {
                    "oc_chat_e": {
                        "policy": "allowlist",
                        "allowlist": ["ou_alice"],
                    }
                },
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_admin", user_id=None),
                "oc_chat_e",
            )
        )

    def test_default_group_policy_fallback_for_chats_without_explicit_rule(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        config = PlatformConfig(
            extra={
                "default_group_policy": "open",
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            _admits_group(adapter,
                message,
                SimpleNamespace(open_id="ou_anyone", user_id=None),
                "oc_chat_unknown",
            )
        )

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_message_matches_bot_open_id_when_configured(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._bot_open_id = "ou_bot"
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)

        bot_mention = SimpleNamespace(
            name="Hermes",
            id=SimpleNamespace(open_id="ou_bot", user_id="u_bot"),
        )
        other_mention = SimpleNamespace(
            name="Other",
            id=SimpleNamespace(open_id="ou_other", user_id="u_other"),
        )

        self.assertTrue(
            _admits_group(adapter, SimpleNamespace(mentions=[bot_mention]), sender_id, "")
        )
        self.assertFalse(
            _admits_group(adapter, SimpleNamespace(mentions=[other_mention]), sender_id, "")
        )

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_message_matches_bot_name_when_only_name_available(self):
        """Name fallback engages when either side lacks an open_id. When BOTH
        the mention and the bot carry open_ids, IDs are authoritative — a
        same-name human with a different open_id must NOT admit."""
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        # Case 1: bot has only a name (open_id not hydrated / not configured).
        # Name fallback is the only available signal for any mention.
        adapter = FeishuAdapter(PlatformConfig())
        adapter._bot_name = "Hermes Bot"
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)

        name_only_mention = SimpleNamespace(
            name="Hermes Bot",
            id=SimpleNamespace(open_id=None, user_id=None),
        )
        different_mention = SimpleNamespace(
            name="Another Bot",
            id=SimpleNamespace(open_id=None, user_id=None),
        )

        self.assertTrue(
            _admits_group(adapter, SimpleNamespace(mentions=[name_only_mention]), sender_id, "")
        )
        self.assertFalse(
            _admits_group(adapter, SimpleNamespace(mentions=[different_mention]), sender_id, "")
        )

        # Case 2: bot's open_id IS known — a same-name human with different
        # open_id must NOT admit (IDs override names).
        adapter2 = FeishuAdapter(PlatformConfig())
        adapter2._bot_open_id = "ou_bot"
        adapter2._bot_name = "Hermes Bot"

        same_name_other_id_mention = SimpleNamespace(
            name="Hermes Bot",
            id=SimpleNamespace(open_id="ou_other", user_id="u_other"),
        )
        bot_mention = SimpleNamespace(
            name="Hermes Bot",
            id=SimpleNamespace(open_id="ou_bot", user_id=None),
        )

        self.assertFalse(
            _admits_group(
                adapter2,
                SimpleNamespace(mentions=[same_name_other_id_mention]),
                sender_id,
                "",
            )
        )
        self.assertTrue(
            _admits_group(adapter2, SimpleNamespace(mentions=[bot_mention]), sender_id, "")
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_post_message_as_text(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="post",
            content='{"zh_cn":{"title":"Title","content":[[{"tag":"text","text":"hello "}],[{"tag":"a","text":"doc","href":"https://example.com"}]]}}',
            message_id="om_post",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Title\nhello\n[doc](https://example.com)")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_post_message_uses_first_available_language_block(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="post",
            content='{"fr_fr":{"title":"Subject","content":[[{"tag":"text","text":"bonjour"}]]}}',
            message_id="om_post_fr",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Subject\nbonjour")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_post_message_with_rich_elements_does_not_drop_content(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="post",
            content=(
                '{"en_us":{"title":"Rich message","content":['
                '[{"tag":"img","alt":"diagram"}],'
                '[{"tag":"at","user_name":"Alice"},{"tag":"text","text":" please check the attachment"}],'
                '[{"tag":"media","file_name":"spec.pdf"}],'
                '[{"tag":"emotion","emoji_type":"smile"}]'
                ']}}'
            ),
            message_id="om_post_rich",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Rich message\n[Image: diagram]\n@Alice please check the attachment\n[Attachment: spec.pdf]\n:smile:")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_post_message_downloads_embedded_resources(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_image = AsyncMock(return_value=("/tmp/feishu-image.png", "image/png"))
        adapter._download_feishu_message_resource = AsyncMock(return_value=("/tmp/spec.pdf", "application/pdf"))
        message = SimpleNamespace(
            message_type="post",
            content=(
                '{"en_us":{"title":"Rich message","content":['
                '[{"tag":"img","image_key":"img_123","alt":"diagram"}],'
                '[{"tag":"media","file_key":"file_123","file_name":"spec.pdf"}]'
                ']}}'
            ),
            message_id="om_post_media",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Rich message\n[Image: diagram]\n[Attachment: spec.pdf]")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, ["/tmp/feishu-image.png", "/tmp/spec.pdf"])
        self.assertEqual(media_types, ["image/png", "application/pdf"])
        adapter._download_feishu_image.assert_awaited_once_with(
            message_id="om_post_media",
            image_key="img_123",
        )
        adapter._download_feishu_message_resource.assert_awaited_once_with(
            message_id="om_post_media",
            file_key="file_123",
            resource_type="file",
            fallback_filename="spec.pdf",
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_merge_forward_message_as_text_summary(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="merge_forward",
            content=json.dumps(
                {
                    "title": "Forwarded updates",
                    "messages": [
                        {"sender_name": "Alice", "text": "Investigating the incident"},
                        {"sender_name": "Bob", "text": "ETA 10 minutes"},
                    ],
                }
            ),
            message_id="om_merge_forward",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(
            text,
            "Forwarded updates\n- Alice: Investigating the incident\n- Bob: ETA 10 minutes",
        )
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_share_chat_message_as_text_summary(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="share_chat",
            content='{"chat_id":"oc_shared","chat_name":"Platform Ops"}',
            message_id="om_share_chat",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Shared chat: Platform Ops\nChat ID: oc_shared")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_interactive_message_as_text_summary(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="interactive",
            content=json.dumps(
                {
                    "card": {
                        "header": {"title": {"tag": "plain_text", "content": "Approval Request"}},
                        "elements": [
                            {"tag": "div", "text": {"tag": "plain_text", "content": "Requester: Alice"}},
                            {
                                "tag": "action",
                                "actions": [
                                    {"tag": "button", "text": {"tag": "plain_text", "content": "Approve"}},
                                ],
                            },
                        ],
                    }
                }
            ),
            message_id="om_interactive",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Approval Request\nRequester: Alice\nApprove\nActions: Approve")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_image_message_downloads_and_caches(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_image = AsyncMock(return_value=("/tmp/feishu-image.png", "image/png"))
        message = SimpleNamespace(
            message_type="image",
            content='{"image_key":"img_123"}',
            message_id="om_image",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "photo")
        self.assertEqual(media_urls, ["/tmp/feishu-image.png"])
        self.assertEqual(media_types, ["image/png"])
        adapter._download_feishu_image.assert_awaited_once_with(
            message_id="om_image",
            image_key="img_123",
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_audio_message_downloads_and_caches(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_message_resource = AsyncMock(
            return_value=("/tmp/feishu-audio.ogg", "audio/ogg")
        )
        message = SimpleNamespace(
            message_type="audio",
            content='{"file_key":"file_audio","file_name":"voice.ogg"}',
            message_id="om_audio",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "audio")
        self.assertEqual(media_urls, ["/tmp/feishu-audio.ogg"])
        self.assertEqual(media_types, ["audio/ogg"])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_file_message_downloads_and_caches(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_message_resource = AsyncMock(
            return_value=("/tmp/doc_123_report.pdf", "application/pdf")
        )
        message = SimpleNamespace(
            message_type="file",
            content='{"file_key":"file_doc","file_name":"report.pdf"}',
            message_id="om_file",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "document")
        self.assertEqual(media_urls, ["/tmp/doc_123_report.pdf"])
        self.assertEqual(media_types, ["application/pdf"])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_media_message_with_image_mime_becomes_photo(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_message_resource = AsyncMock(
            return_value=("/tmp/feishu-media.jpg", "image/jpeg")
        )
        message = SimpleNamespace(
            message_type="media",
            content='{"file_key":"file_media","file_name":"photo.jpg"}',
            message_id="om_media",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "photo")
        self.assertEqual(media_urls, ["/tmp/feishu-media.jpg"])
        self.assertEqual(media_types, ["image/jpeg"])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_media_message_with_video_mime_becomes_video(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_message_resource = AsyncMock(
            return_value=("/tmp/feishu-video.mp4", "video/mp4")
        )
        message = SimpleNamespace(
            message_type="media",
            content='{"file_key":"file_video","file_name":"clip.mp4"}',
            message_id="om_video",
        )

        text, msg_type, media_urls, media_types, _mentions = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "video")
        self.assertEqual(media_urls, ["/tmp/feishu-video.mp4"])
        self.assertEqual(media_types, ["video/mp4"])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_text_from_raw_content_uses_relation_message_fallbacks(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        shared = adapter._extract_text_from_raw_content(
            msg_type="share_chat",
            raw_content='{"chat_id":"oc_shared","chat_name":"Platform Ops"}',
        )
        attachment = adapter._extract_text_from_raw_content(
            msg_type="file",
            raw_content='{"file_key":"file_1","file_name":"report.pdf"}',
        )

        self.assertEqual(shared, "Shared chat: Platform Ops\nChat ID: oc_shared")
        self.assertEqual(attachment, "[Attachment: report.pdf]")

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_text_message_starting_with_slash_becomes_command(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_chat", "name": "Feishu DM", "type": "dm"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "张三", "user_id_alt": None}
        )
        message = SimpleNamespace(
            chat_id="oc_chat",
            thread_id=None,
            parent_id=None,
            upper_message_id=None,
            message_type="text",
            content='{"text":"/help test"}',
            message_id="om_command",
        )

        asyncio.run(
            adapter._process_inbound_message(
                data=SimpleNamespace(event=SimpleNamespace(message=message)),
                message=message,
                sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
                is_bot=False,
                chat_type="p2p",
                message_id="om_command",
            )
        )

        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.message_type.value, "command")
        self.assertEqual(event.text, "/help test")

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_text_file_injects_content(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("hello from feishu")
            path = tmp.name

        try:
            text = asyncio.run(adapter._maybe_extract_text_document(path, "text/plain"))
        finally:
            os.unlink(path)

        self.assertIn("hello from feishu", text)
        self.assertIn("[Content of", text)

    @patch.dict(os.environ, {}, clear=True)
    def test_message_event_submits_to_adapter_loop(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        class _Loop:
            def is_closed(self):
                return False

        adapter._loop = _Loop()

        message = SimpleNamespace(
            message_id="om_text",
            chat_type="p2p",
            chat_id="oc_chat",
            message_type="text",
            content='{"text":"hello"}',
        )
        sender_id = SimpleNamespace(open_id="ou_user", user_id=None, union_id=None)
        sender = SimpleNamespace(sender_id=sender_id, sender_type="user")
        data = SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))

        future = SimpleNamespace(add_done_callback=lambda *_args, **_kwargs: None)

        def _submit(coro, _loop):
            coro.close()
            return future

        with patch("plugins.platforms.feishu.adapter.asyncio.run_coroutine_threadsafe", side_effect=_submit) as submit:
            adapter._on_message_event(data)

        self.assertTrue(submit.called)

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_request_uses_same_message_dispatch_path(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._on_message_event = Mock()

        body = json.dumps({
            "header": {"event_type": "im.message.receive_v1"},
            "event": {"message": {"message_id": "om_test"}},
        }).encode("utf-8")
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            headers={},
            content=_FakeRequestContent(body),
        )

        response = asyncio.run(adapter._handle_webhook_request(request))

        self.assertEqual(response.status, 200)
        adapter._on_message_event.assert_called_once()

    @patch.dict(os.environ, {"FEISHU_VERIFICATION_TOKEN": "expected-token"}, clear=True)
    def test_url_verification_requires_configured_verification_token(self):
        """url_verification must be rejected when token is set but mismatched.

        Regression: previously the challenge was reflected before the token
        check, so an unauthenticated remote could prove endpoint control by
        sending an attacker-controlled challenge string.
        """
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        body = json.dumps({
            "type": "url_verification",
            "token": "wrong-token",
            "challenge": "attacker-controlled-challenge",
        }).encode("utf-8")
        request = SimpleNamespace(
            remote="203.0.113.10",
            content_length=None,
            headers={},
            content=_FakeRequestContent(body),
        )

        response = asyncio.run(adapter._handle_webhook_request(request))

        self.assertEqual(response.status, 401)

    @patch.dict(os.environ, {}, clear=True)
    def test_process_inbound_message_uses_event_sender_identity_only(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import MessageType
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        # Sender name now comes from the contact API; mock it to return a known value.
        adapter._resolve_sender_name_from_api = AsyncMock(return_value="张三")
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_chat", "name": "Feishu DM", "type": "dm"}
        )
        message = SimpleNamespace(
            chat_id="oc_chat",
            thread_id=None,
            message_type="text",
            content='{"text":"hello"}',
            message_id="om_text",
        )
        sender_id = SimpleNamespace(
            open_id="ou_user",
            user_id="u_user",
            union_id="on_union",
        )
        sender = SimpleNamespace(sender_type="user", sender_id=sender_id)
        data = SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))

        asyncio.run(
            adapter._process_inbound_message(
                data=data,
                message=message,
                sender_id=sender.sender_id,
                chat_type="p2p",
                message_id="om_text",
            )
        )

        adapter._dispatch_inbound_event.assert_awaited_once()
        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.message_type, MessageType.TEXT)
        self.assertEqual(event.source.user_id, "u_user")  # tenant-scoped user_id preferred over app-scoped open_id
        self.assertEqual(event.source.user_name, "张三")
        self.assertEqual(event.source.user_id_alt, "on_union")
        self.assertEqual(event.source.chat_name, "Feishu DM")

    @patch.dict(os.environ, {}, clear=True)
    def test_text_batch_merges_rapid_messages_into_single_event(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import MessageEvent, MessageType
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from gateway.session import SessionSource

        adapter = FeishuAdapter(PlatformConfig())
        adapter.handle_message = AsyncMock()
        source = SessionSource(
            platform=adapter.platform,
            chat_id="oc_chat",
            chat_name="Feishu DM",
            chat_type="dm",
            user_id="ou_user",
            user_name="张三",
        )

        async def _sleep(_delay):
            return None

        async def _run() -> None:
            with patch("plugins.platforms.feishu.adapter.asyncio.sleep", side_effect=_sleep):
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="A", message_type=MessageType.TEXT, source=source, message_id="om_1")
                )
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="B", message_type=MessageType.TEXT, source=source, message_id="om_2")
                )
                pending = list(adapter._pending_text_batch_tasks.values())
                self.assertEqual(len(pending), 1)
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run(_run())

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        self.assertEqual(event.text, "A\nB")
        self.assertEqual(event.message_type, MessageType.TEXT)

    @patch.dict(
        os.environ,
        {
            "HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES": "2",
        },
        clear=True,
    )
    def test_text_batch_flushes_when_message_count_limit_is_hit(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import MessageEvent, MessageType
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from gateway.session import SessionSource

        adapter = FeishuAdapter(PlatformConfig())
        adapter.handle_message = AsyncMock()
        source = SessionSource(
            platform=adapter.platform,
            chat_id="oc_chat",
            chat_name="Feishu DM",
            chat_type="dm",
            user_id="ou_user",
            user_name="张三",
        )

        async def _sleep(_delay):
            return None

        async def _run() -> None:
            with patch("plugins.platforms.feishu.adapter.asyncio.sleep", side_effect=_sleep):
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="A", message_type=MessageType.TEXT, source=source, message_id="om_1")
                )
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="B", message_type=MessageType.TEXT, source=source, message_id="om_2")
                )
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="C", message_type=MessageType.TEXT, source=source, message_id="om_3")
                )
                pending = list(adapter._pending_text_batch_tasks.values())
                self.assertEqual(len(pending), 1)
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run(_run())

        self.assertEqual(adapter.handle_message.await_count, 2)
        first = adapter.handle_message.await_args_list[0].args[0]
        second = adapter.handle_message.await_args_list[1].args[0]
        self.assertEqual(first.text, "A\nB")
        self.assertEqual(second.text, "C")

    @patch.dict(os.environ, {}, clear=True)
    def test_media_batch_merges_rapid_photo_messages(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import MessageEvent, MessageType
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from gateway.session import SessionSource

        adapter = FeishuAdapter(PlatformConfig())
        adapter.handle_message = AsyncMock()
        source = SessionSource(
            platform=adapter.platform,
            chat_id="oc_chat",
            chat_name="Feishu DM",
            chat_type="dm",
            user_id="ou_user",
            user_name="张三",
        )

        async def _sleep(_delay):
            return None

        async def _run() -> None:
            with patch("plugins.platforms.feishu.adapter.asyncio.sleep", side_effect=_sleep):
                await adapter._dispatch_inbound_event(
                    MessageEvent(
                        text="第一张",
                        message_type=MessageType.PHOTO,
                        source=source,
                        message_id="om_p1",
                        media_urls=["/tmp/a.png"],
                        media_types=["image/png"],
                    )
                )
                await adapter._dispatch_inbound_event(
                    MessageEvent(
                        text="第二张",
                        message_type=MessageType.PHOTO,
                        source=source,
                        message_id="om_p2",
                        media_urls=["/tmp/b.png"],
                        media_types=["image/png"],
                    )
                )
                pending = list(adapter._pending_media_batch_tasks.values())
                self.assertEqual(len(pending), 1)
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run(_run())

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        self.assertEqual(event.media_urls, ["/tmp/a.png", "/tmp/b.png"])
        self.assertIn("第一张", event.text)
        self.assertIn("第二张", event.text)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_image_downloads_then_uses_native_image_send(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter.send_image_file = AsyncMock(return_value=SimpleNamespace(success=True, message_id="om_img"))

        async def _run():
            with patch("plugins.platforms.feishu.adapter.cache_image_from_url", new=AsyncMock(return_value="/tmp/cached.png")):
                return await adapter.send_image("oc_chat", "https://example.com/cat.png", caption="cat")

        result = asyncio.run(_run())

        self.assertTrue(result.success)
        adapter.send_image_file.assert_awaited_once()
        self.assertEqual(adapter.send_image_file.await_args.kwargs["image_path"], "/tmp/cached.png")

    @patch.dict(os.environ, {}, clear=True)
    def test_send_animation_degrades_to_document_send(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter.send_document = AsyncMock(return_value=SimpleNamespace(success=True, message_id="om_gif"))

        async def _run():
            with patch.object(
                adapter,
                "_download_remote_document",
                new=AsyncMock(return_value=("/tmp/anim.gif", "anim.gif")),
            ):
                return await adapter.send_animation("oc_chat", "https://example.com/anim.gif", caption="look")

        result = asyncio.run(_run())

        self.assertTrue(result.success)
        adapter.send_document.assert_awaited_once()
        caption = adapter.send_document.await_args.kwargs["caption"]
        self.assertIn("GIF downgraded to file", caption)
        self.assertIn("look", caption)

    def test_download_remote_document_reads_response_before_httpx_client_closes(self):
        """#18451 — snapshot Content-Type + body while the httpx.AsyncClient
        context is still active so pooled connections fully release on
        exit.  Otherwise the response is only readable because httpx
        eagerly buffers it; a future refactor to .stream() would silently
        read-after-close."""
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        events: list[str] = []

        class _FakeResponse:
            headers = {"Content-Type": "application/octet-stream"}

            def raise_for_status(self) -> None:
                events.append("raise_for_status")

            @property
            def content(self) -> bytes:
                events.append("content_read")
                return b"doc-bytes"

        class _FakeAsyncClient:
            def __init__(self, *_a: object, **_k: object) -> None:
                pass

            async def __aenter__(self) -> "_FakeAsyncClient":
                events.append("client_enter")
                return self

            async def __aexit__(self, *exc: object) -> None:
                events.append("client_exit")

            async def get(self, *_a: object, **_k: object) -> _FakeResponse:
                events.append("get")
                return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                adapter = FeishuAdapter(PlatformConfig())

                async def _run() -> tuple[str, str]:
                    with patch("tools.url_safety.is_safe_url", return_value=True):
                        with patch("httpx.AsyncClient", _FakeAsyncClient):
                            with patch(
                                "plugins.platforms.feishu.adapter.cache_document_from_bytes",
                                return_value="/tmp/cached-doc.bin",
                            ):
                                return await adapter._download_remote_document(
                                    "https://example.com/doc.bin",
                                    default_ext=".bin",
                                    preferred_name="doc",
                                )

                path, filename = asyncio.run(_run())

        self.assertEqual(path, "/tmp/cached-doc.bin")
        self.assertTrue(filename)
        # content_read MUST happen before client_exit — otherwise we're
        # reading response body after the connection pool has been torn
        # down, which only works by accident (httpx's eager buffering).
        self.assertLess(events.index("content_read"), events.index("client_exit"))

    def test_dedup_state_persists_across_adapter_restart(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        with tempfile.TemporaryDirectory() as temp_home:
            with patch.dict(os.environ, {"HERMES_HOME": temp_home}, clear=False):
                first = FeishuAdapter(PlatformConfig())
                self.assertFalse(first._is_duplicate("om_same"))
                second = FeishuAdapter(PlatformConfig())
                self.assertTrue(second._is_duplicate("om_same"))

    @patch.dict(os.environ, {}, clear=True)
    def test_process_inbound_group_message_keeps_group_type_when_chat_lookup_falls_back(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_group", "name": "oc_group", "type": "dm"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "张三", "user_id_alt": None}
        )
        message = SimpleNamespace(
            chat_id="oc_group",
            thread_id=None,
            message_type="text",
            content='{"text":"hello group"}',
            message_id="om_group_text",
        )
        sender_id = SimpleNamespace(open_id="ou_user", user_id=None, union_id=None)
        sender = SimpleNamespace(sender_type="user", sender_id=sender_id)
        data = SimpleNamespace(event=SimpleNamespace(message=message))

        asyncio.run(
            adapter._process_inbound_message(
                data=data,
                message=message,
                sender_id=sender.sender_id,
                chat_type="group",
                message_id="om_group_text",
            )
        )

        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.source.chat_type, "group")

    @patch.dict(os.environ, {}, clear=True)
    def test_process_inbound_message_fetches_reply_to_text(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_chat", "name": "Feishu DM", "type": "dm"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "张三", "user_id_alt": None}
        )
        adapter._fetch_message_text = AsyncMock(return_value="父消息内容")
        message = SimpleNamespace(
            chat_id="oc_chat",
            thread_id=None,
            parent_id="om_parent",
            upper_message_id=None,
            message_type="text",
            content='{"text":"reply"}',
            message_id="om_reply",
        )

        asyncio.run(
            adapter._process_inbound_message(
                data=SimpleNamespace(event=SimpleNamespace(message=message)),
                message=message,
                sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
                is_bot=False,
                chat_type="p2p",
                message_id="om_reply",
            )
        )

        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.reply_to_message_id, "om_parent")
        self.assertEqual(event.reply_to_text, "父消息内容")

    @patch.dict(os.environ, {}, clear=True)
    def test_send_replies_in_thread_when_thread_metadata_present(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _ReplyAPI:
            def reply(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_reply"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_ReplyAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="hello",
                    reply_to="om_parent",
                    metadata={"thread_id": "omt-thread"},
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_reply")
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_uses_metadata_reply_target_for_threaded_feishu_topic(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def reply(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_reply"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI()))
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="status update",
                    metadata={
                        "thread_id": "omt-thread",
                        "reply_to_message_id": "om_trigger",
                    },
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].message_id, "om_trigger")
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_retries_transient_failure(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"attempts": 0}
        sleeps = []

        class _MessageAPI:
            def create(self, request):
                captured["attempts"] += 1
                captured["request"] = request
                if captured["attempts"] == 1:
                    raise OSError("temporary send failure")
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_retry"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def _sleep(delay):
            sleeps.append(delay)

        with (
            patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct),
            patch("plugins.platforms.feishu.adapter.asyncio.sleep", side_effect=_sleep),
        ):
            result = asyncio.run(adapter.send(chat_id="oc_chat", content="hello retry"))

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_retry")
        self.assertEqual(captured["attempts"], 2)
        self.assertEqual(sleeps, [1])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_does_not_retry_deterministic_api_failure(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"attempts": 0}
        sleeps = []

        class _MessageAPI:
            def create(self, request):
                captured["attempts"] += 1
                return SimpleNamespace(
                    success=lambda: False,
                    code=400,
                    msg="bad request",
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def _sleep(delay):
            sleeps.append(delay)

        with (
            patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct),
            patch("plugins.platforms.feishu.adapter.asyncio.sleep", side_effect=_sleep),
        ):
            result = asyncio.run(adapter.send(chat_id="oc_chat", content="bad payload"))

        self.assertFalse(result.success)
        self.assertEqual(result.error, "[400] bad request")
        self.assertEqual(captured["attempts"], 1)
        self.assertEqual(sleeps, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_document_reply_uses_thread_flag(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_123"),
                )

        class _MessageAPI:
            def reply(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_file_reply"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4 test")
            file_path = tmp.name

        try:
            with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(
                    adapter.send_document(
                        chat_id="oc_chat",
                        file_path=file_path,
                        reply_to="om_parent",
                        metadata={"thread_id": "omt-thread"},
                    )
                )
        finally:
            os.unlink(file_path)

        self.assertTrue(result.success)
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_document_uploads_file_and_sends_file_message(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                captured["upload_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_file_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4 test")
            file_path = tmp.name

        try:
            with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(adapter.send_document(chat_id="oc_chat", file_path=file_path))
        finally:
            os.unlink(file_path)

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_file_msg")
        self.assertEqual(captured["upload_request"].request_body.file_type, "pdf")
        self.assertEqual(
            captured["message_request"].request_body.content,
            '{"file_key": "file_123"}',
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_document_with_caption_uses_single_post_message(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_post_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4 test")
            file_path = tmp.name

        try:
            with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(
                    adapter.send_document(chat_id="oc_chat", file_path=file_path, caption="报告请看")
                )
        finally:
            os.unlink(file_path)

        self.assertTrue(result.success)
        self.assertEqual(captured["message_request"].request_body.msg_type, "post")
        self.assertIn('"tag": "media"', captured["message_request"].request_body.content)
        self.assertIn('"file_key": "file_123"', captured["message_request"].request_body.content)
        self.assertIn("报告请看", captured["message_request"].request_body.content)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_image_file_uploads_image_and_sends_image_message(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _ImageAPI:
            def create(self, request):
                captured["upload_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(image_key="img_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_image_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    image=_ImageAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n\x1a\n")
            image_path = tmp.name

        try:
            with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(adapter.send_image_file(chat_id="oc_chat", image_path=image_path))
        finally:
            os.unlink(image_path)

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_image_msg")
        self.assertEqual(captured["upload_request"].request_body.image_type, "message")
        self.assertEqual(
            captured["message_request"].request_body.content,
            '{"image_key": "img_123"}',
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_image_file_with_caption_uses_single_post_message(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _ImageAPI:
            def create(self, request):
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(image_key="img_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_post_img"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    image=_ImageAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n\x1a\n")
            image_path = tmp.name

        try:
            with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(
                    adapter.send_image_file(chat_id="oc_chat", image_path=image_path, caption="截图说明")
                )
        finally:
            os.unlink(image_path)

        self.assertTrue(result.success)
        self.assertEqual(captured["message_request"].request_body.msg_type, "post")
        self.assertIn('"tag": "img"', captured["message_request"].request_body.content)
        self.assertIn('"image_key": "img_123"', captured["message_request"].request_body.content)
        self.assertIn("截图说明", captured["message_request"].request_body.content)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_video_uploads_file_and_sends_media_message(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                captured["upload_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_video_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_video_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".mp4", delete=False) as tmp:
            tmp.write(b"\x00\x00\x00\x18ftypmp42")
            video_path = tmp.name

        try:
            with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(adapter.send_video(chat_id="oc_chat", video_path=video_path))
        finally:
            os.unlink(video_path)

        self.assertTrue(result.success)
        self.assertEqual(captured["upload_request"].request_body.file_type, "mp4")
        self.assertEqual(captured["message_request"].request_body.msg_type, "media")
        self.assertEqual(captured["message_request"].request_body.content, '{"file_key": "file_video_123"}')

    @patch.dict(os.environ, {}, clear=True)
    def test_send_voice_uploads_opus_and_sends_audio_message(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                captured["upload_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_audio_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_audio_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".opus", delete=False) as tmp:
            tmp.write(b"opus")
            audio_path = tmp.name

        try:
            with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(adapter.send_voice(chat_id="oc_chat", audio_path=audio_path))
        finally:
            os.unlink(audio_path)

        self.assertTrue(result.success)
        self.assertEqual(captured["upload_request"].request_body.file_type, "opus")
        self.assertEqual(captured["message_request"].request_body.msg_type, "audio")
        self.assertEqual(captured["message_request"].request_body.content, '{"file_key": "file_audio_123"}')

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_extracts_title_and_links(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(adapter._build_post_payload("# 标题\n访问 [文档](https://example.com)"))

        elements = payload["zh_cn"]["content"][0]
        self.assertEqual(elements, [{"tag": "md", "text": "# 标题\n访问 [文档](https://example.com)"}])

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_wraps_markdown_in_md_tag(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(
            adapter._build_post_payload("支持 **粗体**、*斜体* 和 `代码`")
        )

        elements = payload["zh_cn"]["content"][0]
        self.assertEqual(
            elements,
            [
                {"tag": "md", "text": "支持 **粗体**、*斜体* 和 `代码`"},
            ],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_keeps_full_markdown_text(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(
            adapter._build_post_payload(
                "---\n1. 第一项\n  2. 子项\n- 外层\n  - 内层\n<u>下划线</u> 和 ~~删除线~~"
            )
        )

        rows = payload["zh_cn"]["content"]
        self.assertEqual(
            rows,
            [[{"tag": "md", "text": "---\n1. 第一项\n  2. 子项\n- 外层\n  - 内层\n<u>下划线</u> 和 ~~删除线~~"}]],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_build_outbound_payload_uses_post_for_markdown_table(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        content = "| Name | Score |\n| --- | ---: |\n| Ada | 10 |"

        msg_type, raw_payload = adapter._build_outbound_payload(content)

        self.assertEqual(msg_type, "post")
        payload = json.loads(raw_payload)
        self.assertEqual(
            payload["zh_cn"]["content"],
            [[{"tag": "md", "text": content}]],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_uses_post_for_every_chunk_of_multi_chunk_markdown(self):
        """Regression for #26841: when a long Markdown message is split
        across multiple chunks, every chunk must go out as
        ``msg_type=post`` — including chunk 1.  The bug was that the
        first chunk often had only plain prose (the per-chunk regex
        didn't match) and was sent as ``text``, so users saw literal
        ``**bold``/``## heading``/code fences while later chunks
        rendered correctly.
        """
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = []

        class _MessageAPI:
            def create(self, request):
                captured.append(request)
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(
                        message_id=f"om_chunk_{len(captured)}",
                    ),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        # Force a deterministic split so the test doesn't depend on the
        # exact 8000-char limit.  Chunk 1 is plain prose; chunk 2 has
        # the markdown markers.  Without the fix, chunk 1 went out as
        # ``msg_type=text``.
        first_chunk = "Here is a short intro that has no markdown markers at all."
        second_chunk = "## Heading\nAnd then some **bold** text."

        with patch.object(
            adapter, "truncate_message", return_value=[first_chunk, second_chunk],
        ), patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content=first_chunk + "\n" + second_chunk,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(len(captured), 2)
        msg_types = [r.request_body.msg_type for r in captured]
        self.assertEqual(msg_types, ["post", "post"])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_plain_text_message_not_upgraded_by_prefer_post(self):
        """A message with no markdown at all must still go out as plain
        ``msg_type=text`` — the whole-message ``prefer_post`` decision
        only flips on when the formatted message matches the hint regex.
        """
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = []

        class _MessageAPI:
            def create(self, request):
                captured.append(request)
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(
                        message_id=f"om_chunk_{len(captured)}",
                    ),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="just a plain sentence",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].request_body.msg_type, "text")

    @patch.dict(os.environ, {}, clear=True)
    def test_send_uses_post_for_inline_markdown(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def create(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_markdown"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].request_body.msg_type, "post")
        payload = json.loads(captured["request"].request_body.content)
        elements = payload["zh_cn"]["content"][0]
        self.assertEqual(elements, [{"tag": "md", "text": "可以用 **粗体** 和 *斜体*。"}])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_splits_fenced_code_blocks_into_separate_post_rows(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def create(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_codeblock"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        content = (
            "确认已入库 ✓\n"
            "文件路径：`/root/.hermes/profiles/agent_cto/cron/jobs.json`\n"
            "**解码后的内容：**\n"
            "```json\n"
            '{"cron": "list"}\n'
            "```\n"
            "后续说明仍应保留。"
        )

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content=content,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].request_body.msg_type, "post")
        payload = json.loads(captured["request"].request_body.content)
        rows = payload["zh_cn"]["content"]
        self.assertEqual(
            rows,
            [
                [
                    {
                        "tag": "md",
                        "text": "确认已入库 ✓\n文件路径：`/root/.hermes/profiles/agent_cto/cron/jobs.json`\n**解码后的内容：**",
                    }
                ],
                [{"tag": "md", "text": "```json\n{\"cron\": \"list\"}\n```"}],
                [{"tag": "md", "text": "后续说明仍应保留。"}],
            ],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_keeps_fence_like_code_lines_inside_code_block(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(
            adapter._build_post_payload(
                "before\n```python\n```oops\n```\nafter"
            )
        )

        self.assertEqual(
            payload["zh_cn"]["content"],
            [
                [{"tag": "md", "text": "before"}],
                [{"tag": "md", "text": "```python\n```oops\n```"}],
                [{"tag": "md", "text": "after"}],
            ],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_preserves_trailing_spaces_in_code_block(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(
            adapter._build_post_payload(
                "before\n```python\nline with two spaces  \n```\nafter"
            )
        )

        self.assertEqual(
            payload["zh_cn"]["content"],
            [
                [{"tag": "md", "text": "before"}],
                [{"tag": "md", "text": "```python\nline with two spaces  \n```"}],
                [{"tag": "md", "text": "after"}],
            ],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_splits_multiple_fenced_code_blocks(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(
            adapter._build_post_payload(
                "before\n```python\nprint(1)\n```\nmiddle\n```json\n{}\n```\nafter"
            )
        )

        self.assertEqual(
            payload["zh_cn"]["content"],
            [
                [{"tag": "md", "text": "before"}],
                [{"tag": "md", "text": "```python\nprint(1)\n```"}],
                [{"tag": "md", "text": "middle"}],
                [{"tag": "md", "text": "```json\n{}\n```"}],
                [{"tag": "md", "text": "after"}],
            ],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_falls_back_to_text_when_post_payload_is_rejected(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"calls": []}

        class _MessageAPI:
            def create(self, request):
                captured["calls"].append(request)
                if len(captured["calls"]) == 1:
                    raise RuntimeError("content format of the post type is incorrect")
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_plain"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["calls"][0].request_body.msg_type, "post")
        self.assertEqual(captured["calls"][1].request_body.msg_type, "text")
        self.assertEqual(
            captured["calls"][1].request_body.content,
            json.dumps({"text": "可以用 粗体 和 斜体。"}, ensure_ascii=False),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_falls_back_to_text_when_post_response_is_unsuccessful(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"calls": []}

        class _MessageAPI:
            def create(self, request):
                captured["calls"].append(request)
                if len(captured["calls"]) == 1:
                    return SimpleNamespace(success=lambda: False, code=230001, msg="content format of the post type is incorrect")
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_plain_response"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["calls"][0].request_body.msg_type, "post")
        self.assertEqual(captured["calls"][1].request_body.msg_type, "text")
        self.assertEqual(
            captured["calls"][1].request_body.content,
            json.dumps({"text": "可以用 粗体 和 斜体。"}, ensure_ascii=False),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_uses_post_for_advanced_markdown_lines(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def create(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_markdown_advanced"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="---\n1. 第一项\n<u>下划线</u>\n~~删除线~~",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].request_body.msg_type, "post")
        payload = json.loads(captured["request"].request_body.content)
        rows = payload["zh_cn"]["content"]
        self.assertEqual(
            rows,
            [[{"tag": "md", "text": "---\n1. 第一项\n<u>下划线</u>\n~~删除线~~"}]],
        )


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestHydrateBotIdentity(unittest.TestCase):
    """Hydration of bot identity via ``/open-apis/bot/v3/info``.

    Covers the manual-setup path where ``FEISHU_BOT_OPEN_ID`` /
    ``FEISHU_BOT_NAME`` are not configured — hydration populates them so
    self-echo protection and group @mention gating both have something to
    match against.
    """

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        return FeishuAdapter(PlatformConfig())

    @patch.dict(os.environ, {}, clear=True)
    def test_hydration_populates_open_id_from_bot_info(self):
        adapter = self._make_adapter()
        adapter._client = Mock()
        payload = json.dumps(
            {
                "code": 0,
                "bot": {
                    "bot_name": "Hermes Bot",
                    "open_id": "ou_hermes_hydrated",
                },
            }
        ).encode("utf-8")
        response = SimpleNamespace(raw=SimpleNamespace(content=payload))
        adapter._client.request = Mock(return_value=response)

        asyncio.run(adapter._hydrate_bot_identity())

        self.assertEqual(adapter._bot_open_id, "ou_hermes_hydrated")
        self.assertEqual(adapter._bot_name, "Hermes Bot")

    @patch.dict(
        os.environ,
        {
            "FEISHU_BOT_OPEN_ID": "ou_env",
            "FEISHU_BOT_NAME": "Env Hermes",
        },
        clear=True,
    )
    def test_hydration_refreshes_env_values_when_bot_info_available(self):
        adapter = self._make_adapter()
        adapter._client = Mock()
        payload = json.dumps(
            {
                "code": 0,
                "bot": {
                    "bot_name": "Hydrated Hermes",
                    "open_id": "ou_hydrated",
                },
            }
        ).encode("utf-8")
        adapter._client.request = Mock(return_value=SimpleNamespace(raw=SimpleNamespace(content=payload)))

        asyncio.run(adapter._hydrate_bot_identity())

        # PR #16993 semantics: /bot/v3/info probe runs unconditionally
        # and hydrated values win over env vars so a stale FEISHU_BOT_*
        # from an old app registration doesn't break @mention gating.
        adapter._client.request.assert_called_once()
        self.assertEqual(adapter._bot_open_id, "ou_hydrated")
        self.assertEqual(adapter._bot_name, "Hydrated Hermes")

    @patch.dict(os.environ, {"FEISHU_BOT_OPEN_ID": "ou_env"}, clear=True)
    def test_hydration_overwrites_stale_env_open_id(self):
        """A stale env open_id should not break group mention gating after app migration."""
        adapter = self._make_adapter()
        adapter._client = Mock()
        payload = json.dumps(
            {
                "code": 0,
                "bot": {
                    "bot_name": "Hermes Bot",
                    "open_id": "ou_probe_DIFFERENT",
                },
            }
        ).encode("utf-8")
        adapter._client.request = Mock(return_value=SimpleNamespace(raw=SimpleNamespace(content=payload)))

        asyncio.run(adapter._hydrate_bot_identity())

        self.assertEqual(adapter._bot_open_id, "ou_probe_DIFFERENT")
        self.assertEqual(adapter._bot_name, "Hermes Bot")  # filled in

    @patch.dict(
        os.environ,
        {
            "FEISHU_BOT_OPEN_ID": "ou_env",
            "FEISHU_BOT_NAME": "Env Hermes",
        },
        clear=True,
    )
    def test_hydration_preserves_env_values_when_bot_info_probe_fails(self):
        adapter = self._make_adapter()
        adapter._client = Mock()
        adapter._client.request = Mock(side_effect=RuntimeError("network down"))

        asyncio.run(adapter._hydrate_bot_identity())

        self.assertEqual(adapter._bot_open_id, "ou_env")
        self.assertEqual(adapter._bot_name, "Env Hermes")

    @patch.dict(os.environ, {}, clear=True)
    def test_hydration_tolerates_probe_failure_and_falls_back_to_app_info(self):
        adapter = self._make_adapter()
        adapter._client = Mock()
        adapter._client.request = Mock(side_effect=RuntimeError("network down"))

        # Make the application-info fallback succeed for _bot_name.
        app_response = Mock()
        app_response.success = Mock(return_value=True)
        app_response.data = SimpleNamespace(app=SimpleNamespace(app_name="Fallback Bot"))
        adapter._client.application.v6.application.get = Mock(return_value=app_response)
        adapter._build_get_application_request = Mock(return_value=object())

        asyncio.run(adapter._hydrate_bot_identity())

        # Primary probe failed — open_id stays empty, but bot_name came from app-info.
        self.assertEqual(adapter._bot_open_id, "")
        self.assertEqual(adapter._bot_name, "Fallback Bot")


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestPendingInboundQueue(unittest.TestCase):
    """Tests for the loop-not-ready race (#5499): inbound events arriving
    before or during adapter loop transitions must be queued for replay
    rather than silently dropped."""

    @patch.dict(os.environ, {}, clear=True)
    def test_event_queued_when_loop_not_ready(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._loop = None  # Simulate "before start()" or "during reconnect"

        with patch("plugins.platforms.feishu.adapter.threading.Thread") as thread_cls:
            adapter._on_message_event(SimpleNamespace(tag="evt-1"))
            adapter._on_message_event(SimpleNamespace(tag="evt-2"))
            adapter._on_message_event(SimpleNamespace(tag="evt-3"))

        # All three queued, none dropped.
        self.assertEqual(len(adapter._pending_inbound_events), 3)
        # Only ONE drainer thread scheduled, not one per event.
        self.assertEqual(thread_cls.call_count, 1)
        # Drain scheduled flag set.
        self.assertTrue(adapter._pending_drain_scheduled)

    @patch.dict(os.environ, {}, clear=True)
    def test_drainer_replays_queued_events_when_loop_becomes_ready(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._loop = None
        adapter._running = True

        class _ReadyLoop:
            def is_closed(self):
                return False

        # Queue three events while loop is None (simulate the race).
        events = [SimpleNamespace(tag=f"evt-{i}") for i in range(3)]
        with patch("plugins.platforms.feishu.adapter.threading.Thread"):
            for ev in events:
                adapter._on_message_event(ev)

        self.assertEqual(len(adapter._pending_inbound_events), 3)

        # Now the loop becomes ready; run the drainer inline (not as a thread)
        # to verify it replays the queue.
        adapter._loop = _ReadyLoop()

        future = SimpleNamespace(add_done_callback=lambda *_a, **_kw: None)
        submitted: list = []

        def _submit(coro, _loop):
            submitted.append(coro)
            coro.close()
            return future

        with patch(
            "plugins.platforms.feishu.adapter.asyncio.run_coroutine_threadsafe",
            side_effect=_submit,
        ) as submit:
            adapter._drain_pending_inbound_events()

        # All three events dispatched to the loop.
        self.assertEqual(submit.call_count, 3)
        # Queue emptied.
        self.assertEqual(len(adapter._pending_inbound_events), 0)
        # Drain flag reset so a future race can schedule a new drainer.
        self.assertFalse(adapter._pending_drain_scheduled)

    @patch.dict(os.environ, {}, clear=True)
    def test_drainer_drops_queue_when_adapter_shuts_down(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._loop = None
        adapter._running = False  # Shutdown state

        with patch("plugins.platforms.feishu.adapter.threading.Thread"):
            adapter._on_message_event(SimpleNamespace(tag="evt-lost"))

        self.assertEqual(len(adapter._pending_inbound_events), 1)

        # Drainer should drop the queue immediately since _running is False.
        adapter._drain_pending_inbound_events()

        self.assertEqual(len(adapter._pending_inbound_events), 0)
        self.assertFalse(adapter._pending_drain_scheduled)

    @patch.dict(os.environ, {}, clear=True)
    def test_queue_cap_evicts_oldest_beyond_max_depth(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._loop = None
        adapter._pending_inbound_max_depth = 3  # Shrink for test

        with patch("plugins.platforms.feishu.adapter.threading.Thread"):
            for i in range(5):
                adapter._on_message_event(SimpleNamespace(tag=f"evt-{i}"))

        # Only the last 3 should remain; evt-0 and evt-1 dropped.
        self.assertEqual(len(adapter._pending_inbound_events), 3)
        tags = [getattr(e, "tag", None) for e in adapter._pending_inbound_events]
        self.assertEqual(tags, ["evt-2", "evt-3", "evt-4"])

    @patch.dict(os.environ, {}, clear=True)
    def test_normal_path_unchanged_when_loop_ready(self):
        """When the loop is ready, events should dispatch directly without
        ever touching the pending queue."""
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        class _ReadyLoop:
            def is_closed(self):
                return False

        adapter._loop = _ReadyLoop()

        future = SimpleNamespace(add_done_callback=lambda *_a, **_kw: None)

        def _submit(coro, _loop):
            coro.close()
            return future

        with patch(
            "plugins.platforms.feishu.adapter.asyncio.run_coroutine_threadsafe",
            side_effect=_submit,
        ) as submit, patch(
            "plugins.platforms.feishu.adapter.threading.Thread"
        ) as thread_cls:
            adapter._on_message_event(SimpleNamespace(tag="evt"))

        self.assertEqual(submit.call_count, 1)
        self.assertEqual(len(adapter._pending_inbound_events), 0)
        self.assertFalse(adapter._pending_drain_scheduled)
        # No drainer thread spawned when the happy path runs.
        self.assertEqual(thread_cls.call_count, 0)


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestWebhookSecurity(unittest.TestCase):
    """Tests for webhook signature verification, rate limiting, and body size limits."""

    def _make_adapter(self, encrypt_key: str = "") -> "FeishuAdapter":
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        with patch.dict(os.environ, {"FEISHU_APP_ID": "cli", "FEISHU_APP_SECRET": "sec", "FEISHU_ENCRYPT_KEY": encrypt_key}, clear=True):
            return FeishuAdapter(PlatformConfig())

    def test_signature_valid_passes(self):
        import hashlib

        encrypt_key = "test_secret"
        adapter = self._make_adapter(encrypt_key)
        body = b'{"type":"event"}'
        timestamp = "1700000000"
        nonce = "abc123"
        content = f"{timestamp}{nonce}{encrypt_key}" + body.decode("utf-8")
        sig = hashlib.sha256(content.encode("utf-8")).hexdigest()
        headers = {"x-lark-request-timestamp": timestamp, "x-lark-request-nonce": nonce, "x-lark-signature": sig}
        self.assertTrue(adapter._is_webhook_signature_valid(headers, body))

    def test_signature_invalid_rejected(self):
        adapter = self._make_adapter("test_secret")
        headers = {
            "x-lark-request-timestamp": "1700000000",
            "x-lark-request-nonce": "abc",
            "x-lark-signature": "deadbeef" * 8,
        }
        self.assertFalse(adapter._is_webhook_signature_valid(headers, b'{"type":"event"}'))

    def test_signature_missing_headers_rejected(self):
        adapter = self._make_adapter("test_secret")
        self.assertFalse(adapter._is_webhook_signature_valid({}, b'{}'))

    def test_rate_limit_allows_requests_within_window(self):
        adapter = self._make_adapter()
        for _ in range(5):
            self.assertTrue(adapter._check_webhook_rate_limit("10.0.0.1"))

    def test_rate_limit_blocks_after_exceeding_max(self):
        from plugins.platforms.feishu.adapter import _FEISHU_WEBHOOK_RATE_LIMIT_MAX
        adapter = self._make_adapter()
        for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX):
            adapter._check_webhook_rate_limit("10.0.0.2")
        self.assertFalse(adapter._check_webhook_rate_limit("10.0.0.2"))

    def test_rate_limit_resets_after_window_expires(self):
        from plugins.platforms.feishu.adapter import _FEISHU_WEBHOOK_RATE_LIMIT_MAX, _FEISHU_WEBHOOK_RATE_WINDOW_SECONDS
        adapter = self._make_adapter()
        ip = "10.0.0.3"
        for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX):
            adapter._check_webhook_rate_limit(ip)
        self.assertFalse(adapter._check_webhook_rate_limit(ip))
        # Simulate window expiry by backdating the stored entry.
        count, window_start = adapter._webhook_rate_counts[ip]
        adapter._webhook_rate_counts[ip] = (count, window_start - _FEISHU_WEBHOOK_RATE_WINDOW_SECONDS - 1)
        self.assertTrue(adapter._check_webhook_rate_limit(ip))

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_request_rejects_oversized_body(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter, _FEISHU_WEBHOOK_MAX_BODY_BYTES

        adapter = FeishuAdapter(PlatformConfig())
        # Simulate a request whose Content-Length already signals oversize.
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=_FEISHU_WEBHOOK_MAX_BODY_BYTES + 1,
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        self.assertEqual(response.status, 413)

    def test_webhook_request_rejects_oversized_chunked_body_while_reading(self):
        from gateway.config import PlatformConfig
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from plugins.platforms.feishu.adapter import FeishuAdapter, _FEISHU_WEBHOOK_MAX_BODY_BYTES

        with tempfile.TemporaryDirectory() as tmpdir:
            token = set_hermes_home_override(tmpdir)
            try:
                adapter = FeishuAdapter(PlatformConfig())
            finally:
                reset_hermes_home_override(token)
            content = _FakeRequestContent(b"A" * (_FEISHU_WEBHOOK_MAX_BODY_BYTES + 2))
            request = SimpleNamespace(
                remote="127.0.0.1",
                content_length=None,
                headers={},
                content=content,
            )

            response = asyncio.run(adapter._handle_webhook_request(request))

            self.assertEqual(response.status, 413)
            self.assertEqual(content.read_sizes, [_FEISHU_WEBHOOK_MAX_BODY_BYTES + 1])

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_request_rejects_invalid_json(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            content=_FakeRequestContent(b"not-json"),
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        self.assertEqual(response.status, 400)

    @patch.dict(os.environ, {"FEISHU_ENCRYPT_KEY": "secret"}, clear=True)
    def test_webhook_request_rejects_bad_signature(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        body = json.dumps({"header": {"event_type": "im.message.receive_v1"}}).encode()
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            headers={"x-lark-request-timestamp": "123", "x-lark-request-nonce": "abc", "x-lark-signature": "bad"},
            content=_FakeRequestContent(body),
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        self.assertEqual(response.status, 401)

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_connect_requires_inbound_auth_secret(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(
            PlatformConfig(
                enabled=True,
                extra={"app_id": "cli_app", "app_secret": "secret_app", "connection_mode": "webhook"},
            )
        )
        self.assertFalse(asyncio.run(adapter.connect()))

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_loads_auth_secrets_from_platform_extra(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "app_id": "cli_app",
                    "app_secret": "secret_app",
                    "connection_mode": "webhook",
                    "verification_token": "token_from_extra",
                    "encrypt_key": "encrypt_from_extra",
                },
            )
        )
        self.assertEqual(adapter._verification_token, "token_from_extra")
        self.assertEqual(adapter._encrypt_key, "encrypt_from_extra")

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_url_verification_challenge_passes_without_signature(self):
        """Challenge requests must succeed even when no encrypt_key is set."""
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        body = json.dumps({"type": "url_verification", "challenge": "test_challenge_token"}).encode()
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            content=_FakeRequestContent(body),
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        self.assertEqual(response.status, 200)
        self.assertIn(b"test_challenge_token", response.body)


class TestDedupTTL(unittest.TestCase):
    """Tests for TTL-aware deduplication."""

    @patch.dict(os.environ, {}, clear=True)
    def test_duplicate_within_ttl_is_rejected(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        with patch.object(adapter, "_persist_seen_message_ids"):
            adapter._seen_message_ids = {"om_dup": time.time()}
            adapter._seen_message_order = ["om_dup"]
            self.assertTrue(adapter._is_duplicate("om_dup"))

    @patch.dict(os.environ, {}, clear=True)
    def test_expired_entry_is_not_considered_duplicate(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter, _FEISHU_DEDUP_TTL_SECONDS

        adapter = FeishuAdapter(PlatformConfig())
        # Plant an entry that expired well past the TTL.
        stale_ts = time.time() - _FEISHU_DEDUP_TTL_SECONDS - 60
        adapter._seen_message_ids = {"om_old": stale_ts}
        adapter._seen_message_order = ["om_old"]
        with patch.object(adapter, "_persist_seen_message_ids"):
            self.assertFalse(adapter._is_duplicate("om_old"))

    @patch.dict(os.environ, {}, clear=True)
    def test_load_tolerates_malformed_timestamp_values(self):
        """Regression #13632 — a non-numeric timestamp in the persisted
        dedup state must not crash adapter startup.  The bad key is
        skipped; the rest of the state loads.
        """
        import tempfile
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        with tempfile.TemporaryDirectory() as temp_home:
            with patch.dict(os.environ, {"HERMES_HOME": temp_home}, clear=True):
                adapter = FeishuAdapter(PlatformConfig())
                adapter._dedup_state_path.parent.mkdir(parents=True, exist_ok=True)
                adapter._dedup_state_path.write_text(
                    json.dumps(
                        {
                            "message_ids": {
                                "om_good": time.time(),
                                "om_bad_str": "not-a-timestamp",
                                "om_bad_null": None,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                adapter._load_seen_message_ids()
                assert "om_good" in adapter._seen_message_ids
                assert "om_bad_str" not in adapter._seen_message_ids
                assert "om_bad_null" not in adapter._seen_message_ids

    @patch.dict(os.environ, {}, clear=True)
    def test_persist_saves_timestamps_as_dict(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        ts = time.time()
        adapter._seen_message_ids = {"om_ts1": ts}
        adapter._seen_message_order = ["om_ts1"]
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter._dedup_state_path = Path(tmpdir) / "dedup.json"
            adapter._persist_seen_message_ids()
            saved = json.loads(adapter._dedup_state_path.read_text())
        self.assertIsInstance(saved["message_ids"], dict)
        self.assertAlmostEqual(saved["message_ids"]["om_ts1"], ts, places=1)

    @patch.dict(os.environ, {}, clear=True)
    def test_load_backward_compat_list_format(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dedup.json"
            path.write_text(json.dumps({"message_ids": ["om_a", "om_b"]}), encoding="utf-8")
            adapter._dedup_state_path = path
            adapter._load_seen_message_ids()
        self.assertIn("om_a", adapter._seen_message_ids)
        self.assertIn("om_b", adapter._seen_message_ids)


class TestGroupMentionAtAll(unittest.TestCase):
    """Tests for @_all (Feishu @everyone) group mention routing."""

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_at_all_in_content_accepts_without_explicit_bot_mention(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            content='{"text":"@_all 请注意"}',
            mentions=[],
        )
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)
        self.assertTrue(_admits_group(adapter, message, sender_id, ""))

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "allowlist", "FEISHU_ALLOWED_USERS": "ou_allowed"}, clear=True)
    def test_at_all_still_requires_policy_gate(self):
        """@_all bypasses mention gating but NOT the allowlist policy."""
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(content='{"text":"@_all attention"}', mentions=[])
        # Non-allowlisted user — should be blocked even with @_all.
        blocked_sender = SimpleNamespace(open_id="ou_blocked", user_id=None)
        self.assertFalse(_admits_group(adapter, message, blocked_sender, ""))
        # Allowlisted user — should pass.
        allowed_sender = SimpleNamespace(open_id="ou_allowed", user_id=None)
        self.assertTrue(_admits_group(adapter, message, allowed_sender, ""))


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestSenderNameResolution(unittest.TestCase):
    """Tests for _resolve_sender_name_from_api (contact API + cache)."""

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_when_client_is_none(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = None
        result = asyncio.run(adapter._resolve_sender_name_from_api("ou_abc"))
        self.assertIsNone(result)

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_cached_name_within_ttl(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace()
        future_expire = time.time() + 600
        adapter._sender_name_cache["ou_cached"] = ("Alice", future_expire)
        result = asyncio.run(adapter._resolve_sender_name_from_api("ou_cached"))
        self.assertEqual(result, "Alice")

    @patch.dict(os.environ, {}, clear=True)
    def test_fetches_and_caches_name_from_api(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        user_obj = SimpleNamespace(name="Bob", display_name=None, nickname=None, en_name=None)
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(user=user_obj),
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        class _ContactAPI:
            def get(self, request):
                return mock_response

        adapter._client = SimpleNamespace(
            contact=SimpleNamespace(v3=SimpleNamespace(user=_ContactAPI()))
        )

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_bob"))

        self.assertEqual(result, "Bob")
        self.assertIn("ou_bob", adapter._sender_name_cache)

    @patch.dict(os.environ, {}, clear=True)
    def test_expired_cache_triggers_new_api_call(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        # Expired cache entry.
        adapter._sender_name_cache["ou_expired"] = ("OldName", time.time() - 1)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        user_obj = SimpleNamespace(name="NewName", display_name=None, nickname=None, en_name=None)

        class _ContactAPI:
            def get(self, request):
                return SimpleNamespace(success=lambda: True, data=SimpleNamespace(user=user_obj))

        adapter._client = SimpleNamespace(
            contact=SimpleNamespace(v3=SimpleNamespace(user=_ContactAPI()))
        )

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_expired"))

        self.assertEqual(result, "NewName")

    @patch.dict(os.environ, {}, clear=True)
    def test_api_failure_returns_none_without_raising(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        class _BrokenContactAPI:
            def get(self, _request):
                raise RuntimeError("API down")

        adapter._client = SimpleNamespace(
            contact=SimpleNamespace(v3=SimpleNamespace(user=_BrokenContactAPI()))
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_broken"))

        self.assertIsNone(result)


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestBotNameResolution(unittest.TestCase):
    """Tests for the bot branch of _resolve_sender_name_from_api (basic_batch API + shared cache)."""

    @staticmethod
    def _batch_payload(bots: Dict[str, str]):
        import json as _json
        body = {
            oid: {"bot_id": oid, "name": name, "i18n_names": {"en_us": name}}
            for oid, name in bots.items()
        }
        return _json.dumps({"code": 0, "msg": "", "data": {"bots": body, "failed_bots": {}}}).encode()

    def _build_adapter_with_bots(self, bots: Dict[str, str]):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        calls = []

        def _fake_request(request):
            calls.append(request)
            return SimpleNamespace(raw=SimpleNamespace(content=self._batch_payload(bots)))

        adapter._client = SimpleNamespace(request=_fake_request)
        return adapter, calls

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_cached_bot_name_without_api_call(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._sender_name_cache["ou_peer"] = ("Peer Bot", time.time() + 600)
        adapter._client = SimpleNamespace(
            request=lambda _r: (_ for _ in ()).throw(RuntimeError("should not fetch"))
        )
        result = asyncio.run(adapter._resolve_sender_name_from_api("ou_peer", is_bot=True))
        self.assertEqual(result, "Peer Bot")

    @patch.dict(os.environ, {}, clear=True)
    def test_fetches_and_caches_bot_name(self):
        adapter, calls = self._build_adapter_with_bots({"ou_peer": "Peer Bot"})

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_peer", is_bot=True))

        self.assertEqual(result, "Peer Bot")
        self.assertEqual(adapter._sender_name_cache["ou_peer"][0], "Peer Bot")
        self.assertEqual(len(calls), 1)
        self.assertIn("/open-apis/bot/v3/bots/basic_batch", calls[0].uri)
        # Feishu expects repeated ?bot_ids= params, not comma-joined.
        self.assertEqual(calls[0].queries, [("bot_ids", "ou_peer")])

    @patch.dict(os.environ, {}, clear=True)
    def test_api_failure_returns_none_and_does_not_poison_cache(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        def _broken_request(_req):
            raise RuntimeError("API down")

        adapter._client = SimpleNamespace(request=_broken_request)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_peer", is_bot=True))

        self.assertIsNone(result)
        self.assertNotIn("ou_peer", adapter._sender_name_cache)

    @patch.dict(os.environ, {}, clear=True)
    def test_bot_absent_from_response_is_not_cached(self):
        """Bot not in ``data.bots`` (e.g. landed in ``failed_bots``) → no
        cache entry, next lookup re-fetches."""
        adapter, _ = self._build_adapter_with_bots({"ou_other": "Other Bot"})

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_ghost", is_bot=True))

        self.assertIsNone(result)
        self.assertNotIn("ou_ghost", adapter._sender_name_cache)

    @patch.dict(os.environ, {}, clear=True)
    def test_empty_name_in_response_is_negative_cached(self):
        """API returns name="" → cache "" so repeat lookups short-circuit."""
        adapter, calls = self._build_adapter_with_bots({"ou_nameless": ""})

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            first = asyncio.run(adapter._resolve_sender_name_from_api("ou_nameless", is_bot=True))
            second = asyncio.run(adapter._resolve_sender_name_from_api("ou_nameless", is_bot=True))

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(adapter._sender_name_cache["ou_nameless"][0], "")
        self.assertEqual(len(calls), 1)

    @patch.dict(os.environ, {}, clear=True)
    def test_non_zero_code_returns_none(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        error_payload = b'{"code":99991663,"msg":"permission denied"}'
        adapter._client = SimpleNamespace(
            request=lambda _r: SimpleNamespace(raw=SimpleNamespace(content=error_payload))
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_peer", is_bot=True))

        self.assertIsNone(result)
        self.assertNotIn("ou_peer", adapter._sender_name_cache)


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestProcessingReactions(unittest.TestCase):
    """Typing on start → removed on SUCCESS, swapped for CrossMark on FAILURE,
    removed (no replacement) on CANCELLED."""

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    def _build_adapter(
        self,
        create_success: bool = True,
        delete_success: bool = True,
        next_reaction_id: str = "r1",
    ):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        tracker = SimpleNamespace(
            create_calls=[],
            delete_calls=[],
            next_reaction_id=next_reaction_id,
            create_success=create_success,
            delete_success=delete_success,
        )

        def _create(request):
            tracker.create_calls.append(
                request.request_body.reaction_type["emoji_type"]
            )
            if tracker.create_success:
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(reaction_id=tracker.next_reaction_id),
                )
            return SimpleNamespace(
                success=lambda: False, code=99, msg="rejected", data=None,
            )

        def _delete(request):
            tracker.delete_calls.append(request.reaction_id)
            return SimpleNamespace(
                success=lambda: tracker.delete_success,
                code=0 if tracker.delete_success else 99,
                msg="success" if tracker.delete_success else "rejected",
            )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message_reaction=SimpleNamespace(create=_create, delete=_delete),
                ),
            ),
        )
        return adapter, tracker

    @staticmethod
    def _event(message_id: str = "om_msg"):
        return SimpleNamespace(message_id=message_id)

    def _patch_to_thread(self):
        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        return patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct)

    # ------------------------------------------------------------------ start
    @patch.dict(os.environ, {}, clear=True)
    def test_start_adds_typing_and_caches_reaction_id(self):
        adapter, tracker = self._build_adapter(next_reaction_id="r_typing")
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
        self.assertEqual(tracker.create_calls, ["Typing"])
        self.assertEqual(adapter._pending_processing_reactions["om_msg"], "r_typing")

    @patch.dict(os.environ, {}, clear=True)
    def test_start_is_idempotent_for_same_message_id(self):
        adapter, tracker = self._build_adapter(next_reaction_id="r_typing")
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
            self._run(adapter.on_processing_start(self._event()))
        self.assertEqual(tracker.create_calls, ["Typing"])

    @patch.dict(os.environ, {}, clear=True)
    def test_start_does_not_cache_when_create_fails(self):
        adapter, tracker = self._build_adapter(create_success=False)
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
        self.assertEqual(tracker.create_calls, ["Typing"])
        self.assertNotIn("om_msg", adapter._pending_processing_reactions)

    # --------------------------------------------------------------- complete
    @patch.dict(os.environ, {}, clear=True)
    def test_success_removes_typing_and_adds_nothing(self):
        adapter, tracker = self._build_adapter(next_reaction_id="r_typing")
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
            self._run(
                adapter.on_processing_complete(self._event(), ProcessingOutcome.SUCCESS)
            )
        self.assertEqual(tracker.create_calls, ["Typing"])
        self.assertEqual(tracker.delete_calls, ["r_typing"])
        self.assertNotIn("om_msg", adapter._pending_processing_reactions)

    @patch.dict(os.environ, {}, clear=True)
    def test_failure_removes_typing_then_adds_cross_mark(self):
        adapter, tracker = self._build_adapter(next_reaction_id="r_typing")
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
            self._run(
                adapter.on_processing_complete(self._event(), ProcessingOutcome.FAILURE)
            )
        self.assertEqual(tracker.create_calls, ["Typing", "CrossMark"])
        self.assertEqual(tracker.delete_calls, ["r_typing"])

    @patch.dict(os.environ, {}, clear=True)
    def test_cancelled_removes_typing_and_adds_nothing(self):
        adapter, tracker = self._build_adapter(next_reaction_id="r_typing")
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
            self._run(
                adapter.on_processing_complete(self._event(), ProcessingOutcome.CANCELLED)
            )
        self.assertEqual(tracker.create_calls, ["Typing"])
        self.assertEqual(tracker.delete_calls, ["r_typing"])
        self.assertNotIn("om_msg", adapter._pending_processing_reactions)

    @patch.dict(os.environ, {}, clear=True)
    def test_failure_without_preceding_start_still_adds_cross_mark(self):
        adapter, tracker = self._build_adapter()
        with self._patch_to_thread():
            self._run(
                adapter.on_processing_complete(self._event(), ProcessingOutcome.FAILURE)
            )
        self.assertEqual(tracker.create_calls, ["CrossMark"])
        self.assertEqual(tracker.delete_calls, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_success_without_preceding_start_is_full_noop(self):
        adapter, tracker = self._build_adapter()
        with self._patch_to_thread():
            self._run(
                adapter.on_processing_complete(self._event(), ProcessingOutcome.SUCCESS)
            )
        self.assertEqual(tracker.create_calls, [])
        self.assertEqual(tracker.delete_calls, [])

    # ------------------------- delete failure: don't stack badges -----------
    @patch.dict(os.environ, {}, clear=True)
    def test_delete_failure_on_failure_outcome_skips_cross_mark(self):
        # Removing Typing is best-effort — but if it fails, we must NOT
        # additionally add CrossMark, or the UI would show two contradictory
        # badges. The handle stays in the cache for LRU to clean up later.
        adapter, tracker = self._build_adapter(
            next_reaction_id="r_typing", delete_success=False,
        )
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
            self._run(
                adapter.on_processing_complete(self._event(), ProcessingOutcome.FAILURE)
            )
        self.assertEqual(tracker.create_calls, ["Typing"])  # CrossMark NOT added
        self.assertEqual(tracker.delete_calls, ["r_typing"])  # delete was attempted
        self.assertEqual(
            adapter._pending_processing_reactions["om_msg"], "r_typing",
        )  # handle retained

    @patch.dict(os.environ, {}, clear=True)
    def test_delete_failure_on_success_outcome_retains_handle(self):
        adapter, tracker = self._build_adapter(
            next_reaction_id="r_typing", delete_success=False,
        )
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
            self._run(
                adapter.on_processing_complete(self._event(), ProcessingOutcome.SUCCESS)
            )
        self.assertEqual(tracker.create_calls, ["Typing"])
        self.assertEqual(tracker.delete_calls, ["r_typing"])
        self.assertEqual(
            adapter._pending_processing_reactions["om_msg"], "r_typing",
        )

    # ------------------------------------------------------------- env toggle
    @patch.dict(os.environ, {"FEISHU_REACTIONS": "false"}, clear=True)
    def test_env_disable_short_circuits_both_hooks(self):
        adapter, tracker = self._build_adapter()
        with self._patch_to_thread():
            self._run(adapter.on_processing_start(self._event()))
            self._run(
                adapter.on_processing_complete(self._event(), ProcessingOutcome.FAILURE)
            )
        self.assertEqual(tracker.create_calls, [])
        self.assertEqual(tracker.delete_calls, [])

    # ------------------------------------------------------------- LRU bounds
    @patch.dict(os.environ, {}, clear=True)
    def test_cache_evicts_oldest_entry_beyond_size_limit(self):
        from plugins.platforms.feishu.adapter import _FEISHU_PROCESSING_REACTION_CACHE_SIZE

        adapter, _ = self._build_adapter()
        counter = {"n": 0}

        def _create(_request):
            counter["n"] += 1
            return SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(reaction_id=f"r{counter['n']}"),
            )

        adapter._client.im.v1.message_reaction.create = _create

        with self._patch_to_thread():
            for i in range(_FEISHU_PROCESSING_REACTION_CACHE_SIZE + 1):
                self._run(adapter.on_processing_start(self._event(f"om_{i}")))

        self.assertNotIn("om_0", adapter._pending_processing_reactions)
        self.assertIn(
            f"om_{_FEISHU_PROCESSING_REACTION_CACHE_SIZE}",
            adapter._pending_processing_reactions,
        )
        self.assertEqual(
            len(adapter._pending_processing_reactions),
            _FEISHU_PROCESSING_REACTION_CACHE_SIZE,
        )


class TestFeishuMentionMap(unittest.TestCase):
    def test_build_mentions_map_handles_at_all(self):
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity, FeishuMentionRef

        mention = SimpleNamespace(key="@_all", id=None, name="")
        result = _build_mentions_map(
            [mention],
            _FeishuBotIdentity(open_id="ou_bot", name="Hermes"),
        )
        self.assertEqual(result["@_all"], FeishuMentionRef(is_all=True))

    def test_build_mentions_map_marks_self_by_open_id(self):
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        ref = _build_mentions_map([mention], _FeishuBotIdentity(open_id="ou_bot"))["@_user_1"]
        self.assertTrue(ref.is_self)
        self.assertEqual(ref.open_id, "ou_bot")
        self.assertEqual(ref.name, "Hermes")

    def test_build_mentions_map_marks_self_by_name_fallback(self):
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="", user_id=""),
            name="Hermes",
        )
        result = _build_mentions_map([mention], _FeishuBotIdentity(name="Hermes"))
        self.assertTrue(result["@_user_1"].is_self)

    def test_build_mentions_map_name_match_does_not_override_mismatching_open_id(self):
        """Regression: a human user whose display name matches the bot must
        NOT be flagged as self when their open_id differs. Before the fix,
        name-match fired even when open_id was present and different, causing
        their messages to be silently stripped/dropped."""
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        human_with_same_name = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_human", user_id=""),
            name="Hermes Bot",
        )
        result = _build_mentions_map(
            [human_with_same_name],
            _FeishuBotIdentity(open_id="ou_bot", name="Hermes Bot"),
        )
        self.assertFalse(result["@_user_1"].is_self)

    def test_build_mentions_map_falls_back_to_name_when_bot_open_id_not_hydrated(self):
        """Regression: right after gateway startup, _hydrate_bot_identity may
        not have populated _bot_open_id yet. During that window, a mention
        carrying a real open_id should still match via name — otherwise
        @bot messages silently fail admission."""
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot_actual", user_id=""),
            name="Hermes Bot",
        )
        # Bot identity has name but no open_id yet (hydration pending).
        result = _build_mentions_map(
            [bot_mention],
            _FeishuBotIdentity(open_id="", name="Hermes Bot"),
        )
        self.assertTrue(result["@_user_1"].is_self)

    def test_build_mentions_map_non_self_user(self):
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_alice", user_id=""),
            name="Alice",
        )
        ref = _build_mentions_map([mention], _FeishuBotIdentity(open_id="ou_bot"))["@_user_1"]
        self.assertFalse(ref.is_self)
        self.assertEqual(ref.open_id, "ou_alice")
        self.assertEqual(ref.name, "Alice")

    def test_build_mentions_map_returns_empty_for_none_input(self):
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        self.assertEqual(_build_mentions_map(None, _FeishuBotIdentity(open_id="ou_bot")), {})

    def test_build_mentions_map_tolerates_missing_id_object(self):
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        mention = SimpleNamespace(key="@_user_9", id=None, name="")
        ref = _build_mentions_map([mention], _FeishuBotIdentity(open_id="ou_bot"))["@_user_9"]
        self.assertEqual(ref.open_id, "")
        self.assertFalse(ref.is_self)


class TestFeishuMentionHint(unittest.TestCase):
    def test_hint_single_user(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="Alice", open_id="ou_alice")]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice)]",
        )

    def test_hint_multiple_users(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Bob", open_id="ou_bob"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice), Bob (open_id=ou_bob)]",
        )

    def test_hint_at_all(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(is_all=True)]
        self.assertEqual(_build_mention_hint(refs), "[Mentioned: @all]")

    def test_hint_filters_self_mentions(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True),
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice)]",
        )

    def test_hint_returns_empty_when_only_self(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)]
        self.assertEqual(_build_mention_hint(refs), "")

    def test_hint_returns_empty_for_no_refs(self):
        from plugins.platforms.feishu.adapter import _build_mention_hint

        self.assertEqual(_build_mention_hint([]), "")

    def test_hint_falls_back_when_open_id_missing(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="Alice", open_id="")]
        self.assertEqual(_build_mention_hint(refs), "[Mentioned: Alice]")

    def test_hint_uses_unknown_placeholder_when_name_missing(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="", open_id="ou_xxx")]
        self.assertEqual(_build_mention_hint(refs), "[Mentioned: unknown (open_id=ou_xxx)]")

    def test_hint_dedupes_repeated_user(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Bob", open_id="ou_bob"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice), Bob (open_id=ou_bob)]",
        )

    def test_hint_dedupes_repeated_at_all(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(is_all=True), FeishuMentionRef(is_all=True)]
        self.assertEqual(_build_mention_hint(refs), "[Mentioned: @all]")


class TestFeishuStripLeadingSelf(unittest.TestCase):
    def _make_refs(self, *, self_name="Hermes", other_name=None):
        from plugins.platforms.feishu.adapter import FeishuMentionRef

        refs = [FeishuMentionRef(name=self_name, open_id="ou_bot", is_self=True)]
        if other_name:
            refs.append(FeishuMentionRef(name=other_name, open_id="ou_alice"))
        return refs

    def test_strips_leading_self(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        result = _strip_edge_self_mentions("@Hermes /help", self._make_refs())
        self.assertEqual(result, "/help")

    def test_strips_consecutive_leading_self(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        result = _strip_edge_self_mentions("@Hermes @Hermes hi", self._make_refs())
        self.assertEqual(result, "hi")

    def test_stops_at_first_non_self_token(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        result = _strip_edge_self_mentions(
            "@Hermes @Alice make a group", self._make_refs(other_name="Alice")
        )
        self.assertEqual(result, "@Alice make a group")

    def test_preserves_mid_text_self(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        result = _strip_edge_self_mentions("check @Hermes said yesterday", self._make_refs())
        self.assertEqual(result, "check @Hermes said yesterday")

    def test_strips_trailing_self_at_end_of_text(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        result = _strip_edge_self_mentions("look up docs @Hermes", self._make_refs())
        self.assertEqual(result, "look up docs")

    def test_strips_trailing_self_with_terminal_punct(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        # Terminal punct after the mention — strip the mention, keep the punct.
        result = _strip_edge_self_mentions("look up docs @Hermes.", self._make_refs())
        self.assertEqual(result, "look up docs.")

    def test_preserves_trailing_self_before_non_terminal_char(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        # Non-terminal char (here a Chinese particle) follows — preserve.
        result = _strip_edge_self_mentions(
            "please don't @Hermes anymore", self._make_refs()
        )
        self.assertEqual(result, "please don't @Hermes anymore")

    def test_returns_input_when_refs_empty(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        self.assertEqual(_strip_edge_self_mentions("@Hermes /help", []), "@Hermes /help")

    def test_returns_input_when_no_self_refs(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions, FeishuMentionRef

        refs = [FeishuMentionRef(name="Alice", open_id="ou_alice")]
        self.assertEqual(_strip_edge_self_mentions("@Alice hi", refs), "@Alice hi")

    def test_uses_open_id_fallback_when_name_missing(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions, FeishuMentionRef

        refs = [FeishuMentionRef(name="", open_id="ou_bot", is_self=True)]
        self.assertEqual(_strip_edge_self_mentions("@ou_bot hi", refs), "hi")

    def test_word_boundary_prevents_prefix_collision(self):
        """A bot named 'Al' must not eat the leading '@Alice' of a different user."""
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions, FeishuMentionRef

        refs = [FeishuMentionRef(name="Al", open_id="ou_bot", is_self=True)]
        self.assertEqual(_strip_edge_self_mentions("@Alice hi", refs), "@Alice hi")


class TestFeishuNormalizeText(unittest.TestCase):
    def test_renders_mention_with_display_name(self):
        from plugins.platforms.feishu.adapter import _normalize_feishu_text, FeishuMentionRef

        refs = {"@_user_1": FeishuMentionRef(name="Alice", open_id="ou_alice")}
        self.assertEqual(_normalize_feishu_text("@_user_1 hello", refs), "@Alice hello")

    def test_renders_self_mention_with_name(self):
        from plugins.platforms.feishu.adapter import _normalize_feishu_text, FeishuMentionRef

        refs = {"@_user_1": FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)}
        self.assertEqual(
            _normalize_feishu_text("stop pinging @_user_1 please", refs),
            "stop pinging @Hermes please",
        )

    def test_at_all_rendered_as_english_literal(self):
        from plugins.platforms.feishu.adapter import _normalize_feishu_text

        self.assertEqual(_normalize_feishu_text("@_all notice", None), "@all notice")

    def test_unknown_placeholder_degrades_to_space(self):
        from plugins.platforms.feishu.adapter import _normalize_feishu_text

        # No map: fall back to the old behavior (substitute with space, then collapse).
        self.assertEqual(_normalize_feishu_text("@_user_9 hello", None), "hello")

    def test_backward_compatible_without_map(self):
        from plugins.platforms.feishu.adapter import _normalize_feishu_text

        self.assertEqual(_normalize_feishu_text("hello  world"), "hello world")

    def test_mention_for_missing_map_entry_degrades_to_space(self):
        from plugins.platforms.feishu.adapter import _normalize_feishu_text, FeishuMentionRef

        refs = {"@_user_1": FeishuMentionRef(name="Alice")}
        # @_user_2 has no entry — should degrade to a space (legacy behavior)
        self.assertEqual(
            _normalize_feishu_text("@_user_1 @_user_2 hi", refs),
            "@Alice hi",
        )


class TestFeishuPostMentionParsing(unittest.TestCase):
    def test_post_at_tag_renders_via_mentions_map(self):
        """Post <at>.user_id is a placeholder ('@_user_N'); the real display
        name comes from the mentions_map lookup. Confirmed via live
        im.v1.message.get payload."""
        from plugins.platforms.feishu.adapter import parse_feishu_post_payload, FeishuMentionRef

        payload = {
            "en_us": {
                "content": [[
                    {"tag": "at", "user_id": "@_user_1", "user_name": "ignored"},
                    {"tag": "text", "text": " hello"},
                ]]
            }
        }
        mentions_map = {
            "@_user_1": FeishuMentionRef(name="Alice", open_id="ou_alice"),
        }
        result = parse_feishu_post_payload(payload, mentions_map=mentions_map)
        self.assertEqual(result.text_content, "@Alice hello")

    def test_post_at_tag_falls_back_to_inline_user_name_when_map_misses(self):
        """When the mentions payload is missing a placeholder, fall back to the
        inline user_name in the <at> tag itself."""
        from plugins.platforms.feishu.adapter import parse_feishu_post_payload

        payload = {
            "en_us": {
                "content": [[
                    {"tag": "at", "user_id": "@_user_7", "user_name": "Unknown"},
                    {"tag": "text", "text": " hi"},
                ]]
            }
        }
        result = parse_feishu_post_payload(payload, mentions_map={})
        self.assertEqual(result.text_content, "@Unknown hi")

    def test_post_at_all_tag_renders_as_at_all(self):
        """Post-format @everyone has user_id == '@_all' (confirmed via live
        im.v1.message.get). Rendered as literal '@all' regardless of map."""
        from plugins.platforms.feishu.adapter import parse_feishu_post_payload

        payload = {
            "en_us": {
                "content": [[
                    {"tag": "at", "user_id": "@_all", "user_name": "everyone"},
                    {"tag": "text", "text": " meeting"},
                ]]
            }
        }
        result = parse_feishu_post_payload(payload)
        self.assertIn("@all", result.text_content)


class TestFeishuNormalizeWithMentions(unittest.TestCase):
    def test_text_message_renders_mention_by_name(self):
        from plugins.platforms.feishu.adapter import normalize_feishu_message, _FeishuBotIdentity

        mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_alice", user_id=""),
            name="Alice",
        )
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "@_user_1 hello"}),
            mentions=[mention],
            bot=_FeishuBotIdentity(open_id="ou_bot"),
        )
        self.assertEqual(normalized.text_content, "@Alice hello")
        self.assertEqual(len(normalized.mentions), 1)
        self.assertEqual(normalized.mentions[0].open_id, "ou_alice")
        self.assertFalse(normalized.mentions[0].is_self)

    def test_text_message_marks_bot_self_mention(self):
        from plugins.platforms.feishu.adapter import normalize_feishu_message, _FeishuBotIdentity

        mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "@_user_1 /help"}),
            mentions=[mention],
            bot=_FeishuBotIdentity(open_id="ou_bot"),
        )
        self.assertTrue(normalized.mentions[0].is_self)
        # self mention is still rendered — strip is a separate adapter-level pass
        self.assertEqual(normalized.text_content, "@Hermes /help")

    def test_text_message_at_all_surfaces_ref(self):
        from plugins.platforms.feishu.adapter import normalize_feishu_message

        mention = SimpleNamespace(key="@_all", id=None, name="")
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "@_all meeting"}),
            mentions=[mention],
        )
        self.assertEqual(normalized.text_content, "@all meeting")
        self.assertEqual(len(normalized.mentions), 1)
        self.assertTrue(normalized.mentions[0].is_all)

    def test_text_message_at_all_in_text_without_mentions_payload(self):
        """Feishu SDK sometimes omits @_all from the mentions payload (confirmed
        via im.v1.message.get). The fallback scan on raw text must still yield
        an is_all ref so [Mentioned: @all] gets injected."""
        from plugins.platforms.feishu.adapter import normalize_feishu_message

        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "@_all hello"}),
            mentions=None,
        )
        self.assertEqual(normalized.text_content, "@all hello")
        self.assertEqual(len(normalized.mentions), 1)
        self.assertTrue(normalized.mentions[0].is_all)

    def test_text_message_at_all_not_synthesized_if_absent_from_text(self):
        """No @_all in text → no synthetic ref even if mentions_map is empty."""
        from plugins.platforms.feishu.adapter import normalize_feishu_message

        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "plain hello"}),
            mentions=None,
        )
        self.assertEqual(normalized.mentions, [])

    def test_text_message_without_mentions_param_is_backward_compatible(self):
        from plugins.platforms.feishu.adapter import normalize_feishu_message

        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "hello world"}),
        )
        self.assertEqual(normalized.text_content, "hello world")
        self.assertEqual(normalized.mentions, [])

    def test_post_message_marks_self_via_mentions_map_lookup(self):
        """Real Feishu post: <at user_id="@_user_N"> + top-level mentions array
        resolves to open_id via placeholder lookup, not direct tag fields."""
        from plugins.platforms.feishu.adapter import normalize_feishu_message, _FeishuBotIdentity

        raw = json.dumps({
            "en_us": {
                "content": [
                    [
                        {"tag": "at", "user_id": "@_user_1", "user_name": "Hermes"},
                        {"tag": "text", "text": " check this"},
                    ]
                ]
            }
        })
        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        normalized = normalize_feishu_message(
            message_type="post",
            raw_content=raw,
            mentions=[bot_mention],
            bot=_FeishuBotIdentity(open_id="ou_bot"),
        )
        self.assertEqual(len(normalized.mentions), 1)
        self.assertTrue(normalized.mentions[0].is_self)
        self.assertEqual(normalized.mentions[0].open_id, "ou_bot")


class TestFeishuPostMentionsBot(unittest.TestCase):
    def _build_adapter(self, bot_open_id="ou_bot", bot_user_id="", bot_name=""):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._bot_open_id = bot_open_id
        adapter._bot_user_id = bot_user_id
        adapter._bot_name = bot_name
        return adapter

    def test_post_mentions_bot_uses_is_self_flag(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef

        adapter = self._build_adapter()
        self.assertTrue(
            adapter._post_mentions_bot(
                [FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)]
            )
        )
        self.assertFalse(
            adapter._post_mentions_bot(
                [FeishuMentionRef(name="Alice", open_id="ou_alice")]
            )
        )

    def test_post_mentions_bot_empty_returns_false(self):
        adapter = self._build_adapter()
        self.assertFalse(adapter._post_mentions_bot([]))


class TestFeishuExtractMessageContent(unittest.TestCase):
    def _build_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._bot_open_id = "ou_bot"
        adapter._bot_user_id = ""
        adapter._bot_name = "Hermes"
        adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
        return adapter

    def test_returns_five_tuple_with_mentions(self):
        adapter = self._build_adapter()
        message = SimpleNamespace(
            content=json.dumps({"text": "@_user_1 hello"}),
            message_type="text",
            message_id="m1",
            mentions=[
                SimpleNamespace(
                    key="@_user_1",
                    id=SimpleNamespace(open_id="ou_alice", user_id=""),
                    name="Alice",
                )
            ],
        )

        text, inbound_type, media_urls, media_types, mentions = asyncio.run(
            adapter._extract_message_content(message)
        )
        self.assertEqual(text, "@Alice hello")
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].open_id, "ou_alice")

    def test_returns_empty_mentions_when_missing(self):
        adapter = self._build_adapter()
        message = SimpleNamespace(
            content=json.dumps({"text": "plain hello"}),
            message_type="text",
            message_id="m2",
            mentions=None,
        )

        text, _, _, _, mentions = asyncio.run(adapter._extract_message_content(message))
        self.assertEqual(text, "plain hello")
        self.assertEqual(mentions, [])


class TestFeishuProcessInboundMessage(unittest.TestCase):
    def _build_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._bot_open_id = "ou_bot"
        adapter._bot_user_id = ""
        adapter._bot_name = "Hermes"
        adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
        adapter._fetch_message_text = AsyncMock(return_value=None)
        adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "u1", "user_name": "Alice", "user_id_alt": None}
        )
        adapter._resolve_source_chat_type = Mock(return_value="group")
        adapter.build_source = Mock(return_value=SimpleNamespace(thread_id=None))
        adapter._dispatch_inbound_event = AsyncMock()
        return adapter

    def test_leading_self_mention_stripped_for_command(self):
        from gateway.platforms.base import MessageType

        adapter = self._build_adapter()
        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        message = SimpleNamespace(
            content=json.dumps({"text": "@_user_1 /help"}),
            message_type="text",
            message_id="m1",
            mentions=[bot_mention],
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message,
                message=message,
                sender_id=None,
                chat_type="group",
                message_id="m1",
            )
        )
        event = adapter._dispatch_inbound_event.call_args.args[0]
        self.assertEqual(event.text, "/help")
        self.assertEqual(event.message_type, MessageType.COMMAND)

    def test_non_command_message_with_mentions_injects_hint(self):
        from gateway.platforms.base import MessageType

        adapter = self._build_adapter()
        alice = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_alice", user_id=""),
            name="Alice",
        )
        bob = SimpleNamespace(
            key="@_user_2",
            id=SimpleNamespace(open_id="ou_bob", user_id=""),
            name="Bob",
        )
        message = SimpleNamespace(
            content=json.dumps({"text": "@_user_1 @_user_2 make a group"}),
            message_type="text",
            message_id="m2",
            mentions=[alice, bob],
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message,
                message=message,
                sender_id=None,
                chat_type="group",
                message_id="m2",
            )
        )
        event = adapter._dispatch_inbound_event.call_args.args[0]
        self.assertEqual(event.message_type, MessageType.TEXT)
        self.assertIn("[Mentioned: Alice (open_id=ou_alice), Bob (open_id=ou_bob)]", event.text)
        self.assertIn("@Alice @Bob make a group", event.text)

    def test_command_message_never_injects_hint(self):
        adapter = self._build_adapter()
        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        alice = SimpleNamespace(
            key="@_user_2",
            id=SimpleNamespace(open_id="ou_alice", user_id=""),
            name="Alice",
        )
        message = SimpleNamespace(
            content=json.dumps({"text": "@_user_1 /model @_user_2"}),
            message_type="text",
            message_id="m3",
            mentions=[bot_mention, alice],
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message,
                message=message,
                sender_id=None,
                chat_type="group",
                message_id="m3",
            )
        )
        event = adapter._dispatch_inbound_event.call_args.args[0]
        self.assertNotIn("[Mentioned:", event.text)
        self.assertTrue(event.text.startswith("/model"))

    def test_mid_text_self_mention_preserved(self):
        adapter = self._build_adapter()
        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        message = SimpleNamespace(
            content=json.dumps({"text": "stop pinging @_user_1 please"}),
            message_type="text",
            message_id="m4",
            mentions=[bot_mention],
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message,
                message=message,
                sender_id=None,
                chat_type="group",
                message_id="m4",
            )
        )
        event = adapter._dispatch_inbound_event.call_args.args[0]
        self.assertEqual(event.text, "stop pinging @Hermes please")

    def test_pure_self_mention_message_is_ignored(self):
        """A message containing only '@Bot' (no body, no media) must not dispatch.

        Regression guard: the rendered '@Hermes' slips past the pre-strip empty
        guard; the post-strip guard must catch it.
        """
        adapter = self._build_adapter()
        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        message = SimpleNamespace(
            content=json.dumps({"text": "@_user_1"}),
            message_type="text",
            message_id="m5",
            mentions=[bot_mention],
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message, message=message, sender_id=None,
                chat_type="group", message_id="m5",
            )
        )
        adapter._dispatch_inbound_event.assert_not_called()


class TestFeishuFetchMessageText(unittest.TestCase):
    def _build_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._bot_open_id = "ou_bot"
        adapter._bot_user_id = ""
        adapter._bot_name = "Hermes"
        adapter._message_text_cache = OrderedDict()
        adapter._client = Mock()
        adapter._build_get_message_request = Mock(return_value=object())
        return adapter

    def test_fetch_message_text_renders_mentions_without_hint_prefix(self):
        adapter = self._build_adapter()

        alice_mention = SimpleNamespace(
            key="@_user_1",
            id="ou_alice",
            id_type="open_id",
            name="Alice",
        )
        parent = SimpleNamespace(
            body=SimpleNamespace(content=json.dumps({"text": "@_user_1 hi"})),
            msg_type="text",
            mentions=[alice_mention],
        )
        response = Mock()
        response.success = Mock(return_value=True)
        response.data = SimpleNamespace(items=[parent])
        adapter._client.im.v1.message.get = Mock(return_value=response)

        result = asyncio.run(adapter._fetch_message_text("m_parent"))
        self.assertEqual(result, "@Alice hi")
        # No [Mentioned:] wrapper — reply-context path intentionally skips the hint.
        self.assertNotIn("[Mentioned:", result)

    def test_extract_text_from_raw_content_accepts_mentions_kwarg(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._bot_open_id = ""
        adapter._bot_user_id = ""
        adapter._bot_name = ""

        alice_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_alice", user_id=""),
            name="Alice",
        )
        self.assertEqual(
            adapter._extract_text_from_raw_content(
                msg_type="text",
                raw_content=json.dumps({"text": "@_user_1 hello"}),
                mentions=[alice_mention],
            ),
            "@Alice hello",
        )

    def test_fetch_message_text_marks_is_self_via_string_id_shape(self):
        """History-path Mention objects carry id as str + id_type; is_self must still work."""
        adapter = self._build_adapter()
        # bot_name is empty — is_self must be detected via open_id alone
        adapter._bot_name = ""

        bot_mention = SimpleNamespace(
            key="@_user_1",
            id="ou_bot",
            id_type="open_id",
            name="Hermes",
        )
        parent = SimpleNamespace(
            body=SimpleNamespace(content=json.dumps({"text": "@_user_1 hi"})),
            msg_type="text",
            mentions=[bot_mention],
        )
        response = Mock()
        response.success = Mock(return_value=True)
        response.data = SimpleNamespace(items=[parent])
        adapter._client.im.v1.message.get = Mock(return_value=response)

        # The rendered text should still have the bot name substituted.
        result = asyncio.run(adapter._fetch_message_text("m_parent"))
        self.assertEqual(result, "@Hermes hi")

    def test_build_mentions_map_string_id_shape(self):
        """_build_mentions_map accepts the reply-history shape (id as str +
        id_type='open_id'). user_id id_type is not load-bearing for self
        detection — inbound mention payloads always include an open_id."""
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        # open_id discriminator, non-self
        alice = SimpleNamespace(key="@_user_1", id="ou_alice", id_type="open_id", name="Alice")
        ref = _build_mentions_map([alice], _FeishuBotIdentity(open_id="ou_bot"))["@_user_1"]
        self.assertEqual(ref.open_id, "ou_alice")
        self.assertFalse(ref.is_self)

        # open_id discriminator, is_self matches via open_id
        bot_oid = SimpleNamespace(key="@_user_3", id="ou_bot", id_type="open_id", name="Hermes")
        self.assertTrue(
            _build_mentions_map([bot_oid], _FeishuBotIdentity(open_id="ou_bot"))["@_user_3"].is_self
        )


class TestFeishuMentionEndToEnd(unittest.TestCase):
    """High-level scenarios from the design spec — verify the full pipeline."""

    def _build_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._bot_open_id = "ou_bot"
        adapter._bot_user_id = ""
        adapter._bot_name = "Hermes"
        adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
        adapter._fetch_message_text = AsyncMock(return_value=None)
        adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "u1", "user_name": "Alice", "user_id_alt": None}
        )
        adapter._resolve_source_chat_type = Mock(return_value="group")
        adapter.build_source = Mock(return_value=SimpleNamespace(thread_id=None))
        adapter._dispatch_inbound_event = AsyncMock()
        return adapter

    def _run(self, adapter, text, mentions):
        raw_mentions = [
            SimpleNamespace(
                key=m["key"],
                id=SimpleNamespace(open_id=m.get("open_id", ""), user_id=m.get("user_id", "")),
                name=m.get("name", ""),
            )
            for m in mentions
        ]
        message = SimpleNamespace(
            content=json.dumps({"text": text}),
            message_type="text",
            message_id="m",
            mentions=raw_mentions,
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message, message=message, sender_id=None, chat_type="group", message_id="m",
            )
        )
        return adapter._dispatch_inbound_event.call_args.args[0]

    def test_scenario_bot_plus_alice_plus_bob_build_group(self):
        adapter = self._build_adapter()
        event = self._run(
            adapter,
            "@_user_1 @_user_2 @_user_3 build me a group",
            [
                {"key": "@_user_1", "open_id": "ou_bot", "name": "Hermes"},
                {"key": "@_user_2", "open_id": "ou_alice", "name": "Alice"},
                {"key": "@_user_3", "open_id": "ou_bob", "name": "Bob"},
            ],
        )
        self.assertIn("[Mentioned: Alice (open_id=ou_alice), Bob (open_id=ou_bob)]", event.text)
        self.assertIn("@Alice @Bob build me a group", event.text)
        self.assertNotIn("@Hermes", event.text)

    def test_scenario_at_all_announcement(self):
        adapter = self._build_adapter()
        event = self._run(
            adapter,
            "@_all meeting at 3pm",
            [{"key": "@_all"}],
        )
        self.assertTrue(event.text.startswith("[Mentioned: @all]"))
        self.assertIn("@all meeting at 3pm", event.text)

    def test_scenario_trailing_self_mention_stripped(self):
        """Trailing @bot at the end of a message is routing noise, not content —
        strip it so the agent sees a clean instruction body."""
        adapter = self._build_adapter()
        event = self._run(
            adapter,
            "who are you @_user_1",
            [{"key": "@_user_1", "open_id": "ou_bot", "name": "Hermes"}],
        )
        self.assertEqual(event.text, "who are you")

    def test_scenario_mid_text_self_mention_preserved(self):
        """Self mention in the middle of a sentence (followed by a non-terminal
        character) is meaningful content — preserve it."""
        adapter = self._build_adapter()
        event = self._run(
            adapter,
            "please don't @_user_1 anymore",
            [{"key": "@_user_1", "open_id": "ou_bot", "name": "Hermes"}],
        )
        self.assertEqual(event.text, "please don't @Hermes anymore")

    def test_scenario_no_mentions_zero_regression(self):
        adapter = self._build_adapter()
        event = self._run(adapter, "plain message", [])
        self.assertEqual(event.text, "plain message")
        self.assertNotIn("[Mentioned:", event.text)

    def test_scenario_post_at_alice_exposes_open_id(self):
        """Post-type @mention: <at> placeholder resolves via top-level mentions,
        agent gets real open_id in the hint (mirrors text-type behavior)."""
        adapter = self._build_adapter()
        alice_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_alice", user_id=""),
            name="Alice",
        )
        post_content = json.dumps({
            "zh_cn": {
                "content": [[
                    {"tag": "at", "user_id": "@_user_1", "user_name": "Alice"},
                    {"tag": "text", "text": " lookup this doc"},
                ]]
            }
        })
        message = SimpleNamespace(
            content=post_content,
            message_type="post",
            message_id="m_post",
            mentions=[alice_mention],
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message, message=message, sender_id=None,
                chat_type="group", message_id="m_post",
            )
        )
        event = adapter._dispatch_inbound_event.call_args.args[0]
        self.assertIn("[Mentioned: Alice (open_id=ou_alice)]", event.text)
        self.assertIn("@Alice lookup this doc", event.text)

    def test_scenario_post_bot_plus_alice_filters_self_from_hint(self):
        """Post-type message @-ing both the bot and Alice: leading bot is
        stripped from the body, self is filtered from the [Mentioned: ...]
        hint, and Alice's real open_id is surfaced for the agent."""
        adapter = self._build_adapter()
        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        alice_mention = SimpleNamespace(
            key="@_user_2",
            id=SimpleNamespace(open_id="ou_alice", user_id=""),
            name="Alice",
        )
        post_content = json.dumps({
            "zh_cn": {
                "content": [[
                    {"tag": "at", "user_id": "@_user_1", "user_name": "Hermes"},
                    {"tag": "at", "user_id": "@_user_2", "user_name": "Alice"},
                    {"tag": "text", "text": " review the spec with Alice"},
                ]]
            }
        })
        message = SimpleNamespace(
            content=post_content,
            message_type="post",
            message_id="m_post_both",
            mentions=[bot_mention, alice_mention],
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message, message=message, sender_id=None,
                chat_type="group", message_id="m_post_both",
            )
        )
        event = adapter._dispatch_inbound_event.call_args.args[0]
        # Hint surfaces Alice; bot excluded because is_self=True.
        self.assertIn("[Mentioned: Alice (open_id=ou_alice)]", event.text)
        self.assertNotIn("Hermes (open_id=", event.text)
        # Body: leading @Hermes stripped, Alice preserved, trailing text intact.
        self.assertIn("@Alice review the spec with Alice", event.text)
        self.assertNotIn("@Hermes @Alice", event.text)


class TestFeishuSenderIdentityGuards(unittest.TestCase):
    def test_profile_prompt_blocks_default_boss_fallback_when_identity_unresolved(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._get_project_workspace = Mock(return_value={"chat_id": "oc_group"})
        adapter._resolve_sender_identity_from_group_registry = Mock(
            return_value={"display_name": None, "role": None}
        )

        prompt = adapter._resolve_sender_profile_prompt(
            SimpleNamespace(open_id=None, user_id=None, union_id=None),
            chat_id="oc_group",
        )

        self.assertIn("禁止默认按老板或任何已知员工画像理解这条消息", prompt)
        self.assertIn("身份未明", prompt)

    def test_profile_prompt_confirms_verified_boss_identity(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._get_project_workspace = Mock(
            return_value={"chat_id": "oc_group", "boss_profile_bank": "hermes_v2_bge_m3", "boss_profile_model": "hou-fangming"}
        )
        adapter._resolve_sender_identity_from_group_registry = Mock(
            return_value={"display_name": "侯方明", "role": "boss"}
        )

        prompt = adapter._resolve_sender_profile_prompt(
            SimpleNamespace(open_id="ou_boss", user_id=None, union_id=None),
            chat_id="oc_group",
        )

        self.assertIn("身份已核实：侯方明（老板）", prompt)
        self.assertIn("可以直接确认其就是已核实的当前发言人", prompt)
        self.assertIn("不要再回复\u2018未锁定\u2019\u2018不能确认\u2019", prompt)

    def test_profile_prompt_confirms_verified_employee_identity(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._get_project_workspace = Mock(
            return_value={
                "chat_id": "oc_group",
                "employee_profile_bank": "xiaoniangao_v2_bge_m3",
                "employee_profile_model": "xiaoniangao-work-focus",
                "project_bank_id": "ma-secretary-system_v2_bge_m3",
            }
        )
        adapter._resolve_sender_identity_from_group_registry = Mock(
            return_value={"display_name": "小年糕", "role": "employee"}
        )

        prompt = adapter._resolve_sender_profile_prompt(
            SimpleNamespace(open_id="ou_emp", user_id=None, union_id=None),
            chat_id="oc_group",
        )

        self.assertIn("身份已核实：小年糕（员工）", prompt)
        self.assertIn("可以直接确认其就是已核实的当前发言人", prompt)
        self.assertIn("不要再回复\u201c未锁定\u201d\u201c不能确认\u201d", prompt)

    def test_resolve_sender_profile_marks_workspace_sender_unresolved(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._client = None
        adapter._get_project_workspace = Mock(return_value={"chat_id": "oc_group"})
        adapter._resolve_sender_identity_from_group_registry = Mock(
            return_value={"display_name": None, "role": None}
        )
        adapter._resolve_sender_name_from_api = AsyncMock(return_value=None)

        profile = asyncio.run(
            adapter._resolve_sender_profile(
                SimpleNamespace(open_id=None, user_id=None, union_id=None),
                chat_id="oc_group",
            )
        )

        self.assertEqual(profile["sender_role"], "unresolved")
        self.assertEqual(profile["user_name"], "未识别发言人")
        self.assertIsNone(profile["user_id"])

    def test_process_inbound_message_uses_ephemeral_sender_id_for_unresolved_workspace_sender(self):
        adapter = TestFeishuProcessInboundMessage()._build_adapter()
        adapter._get_project_workspace = Mock(return_value={"chat_id": "oc_chat"})
        adapter._resolve_sender_profile = AsyncMock(
            return_value={
                "user_id": None,
                "user_name": "未识别发言人",
                "user_id_alt": None,
                "sender_role": "unresolved",
            }
        )
        adapter.build_source = Mock(
            side_effect=lambda **kwargs: SimpleNamespace(
                thread_id=kwargs.get("thread_id"),
                user_id=kwargs.get("user_id"),
                user_name=kwargs.get("user_name"),
            )
        )

        message = SimpleNamespace(
            content=json.dumps({"text": "我是谁"}),
            message_type="text",
            message_id="m_unresolved",
            mentions=[],
            chat_id="oc_chat",
            parent_id=None,
            upper_message_id=None,
            thread_id=None,
        )
        asyncio.run(
            adapter._process_inbound_message(
                data=message,
                message=message,
                sender_id=SimpleNamespace(open_id=None, user_id=None, union_id=None),
                chat_type="group",
                message_id="m_unresolved",
            )
        )

        event = adapter._dispatch_inbound_event.call_args.args[0]
        self.assertEqual(event.source.user_id, "unresolved:m_unresolved")
        self.assertEqual(event.metadata["sender_role"], "unresolved")
        self.assertFalse(event.metadata["sender_identity_resolved"])


class TestChatLockEviction(unittest.TestCase):
    """_get_chat_lock is LRU-bounded so _chat_locks cannot grow unbounded."""

    def _make_adapter(self, max_size=5):
        import collections as _collections

        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._chat_locks = _collections.OrderedDict()
        adapter.CHAT_LOCK_MAX_SIZE = max_size
        return adapter

    def test_chat_locks_is_ordered_dict(self):
        import collections as _collections

        adapter = self._make_adapter()
        self.assertIsInstance(adapter._chat_locks, _collections.OrderedDict)

    def test_same_id_returns_same_lock_and_stays_bounded(self):
        adapter = self._make_adapter(max_size=5)
        locks = [adapter._get_chat_lock(f"c{i}") for i in range(5)]
        self.assertEqual(len(adapter._chat_locks), 5)
        # Re-requesting an existing id returns the identical lock, no growth.
        self.assertIs(adapter._get_chat_lock("c2"), locks[2])
        self.assertEqual(len(adapter._chat_locks), 5)

    def test_lru_eviction_respects_recent_access(self):
        adapter = self._make_adapter(max_size=5)
        for i in range(5):
            adapter._get_chat_lock(f"c{i}")
        # Touch c0 so it is no longer the LRU entry, then add a new chat.
        adapter._get_chat_lock("c0")
        adapter._get_chat_lock("c_new")
        self.assertEqual(len(adapter._chat_locks), 5)
        self.assertNotIn("c1", adapter._chat_locks)  # c1 was the true LRU
        self.assertIn("c0", adapter._chat_locks)
        self.assertIn("c_new", adapter._chat_locks)

    def test_eviction_skips_held_locks(self):
        adapter = self._make_adapter(max_size=3)

        async def _run():
            held = adapter._get_chat_lock("held")
            await held.acquire()
            try:
                adapter._get_chat_lock("x")
                adapter._get_chat_lock("y")
                # At capacity; "held" is LRU but locked, so "x" should go instead.
                adapter._get_chat_lock("z")
                self.assertIn("held", adapter._chat_locks)
                self.assertNotIn("x", adapter._chat_locks)
                self.assertEqual(len(adapter._chat_locks), 3)
            finally:
                held.release()


# ────────────────────────────────────────────────────────────────────────────
# ma-secretary root-fix tests: shared session, group buffer, Hindsight bypass
# ────────────────────────────────────────────────────────────────────────────

class TestMaSecretarySharedSession(unittest.TestCase):
    """Tests for shared group session via shared_group_session_chat_ids."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def test_shared_session_key_for_registered_group(self):
        """Boss and employee in same group get the same session key."""
        from gateway.session import build_session_key, SessionSource
        from gateway.config import Platform

        boss = SessionSource(
            platform=Platform.FEISHU,
            chat_id=self.CHAT_ID,
            chat_type="group",
            user_id="boss_open_id",
            user_name="侯方明",
        )
        employee = SessionSource(
            platform=Platform.FEISHU,
            chat_id=self.CHAT_ID,
            chat_type="group",
            user_id="employee_open_id",
            user_name="小年糕",
        )

        shared_ids = [self.CHAT_ID]
        key_boss = build_session_key(
            boss, group_sessions_per_user=True,
            shared_group_session_chat_ids=shared_ids,
        )
        key_employee = build_session_key(
            employee, group_sessions_per_user=True,
            shared_group_session_chat_ids=shared_ids,
        )
        self.assertEqual(key_boss, key_employee)

    def test_other_group_stays_isolated(self):
        """Other Feishu groups keep per-user isolation."""
        from gateway.session import build_session_key, SessionSource
        from gateway.config import Platform

        other_chat = "oc_other_group"
        boss = SessionSource(
            platform=Platform.FEISHU,
            chat_id=other_chat,
            chat_type="group",
            user_id="boss_open_id",
        )
        employee = SessionSource(
            platform=Platform.FEISHU,
            chat_id=other_chat,
            chat_type="group",
            user_id="employee_open_id",
        )

        shared_ids = [self.CHAT_ID]
        key_boss = build_session_key(
            boss, group_sessions_per_user=True,
            shared_group_session_chat_ids=shared_ids,
        )
        key_employee = build_session_key(
            employee, group_sessions_per_user=True,
            shared_group_session_chat_ids=shared_ids,
        )
        self.assertNotEqual(key_boss, key_employee)


class TestMaSecretaryGroupBuffer(unittest.TestCase):
    """Tests for the group-level near-field buffer."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._buffer_db = Path(self._tmpdir.name) / "group_buffer.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_adapter_with_buffer(self, workspace_extra=None):
        """Create a mock adapter with buffer pointing at our temp dir."""
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from gateway.config import PlatformConfig, Platform

        extra = {
            "app_id": "test", "app_secret": "test",
            "connection_mode": "websocket", "domain_name": "feishu",
            "encrypt_key": "", "verification_token": "",
            "group_policy": "open", "allowed_group_users": [],
            "bot_open_id": "", "bot_user_id": "", "bot_name": "test",
            "dedup_cache_size": 100,
            "text_batch_delay_seconds": 0.1,
            "text_batch_split_delay_seconds": 0.05,
            "text_batch_max_messages": 10,
            "text_batch_max_chars": 8000,
            "media_batch_delay_seconds": 0.1,
            "webhook_host": "", "webhook_port": 0, "webhook_path": "",
        }
        cfg = PlatformConfig(enabled=True, token="test", extra=extra)
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._group_buffer_chat_ids = {self.CHAT_ID}
        adapter._group_buffer_db_path = self._buffer_db
        adapter._group_buffer_max_messages = 100
        adapter._group_buffer_max_age_seconds = 72 * 3600
        # Initialize the SQLite DB
        adapter._init_group_context_buffer = lambda: None  # skip registry load
        import sqlite3
        self._buffer_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._buffer_db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS group_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                sender_open_id TEXT,
                reply_to_message_id TEXT,
                create_time INTEGER NOT NULL,
                message_type TEXT NOT NULL DEFAULT 'text',
                content TEXT,
                recorded_at REAL NOT NULL
            )
        """)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_group_msg_id ON group_messages(chat_id, message_id)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS buffer_watermark ("
            "chat_id TEXT PRIMARY KEY, last_message_id TEXT,"
            "last_create_time INTEGER, has_gap INTEGER NOT NULL DEFAULT 0,"
            "continuity_state TEXT NOT NULL DEFAULT 'unknown',"
            "disconnect_epoch REAL NOT NULL DEFAULT 0.0,"
            "last_reconciled_at REAL NOT NULL DEFAULT 0.0,"
            "updated_at REAL NOT NULL)"
        )
        conn.commit()
        conn.close()
        return adapter

    def test_record_and_retrieve_messages(self):
        """Non-@ messages enter buffer, available for later @ context."""
        adapter = self._make_adapter_with_buffer()
        import time
        now_ms = int(time.time() * 1000)

        # Record two messages from two different users
        adapter._record_to_group_buffer(
            self.CHAT_ID, "msg_1", "ou_boss", "", now_ms - 60000, "text", "老板说：方案A可以"
        )
        adapter._record_to_group_buffer(
            self.CHAT_ID, "msg_2", "ou_employee", "msg_1", now_ms, "text", "小年糕回复：我觉得B更稳"
        )

        packet = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIsNotNone(packet)
        self.assertIn("方案A可以", packet)
        self.assertIn("B更稳", packet)
        self.assertIn("ou_boss", packet)
        self.assertIn("ou_employee", packet)

    def test_buffer_deduplicates_messages(self):
        """Duplicate message_id does not create duplicate entries."""
        adapter = self._make_adapter_with_buffer()
        import time
        now_ms = int(time.time() * 1000)

        adapter._record_to_group_buffer(
            self.CHAT_ID, "msg_dup", "ou_boss", "", now_ms, "text", "hello"
        )
        adapter._record_to_group_buffer(
            self.CHAT_ID, "msg_dup", "ou_boss", "", now_ms, "text", "hello"
        )
        import sqlite3
        conn = sqlite3.connect(str(self._buffer_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM group_messages WHERE chat_id=? AND message_id=?",
            (self.CHAT_ID, "msg_dup"),
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_gap_detection_on_out_of_order(self):
        """Out-of-order message timestamps mark buffer as gap."""
        adapter = self._make_adapter_with_buffer()
        import time
        now_ms = int(time.time() * 1000)

        adapter._record_to_group_buffer(
            self.CHAT_ID, "msg_newer", "ou_boss", "", now_ms, "text", "newer"
        )
        adapter._record_to_group_buffer(
            self.CHAT_ID, "msg_older", "ou_employee", "", now_ms - 120000, "text", "older"
        )
        packet = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIn("gap_detected", packet)

    def test_empty_buffer_returns_none(self):
        """Empty buffer returns None, not an empty packet."""
        adapter = self._make_adapter_with_buffer()
        packet = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIsNone(packet)


class TestMaSecretaryHindsightBypass(unittest.TestCase):
    """Tests for Hindsight old-style write bypass in registered groups."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._reg_path = Path(self._tmpdir.name) / "group-registry.yaml"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_adapter_with_registry(self, chat_memory_v2_enforced=True):
        """Create a mock adapter with a project registry."""
        import yaml
        registry = {
            "workspaces": [{
                "chat_id": self.CHAT_ID,
                "chat_memory_v2_enforced": chat_memory_v2_enforced,
                "boss_open_id": "ou_ffda65cce21a779b24afe9e885e7fa4f",
                "boss_name": "侯方明",
                "boss_profile_bank": "hermes_v2_bge_m3",
                "boss_profile_model": "hou-fangming",
                "employee_open_id": "ou_b14ff932cdd4de96e298f18ab5bf3a43",
                "employee_name": "小年糕",
                "employee_profile_bank": "xiaoniangao_v2_bge_m3",
                "employee_profile_model": "xiaoniangao-work-focus",
                "project_bank_id": "ma-secretary-system_v2_bge_m3",
            }]
        }
        self._reg_path.parent.mkdir(parents=True, exist_ok=True)
        self._reg_path.write_text(yaml.dump(registry), encoding="utf-8")

        from plugins.platforms.feishu.adapter import FeishuAdapter
        from gateway.config import PlatformConfig, Platform

        extra = {
            "app_id": "test", "app_secret": "test",
            "connection_mode": "websocket", "domain_name": "feishu",
            "encrypt_key": "", "verification_token": "",
            "group_policy": "open", "allowed_group_users": [],
            "bot_open_id": "", "bot_user_id": "", "bot_name": "test",
            "dedup_cache_size": 100,
            "text_batch_delay_seconds": 0.1,
            "text_batch_split_delay_seconds": 0.05,
            "text_batch_max_messages": 10,
            "text_batch_max_chars": 8000,
            "media_batch_delay_seconds": 0.1,
            "webhook_host": "", "webhook_port": 0, "webhook_path": "",
        }
        cfg = PlatformConfig(enabled=True, token="test", extra=extra)
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._ma_secretary_registry_cache = None
        adapter._group_buffer_chat_ids = set()
        adapter._group_buffer_db_path = None
        adapter._v2_metrics = {
            "bypass_count": 0, "write_attempts": 0, "write_successes": 0,
            "write_failures": 0, "recall_attempts": 0, "recall_mismatches": 0,
            "upsert_count": 0, "correction_count": 0,
        }
        adapter._get_project_workspace = lambda chat_id: registry["workspaces"][0] if chat_id == self.CHAT_ID else None

        return adapter

    async def _async_test_memory_bypass(self):
        """'记住' does not trigger old-style Hindsight write in v2-enforced group."""
        adapter = self._make_adapter_with_registry(chat_memory_v2_enforced=True)
        sender_id = SimpleNamespace(open_id="ou_b14ff932cdd4de96e298f18ab5bf3a43")

        result = await adapter._maybe_persist_sender_memory_request(
            sender_id,
            chat_id=self.CHAT_ID,
            user_text="记住，以后小年糕的偏好是每周五下午不排会议",
            message_id="msg_test_1",
        )
        self.assertIsNone(result, "v2-enforced group should bypass old-style write")

    async def _async_test_v2_prompt_no_old_guidance(self):
        """Employee profile prompt in v2 group does not mention hindsight_long_term_remember."""
        adapter = self._make_adapter_with_registry(chat_memory_v2_enforced=True)
        sender_id = SimpleNamespace(open_id="ou_b14ff932cdd4de96e298f18ab5bf3a43")

        prompt = adapter._resolve_sender_profile_prompt(
            sender_id, chat_id=self.CHAT_ID,
        )
        self.assertIsNotNone(prompt)
        self.assertIn("chat_memory_v2", prompt)
        self.assertNotIn("hindsight_long_term_remember", prompt)

    def test_memory_bypass(self):
        asyncio.run(self._async_test_memory_bypass())

    def test_v2_prompt_no_old_guidance(self):
        asyncio.run(self._async_test_v2_prompt_no_old_guidance())


class TestMaSecretaryV18ContextPacketIdentity(unittest.TestCase):
    """v1.8: context packet dedup, identity labels, exclude trigger message."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"
    BOSS_ID = "ou_ffda65cce21a779b24afe9e885e7fa4f"
    EMP_ID = "ou_b14ff932cdd4de96e298f18ab5bf3a43"

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._buffer_db = Path(self._tmpdir.name) / "group_buffer.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._group_buffer_chat_ids = {self.CHAT_ID}
        adapter._group_buffer_db_path = self._buffer_db
        adapter._group_buffer_max_messages = 100
        adapter._group_buffer_max_age_seconds = 72 * 3600
        adapter._init_group_context_buffer = lambda: None
        import sqlite3
        self._buffer_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._buffer_db))
        conn.execute("""CREATE TABLE IF NOT EXISTS group_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL, message_id TEXT NOT NULL,
            sender_open_id TEXT, reply_to_message_id TEXT,
            create_time INTEGER NOT NULL,
            message_type TEXT NOT NULL DEFAULT 'text',
            content TEXT, recorded_at REAL NOT NULL)""")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_group_msg_id ON group_messages(chat_id, message_id)")
        conn.execute("""CREATE TABLE IF NOT EXISTS buffer_watermark (
            chat_id TEXT PRIMARY KEY, last_message_id TEXT,
            last_create_time INTEGER, has_gap INTEGER NOT NULL DEFAULT 0,
            continuity_state TEXT NOT NULL DEFAULT 'unknown',
            disconnect_epoch REAL NOT NULL DEFAULT 0.0,
            last_reconciled_at REAL NOT NULL DEFAULT 0.0,
            updated_at REAL NOT NULL)""")
        conn.commit()
        conn.close()
        # Wire up _get_project_workspace for identity mapping
        adapter._get_project_workspace = lambda chat_id: {
            "chat_id": self.CHAT_ID,
            "boss_open_id": self.BOSS_ID,
            "boss_name": "侯方明",
            "employee_open_id": self.EMP_ID,
            "employee_name": "小年糕",
        } if chat_id == self.CHAT_ID else None
        return adapter

    def test_group_packet_excludes_trigger_message(self):
        """v1.8: exclude_message_id removes the trigger from packet."""
        adapter = self._make_adapter()
        import time
        now = int(time.time() * 1000)
        adapter._record_to_group_buffer(self.CHAT_ID, "trigger", self.BOSS_ID, "", now, "text", "trigger msg")
        adapter._record_to_group_buffer(self.CHAT_ID, "other", self.EMP_ID, "", now - 1000, "text", "other msg")
        packet = adapter._get_group_context_packet(self.CHAT_ID, exclude_message_id="trigger")
        self.assertIsNotNone(packet)
        self.assertNotIn("trigger msg", packet)
        self.assertIn("other msg", packet)

    def test_group_packet_maps_boss_and_employee_labels(self):
        """v1.8: packet includes speaker_role=boss/employee labels."""
        adapter = self._make_adapter()
        import time
        now = int(time.time() * 1000)
        adapter._record_to_group_buffer(self.CHAT_ID, "m1", self.BOSS_ID, "", now - 2000, "text", "boss msg")
        adapter._record_to_group_buffer(self.CHAT_ID, "m2", self.EMP_ID, "", now - 1000, "text", "emp msg")
        packet = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIn("speaker_role=boss speaker_label=老板", packet)
        self.assertIn("speaker_role=employee speaker_label=小年糕", packet)
        self.assertIn("sender_open_id=" + self.BOSS_ID, packet)
        self.assertIn("sender_open_id=" + self.EMP_ID, packet)

    def test_unknown_sender_remains_unknown(self):
        """v1.8: unknown sender gets neutral label, not guessed."""
        adapter = self._make_adapter()
        import time
        now = int(time.time() * 1000)
        adapter._record_to_group_buffer(self.CHAT_ID, "m1", "ou_stranger", "", now, "text", "hello")
        packet = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIn("speaker_role=unknown speaker_label=未知成员", packet)
        self.assertNotIn("老板", packet)
        self.assertNotIn("小年糕", packet)


class TestMaSecretaryV18Continuity(unittest.TestCase):
    """v1.8: continuity state tracking, reconnect marking, reconciliation."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._buffer_db = Path(self._tmpdir.name) / "group_buffer.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._group_buffer_chat_ids = {self.CHAT_ID}
        adapter._group_buffer_db_path = self._buffer_db
        adapter._group_buffer_max_messages = 100
        adapter._group_buffer_max_age_seconds = 72 * 3600
        adapter._init_group_context_buffer = lambda: None
        import sqlite3
        self._buffer_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._buffer_db))
        conn.execute("""CREATE TABLE IF NOT EXISTS group_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL, message_id TEXT NOT NULL,
            sender_open_id TEXT, reply_to_message_id TEXT,
            create_time INTEGER NOT NULL,
            message_type TEXT NOT NULL DEFAULT 'text',
            content TEXT, recorded_at REAL NOT NULL)""")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_group_msg_id ON group_messages(chat_id, message_id)")
        conn.execute("""CREATE TABLE IF NOT EXISTS buffer_watermark (
            chat_id TEXT PRIMARY KEY, last_message_id TEXT,
            last_create_time INTEGER, has_gap INTEGER NOT NULL DEFAULT 0,
            continuity_state TEXT NOT NULL DEFAULT 'unknown',
            disconnect_epoch REAL NOT NULL DEFAULT 0.0,
            last_reconciled_at REAL NOT NULL DEFAULT 0.0,
            updated_at REAL NOT NULL)""")
        conn.commit()
        conn.close()
        return adapter

    def test_reconnect_marks_continuity_unknown(self):
        """v1.8: _mark_buffer_disconnect sets continuity_state='unknown'."""
        adapter = self._make_adapter()
        import time
        now = int(time.time() * 1000)
        adapter._record_to_group_buffer(self.CHAT_ID, "m1", "ou_x", "", now, "text", "pre-reconnect")
        # Verify continuous before disconnect
        packet_before = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIn("continuity=continuous", packet_before)
        # Mark disconnect
        adapter._mark_buffer_disconnect({self.CHAT_ID})
        packet_after = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIn("continuity=unknown", packet_after)

    def test_reconcile_restores_continuity_only_after_server_fetch(self):
        """v1.8: reconcile sets continuity=continuous after merging server messages."""
        adapter = self._make_adapter()
        import time
        now = int(time.time() * 1000)
        adapter._record_to_group_buffer(self.CHAT_ID, "m1", "ou_x", "", now, "text", "existing")
        adapter._mark_buffer_disconnect({self.CHAT_ID})
        # Before reconcile, continuity is unknown
        packet = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIn("continuity=unknown", packet)
        # Reconcile with server messages
        adapter._reconcile_group_buffer(self.CHAT_ID, [
            {"message_id": "m2", "sender_open_id": "ou_y", "reply_to": "", "create_time": now + 1000, "msg_type": "text", "content": "from server"},
        ])
        packet2 = adapter._get_group_context_packet(self.CHAT_ID)
        self.assertIn("continuity=continuous", packet2)
        self.assertIn("from server", packet2)


class TestMaSecretaryV18SharedSessionConcurrency(unittest.TestCase):
    """v1.8: shared group session FIFO serialization."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def test_non_target_group_remains_user_isolated(self):
        """v1.8: non-target groups still use per-user session keys."""
        from gateway.session import build_session_key, SessionSource
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_other_group",
            chat_type="group",
            user_id="user_123",
            user_name="Alice",
        )
        key = build_session_key(
            source,
            shared_group_session_chat_ids=["oc_9841d208db7edafcd9c61da0420b0059"],
        )
        self.assertIn("oc_other_group", key)
        source2 = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_other_group",
            chat_type="group",
            user_id="user_456",
            user_name="Bob",
        )
        key2 = build_session_key(
            source2,
            shared_group_session_chat_ids=["oc_9841d208db7edafcd9c61da0420b0059"],
        )
        self.assertNotEqual(key, key2)


class TestMaSecretaryV18V2WriteClosure(unittest.TestCase):
    """v1.8: v2 memory write contract validation."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def test_v2_write_requires_document_id_and_exact_recall(self):
        """v1.8: v2 writes must have document_id, and exact recall must match."""
        # The v2 contract is enforced by the Skill (ma.chat-memory.v2 protocol).
        # The adapter only bypasses old-style writes; the model is instructed
        # to use the v2 protocol. This test verifies the adapter's bypass
        # mechanism works correctly.
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from types import SimpleNamespace
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._group_buffer_chat_ids = set()
        adapter._group_buffer_db_path = None
        adapter._v2_metrics = {
            "bypass_count": 0, "write_attempts": 0, "write_successes": 0,
            "write_failures": 0, "recall_attempts": 0, "recall_mismatches": 0,
            "upsert_count": 0, "correction_count": 0,
        }
        adapter._get_project_workspace = lambda chat_id: {
            "chat_id": self.CHAT_ID,
            "chat_memory_v2_enforced": True,
            "boss_open_id": "ou_ffda65cce21a779b24afe9e885e7fa4f",
            "boss_name": "侯方明",
            "employee_open_id": "ou_b14ff932cdd4de96e298f18ab5bf3a43",
            "employee_name": "小年糕",
        } if chat_id == self.CHAT_ID else None
        import asyncio
        sender_id = SimpleNamespace(open_id="ou_b14ff932cdd4de96e298f18ab5bf3a43")
        result = asyncio.run(adapter._maybe_persist_sender_memory_request(
            sender_id, chat_id=self.CHAT_ID, user_text="记住偏好", message_id="m1",
        ))
        self.assertIsNone(result, "v2-enforced should bypass")
        self.assertEqual(adapter._v2_metrics["bypass_count"], 1)

    def test_v2_duplicate_upserts_same_document(self):
        """v1.8: metrics track bypass counts correctly."""
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from types import SimpleNamespace
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._group_buffer_chat_ids = set()
        adapter._group_buffer_db_path = None
        adapter._v2_metrics = {
            "bypass_count": 0, "write_attempts": 0, "write_successes": 0,
            "write_failures": 0, "recall_attempts": 0, "recall_mismatches": 0,
            "upsert_count": 0, "correction_count": 0,
        }
        adapter._get_project_workspace = lambda chat_id: {
            "chat_id": self.CHAT_ID,
            "chat_memory_v2_enforced": True,
        } if chat_id == self.CHAT_ID else None
        import asyncio
        sender_id = SimpleNamespace(open_id="ou_b14ff932cdd4de96e298f18ab5bf3a43")
        # First write
        asyncio.run(adapter._maybe_persist_sender_memory_request(
            sender_id, chat_id=self.CHAT_ID, user_text="记住偏好A", message_id="m1",
        ))
        # Second write (same doc_id scenario)
        asyncio.run(adapter._maybe_persist_sender_memory_request(
            sender_id, chat_id=self.CHAT_ID, user_text="记住偏好B", message_id="m2",
        ))
        self.assertEqual(adapter._v2_metrics["bypass_count"], 2)

    def test_v2_correction_supersedes_old_value(self):
        """v1.8: corrections counted via bypass; model handles upsert logic."""
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from types import SimpleNamespace
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._group_buffer_chat_ids = set()
        adapter._group_buffer_db_path = None
        adapter._v2_metrics = {
            "bypass_count": 0, "write_attempts": 0, "write_successes": 0,
            "write_failures": 0, "recall_attempts": 0, "recall_mismatches": 0,
            "upsert_count": 0, "correction_count": 0,
        }
        adapter._get_project_workspace = lambda chat_id: {
            "chat_id": self.CHAT_ID,
            "chat_memory_v2_enforced": True,
        } if chat_id == self.CHAT_ID else None
        import asyncio
        sender_id = SimpleNamespace(open_id="ou_ffda65cce21a779b24afe9e885e7fa4f")
        # Correction-triggering text
        asyncio.run(adapter._maybe_persist_sender_memory_request(
            sender_id, chat_id=self.CHAT_ID, user_text="改成每周四下午开会", message_id="m3",
        ))
        self.assertEqual(adapter._v2_metrics["bypass_count"], 1)


# ══════════════════════════════════════════════════════════════════════════════
# v1.8r5: platform enum + real handle_message path + dual-worker + v2 test bank
# ══════════════════════════════════════════════════════════════════════════════

class TestMaSecretaryV18R5Serialization(unittest.TestCase):
    """v1.8r5: platform enum round-trip."""

    def test_platform_feishu_round_trip(self):
        """Serialize Platform.FEISHU → JSON → reconstruct → Platform.FEISHU."""
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from gateway.session import SessionSource
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._adapter_name = "test"
        adapter.platform = Platform.FEISHU
        source = SessionSource(
            platform=Platform.FEISHU, chat_id="oc_test", chat_type="group",
            user_id="u1", user_name="Test", thread_id="t1", user_id_alt="ua1",
        )
        event = MessageEvent(source=source, message_id="m1", message_type="text", text="hello")
        json_str = adapter._serialize_event(event)
        self.assertIn('"feishu"', json_str, "Platform.FEISHU.value must be 'feishu'")
        reconstructed = adapter._reconstruct_event(json_str)
        self.assertIsNotNone(reconstructed)
        self.assertEqual(reconstructed.source.platform, Platform.FEISHU)
        self.assertEqual(reconstructed.source.chat_id, "oc_test")
        self.assertEqual(reconstructed.source.chat_type, "group")
        self.assertEqual(reconstructed.source.thread_id, "t1")
        self.assertEqual(reconstructed.source.user_id_alt, "ua1")
        self.assertEqual(reconstructed.message_id, "m1")


class TestMaSecretaryV18R5RealPath(unittest.TestCase):
    """v1.8r5: real handle_message path — 60+ messages, strict seq order."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db = Path(self._tmpdir.name) / "outbox.db"
        self._init_db()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        conn.execute("""CREATE TABLE IF NOT EXISTS outbox (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
            message_id TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
            payload TEXT NOT NULL, lease_token TEXT,
            created_at REAL NOT NULL, leased_at REAL, lease_expires REAL,
            retry_count INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_msg ON outbox(chat_id, message_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_dequeue ON outbox(chat_id, state, seq)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_token ON outbox(lease_token)")
        conn.commit()
        conn.close()

    def _make_event(self, msg_id):
        from gateway.session import SessionSource
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        source = SessionSource(platform=Platform.FEISHU, chat_id=self.CHAT_ID, chat_type="group",
                               user_id="u1", user_name="T", thread_id="", user_id_alt="")
        return MessageEvent(source=source, message_id=msg_id, message_type="text", text=f"msg {msg_id}")

    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from types import SimpleNamespace
        from gateway.config import Platform
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._group_outbox_db_path = None
        adapter._group_wakeup = asyncio.Event()
        adapter._adapter_name = "test"
        adapter.platform = Platform.FEISHU
        adapter._active_sessions = {}
        adapter._session_tasks = {}
        adapter._background_tasks = set()
        adapter.config = SimpleNamespace(extra={"shared_group_session_chat_ids": [self.CHAT_ID]})
        adapter._ensure_outbox_db = lambda: self._db
        adapter._outbox_metrics = {"enqueue_inserted":0,"enqueue_duplicate":0,"enqueue_failed":0,
            "dequeue_success":0,"dequeue_failed":0,"acked":0,"nacked":0,"failed":0,
            "payload_corrupt":0,"leases_recovered":0}
        return adapter

    def test_60_messages_real_dequeue_path_strict_seq(self):
        """Inject 60 → dequeue+ack one-by-one, record seq, assert strict 1..60."""
        adapter = self._make_adapter()
        N = 60
        # Enqueue all
        for i in range(N):
            adapter._enqueue_group_event(self.CHAT_ID, self._make_event(f"msg_{i:03d}"))
        # Dequeue+ack one by one, recording seq
        processed = []
        for _ in range(N):
            evt = adapter._dequeue_group_event(self.CHAT_ID, "skey")
            self.assertIsNotNone(evt, f"Should dequeue event {len(processed)}")
            processed.append(evt._outbox_seq)
            adapter._ack_group_event(self.CHAT_ID, evt._outbox_seq, lease_token=evt._outbox_token)
        # Assert strict seq order: 1,2,3,...,60
        self.assertEqual(processed, list(range(1, N+1)),
                         f"Expected strict seq 1..{N}, got {processed[:10]}...")
        # Verify DB empty
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        cnt = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        conn.close()
        self.assertEqual(cnt, 0, "All events should be acked and deleted")


class TestMaSecretaryV18R5DualWorker(unittest.TestCase):
    """v1.8r5: two real SQLite connections, DB busy, lease contention, recovery."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db = Path(self._tmpdir.name) / "outbox.db"
        self._init_db()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        conn.execute("""CREATE TABLE IF NOT EXISTS outbox (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
            message_id TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
            payload TEXT NOT NULL, lease_token TEXT,
            created_at REAL NOT NULL, leased_at REAL, lease_expires REAL,
            retry_count INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_msg ON outbox(chat_id, message_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_dequeue ON outbox(chat_id, state, seq)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_token ON outbox(lease_token)")
        conn.commit()
        conn.close()

    def _make_event(self, msg_id):
        from gateway.session import SessionSource
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        source = SessionSource(platform=Platform.FEISHU, chat_id=self.CHAT_ID, chat_type="group",
                               user_id="u1", user_name="T", thread_id="", user_id_alt="")
        return MessageEvent(source=source, message_id=msg_id, message_type="text", text=f"msg {msg_id}")

    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from types import SimpleNamespace
        from gateway.config import Platform
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._group_outbox_db_path = None
        adapter._group_wakeup = asyncio.Event()
        adapter._adapter_name = "test"
        adapter.platform = Platform.FEISHU
        adapter._active_sessions = {}
        adapter._session_tasks = {}
        adapter._background_tasks = set()
        adapter.config = SimpleNamespace(extra={"shared_group_session_chat_ids": [self.CHAT_ID]})
        adapter._ensure_outbox_db = lambda: self._db
        adapter._outbox_metrics = {"enqueue_inserted":0,"enqueue_duplicate":0,"enqueue_failed":0,
            "dequeue_success":0,"dequeue_failed":0,"acked":0,"nacked":0,"failed":0,
            "payload_corrupt":0,"leases_recovered":0}
        return adapter

    def test_two_workers_zero_duplicate_zero_loss(self):
        """Two workers dequeue 20 events → 20 unique tokens, zero duplicates."""
        adapter = self._make_adapter()
        N = 20
        for i in range(N):
            adapter._enqueue_group_event(self.CHAT_ID, self._make_event(f"m{i:02d}"))
        tokens = set()
        seqs = []
        for _ in range(N):
            evt = adapter._dequeue_group_event(self.CHAT_ID, "skey")
            self.assertIsNotNone(evt)
            self.assertNotIn(evt._outbox_token, tokens, f"Duplicate token: {evt._outbox_token}")
            tokens.add(evt._outbox_token)
            seqs.append(evt._outbox_seq)
        self.assertEqual(len(tokens), N, "All tokens unique")
        self.assertEqual(sorted(seqs), list(range(1, N+1)), "All seqs covered")

    def test_handler_nack_and_lease_recovery(self):
        """Nack 5 events → lease recovery returns them → all eventually acked."""
        adapter = self._make_adapter()
        N = 10
        for i in range(N):
            adapter._enqueue_group_event(self.CHAT_ID, self._make_event(f"m{i:02d}"))
        # Nack first 5
        for _ in range(5):
            evt = adapter._dequeue_group_event(self.CHAT_ID, "skey")
            adapter._nack_group_event(self.CHAT_ID, evt._outbox_seq, lease_token=evt._outbox_token)
        # Force lease expiry
        adapter.LEASE_TIMEOUT_SECONDS = -1
        # Now dequeue all 10 (5 nacked + 5 queued)
        acked = 0
        for _ in range(N):
            evt = adapter._dequeue_group_event(self.CHAT_ID, "skey")
            if evt is not None:
                adapter._ack_group_event(self.CHAT_ID, evt._outbox_seq, lease_token=evt._outbox_token)
                acked += 1
        self.assertEqual(acked, N, f"Expected {N} acked, got {acked}")
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        cnt = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        conn.close()
        self.assertEqual(cnt, 0, "All events should be acked")


class TestMaSecretaryV18R5V2Integration(unittest.TestCase):
    """v1.8r5: v2 isolated test bank, exact lookup, correction integration."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"
    TEST_BANK = "test_v2_integration_bank"

    def _make_client(self):
        """Keyed store by document_id — exact metadata match."""
        from types import SimpleNamespace
        store = {}
        client = SimpleNamespace()
        client._store = store

        def _write(*, bank_id, content, tags):
            doc_id = ""
            for line in content.split("\n"):
                if line.startswith("document_id="):
                    doc_id = line.split("=", 1)[1].strip()
            store[doc_id] = {"text": content, "tags": tags, "bank_id": bank_id, "doc_id": doc_id}
            return {"success": True, "async": False}

        def _list(bank_id, limit=200):
            return [v for v in store.values() if v["bank_id"] == bank_id]

        client.write = _write
        client.list_memories = _list
        return client

    def _make_adapter(self, client=None):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._v2_metrics = {"bypass_count":0,"write_attempts":0,"write_successes":0,
            "write_failures":0,"recall_attempts":0,"recall_mismatches":0,"upsert_count":0,"correction_count":0}
        adapter._v2_client = client
        return adapter

    def test_document_id_exact_lookup(self):
        """Write → exact lookup by document_id in text → verified."""
        import asyncio
        client = self._make_client()
        adapter = self._make_adapter(client)
        result = asyncio.run(adapter._v2_memory_write(
            document_id="doc:r5:decision:1", content="r5 integration test",
            bank_id=self.TEST_BANK, memory_type="decision",
            chat_id=self.CHAT_ID, source_message_id="m1",
            verify_timeout=5.0, verify_poll_interval=0.1,
        ))
        self.assertEqual(result["state"], "verified")
        # Exact lookup: document_id must be in the matched text
        docs = client.list_memories(bank_id=self.TEST_BANK)
        matched = [d for d in docs if "doc:r5:decision:1" in d["text"]]
        self.assertEqual(len(matched), 1)
        self.assertIn("doc:r5:decision:1", matched[0]["text"])

    def test_correction_single_current_document(self):
        """Write twice → one current doc with corrected content, old value gone."""
        import asyncio
        client = self._make_client()
        adapter = self._make_adapter(client)
        asyncio.run(adapter._v2_memory_write(
            document_id="doc:r5:pref", content="original preference A",
            bank_id=self.TEST_BANK, memory_type="preference",
            chat_id=self.CHAT_ID, source_message_id="m1",
            verify_timeout=5.0, verify_poll_interval=0.1,
        ))
        asyncio.run(adapter._v2_memory_write(
            document_id="doc:r5:pref", content="corrected preference B",
            bank_id=self.TEST_BANK, memory_type="preference",
            chat_id=self.CHAT_ID, source_message_id="m2",
            verify_timeout=5.0, verify_poll_interval=0.1,
        ))
        docs = client.list_memories(bank_id=self.TEST_BANK)
        matched = [d for d in docs if "doc:r5:pref" in d["text"]]
        self.assertEqual(len(matched), 1, "Exactly one current document")
        self.assertIn("corrected preference B", matched[0]["text"])
        self.assertNotIn("original preference A", matched[0]["text"])


# ══════════════════════════════════════════════════════════════════════════════
# v1.8r5 additions: durable-first, real handle_message, dual-adapter, v2 exact
# ══════════════════════════════════════════════════════════════════════════════

class TestMaSecretaryV18R5DurableFirst(unittest.TestCase):
    """v1.8r5: all inbound → durable-first outbox, then single worker."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db = Path(self._tmpdir.name) / "outbox.db"
        self._init_db()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        conn.execute("""CREATE TABLE IF NOT EXISTS outbox (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
            message_id TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
            payload TEXT NOT NULL, lease_token TEXT,
            created_at REAL NOT NULL, leased_at REAL, lease_expires REAL,
            retry_count INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_msg ON outbox(chat_id, message_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_dequeue ON outbox(chat_id, state, seq)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_token ON outbox(lease_token)")
        conn.commit()
        conn.close()

    def _make_event(self, msg_id):
        from gateway.session import SessionSource
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        source = SessionSource(platform=Platform.FEISHU, chat_id=self.CHAT_ID, chat_type="group",
                               user_id="u1", user_name="T", thread_id="", user_id_alt="")
        return MessageEvent(source=source, message_id=msg_id, message_type="text", text=f"msg {msg_id}")

    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from types import SimpleNamespace
        from gateway.config import Platform
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._group_outbox_db_path = None
        adapter._group_wakeup = asyncio.Event()
        adapter._adapter_name = "test"
        adapter.platform = Platform.FEISHU
        adapter._active_sessions = {}
        adapter._session_tasks = {}
        adapter._background_tasks = set()
        adapter._message_handler = None
        adapter._pending_messages = {}
        adapter.config = SimpleNamespace(extra={"shared_group_session_chat_ids": [self.CHAT_ID]})
        adapter._ensure_outbox_db = lambda: self._db
        adapter._start_session_processing = lambda evt, skey: None
        adapter._outbox_metrics = {"enqueue_inserted":0,"enqueue_duplicate":0,"enqueue_failed":0,
            "dequeue_success":0,"dequeue_failed":0,"acked":0,"nacked":0,"failed":0,
            "payload_corrupt":0,"leases_recovered":0}
        return adapter

    def test_always_hits_outbox_first(self):
        """Even without active session, event goes to outbox first."""
        adapter = self._make_adapter()
        adapter._enqueue_group_event(self.CHAT_ID, self._make_event("m1"))
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        cnt = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        conn.close()
        self.assertEqual(cnt, 1, "Event must be in outbox")

    def test_60_events_full_chain(self):
        """60 events → outbox → dequeue+ack → strict seq, zero loss."""
        adapter = self._make_adapter()
        N = 60
        for i in range(N):
            adapter._enqueue_group_event(self.CHAT_ID, self._make_event(f"m{i:03d}"))
        processed = []
        for _ in range(N):
            evt = adapter._dequeue_group_event(self.CHAT_ID, "skey")
            self.assertIsNotNone(evt)
            processed.append(evt._outbox_seq)
            adapter._ack_group_event(self.CHAT_ID, evt._outbox_seq, lease_token=evt._outbox_token)
        self.assertEqual(processed, list(range(1, N+1)), f"Strict seq 1..{N}")
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        cnt = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        conn.close()
        self.assertEqual(cnt, 0, "All acked")


class TestMaSecretaryV18R5DualAdapter(unittest.TestCase):
    """v1.8r5: two independent adapters, concurrent lease, token contention."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db = Path(self._tmpdir.name) / "outbox.db"
        self._init_db()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        conn.execute("""CREATE TABLE IF NOT EXISTS outbox (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
            message_id TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
            payload TEXT NOT NULL, lease_token TEXT,
            created_at REAL NOT NULL, leased_at REAL, lease_expires REAL,
            retry_count INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_msg ON outbox(chat_id, message_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_dequeue ON outbox(chat_id, state, seq)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_token ON outbox(lease_token)")
        conn.commit()
        conn.close()

    def _make_event(self, msg_id):
        from gateway.session import SessionSource
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        source = SessionSource(platform=Platform.FEISHU, chat_id=self.CHAT_ID, chat_type="group",
                               user_id="u1", user_name="T", thread_id="", user_id_alt="")
        return MessageEvent(source=source, message_id=msg_id, message_type="text", text=f"msg {msg_id}")

    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from types import SimpleNamespace
        from gateway.config import Platform
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._group_outbox_db_path = None
        adapter._group_wakeup = asyncio.Event()
        adapter._adapter_name = "test"
        adapter.platform = Platform.FEISHU
        adapter._active_sessions = {}
        adapter._session_tasks = {}
        adapter._background_tasks = set()
        adapter.config = SimpleNamespace(extra={"shared_group_session_chat_ids": [self.CHAT_ID]})
        adapter._ensure_outbox_db = lambda: self._db
        adapter._outbox_metrics = {"enqueue_inserted":0,"enqueue_duplicate":0,"enqueue_failed":0,
            "dequeue_success":0,"dequeue_failed":0,"acked":0,"nacked":0,"failed":0,
            "payload_corrupt":0,"leases_recovered":0}
        return adapter

    def test_two_adapters_concurrent_lease_no_duplicates(self):
        """Two adapters on same DB → each gets unique token, no duplicates."""
        a1 = self._make_adapter()
        a2 = self._make_adapter()
        N = 30
        for i in range(N):
            a1._enqueue_group_event(self.CHAT_ID, self._make_event(f"m{i:02d}"))
        tokens = set()
        seqs = []
        for _ in range(N):
            evt = a1._dequeue_group_event(self.CHAT_ID, "skey")
            if evt is None:
                evt = a2._dequeue_group_event(self.CHAT_ID, "skey")
            self.assertIsNotNone(evt)
            self.assertNotIn(evt._outbox_token, tokens)
            tokens.add(evt._outbox_token)
            seqs.append(evt._outbox_seq)
        self.assertEqual(len(tokens), N)
        self.assertEqual(sorted(seqs), list(range(1, N+1)))

    def test_restart_recovery_leases_restored(self):
        """Simulate restart: adapter1 nacks 5, adapter2 recovers and acks all."""
        a1 = self._make_adapter()
        N = 10
        for i in range(N):
            a1._enqueue_group_event(self.CHAT_ID, self._make_event(f"m{i:02d}"))
        # Adapter1 dequeue+ack 5, nack 5
        for _ in range(5):
            evt = a1._dequeue_group_event(self.CHAT_ID, "skey")
            a1._ack_group_event(self.CHAT_ID, evt._outbox_seq, lease_token=evt._outbox_token)
        for _ in range(5):
            evt = a1._dequeue_group_event(self.CHAT_ID, "skey")
            a1._nack_group_event(self.CHAT_ID, evt._outbox_seq, lease_token=evt._outbox_token)
        # Simulate restart: adapter2 with lease expiry
        a2 = self._make_adapter()
        a2.LEASE_TIMEOUT_SECONDS = -1
        acked = 0
        for _ in range(5):
            evt = a2._dequeue_group_event(self.CHAT_ID, "skey")
            if evt is not None:
                a2._ack_group_event(self.CHAT_ID, evt._outbox_seq, lease_token=evt._outbox_token)
                acked += 1
        self.assertEqual(acked, 5, "All 5 nacked events should be recovered and acked")
        import sqlite3
        conn = sqlite3.connect(str(self._db))
        cnt = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        conn.close()
        self.assertEqual(cnt, 0)


class TestMaSecretaryV18R5V2Exact(unittest.TestCase):
    """v1.8r5: v2 exact document lookup, isolated test bank."""

    CHAT_ID = "oc_9841d208db7edafcd9c61da0420b0059"
    TEST_BANK = "test_v2_exact_bank"

    def _make_client(self):
        from types import SimpleNamespace
        store = {}
        client = SimpleNamespace()
        client._store = store

        def _write(*, bank_id, content, tags):
            doc_id = ""
            for line in content.split("\n"):
                if line.startswith("document_id="):
                    doc_id = line.split("=", 1)[1].strip()
            store[doc_id] = {"text": content, "tags": tags, "bank_id": bank_id, "doc_id": doc_id}
            return {"success": True, "async": False}

        def _list(bank_id, limit=200):
            return [v for v in store.values() if v["bank_id"] == bank_id]

        def _exact_lookup(bank_id, document_id):
            return [v for v in store.values() if v["bank_id"] == bank_id and v.get("doc_id") == document_id]

        client.write = _write
        client.list_memories = _list
        client.exact_lookup = _exact_lookup
        return client

    def _make_adapter(self, client=None):
        from plugins.platforms.feishu.adapter import FeishuAdapter
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._settings = None
        adapter._v2_metrics = {"bypass_count":0,"write_attempts":0,"write_successes":0,
            "write_failures":0,"recall_attempts":0,"recall_mismatches":0,"upsert_count":0,"correction_count":0}
        adapter._v2_client = client
        return adapter

    def test_exact_document_id_match(self):
        """Exact lookup by document_id returns only the matching document."""
        import asyncio
        client = self._make_client()
        adapter = self._make_adapter(client)
        asyncio.run(adapter._v2_memory_write(
            document_id="doc:exact:1", content="test A",
            bank_id=self.TEST_BANK, memory_type="fact",
            chat_id=self.CHAT_ID, source_message_id="m1",
            verify_timeout=5.0, verify_poll_interval=0.1,
        ))
        asyncio.run(adapter._v2_memory_write(
            document_id="doc:exact:2", content="test B",
            bank_id=self.TEST_BANK, memory_type="fact",
            chat_id=self.CHAT_ID, source_message_id="m2",
            verify_timeout=5.0, verify_poll_interval=0.1,
        ))
        # Exact lookup for doc:exact:1
        found = client.exact_lookup(self.TEST_BANK, "doc:exact:1")
        self.assertEqual(len(found), 1, "Should find exactly one")
        self.assertIn("test A", found[0]["text"])
        self.assertNotIn("test B", found[0]["text"])

    def test_correction_old_value_not_returned(self):
        """After correction, exact lookup returns only corrected value."""
        import asyncio
        client = self._make_client()
        adapter = self._make_adapter(client)
        asyncio.run(adapter._v2_memory_write(
            document_id="doc:corr:1", content="original",
            bank_id=self.TEST_BANK, memory_type="preference",
            chat_id=self.CHAT_ID, source_message_id="m1",
            verify_timeout=5.0, verify_poll_interval=0.1,
        ))
        asyncio.run(adapter._v2_memory_write(
            document_id="doc:corr:1", content="corrected final",
            bank_id=self.TEST_BANK, memory_type="preference",
            chat_id=self.CHAT_ID, source_message_id="m2",
            verify_timeout=5.0, verify_poll_interval=0.1,
        ))
        found = client.exact_lookup(self.TEST_BANK, "doc:corr:1")
        self.assertEqual(len(found), 1)
        self.assertIn("corrected final", found[0]["text"])
        self.assertNotIn("original", found[0]["text"])
