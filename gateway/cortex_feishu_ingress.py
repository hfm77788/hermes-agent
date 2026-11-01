"""Bounded deterministic Feishu -> existing Hermes Cortex publisher.

Runs *after* gateway auth/admission, before an AI model turn. This is task
publication, NOT ChatGPT activation or a new authorizing surface.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import re

logger = logging.getLogger(__name__)

_REGISTRY = Path("/home/ubuntu/.hermes/projects/ma-secretary-interaction-system/group-registry.yaml")
_MCP_REPOSITORY = Path("/home/ubuntu/hermes-mcp-server/repository")
_MCP_PYTHON = _MCP_REPOSITORY / ".venv/bin/python"
_WORKER = Path(__file__).with_name("cortex_feishu_publish.py")
_TASK_INTENT = re.compile(
    r"^\s*(?:这是一次性任务|任务\s*[:：]|请(?:帮我|让|安排|执行|处理|检查|分析|生成|完成)"
    r"|帮我|安排|执行|处理|每天|每日|每周|每月|每小时|定期|一次性任务)",
    re.IGNORECASE,
)


def authenticated_goal(source: dict, text: str, message_id: str) -> dict | None:
    """Restrict direct publication to configured group OWNER; no auth from text."""
    if not isinstance(source, dict) or not isinstance(text, str):
        return None
    if not isinstance(message_id, str) or not re.fullmatch(r"om_[a-zA-Z0-9]{12,120}", message_id):
        return None
    if source.get("platform") != "feishu" or source.get("chat_type") != "group":
        return None
    if source.get("is_bot") or not 1 <= len(text.strip()) <= 2000:
        return None
    if text.lstrip().startswith("/") or not _TASK_INTENT.match(text):
        return None
    try:
        import yaml
        registry = yaml.safe_load(_REGISTRY.read_text(encoding="utf-8"))
        groups = registry.get("workspaces")
        if not isinstance(groups, list):
            # Other registry layouts are not silently guessed.
            return None
        matches = [g for g in groups if isinstance(g, dict)
                   and g.get("chat_id") == source.get("chat_id")
                   and g.get("chat_name") in {"GPT 河马协作群", "微信 ai 秘书"}]
        if len(matches) != 1:
            return None
        group = matches[0]
        if group.get("status") != "active" or group.get("cortex_direct_dispatch") is not True:
            return None
        if (not isinstance(group.get("boss_open_id"), str) or
                source.get("user_id") != group["boss_open_id"]):
            return None
    except Exception as exc:
        # Fail without routing when policy cannot be loaded/parsed. The user
        # identity was not established from a trusted local group registry.
        logger.warning("Cortex Feishu group registry unavailable: %s", type(exc).__name__)
        return None
    return {
        "chat_id": source["chat_id"],
        "message_id": message_id,
        "goal": text.strip(),
    }


async def maybe_publish_owner_goal(*, platform: str, source: dict, text: str,
                                   message_id: str, **_unused) -> dict | None:
    if platform != "feishu":
        return None
    payload = authenticated_goal(source, text, message_id)
    if payload is None:
        return None
    # Avoid inheriting the gateway's provider credentials into the local
    # publisher. CLI and paths are static, not derived from message content.
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/ubuntu",
        "PYTHONPATH": str(_MCP_REPOSITORY),
        "PYTHONUNBUFFERED": "1",
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            str(_MCP_PYTHON), str(_WORKER),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=str(_MCP_REPOSITORY), env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
                timeout=20,
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            raise
        if proc.returncode != 0:
            logger.error("Cortex Feishu publish failed: exit=%s stderr_class=%s",
                         proc.returncode, "publisher_error" if stderr else "unknown")
            raise RuntimeError("cortex_publish_failed")
        result = json.loads(stdout)
        task_id = result["task_id"]
        if not isinstance(task_id, str) or not task_id.startswith(
                ("clt-cortex-chat-", "gpt-inbox-agent-cortex-")):
            raise ValueError("invalid_cortex_task_id")
        logger.info("Cortex Feishu publication acknowledged: task_id=%s replay=%s",
                    task_id, bool(result.get("idempotent_replay")))
        # The durable ID is the only authoritative receipt. Registration
        # is not execution or business verification, and replay is not
        # another task.
        replay = "（原消息重复投递，复用已有任务）" if result.get("idempotent_replay") else ""
        if result["profile"] == "chat_handoff":
            status = "一次性任务已登记，等待接管（须由专项 GPT Chat 真实激活）；未执行、未验收。"
        else:
            status = "周期任务已登记到原 General Worker 队列；等待真实领取、执行与验收。"
        return {"action": "handled",
                "reply": f"已受理{replay}。任务编号：{task_id}。{status}"}
    except asyncio.CancelledError:
        raise
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        logger.error("Cortex Feishu publication blocked: %s", type(exc).__name__)
        # Fail CLOSED for matched owner task. Do not fall into a slow model
        # conversation that might perform unauthorized/untraceable side effects.
        return {"action": "handled",
                "reply": "Hermes 任务派单暂时受阻，未确认创建成功；已保留原飞书消息，请勿重复发送。"}
