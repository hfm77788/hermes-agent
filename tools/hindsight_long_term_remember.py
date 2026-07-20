"""Agent-facing tool: write a fact to Hindsight long-term memory (synchronous, verifiable).

老板说"记到 Hindsight 长期记忆"时调这个工具。

正确通道（与 hindsight_retain 工具不同）：
- hindsight_retain 工具 = 异步、行为不可见、可能没真存
- memory 工具 = session 临时、跨 session 失效
- 本工具 = 直接调 Hindsight API + 同步返回 + 验证 async:false + success:true 才算真存

调用方式：hindsight_long_term_remember(content, tags, fact_type="experience")
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


HINDSIGHT_API_URL = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8889")
DEFAULT_BANK_ID = os.environ.get("HINDSIGHT_BANK_ID", "hermes")
DEFAULT_TIMEOUT = 30


SCHEMA: Dict[str, Any] = {
    "name": "hindsight_long_term_remember",
    "description": (
        "Store a fact to Hindsight long-term memory (synchronous, verifiable).\n\n"
        "Use this when the user says: '记到 Hindsight', '记到长期记忆', '沉淀到 Hindsight bank', "
        "'存到 Hindsight 长期记忆'.\n\n"
        "This is the CORRECT channel for long-term memory:\n"
        "- ❌ DO NOT use hindsight_retain (async, behavior invisible, may not actually save)\n"
        "- ❌ DO NOT use memory tool (session-temporary, lost across sessions)\n"
        "- ❌ DO NOT write to local md files (user won't see them)\n\n"
        "Returns: {success, async, bank_id, items_count, operation_id} on success.\n"
        "Raises RuntimeError if async!=false (means the fact is in queue, not actually stored)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The fact to remember. Use a unique marker like [LESSON_YYYY_MM_DD] at the start "
                    "so it can be retrieved by recall later. Keep concise but include enough context."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tags for retrieval. Recommended: ['long-term-lesson', '<topic>', '<YYYY-MM-DD>']. "
                    "Use 'root-cause-insight' for architectural insights, 'playbook' for workflows, "
                    "'hindsight-fix' for Hindsight-specific lessons."
                ),
            },
            "fact_type": {
                "type": "string",
                "enum": ["experience", "observation", "world"],
                "default": "experience",
                "description": (
                    "fact_type=experience means Hindsight treats it as a consolidated lesson "
                    "(recommended for lessons). observation = raw observation. world = world knowledge."
                ),
            },
            "bank_id": {
                "type": "string",
                "default": "hermes",
                "description": "Hindsight bank id. Default: 'hermes'.",
            },
        },
        "required": ["content", "tags"],
    },
}


def _post_memories(
    bank_id: str,
    items: List[Dict[str, Any]],
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """POST to Hindsight /v1/default/banks/{bank}/memories. Returns parsed JSON."""
    url = f"{HINDSIGHT_API_URL.rstrip('/')}/v1/default/banks/{bank_id}/memories"
    payload = json.dumps({"items": items}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def hindsight_long_term_remember(
    content: str,
    tags: List[str],
    fact_type: str = "experience",
    bank_id: Optional[str] = None,
) -> str:
    """Write a fact to Hindsight long-term memory. Synchronous, returns JSON."""
    if not content or not content.strip():
        return json.dumps({"success": False, "error": "content is required and cannot be empty"})
    if not tags:
        return json.dumps({"success": False, "error": "tags is required (use ['long-term-lesson', '<topic>', '<date>'])"})

    bank = bank_id or DEFAULT_BANK_ID
    items = [{"content": content, "fact_type": fact_type, "tags": tags}]

    try:
        resp = _post_memories(bank, items)
    except urllib.error.URLError as e:
        return json.dumps({"success": False, "error": f"network error: {e}"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"unexpected error: {e}"})

    # CRITICAL: verify async=false + success=true. async=true means the fact is in queue, not actually stored.
    if not resp.get("success"):
        return json.dumps({"success": False, "error": f"server returned success=false: {resp}"})
    if resp.get("async") is True:
        return json.dumps({
            "success": False,
            "error": "server returned async=true — fact is in queue, not actually stored. This tool only supports sync mode.",
            "server_response": resp,
        })

    return json.dumps({
        "success": True,
        "async": False,
        "bank_id": resp.get("bank_id"),
        "items_count": resp.get("items_count"),
        "operation_id": resp.get("operation_id"),
        "fact_type": fact_type,
        "tags": tags,
    })


def _hindsight_long_term_remember_check() -> bool:
    """Gate: only show this tool when Hindsight API is reachable."""
    try:
        url = f"{HINDSIGHT_API_URL.rstrip('/')}/health"
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


registry.register(
    name="hindsight_long_term_remember",
    toolset="memory",
    schema=SCHEMA,
    handler=lambda args, **kw: hindsight_long_term_remember(
        content=args.get("content", ""),
        tags=args.get("tags", []),
        fact_type=args.get("fact_type", "experience"),
        bank_id=args.get("bank_id"),
    ),
    check_fn=_hindsight_long_term_remember_check,
    emoji="🧠",
)
