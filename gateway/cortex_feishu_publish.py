"""Isolated Hermes-venv child for admitted owner Feishu task publication.

Input is one JSON object on stdin. Does not activate ChatGPT or issue L4
authorization: uses existing supervisor.db and its atomic idempotency checks.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def publish_one(payload: dict, *, store_path: Path) -> dict:
    from app.tools.closed_loop_task_control import ClosedLoopTaskStore
    from app.tools.cortex_control_actions import publish_goal
    from app.tools.cortex_bridge_routing import feishu_delivery_key

    if not isinstance(payload, dict) or set(payload) != {"chat_id", "message_id", "goal"}:
        raise ValueError("invalid_cortex_feishu_payload")
    key = feishu_delivery_key(payload["chat_id"], payload["message_id"])
    result = publish_goal(
        ClosedLoopTaskStore(store_path), goal=payload["goal"],
        profile="auto", priority="normal", confirm_full_delivery=False,
        idempotency_key=key,
        **({
            "source_chat_id": payload["chat_id"],
            "source_message_id": payload["message_id"],
        } if payload["chat_id"] == "oc_b044615532e0d934ae8198743707549c" else {}),
    )
    return {"task_id": result["task_id"], "profile": result["profile"],
            "idempotent_replay": bool(result.get("idempotent_replay"))}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read(10000))
        result = publish_one(payload, store_path=Path(
            "/home/ubuntu/hermes-mcp-server/state/closed-loop/supervisor.db"))
        sys.stdout.write(json.dumps(result))
        return 0
    except Exception as exc:
        # Do not print the inbound message, files, credentials, or traceback.
        sys.stderr.write(type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
