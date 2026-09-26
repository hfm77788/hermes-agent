#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

LOG = logging.getLogger("dingtalk_free_response_bridge")


@dataclass(frozen=True)
class MemberRoute:
    open_dingtalk_id: str
    hermes_session_user_id: str
    role: str


@dataclass(frozen=True)
class GroupRoute:
    name: str
    chat_id: str
    skill: str
    allowed_members: dict[str, MemberRoute]
    ignored_sender_ids: set[str]
    skip_text_markers: tuple[str, ...]
    resume_existing_session: bool
    recent_context_messages: int


class Bridge:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self.profile = self.cfg.get("profile", "hema-teacher")
        self.robot_code = self.cfg["robot_code"]
        self.bot_title = self.cfg.get("bot_title", "河马老师")
        self.poll_interval = float(self.cfg.get("poll_interval_seconds", 3))
        self.timezone = ZoneInfo(self.cfg.get("timezone", "Asia/Shanghai"))
        self.dws = self.cfg.get("dws_bin", "/home/ubuntu/.local/bin/dws")
        self.hermes = self.cfg.get("hermes_bin", "/home/ubuntu/.local/bin/hermes")
        self.sessions_json = Path(
            self.cfg.get(
                "sessions_json",
                f"/home/ubuntu/.hermes/profiles/{self.profile}/sessions/sessions.json",
            )
        )
        self.state_path = Path(
            self.cfg.get(
                "state_file",
                f"/home/ubuntu/.hermes/state/dingtalk-free-response-{self.profile}.json",
            )
        )
        self.max_attempts = int(self.cfg.get("max_attempts", 3))
        self.hermes_timeout = int(self.cfg.get("hermes_timeout_seconds", 180))
        self.groups = self._parse_groups(self.cfg.get("groups") or [])
        self.state = self._load_state()

    @staticmethod
    def _parse_groups(rows: list[dict[str, Any]]) -> list[GroupRoute]:
        out: list[GroupRoute] = []
        for row in rows:
            members: dict[str, MemberRoute] = {}
            for member in row.get("allowed_members") or []:
                route = MemberRoute(
                    open_dingtalk_id=member["open_dingtalk_id"],
                    hermes_session_user_id=member["hermes_session_user_id"],
                    role=member.get("role", "verified_member"),
                )
                members[route.open_dingtalk_id] = route
            out.append(
                GroupRoute(
                    name=row.get("name") or row["chat_id"],
                    chat_id=row["chat_id"],
                    skill=row["skill"],
                    allowed_members=members,
                    ignored_sender_ids=set(row.get("ignored_sender_ids") or []),
                    skip_text_markers=tuple(row.get("skip_text_markers") or ["@河马老师"]),
                    resume_existing_session=bool(row.get("resume_existing_session", False)),
                    recent_context_messages=max(0, int(row.get("recent_context_messages", 6))),
                )
            )
        return out

    def _default_state(self) -> dict[str, Any]:
        now = datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
        return {
            "version": 1,
            "groups": {
                g.chat_id: {
                    "cursor_time": now,
                    "processed_ids": [],
                    "attempts": {},
                    "pending": {},
                }
                for g in self.groups
            },
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._default_state()
            self._save_state(state)
            return state
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            LOG.exception("state read failed; refusing to replay history")
            raise
        for g in self.groups:
            state.setdefault("groups", {}).setdefault(
                g.chat_id,
                {
                    "cursor_time": datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S"),
                    "processed_ids": [],
                    "attempts": {},
                    "pending": {},
                },
            )
        return state

    def _save_state(self, state: dict[str, Any] | None = None) -> None:
        state = state or self.state
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.state_path.name + ".", dir=str(self.state_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.state_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @staticmethod
    def _run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "HOME": "/home/ubuntu"},
        )

    def _fetch_messages(self, group: GroupRoute, cursor_time: str) -> list[dict[str, Any]]:
        cmd = [
            self.dws,
            "chat",
            "+chat-messages",
            "--group",
            group.chat_id,
            "--start",
            cursor_time,
            "--order",
            "asc",
            "--page-all",
            "--page-limit",
            "2",
            "--max-items",
            "100",
            "--format",
            "json",
        ]
        proc = self._run(cmd, timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"dws fetch failed rc={proc.returncode}: {proc.stderr[-500:]}")
        payload = json.loads(proc.stdout or "{}")
        if isinstance(payload, dict):
            rows = payload.get("messages") or payload.get("items") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        return [x for x in rows if isinstance(x, dict)]

    def _session_id(self, group: GroupRoute, member: MemberRoute) -> str:
        sessions = json.loads(self.sessions_json.read_text(encoding="utf-8"))
        key = (
            f"agent:main:dingtalk:group:{group.chat_id}:"
            f"{member.hermes_session_user_id}"
        )
        row = sessions.get(key)
        if not row or not row.get("session_id"):
            raise RuntimeError(f"missing Hermes session binding for {group.name} / {member.role}")
        return row["session_id"]

    def _recent_context(self, group: GroupRoute, created_time: str) -> str:
        if not created_time or group.recent_context_messages <= 0:
            return ""
        proc = self._run(
            [
                self.dws,
                "chat",
                "+chat-messages",
                "--group",
                group.chat_id,
                "--time",
                created_time,
                "--direction",
                "older",
                "--limit",
                str(group.recent_context_messages),
                "--format",
                "json",
            ],
            timeout=30,
        )
        if proc.returncode != 0:
            LOG.warning("recent context fetch failed group=%s rc=%s", group.name, proc.returncode)
            return ""
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            LOG.warning("recent context parse failed group=%s", group.name)
            return ""
        if isinstance(payload, dict):
            rows = payload.get("messages") or payload.get("items") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        lines: list[str] = []
        for row in sorted(
            (x for x in rows if isinstance(x, dict)),
            key=lambda x: str(x.get("createTime") or ""),
        ):
            body = str(row.get("text") or "").strip()
            if not body:
                continue
            sender = str(row.get("sender") or "成员")
            lines.append(f"{sender}: {body[:800]}")
        return "\n".join(lines[-group.recent_context_messages :])

    def _generate_reply(
        self,
        group: GroupRoute,
        member: MemberRoute,
        text: str,
        created_time: str,
    ) -> str:
        recent = self._recent_context(group, created_time)
        context_block = (
            f"最近群聊上下文（按时间顺序，仅用于衔接当前对话）：\n{recent}\n"
            if recent
            else ""
        )
        prompt = (
            "【DingTalk免@桥接入站】这条消息已由受信任的免@监听桥读取并通过硬身份校验，"
            "等价于正常课堂入站；不要要求用户再次@河马老师，也不要解释技术机制。"
            f"来源群={group.name}；发送者角色={member.role}。\n"
            f"{context_block}"
            f"当前原始消息：{text}\n"
            "请按当前课堂Skill、持久学习状态和以上最近上下文直接回应，只输出用户可见正文。"
        )
        cmd = [self.hermes, "-p", self.profile]
        if group.resume_existing_session:
            cmd.extend(["--resume", self._session_id(group, member)])
        cmd.extend(
            [
                "-z",
                prompt,
                "--skills",
                f"{group.skill},dingtalk-inbound-identity",
            ]
        )
        proc = self._run(
            cmd,
            timeout=self.hermes_timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Hermes generation failed rc={proc.returncode}: {proc.stderr[-800:]}"
            )
        reply = (proc.stdout or "").strip()
        if not reply:
            raise RuntimeError("Hermes returned empty reply")
        return reply

    def _send_reply(self, group: GroupRoute, reply: str) -> None:
        proc = self._run(
            [
                self.dws,
                "chat",
                "+messages-send-by-bot",
                "--robot-code",
                self.robot_code,
                "--group",
                group.chat_id,
                "--title",
                self.bot_title,
                "--content",
                reply,
                "--format",
                "json",
                "--yes",
            ],
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"dws send failed rc={proc.returncode}: {proc.stderr[-500:]}")
        payload = json.loads(proc.stdout or "{}")
        success = payload.get("success")
        if success is False:
            raise RuntimeError(f"dws send returned failure: {payload}")

    def _mark_processed(self, gs: dict[str, Any], message_id: str) -> None:
        ids = list(gs.get("processed_ids") or [])
        ids.append(message_id)
        gs["processed_ids"] = ids[-500:]
        gs.get("attempts", {}).pop(message_id, None)
        gs.get("pending", {}).pop(message_id, None)

    def _process_message(self, group: GroupRoute, msg: dict[str, Any]) -> bool:
        gs = self.state["groups"][group.chat_id]
        message_id = str(msg.get("messageId") or "")
        if not message_id or message_id in set(gs.get("processed_ids") or []):
            return True

        sender_id = str(msg.get("senderId") or "")
        text = str(msg.get("text") or "").strip()

        if sender_id in group.ignored_sender_ids:
            self._mark_processed(gs, message_id)
            return True

        if any(marker and marker in text for marker in group.skip_text_markers):
            self._mark_processed(gs, message_id)
            return True

        member = group.allowed_members.get(sender_id)
        if member is None:
            LOG.warning("ignored unverified sender group=%s sender_id=%s", group.name, sender_id)
            self._mark_processed(gs, message_id)
            return True

        if not text:
            LOG.info("ignored unsupported non-text message group=%s id=%s", group.name, message_id)
            self._mark_processed(gs, message_id)
            return True

        pending = gs.setdefault("pending", {})
        reply = pending.get(message_id)
        if not reply:
            reply = self._generate_reply(
                group,
                member,
                text,
                str(msg.get("createTime") or ""),
            )
            pending[message_id] = reply
            self._save_state()

        self._send_reply(group, reply)
        self._mark_processed(gs, message_id)
        LOG.info("replied group=%s message_id=%s role=%s", group.name, message_id, member.role)
        return True

    def poll_once(self) -> None:
        changed = False
        for group in self.groups:
            gs = self.state["groups"][group.chat_id]
            rows = self._fetch_messages(group, gs["cursor_time"])
            for msg in rows:
                created = str(msg.get("createTime") or "")
                if created and created > gs.get("cursor_time", ""):
                    gs["cursor_time"] = created
                    changed = True
                message_id = str(msg.get("messageId") or "")
                if not message_id:
                    continue
                if message_id in set(gs.get("processed_ids") or []):
                    continue
                try:
                    self._process_message(group, msg)
                    changed = True
                except Exception:
                    attempts = gs.setdefault("attempts", {})
                    attempts[message_id] = int(attempts.get(message_id, 0)) + 1
                    LOG.exception(
                        "message processing failed group=%s id=%s attempt=%s",
                        group.name,
                        message_id,
                        attempts[message_id],
                    )
                    if attempts[message_id] >= self.max_attempts:
                        LOG.error(
                            "dead-letter group=%s id=%s after %s attempts",
                            group.name,
                            message_id,
                            attempts[message_id],
                        )
                        self._mark_processed(gs, message_id)
                    changed = True
            if changed:
                self._save_state()
        if changed:
            self._save_state()

    def run_forever(self) -> None:
        LOG.info("bridge started profile=%s groups=%s", self.profile, [g.name for g in self.groups])
        while True:
            try:
                self.poll_once()
            except Exception:
                LOG.exception("poll loop failed")
            time.sleep(self.poll_interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bridge = Bridge(Path(args.config))
    if args.once:
        bridge.poll_once()
    else:
        bridge.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
