from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.dingtalk_free_response_bridge import Bridge


def write_cfg(tmp_path: Path) -> Path:
    cfg = {
        "profile": "hema-teacher",
        "robot_code": "robot-code",
        "state_file": str(tmp_path / "state.json"),
        "sessions_json": str(tmp_path / "sessions.json"),
        "groups": [
            {
                "name": "Math",
                "chat_id": "cid-math",
                "skill": "math-skill",
                "ignored_sender_ids": ["bot-id"],
                "skip_text_markers": ["@河马老师"],
                "allowed_members": [
                    {
                        "open_dingtalk_id": "child-id",
                        "hermes_session_user_id": "hard-child-key",
                        "role": "learner",
                    }
                ],
            }
        ],
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


def test_first_start_initializes_cursor_without_replay(tmp_path):
    bridge = Bridge(write_cfg(tmp_path))
    gs = bridge.state["groups"]["cid-math"]
    assert gs["cursor_time"]
    assert gs["processed_ids"] == []


def test_session_resolution_uses_exact_chat_and_hard_user_key(tmp_path):
    cfg = write_cfg(tmp_path)
    sessions = {
        "agent:main:dingtalk:group:cid-math:hard-child-key": {
            "session_id": "sid-123"
        }
    }
    (tmp_path / "sessions.json").write_text(json.dumps(sessions), encoding="utf-8")
    bridge = Bridge(cfg)
    group = bridge.groups[0]
    member = group.allowed_members["child-id"]
    assert bridge._session_id(group, member) == "sid-123"


def test_bot_and_mentioned_messages_are_skipped(tmp_path, monkeypatch):
    bridge = Bridge(write_cfg(tmp_path))
    group = bridge.groups[0]
    monkeypatch.setattr(bridge, "_generate_reply", lambda *a, **k: pytest.fail("should not generate"))
    assert bridge._process_message(group, {"messageId": "m1", "senderId": "bot-id", "text": "x"})
    assert bridge._process_message(group, {"messageId": "m2", "senderId": "child-id", "text": "hi @河马老师"})


def test_unverified_sender_is_fail_closed(tmp_path, monkeypatch):
    bridge = Bridge(write_cfg(tmp_path))
    group = bridge.groups[0]
    monkeypatch.setattr(bridge, "_generate_reply", lambda *a, **k: pytest.fail("should not generate"))
    assert bridge._process_message(group, {"messageId": "m3", "senderId": "stranger", "text": "hello"})
    assert "m3" in bridge.state["groups"]["cid-math"]["processed_ids"]


def test_pending_reply_prevents_duplicate_generation_on_send_retry(tmp_path, monkeypatch):
    bridge = Bridge(write_cfg(tmp_path))
    group = bridge.groups[0]
    calls = {"generate": 0, "send": 0}

    def generate(*_a, **_k):
        calls["generate"] += 1
        return "reply"

    def send(*_a, **_k):
        calls["send"] += 1
        if calls["send"] == 1:
            raise RuntimeError("temporary send failure")

    monkeypatch.setattr(bridge, "_generate_reply", generate)
    monkeypatch.setattr(bridge, "_send_reply", send)

    msg = {"messageId": "m4", "senderId": "child-id", "text": "hello"}
    with pytest.raises(RuntimeError):
        bridge._process_message(group, msg)
    assert bridge.state["groups"]["cid-math"]["pending"]["m4"] == "reply"

    assert bridge._process_message(group, msg)
    assert calls["generate"] == 1
    assert calls["send"] == 2
    assert "m4" in bridge.state["groups"]["cid-math"]["processed_ids"]


def test_recent_context_is_chronological(tmp_path, monkeypatch):
    bridge = Bridge(write_cfg(tmp_path))
    group = bridge.groups[0]

    class Proc:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                "messages": [
                    {"createTime": "2026-09-26 20:00:02", "sender": "Moon", "text": "40"},
                    {"createTime": "2026-09-26 20:00:01", "sender": "河马老师", "text": "Q1"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(bridge, "_run", lambda *a, **k: Proc())
    assert bridge._recent_context(group, "2026-09-26 20:00:03") == "河马老师: Q1\nMoon: 40"


def test_fresh_generation_avoids_heavy_session_resume(tmp_path, monkeypatch):
    bridge = Bridge(write_cfg(tmp_path))
    group = bridge.groups[0]
    member = group.allowed_members["child-id"]
    assert group.resume_existing_session is False

    monkeypatch.setattr(
        bridge,
        "_recent_context",
        lambda *_a, **_k: "河马老师: Q1\nMoon: 40",
    )
    captured = {}

    class Proc:
        returncode = 0
        stderr = ""
        stdout = "继续下一题"

    def fake_run(cmd, *, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return Proc()

    monkeypatch.setattr(bridge, "_run", fake_run)
    reply = bridge._generate_reply(
        group,
        member,
        "继续",
        "2026-09-26 20:00:03",
    )
    assert reply == "继续下一题"
    assert "--resume" not in captured["cmd"]
    prompt = captured["cmd"][captured["cmd"].index("-z") + 1]
    assert "河马老师: Q1" in prompt
    assert "Moon: 40" in prompt
    assert "当前原始消息：继续" in prompt



def test_gateway_injection_reuses_dual_identity_and_bootstraps_context_once(tmp_path, monkeypatch):
    cfg_path = write_cfg(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["gateway_inject_existing_session"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    bridge = Bridge(cfg_path)
    group = bridge.groups[0]
    member = group.allowed_members["child-id"]
    calls = []
    context_calls = {"n": 0}

    def recent(*_a, **_k):
        context_calls["n"] += 1
        return "河马老师: R2 求多少米？"

    def inject(_home, params, *, timeout):
        calls.append((dict(params), timeout))
        return {"accepted": True, "response": "答对了。下一题"}

    monkeypatch.setattr(bridge, "_recent_context", recent)
    monkeypatch.setattr(
        "scripts.dingtalk_free_response_bridge.inject_gateway_local_inbound", inject)

    assert bridge._generate_reply(
        group, member, "4000米", "2026-09-26 21:20:42", "m1") == "答对了。下一题"
    assert calls[0][0]["user_id"] == "child-id"
    assert calls[0][0]["user_id_alt"] == "hard-child-key"
    assert calls[0][0]["skill"] == "math-skill"
    assert calls[0][0]["force_context"] is True
    assert calls[0][0]["recent_context"] == "河马老师: R2 求多少米？"

    assert bridge._generate_reply(
        group, member, "31.4", "2026-09-26 21:22:00", "m2") == "答对了。下一题"
    assert calls[1][0]["force_context"] is False
    assert calls[1][0]["recent_context"] == ""
    assert context_calls["n"] == 1
