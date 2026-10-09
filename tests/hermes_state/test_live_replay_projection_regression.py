"""Regression for missing live-replay columns after adding CAS message snapshots.

Feishu live transcript reads use repair_alternation=True. The model-message
projection must include all columns consumed by transcript_row_snapshot, even
when those columns are not exposed to the LLM message dictionary.
"""
from agent.message_metadata import DB_ROW_SNAPSHOT, MESSAGE_UID
from agent.transcript_repair import _OWNED_COLUMNS
from hermes_state import SessionDB


def test_feishu_live_replay_fetches_cas_owned_columns(tmp_path):
    from hermes_state_coverage import SessionCoverageMixin
    assert SessionCoverageMixin in SessionDB.__mro__, "coverage helpers missing from live SessionDB"

    store = SessionDB(tmp_path / "state.db")
    try:
        selected = {part.strip() for part in store._CONVERSATION_ROW_COLUMNS.split(",")}
        needed = set(_OWNED_COLUMNS) | {
            "message_uid", "absorbed_message_uids", "tool_call_uids", "tool_call_uid"
        }
        assert needed <= selected, f"Columns missing: {sorted(needed - selected)}"

        session_id = "feishu-live-replay"
        store.create_session(session_id, "feishu")
        written = store.append_messages_batch(session_id, [
            {"role": "user", "content": "Upgrade status?", "timestamp": 1000.0},
            {
                "role": "assistant",
                "content": "Checking the upgrade",
                "timestamp": 1001.0,
                "token_count": 8,
            },
        ])
        assert written == 2

        restored = store.get_messages_as_conversation(
            session_id, repair_alternation=True
        )
        assert [item["role"] for item in restored] == ["user", "assistant"]
        assert all(item.get(MESSAGE_UID) for item in restored)
        assert all(item.get(DB_ROW_SNAPSHOT) for item in restored)
    finally:
        store.close()
