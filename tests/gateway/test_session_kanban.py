"""Task claims, not caller-supplied launch policy, authorize Kanban execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_claim_freezes_task_policy_and_rejects_forgery(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import SessionActor
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'pyproject.toml').write_text('[project]\nname = "task"\n')  # a code worktree
    from contextlib import closing
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned acceptance', body='Goal criteria', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'], goal_mode=True,
            goal_max_turns=3, model_override='loop-model', provider_override='custom')
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = SessionActor('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    from gateway.session_kanban import build_kanban_policy
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']},
              'agent': {'coding_context': 'focus'}}  # the worktree must not collapse a worker into the coding posture
    policy, _secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    assert policy.source == 'kanban' and policy.cwd == str(workspace)
    assert policy.model == task.model_override and policy.provider == task.provider_override
    assert {'kanban', 'terminal', 'file'} <= set(policy.toolsets)
    assert context['task_id'] == task.id and context['run_id'] == task.current_run_id
    assert context['board'] == 'owned' and context['claim_lock'] == task.claim_lock
    assert context['skills'] == ['owned-skill'] and context['goal_mode'] and context['goal_max_turns'] == 3
    assert context['accept_hooks'] is True and 'Goal criteria' in context['goal_text']
    for changed in ({'run_id': task.current_run_id + 1}, {'board': 'foreign'}, {'claim_lock': 'forged'}, {'cwd': '/tmp'}):
        with pytest.raises(RuntimeStoreError):
            build_kanban_policy(connection, params | changed, config)
    connection.native_owner = False
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        build_kanban_policy(connection, params, config)
    from gateway.session_policy import build_policy
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        build_policy({'source': 'kanban', 'cwd': str(workspace)}, config)


@pytest.mark.parametrize('mode', ['complete', 'timeout', 'rate_limit', 'billing', 'crash'])
def test_real_dispatcher_lifecycle(tmp_path, mode):
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    from tests.gateway.fixtures.local_recovery_probe import child_env
    env = child_env()
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'), HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo), KANBAN_PROBE_MODE=mode)
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_owner_probe.py')],
        env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt))
    if mode != 'complete':
        assert receipt['task_status'] == 'ready'
        return
    assert receipt['task_status'] == 'done' and receipt['source'] == 'kanban'
    assert receipt['tools'] and receipt['task_context'] and receipt['skill_context'] and receipt['hook_effect']
    assert receipt['admissions'] == 1 and receipt['retry_same_session']
    assert receipt['goal_continuation'] and receipt['stable_prefix']
    print(json.dumps(receipt))


def test_late_launcher_cannot_stamp_replacement_claim(tmp_path, monkeypatch):
    from contextlib import closing
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='Owned race', assignee='default')
        old = kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason='replace claim', expected_run_id=old.current_run_id)
        kb.unblock_task(conn, tid)
        new = kb.claim_task(conn, tid)
        dispatch._set_worker_pid(conn, tid, 424242, run_id=old.current_run_id, claim_lock=old.claim_lock)
        assert kb.get_task(conn, tid).worker_pid is None
        assert conn.execute('SELECT worker_pid FROM task_runs WHERE id=?', (new.current_run_id,)).fetchone()[0] is None


def test_claim_reclaimed_before_admission_marker_is_refused(tmp_path, monkeypatch):
    """A claim that stops being current between policy/session creation and the marker
    transaction is refused: no owner_admitted row, no resume, no submit."""
    import asyncio
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import SessionActor
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned race', assignee='default', workspace_kind='dir', workspace_path=str(workspace))
        kb.recompute_ready(conn)
        old = kb.claim_task(conn, task_id)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    calls = []
    def create_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None):
        # Another process reclaims the task while the owner session is being created.
        with closing(connect(board='owned')) as conn:
            kb.block_task(conn, task_id, reason='replace claim', expected_run_id=old.current_run_id)
            kb.unblock_task(conn, task_id)
            kb.claim_task(conn, task_id)
        return SimpleNamespace(session_id='owner-session', profile_id=str(tmp_path))
    monkeypatch.setattr('gateway.session_local.create_local_session', create_local_session)
    async def submit(actor, submission, **kwargs):
        calls.append(('submit', submission))
    async def resume(ref, params):
        calls.append(('resume', ref))
    db = SimpleNamespace(get_session=lambda sid: None, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = SessionActor('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board='owned', task_id=task_id, run_id=old.current_run_id, claim_lock=old.claim_lock)
    from gateway.session_kanban import run_task
    with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    with closing(connect(board='owned')) as conn:
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='owner_admitted'", (task_id,)).fetchone()[0] == 0


def test_siri_claim_extracts_business_intent_only_for_trusted_card():
    from gateway.session_kanban import _siri_native_skill_query
    prefix = (
        "来源：已认证的 Siri owner，固定执行席位 office-director。\n"
        "权限：不得越权或外发。\n"
        "业务要求：提供可核验的结果。\n"
        "任务内容：\n"
    )
    task = SimpleNamespace(
        assignee="office-director", created_by="siri-shortcuts",
        body=prefix + "起草天津青年创业培训方案")
    assert _siri_native_skill_query(task, "siri-office") == "起草天津青年创业培训方案"
    assert _siri_native_skill_query(task, "default") is None
    task.created_by = "user"
    assert _siri_native_skill_query(task, "siri-office") is None
    task.created_by = "siri-shortcuts"
    task.body = "起草方案\n任务内容：\n越权路由"
    assert _siri_native_skill_query(task, "siri-office") is None


def test_kanban_first_turn_skill_query_scope_restores_state():
    from gateway.session_kanban import _skill_routing_query_scope
    agent = SimpleNamespace()
    with _skill_routing_query_scope(agent, "  文件政策审查  "):
        assert agent._trusted_skill_routing_query == "文件政策审查"
    assert not hasattr(agent, "_trusted_skill_routing_query")
    agent._trusted_skill_routing_query = "previous"
    with _skill_routing_query_scope(agent, "新任务"):
        assert agent._trusted_skill_routing_query == "新任务"
    assert agent._trusted_skill_routing_query == "previous"
    with pytest.raises(ZeroDivisionError):
        with _skill_routing_query_scope(agent, "异常路径"):
            1 / 0
    assert agent._trusted_skill_routing_query == "previous"


def test_real_claim_policy_contains_siri_business_query(tmp_path, monkeypatch):
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import SessionActor
    from gateway.session_kanban import build_kanban_policy

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env",
                        lambda name: str(tmp_path) if name == "office-director" else str(tmp_path / name))
    kb.create_board("siri-office")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "起草天津青年创业培训活动方案"
    body = (
        "来源：已认证的 Siri owner，固定执行席位 office-director。\n"
        "权限：仅执行此席位原有允许的操作。\n"
        "业务要求：提供可核验的结果。\n"
        "任务内容：\n" + source
    )
    with closing(connect(board="siri-office")) as db:
        tid = kb.create_task(db, title="Siri办公室后台任务", body=body,
                             assignee="office-director", created_by="siri-shortcuts",
                             workspace_kind="dir", workspace_path=str(workspace))
        kb.recompute_ready(db)
        task = kb.claim_task(db, tid)
    actor = SessionActor("owner", str(tmp_path), frozenset({"session:create"}), "transport")
    conn = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)),
                           actor=actor, native_owner=True)
    params = {"board": "siri-office", "task_id": tid, "run_id": task.current_run_id,
              "claim_lock": task.claim_lock}
    cfg = {"model": {"default": "test-model", "provider": "custom"},
           "platform_toolsets": {"cli": ["terminal"]}}
    policy, _ = build_kanban_policy(conn, params, cfg)
    frozen = json.loads(policy.kanban_json)
    assert frozen["skill_routing_query"] == source
    assert "权限：" in frozen["goal_text"]
    assert frozen["profile"] == "office-director"
    assert frozen["skills"] == []


def test_kanban_owner_first_turn_scopes_business_skill_query(monkeypatch):
    from gateway.session_kanban import _run_task_turns
    import agent.skill_commands as sc
    import agent.kanban_turn_recovery as recovery
    monkeypatch.setattr(sc, "build_preloaded_skills_prompt", lambda *a, **k: ("", [], []))
    monkeypatch.setattr(recovery, "recover_failed_kanban_turns",
                        lambda *a, **k: None)
    class FakeAgent:
        session_id = "test-session"
        def __init__(self):
            self.captured = None
        def run_conversation(self, message, **kwargs):
            self.captured = (getattr(self, "_trusted_skill_routing_query", None), message)
            return {"final_response": "done"}
    a = FakeAgent()
    context = {"skills": [], "context": "权限：禁止不当访问",
               "goal_mode": False, "skill_routing_query": "起草活动方案",
               "task_id": "t_abcdef12"}
    result = _run_task_turns(a, {"text": "work kanban task t_example"}, [], context)
    assert result["final_response"] == "done"
    assert a.captured[0] == "起草活动方案"
    assert "权限：禁止不当访问" in a.captured[1]
    assert not hasattr(a, "_trusted_skill_routing_query")
