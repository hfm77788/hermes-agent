"""Shared slash command helpers for skills.

Shared between CLI (cli.py) and gateway (gateway/run.py) so both surfaces
can invoke skills via /skill-name commands.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import display_hermes_home
from agent.skill_preprocessing import (
    expand_inline_shell as _expand_inline_shell,
    load_skills_config as _load_skills_config,
    substitute_template_vars as _substitute_template_vars,
)

logger = logging.getLogger(__name__)

_skill_commands: Dict[str, Dict[str, Any]] = {}
_skill_commands_platform: Optional[str] = None
# Patterns for sanitizing skill names into clean hyphen-separated slugs.
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")


def _resolve_skill_commands_platform() -> Optional[str]:
    """Return the current platform scope used for disabled-skill filtering.

    Used to detect when the active platform has shifted so
    :func:`get_skill_commands` can drop a stale cache that was populated
    for a different platform's ``skills.platform_disabled`` view (#14536).

    Resolves from (in order) ``HERMES_PLATFORM`` env var and
    ``HERMES_SESSION_PLATFORM`` from the gateway session context. Returns
    ``None`` when no platform scope is active (e.g. classic CLI, RL
    rollouts, standalone scripts).
    """
    try:
        from gateway.session_context import get_session_env

        resolved_platform = (
            os.getenv("HERMES_PLATFORM")
            or get_session_env("HERMES_SESSION_PLATFORM")
        )
    except Exception:
        resolved_platform = os.getenv("HERMES_PLATFORM")
    return resolved_platform or None

def _load_skill_payload(skill_identifier: str, task_id: str | None = None) -> tuple[dict[str, Any], Path | None, str] | None:
    """Load a skill by name/path and return (loaded_payload, skill_dir, display_name)."""
    raw_identifier = (skill_identifier or "").strip()
    if not raw_identifier:
        return None

    try:
        from tools.skills_tool import SKILLS_DIR, skill_view
        from agent.skill_utils import get_external_skills_dirs

        identifier_path = Path(raw_identifier).expanduser()
        if identifier_path.is_absolute():
            normalized = None
            trusted_roots = [SKILLS_DIR]
            try:
                trusted_roots.extend(get_external_skills_dirs())
            except Exception:
                pass

            # Prefer the lexical path under a trusted skill root before
            # resolving symlinks.  Slash-command discovery can legitimately
            # find a skill via ~/.hermes/skills/<name> where <name> is a
            # symlink to a checked-out skill elsewhere.  Resolving first turns
            # that trusted visible path into an arbitrary absolute path that
            # skill_view() refuses to load.
            for root in trusted_roots:
                try:
                    normalized = str(identifier_path.relative_to(root))
                    break
                except ValueError:
                    continue

            if normalized is None:
                try:
                    normalized = str(identifier_path.resolve().relative_to(SKILLS_DIR.resolve()))
                except Exception:
                    normalized = raw_identifier
        else:
            normalized = raw_identifier.lstrip("/")

        loaded_skill = json.loads(
            skill_view(normalized, task_id=task_id, preprocess=False)
        )
    except Exception:
        return None

    if not loaded_skill.get("success"):
        return None

    skill_name = str(loaded_skill.get("name") or normalized)
    skill_path = str(loaded_skill.get("path") or "")
    skill_dir = None
    # Prefer the absolute skill_dir returned by skill_view() — this is
    # correct for both local and external skills.  Fall back to the old
    # SKILLS_DIR-relative reconstruction only when skill_dir is absent
    # (e.g. legacy skill_view responses).
    abs_skill_dir = loaded_skill.get("skill_dir")
    if abs_skill_dir:
        skill_dir = Path(abs_skill_dir)
    elif skill_path:
        try:
            skill_dir = SKILLS_DIR / Path(skill_path).parent
        except Exception:
            skill_dir = None

    return loaded_skill, skill_dir, skill_name


def _inject_skill_config(loaded_skill: dict[str, Any], parts: list[str]) -> None:
    """Resolve and inject skill-declared config values into the message parts.

    If the loaded skill's frontmatter declares ``metadata.hermes.config``
    entries, their current values (from config.yaml or defaults) are appended
    as a ``[Skill config: ...]`` block so the agent knows the configured values
    without needing to read config.yaml itself.
    """
    try:
        from agent.skill_utils import (
            extract_skill_config_vars,
            parse_frontmatter,
            resolve_skill_config_values,
        )

        # The loaded_skill dict contains the raw content which includes frontmatter
        raw_content = str(loaded_skill.get("raw_content") or loaded_skill.get("content") or "")
        if not raw_content:
            return

        frontmatter, _ = parse_frontmatter(raw_content)
        config_vars = extract_skill_config_vars(frontmatter)
        if not config_vars:
            return

        resolved = resolve_skill_config_values(config_vars)
        if not resolved:
            return

        lines = ["", f"[Skill config (from {display_hermes_home()}/config.yaml):"]
        for key, value in resolved.items():
            display_val = str(value) if value else "(not set)"
            lines.append(f"  {key} = {display_val}")
        lines.append("]")
        parts.extend(lines)
    except Exception:
        pass  # Non-critical — skill still loads without config injection


def _build_skill_message(
    loaded_skill: dict[str, Any],
    skill_dir: Path | None,
    activation_note: str,
    user_instruction: str = "",
    runtime_note: str = "",
    session_id: str | None = None,
) -> str:
    """Format a loaded skill into a user/system message payload."""
    from tools.skills_tool import SKILLS_DIR

    content = str(loaded_skill.get("content") or "")

    # ── Template substitution and inline-shell expansion ──
    # Done before anything else so downstream blocks (setup notes,
    # supporting-file hints) see the expanded content.
    skills_cfg = _load_skills_config()
    if skills_cfg.get("template_vars", True):
        content = _substitute_template_vars(content, skill_dir, session_id)
    if skills_cfg.get("inline_shell", False):
        timeout = int(skills_cfg.get("inline_shell_timeout", 10) or 10)
        content = _expand_inline_shell(content, skill_dir, timeout)

    parts = [activation_note, "", content.strip()]

    # ── Inject the absolute skill directory so the agent can reference
    #    bundled scripts without an extra skill_view() round-trip. ──
    if skill_dir:
        parts.append("")
        parts.append(f"[Skill directory: {skill_dir}]")
        parts.append(
            "Resolve any relative paths in this skill (e.g. `scripts/foo.js`, "
            "`templates/config.yaml`) against that directory, then run them "
            "with the terminal tool using the absolute path."
        )

    # ── Inject resolved skill config values ──
    _inject_skill_config(loaded_skill, parts)

    if loaded_skill.get("setup_skipped"):
        parts.extend(
            [
                "",
                "[Skill setup note: Required environment setup was skipped. Continue loading the skill and explain any reduced functionality if it matters.]",
            ]
        )
    elif loaded_skill.get("gateway_setup_hint"):
        parts.extend(
            [
                "",
                f"[Skill setup note: {loaded_skill['gateway_setup_hint']}]",
            ]
        )
    elif loaded_skill.get("setup_needed") and loaded_skill.get("setup_note"):
        parts.extend(
            [
                "",
                f"[Skill setup note: {loaded_skill['setup_note']}]",
            ]
        )

    supporting = []
    linked_files = loaded_skill.get("linked_files") or {}
    for entries in linked_files.values():
        if isinstance(entries, list):
            supporting.extend(entries)

    if not supporting and skill_dir:
        for subdir in ("references", "templates", "scripts", "assets"):
            subdir_path = skill_dir / subdir
            if subdir_path.exists():
                for f in sorted(subdir_path.rglob("*")):
                    if f.is_file() and not f.is_symlink():
                        rel = str(f.relative_to(skill_dir))
                        supporting.append(rel)

    if supporting and skill_dir:
        try:
            skill_view_target = str(skill_dir.relative_to(SKILLS_DIR))
        except ValueError:
            # Skill is from an external dir — use the skill name instead
            skill_view_target = skill_dir.name
        parts.append("")
        parts.append("[This skill has supporting files:]")
        for sf in supporting:
            parts.append(f"- {sf}  ->  {skill_dir / sf}")
        parts.append(
            f'\nLoad any of these with skill_view(name="{skill_view_target}", '
            f'file_path="<path>"), or run scripts directly by absolute path '
            f"(e.g. `node {skill_dir}/scripts/foo.js`)."
        )

    if user_instruction:
        parts.append("")
        parts.append(f"The user has provided the following instruction alongside the skill invocation: {user_instruction}")

    if runtime_note:
        parts.append("")
        parts.append(f"[Runtime note: {runtime_note}]")

    return "\n".join(parts)


def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Scan ~/.hermes/skills/ and return a mapping of /command -> skill info.

    Returns:
        Dict mapping "/skill-name" to {name, description, skill_md_path, skill_dir}.
    """
    global _skill_commands, _skill_commands_platform
    _skill_commands_platform = _resolve_skill_commands_platform()
    _skill_commands = {}
    try:
        from tools.skills_tool import SKILLS_DIR, _parse_frontmatter, skill_matches_platform, _get_disabled_skill_names
        from agent.skill_utils import get_external_skills_dirs, iter_skill_index_files
        disabled = _get_disabled_skill_names()
        seen_names: set = set()

        # Scan local dir first, then external dirs
        dirs_to_scan = []
        if SKILLS_DIR.exists():
            dirs_to_scan.append(SKILLS_DIR)
        dirs_to_scan.extend(get_external_skills_dirs())

        for scan_dir in dirs_to_scan:
            for skill_md in iter_skill_index_files(scan_dir, "SKILL.md"):
                if any(part in {'.git', '.github', '.hub', '.archive'} for part in skill_md.parts):
                    continue
                try:
                    content = skill_md.read_text(encoding='utf-8')
                    frontmatter, body = _parse_frontmatter(content)
                    # Skip skills incompatible with the current OS platform
                    if not skill_matches_platform(frontmatter):
                        continue
                    name = frontmatter.get('name', skill_md.parent.name)
                    if name in seen_names:
                        continue
                    # Respect user's disabled skills config
                    if name in disabled:
                        continue
                    description = frontmatter.get('description', '')
                    if not description:
                        for line in body.strip().split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                description = line[:80]
                                break
                    seen_names.add(name)
                    # Normalize to hyphen-separated slug, stripping
                    # non-alnum chars (e.g. +, /) to avoid invalid
                    # Telegram command names downstream.
                    cmd_name = name.lower().replace(' ', '-').replace('_', '-')
                    cmd_name = _SKILL_INVALID_CHARS.sub('', cmd_name)
                    cmd_name = _SKILL_MULTI_HYPHEN.sub('-', cmd_name).strip('-')
                    if not cmd_name:
                        continue
                    _skill_commands[f"/{cmd_name}"] = {
                        "name": name,
                        "description": description or f"Invoke the {name} skill",
                        "skill_md_path": str(skill_md),
                        "skill_dir": str(skill_md.parent),
                    }
                except Exception:
                    continue
    except Exception:
        pass
    return _skill_commands


def get_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Return the current skill commands mapping (scan first if empty).

    Rescans when the active platform scope changes (e.g. a gateway
    process serving Telegram and Discord concurrently) so each platform
    sees its own ``skills.platform_disabled`` view (#14536).
    """
    if (
        not _skill_commands
        or _skill_commands_platform != _resolve_skill_commands_platform()
    ):
        scan_skill_commands()
    return _skill_commands


def reload_skills() -> Dict[str, Any]:
    """Re-scan the skills directory and return a diff of what changed.

    Rescans ``~/.hermes/skills/`` and any ``skills.external_dirs`` so the
    slash-command map (``agent.skill_commands._skill_commands``) reflects
    skills added or removed on disk.

    This does NOT invalidate the skills system-prompt cache. Skills are
    called by name via ``/skill-name``, ``skills_list``, or ``skill_view``
    — they don't need to be in the system prompt for the model to use them.
    Keeping the prompt cache intact preserves prefix caching across the
    reload, so a user invoking ``/reload-skills`` pays no cache-reset cost.

    Returns:
        Dict with keys::

            {
              "added":      [{"name": str, "description": str}, ...],
              "removed":    [{"name": str, "description": str}, ...],
              "unchanged":  [skill names present before and after],
              "total":      total skill count after rescan,
              "commands":   total /slash-skill count after rescan,
            }

        ``description`` is the skill's full SKILL.md frontmatter
        ``description:`` field — the same string the system prompt renders
        as ``    - name: description`` for pre-existing skills.
    """
    # Snapshot pre-reload state (name -> description) from the current
    # slash-command cache. Using dicts lets the post-rescan diff carry
    # descriptions for newly-visible or just-removed skills without a
    # second disk walk.
    def _snapshot(cmds: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for slash_key, info in cmds.items():
            bare = slash_key.lstrip("/")
            out[bare] = (info or {}).get("description") or ""
        return out

    before = _snapshot(_skill_commands)

    # Rescan the skills dir. ``scan_skill_commands`` resets
    # ``_skill_commands = {}`` internally and repopulates it.
    new_commands = scan_skill_commands()

    after = _snapshot(new_commands)

    added_names = sorted(set(after) - set(before))
    removed_names = sorted(set(before) - set(after))
    unchanged = sorted(set(after) & set(before))

    added = [{"name": n, "description": after[n]} for n in added_names]
    # For removed skills, use the description we had cached pre-rescan
    # (the skill file is gone so we can't re-read it).
    removed = [{"name": n, "description": before[n]} for n in removed_names]

    return {
        "added": added,
        "removed": removed,
        "unchanged": unchanged,
        "total": len(after),
        "commands": len(new_commands),
    }


def resolve_skill_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed /command to its canonical skill_cmds key.

    Skills are always stored with hyphens — ``scan_skill_commands`` normalizes
    spaces and underscores to hyphens when building the key. Hyphens and
    underscores are treated interchangeably in user input: this matches
    ``_check_unavailable_skill`` and accommodates Telegram bot-command names
    (which disallow hyphens, so ``/claude-code`` is registered as
    ``/claude_code`` and comes back in the underscored form).

    Returns the matching ``/slug`` key from ``get_skill_commands()`` or
    ``None`` if no match.
    """
    if not command:
        return None
    cmd_key = f"/{command.replace('_', '-')}"
    return cmd_key if cmd_key in get_skill_commands() else None


def build_skill_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
    runtime_note: str = "",
) -> Optional[str]:
    """Build the user message content for a skill slash command invocation.

    Args:
        cmd_key: The command key including leading slash (e.g., "/gif-search").
        user_instruction: Optional text the user typed after the command.

    Returns:
        The formatted message string, or None if the skill wasn't found.
    """
    commands = get_skill_commands()
    skill_info = commands.get(cmd_key)
    if not skill_info:
        return None

    loaded = _load_skill_payload(skill_info["skill_dir"], task_id=task_id)
    if not loaded:
        return None

    loaded_skill, skill_dir, skill_name = loaded

    # Track active usage for Curator lifecycle management (#17782)
    try:
        from tools.skill_usage import bump_use
        bump_use(skill_name)
    except Exception:
        pass  # Non-critical — skill invocation proceeds regardless

    activation_note = (
        f'[IMPORTANT: The user has invoked the "{skill_name}" skill, indicating they want '
        "you to follow its instructions. The full skill content is loaded below.]"
    )
    return _build_skill_message(
        loaded_skill,
        skill_dir,
        activation_note,
        user_instruction=user_instruction,
        runtime_note=runtime_note,
        session_id=task_id,
    )


def build_preloaded_skills_prompt(
    skill_identifiers: list[str],
    task_id: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Load one or more skills for session-wide CLI preloading.

    Returns (prompt_text, loaded_skill_names, missing_identifiers).
    """
    prompt_parts: list[str] = []
    loaded_names: list[str] = []
    missing: list[str] = []

    seen: set[str] = set()
    for raw_identifier in skill_identifiers:
        identifier = (raw_identifier or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)

        loaded = _load_skill_payload(identifier, task_id=task_id)
        if not loaded:
            missing.append(identifier)
            continue

        loaded_skill, skill_dir, skill_name = loaded

        # Track active usage for Curator lifecycle management (#17782)
        try:
            from tools.skill_usage import bump_use
            bump_use(skill_name)
        except Exception:
            pass  # Non-critical

        activation_note = (
            f'[IMPORTANT: The user launched this CLI session with the "{skill_name}" skill '
            "preloaded. Treat its instructions as active guidance for the duration of this "
            "session unless the user overrides them.]"
        )
        prompt_parts.append(
            _build_skill_message(
                loaded_skill,
                skill_dir,
                activation_note,
                session_id=task_id,
            )
        )
        loaded_names.append(skill_name)

    return "\n\n".join(prompt_parts), loaded_names, missing


# -----------------------------------------------------------------------------------------
# Per-turn auto skill preloader
# -----------------------------------------------------------------------------------------
# Runtime-level routing lives here (not only inside skills) so the first model
# call sees the right skill bundle before it has a chance to forget Step 0.
# Rules use OR matching across pipe-separated keywords.  Keep keywords specific:
# broad words like "new" cause noisy false positives and slow turns.

_AUTO_PRELOAD_RULES: list[tuple[str, list[str], str]] = [
    # Base code-task controller. Other code-related rules below also include it
    # explicitly so PR/debug/test/review prompts get the deterministic bundle.
    (
        "执行代码|代码任务|小修|快速修复|严格验证|代码改动|改代码|写代码|"
        "代码复杂度|风险分级|执行代码任务|修 bug|修复代码|implement code|"
        "code task|coding task|fix bug|small fix|quick fix|modify code",
        ["code-task-execution"],
        "code execution task",
    ),
    # GitHub PR / git write workflow bundle.
    (
        "创建PR|提PR|建PR|合PR|PR详情|PR改动|PR审查|提交PR|开PR|"
        "pull request|github pr|gh pr|merge pr|create pr|open pr|"
        "branch|commit|push|rebase|cherry-pick",
        ["code-task-execution", "github-pr-workflow"],
        "GitHub PR workflow",
    ),
    # Debugging bundle.
    (
        "调试|排查问题|找bug|系统调试|复现问题|根因分析|debugging|debug|"
        "root cause|reproduce|failing behavior|traceback|stack trace",
        ["code-task-execution", "systematic-debugging"],
        "systematic debugging",
    ),
    # TDD / regression-test bundle.
    (
        "TDD|红绿重构|测试先行|RED|GREEN|REFACTOR|写测试用例|回归测试|"
        "test driven|red green|regression test|tests first|write tests",
        ["code-task-execution", "test-driven-development"],
        "TDD workflow",
    ),
    # Review / quality-gate bundle.
    (
        "代码审查|PR审查|pre-commit|质量门禁|自动修复|提交前检查|安全扫描|"
        "code review|quality gate|security scan|preflight|verify before commit",
        ["code-task-execution", "requesting-code-review"],
        "code review",
    ),
    # spike
    (
        "spike|throwaway|验证实验|快速验证|quick check|validate idea",
        ["spike"],
        "spike experiment",
    ),
    # writing-plans
    (
        "写计划|写个计划|做计划|实施计划|任务分解|拆解任务|规划|implementation plan|"
        "write plan|break down",
        ["writing-plans"],
        "writing plans",
    ),
    # subagent-driven-development
    (
        "子代理|子Agent|delegate|multi-agent|并行任务|委托任务|subagent|orchestrate|parallel",
        ["subagent-driven-development"],
        "subagent-driven development",
    ),
    # skill authoring / self-repair bundle.
    (
        "skill author|写skill|更新skill|修skill|优化skill|patch skill|"
        "fix skill|update skill|create skill|skill 不对|skill失效",
        ["code-task-execution", "hermes-agent-skill-authoring", "skill-self-repair"],
        "skill authoring",
    ),
    # knowledge base / wiki
    (
        "知识库|资料库|wiki|GET笔记|RaymondWiki|knowledge base|raymond|notes",
        ["wiki-ops", "getnote-ops"],
        "knowledge base management",
    ),
]

_L1_FAST_KEYWORDS = (
    "小修",
    "快速修复",
    "small fix",
    "quick fix",
    "typo",
    "拼写",
    "一行",
    "配置小改",
)

_NON_L1_KEYWORDS = (
    "pr",
    "pull request",
    "merge",
    "branch",
    "commit",
    "push",
    "debug",
    "调试",
    "复现",
    "根因",
    "tdd",
    "测试先行",
    "回归测试",
    "security",
    "安全",
    "auth",
    "权限",
    "migration",
    "迁移",
    "并发",
    "concurrency",
    "public api",
    "review",
    "审查",
)


def _config_bool(section: dict, key: str, default: bool) -> bool:
    value = section.get(key, default)
    return bool(value) if isinstance(value, bool) else default


def _is_l1_fast_message(msg_lower: str) -> bool:
    msg_lower = msg_lower.lower()
    return any(k.lower() in msg_lower for k in _L1_FAST_KEYWORDS) and not any(
        k.lower() in msg_lower for k in _NON_L1_KEYWORDS
    )


def _build_code_task_compact_context(
    user_message: str,
    task_id: str | None = None,
) -> str:
    """Return a small L1 code-task SOP without loading the full skill body.

    This is intentionally narrow: it avoids the large SKILL.md payload for
    obvious low-risk fixes while preserving the marker and safety gates.
    """
    try:
        from tools.skill_usage import bump_use
        bump_use("code-task-execution")
    except Exception:
        pass

    runtime_note = (
        "[AUTO-PRELOADED-COMPACT] The \"code-task-execution\" skill was "
        f"compact-preloaded for an L1 fast-path request: {user_message.strip()[:80]}"
    )
    session_line = f"\nSession/task id: {task_id}" if task_id else ""
    return f"""[IMPORTANT: Compact runtime guidance from the \"code-task-execution\" skill is active for this turn.]

# code-task-execution — L1 Compact Fast Path

Use this compact path only for obvious low-risk 1-2 file fixes. Escalate to the full skill or related skills if the task involves PRs, debugging/root cause, TDD, security/auth/path/state/migrations, public API behavior, unrelated working-tree changes, or two failed attempts.

Required steps:
1. Classify the task as L1 unless risk signals require escalation.
2. Run `git status --short` before editing in a git repo.
3. Locate the exact file/pattern and make the smallest possible patch.
4. Run one targeted test, import/build parse, CLI smoke, or equivalent.
5. Run scope gate: `git diff --name-only` and `git diff --check`.
6. Final report must include status, changed files, verification commands, risk, next step.
7. End the final answer with exactly: `[skill: code-task-execution]`.

Do not refactor unrelated code, format whole files unless requested, add broad abstractions, auto-commit, or auto-open PR unless the user explicitly asks.{session_line}

{runtime_note}"""


def _matched_auto_preload_slugs(msg_lower: str) -> list[str]:
    """Return deterministic, de-duplicated auto-preload skill bundle."""
    matched_slugs: list[str] = []
    for keywords, slugs, _reason in _AUTO_PRELOAD_RULES:
        kw_list = [k.strip().lower() for k in keywords.split("|") if k.strip()]
        if any(kw in msg_lower for kw in kw_list):
            for slug in slugs:
                if slug not in matched_slugs:
                    matched_slugs.append(slug)
    return matched_slugs


def build_auto_skill_preload_context(
    user_message: str,
    task_id: str | None = None,
) -> str:
    """Auto-preload deterministic skill bundles for the current turn.

    Returns an empty string when no skills match or auto-preload is disabled.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        skills_cfg = cfg.get("skills", {}) if isinstance(cfg.get("skills"), dict) else {}
        if not _config_bool(skills_cfg, "auto_preload", False):
            return ""
        compact_l1 = _config_bool(skills_cfg, "auto_preload_compact_l1", True)
    except Exception:
        return ""

    if not user_message or not user_message.strip():
        return ""

    msg_lower = user_message.lower()
    msg_stripped = user_message.strip()
    matched_slugs = _matched_auto_preload_slugs(msg_lower)

    if not matched_slugs:
        return ""

    prompt_parts: list[str] = []
    loaded_names: list[str] = []
    compact_code_task = (
        compact_l1
        and "code-task-execution" in matched_slugs
        and _is_l1_fast_message(msg_lower)
    )

    for slug in matched_slugs:
        if slug == "code-task-execution" and compact_code_task:
            prompt_parts.append(_build_code_task_compact_context(user_message, task_id=task_id))
            loaded_names.append(slug)
            continue

        cmd_key = f"/{slug.replace('_', '-')}"
        invocation = build_skill_invocation_message(
            cmd_key,
            user_instruction="",
            task_id=task_id,
            runtime_note=(
                f"[AUTO-PRELOADED] The \"{slug}\" skill was automatically "
                f"preloaded because the user's request matches: {msg_stripped[:80]}"
            ),
        )
        if invocation:
            prompt_parts.append(invocation)
            loaded_names.append(slug)

    if not prompt_parts:
        return ""

    logger.debug(
        "auto_skill_preload: matched %s for user message: %s",
        loaded_names,
        msg_stripped[:60],
    )
    return "\n\n".join(prompt_parts)
