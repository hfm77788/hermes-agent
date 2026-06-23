"""
WeCom Material Ingestion - code-level PR creation entry point.

Triggered after WeCom confirms a file belongs to the ingestion group and the
file has been saved to staging. Bridges WeCom gateway -> PR creation.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

WIKI_ROOT = Path("/home/ubuntu/raymond-wiki")
STAGING_ROOT = WIKI_ROOT / "projects" / "_staging" / "materials"


def _run_git_cwd(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run_gh_cwd(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
    )


def create_material_pr(
    *,
    batch_id: str,
    topic_key: str,
    topic_name: str,
    sender: str,
    message_id: str,
    file_count: int,
    confidence: str,
    staging_path: Path,
) -> dict:
    """
    Code-level PR creation for material ingestion.

    Steps:
    1. Create a staging branch
    2. git add the staging directory
    3. git commit
    4. git push
    5. gh pr create
    6. Return structured result
    """
    branch_name = f"staging/materials/{topic_key}/{batch_id}"
    commit_message = f"""feat: add material batch {batch_id} for {topic_key}

Source: wecom message {message_id}
Sender: {sender}
Files: {file_count} file(s)
Topic: {topic_name}
Confidence: {confidence}"""

    try:
        result = _run_git_cwd(WIKI_ROOT, "checkout", "-b", branch_name)
        if result.returncode != 0:
            result = _run_git_cwd(WIKI_ROOT, "checkout", branch_name)
            if result.returncode != 0:
                return {
                    "success": False,
                    "pr_number": None,
                    "pr_url": None,
                    "head_sha": None,
                    "changed_files": [],
                    "error": f"git checkout failed: {result.stderr}",
                    "stage": "git_checkout",
                }

        result = _run_git_cwd(WIKI_ROOT, "add", "--", str(staging_path))
        if result.returncode != 0:
            return {
                "success": False,
                "pr_number": None,
                "pr_url": None,
                "head_sha": None,
                "changed_files": [],
                "error": f"git add failed: {result.stderr}",
                "stage": "git_add",
            }

        status = _run_git_cwd(WIKI_ROOT, "status", "--porcelain")
        if not status.stdout.strip():
            return {
                "success": False,
                "pr_number": None,
                "pr_url": None,
                "head_sha": None,
                "changed_files": [],
                "error": "no changes to commit",
                "stage": "git_add_empty",
            }

        result = _run_git_cwd(WIKI_ROOT, "commit", "-m", commit_message)
        if result.returncode != 0:
            return {
                "success": False,
                "pr_number": None,
                "pr_url": None,
                "head_sha": None,
                "changed_files": [],
                "error": f"git commit failed: {result.stderr}",
                "stage": "git_commit",
            }

        head_before_push = _run_git_cwd(WIKI_ROOT, "rev-parse", "HEAD").stdout.strip()

        result = _run_git_cwd(WIKI_ROOT, "push", "-u", "fork", branch_name)
        if result.returncode != 0:
            return {
                "success": False,
                "pr_number": None,
                "pr_url": None,
                "head_sha": head_before_push,
                "changed_files": [],
                "error": f"git push failed: {result.stderr}",
                "stage": "git_push",
            }

        pr_title = f"[自动录入] {topic_name} {batch_id}"
        pr_body = f"""## 资料收录清单

| 项目 | 内容 |
|------|------|
| 批次号 | {batch_id} |
| 主题 | {topic_name} |
| 发送人 | {sender} |
| 文件数 | {file_count} |
| 置信度 | {confidence} |

请审核后合并到 official 分支。
"""
        result = _run_gh_cwd(
            WIKI_ROOT,
            "pr",
            "create",
            "--title",
            pr_title,
            "--body",
            pr_body,
            "--label",
            "auto-ingestion",
        )
        if result.returncode != 0:
            return {
                "success": False,
                "pr_number": None,
                "pr_url": None,
                "head_sha": head_before_push,
                "changed_files": [],
                "error": f"gh pr create failed: {result.stderr}",
                "stage": "gh_pr_create",
            }

        view_result = _run_gh_cwd(WIKI_ROOT, "pr", "view", "--json", "number,url,headRefOid")
        if view_result.returncode != 0:
            return {
                "success": True,
                "pr_number": None,
                "pr_url": None,
                "head_sha": head_before_push,
                "changed_files": [],
                "error": "PR created but could not fetch details",
                "stage": "gh_pr_view",
            }

        pr_details = json.loads(view_result.stdout)
        changed = _run_git_cwd(WIKI_ROOT, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
        changed_files = changed.stdout.strip().splitlines() if changed.stdout.strip() else []

        return {
            "success": True,
            "pr_number": pr_details["number"],
            "pr_url": pr_details["url"],
            "head_sha": pr_details["headRefOid"],
            "changed_files": changed_files,
            "error": None,
            "stage": "complete",
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "head_sha": None,
            "changed_files": [],
            "error": "timeout during git/gh operation",
            "stage": "timeout",
        }
    except Exception as exc:
        logger.exception("Material PR creation failed")
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "head_sha": None,
            "changed_files": [],
            "error": str(exc),
            "stage": "exception",
        }


def batch_id_from_message(msg_id: str, chat_id: str) -> str:
    """Derive a deterministic batch id from message metadata."""
    import hashlib

    return hashlib.md5(f"{chat_id}{msg_id}".encode()).hexdigest()[:12]
