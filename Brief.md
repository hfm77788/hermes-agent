## 任务：修复 PR #23 的 WeCom text/plain 文件误归类 + attribution failure

### 仓库信息
- 仓库：hfm77788/hermes-agent
- 分支：fix/wecom-plaintext-no-material-trigger
- 当前 head: ca08030cc18dab39d4f6cb331d1654710b090713
- 执行端：Codex CLI（必须 `codex exec --dangerously-bypass-approvals-and-sandbox`，不得用 delegate_task）

### 问题 1：WeCom text/plain 文件误归类（Thread wecom.py:849）

**根因：** `_derive_message_type()` 用白名单判断 document_types，但 `text/plain` 不在白名单里，导致 .txt/.log 等文件附件被归为 `TEXT`，丢失 document context。

**修复方案：**
在 `wecom.py` 的 `_derive_message_type` 中，区分"纯文本消息"和"text/plain 文件上传"：

```python
def _derive_message_type(self, body, text, media_types):
    document_types = (
        "application/",
        "text/markdown",
        "text/x-python",
        "text/html",
        "text/xml",
        "text/csv",
    )
    # 文件上传或已缓存 media path 的 text/plain → DOCUMENT
    if any(mtype.startswith(document_types) for mtype in media_types):
        return MessageType.DOCUMENT

    msgtype = str(body.get("msgtype") or "").lower()

    # text/plain 文件附件（.txt/.log 等）来自文件上传 → DOCUMENT
    # 判断依据：media_types 包含 text/plain 且有 text 内容（caption）或非空
    # 关键：纯文本消息（msgtype=text, 无 media_urls, text 非空）→ TEXT
    if "text/plain" in media_types:
        # 如果有 media_urls → 文件上传的 text/plain → DOCUMENT
        if media_urls:
            return MessageType.DOCUMENT
        # 否则是普通文本消息（WeCom 纯文本 msgtype=text）→ TEXT
        return MessageType.TEXT

    if any(mtype.startswith("image/") for mtype in media_types):
        return MessageType.TEXT if text else MessageType.PHOTO
    if msgtype == "voice":
        return MessageType.VOICE
    return MessageType.TEXT
```

**注意：** `media_urls` 字段的存在是关键区分信号——文件上传/缓存路径会有 `media_urls`，纯文本消息不会有。

### 问题 2：tests/gateway/test_wecom.py 补充 .txt/.log 文件测试

在 `test_inbound_material_detection.py` 或 `test_wecom.py` 中补充：

```python
def test_wecom_text_plain_file_attachment_gets_document_context():
    """WeCom file upload with text/plain MIME (.txt/.log) → DOCUMENT."""
    body = {"msgtype": "text"}
    # 有 media_urls = 文件上传的 text/plain → DOCUMENT
    assert WeComAdapter._derive_message_type(body, "notes.txt", ["text/plain"]) == MessageType.DOCUMENT
    assert WeComAdapter._derive_message_type(body, "debug.log", ["text/plain"]) == MessageType.DOCUMENT

def test_wecom_plain_text_message_still_gets_text():
    """WeCom 纯文本消息（无 media_urls）→ TEXT，不会误归为 DOCUMENT。"""
    body = {"msgtype": "text"}
    # 无 media_urls，纯文本消息 → TEXT
    assert WeComAdapter._derive_message_type(body, "hello", ["text/plain"]) == MessageType.TEXT
```

### 问题 3：attribution failure

检查 `scripts/release.py` 和 `AUTHOR_MAP`，确保 hfm77788 的 contributor 映射正确。

### 执行约束
1. 用 `codex exec --dangerously-bypass-approvals-and-sandbox`，terminal(pty=True)
2. 不得通过 delegate_task 包装
3. 完成后 `git push fork fix/wecom-plaintext-no-material-trigger`
4. 验证 pytest 通过后再回报
