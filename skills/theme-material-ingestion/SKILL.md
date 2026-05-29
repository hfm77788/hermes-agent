---
description: Automatically process theme materials submitted to the theme material ingestion group (oc_a19b4f58f14f7bea48a67610eb0bcb33). Two-gate workflow: preview on trigger match, confirmed ingestion on explicit user confirmation.
---

# Theme Material Ingestion Skill

You are processing messages from the **主题资料录入群** (Theme Material Ingestion Group).

## Two-Gate Workflow

### Gate 1 — preview_only (auto on trigger match)
When a message arrives and meets trigger conditions, the gateway already sets:
- `event.ingestion_action = "preview_only"`
- `event.ingestion_trigger_reason` (e.g., "attachment", "keyword", "document_url", "mention_request")
- `event.ingestion_predicted_topic` (e.g., "competition_aild", "competition_emergency_safety", "chuangqingchun", None)
- `event.ingestion_confidence` ("HIGH", "MEDIUM", "LOW", "UNKNOWN")

**Your job at this stage**: Reply in the group with a preview message. Do NOT write files yet.

### Gate 2 — confirmed_ingestion (on user confirmation)
Only when user replies with one of: **确认录入**, **可以录入**, **进入处理**, **开始处理**, **确认入库**, **生成候选 source**

Only then:
1. Convert to Markdown
2. write_file to `projects/_staging/materials/...`
3. Generate `candidate_source.md`
4. Generate `ingestion_report.md`

## Trigger Conditions (handled by gateway — for reference only)
A message triggers `preview_only` when it is from chat_id `oc_a19b4f58f14f7bea48a67610eb0bcb33` AND contains:
- **Attachments**: images, PDF, Word, Excel, PPT, zip
- **Document URLs**: links to feishu docs, PDF/doc/xls/ppt links
- **Trigger keywords**: 录入, 入库, 归档, 整理资料, 转 markdown, 请处理这份资料, 资料录入测试
- **@mention + explicit processing request** (录入/入库/处理/整理/归档/转 markdown)

**Do NOT trigger for**: 你好, 收到, 谢谢, 在吗, OK, 好的, 嗯, 好

## Preview Message Template (Gate 1)
When you receive `preview_only`, reply in the group with:

```
已检测到一份可能需要录入的资料。
初步判断：
- 资料类型：<文件/链接/文本/图片/未知>
- 可能主题：<AILD / 应急安全 / 创青春 / 未确定>
- 建议暂存位置：<projects/_staging/materials/...>
- 后续可能归入：<既有专区路径或待确认>
- 风险提示：<来源不明/需核验/含个人信息/无>
请确认是否进入资料处理流程：
回复"确认录入"后，我再转 Markdown、生成候选 source，并等待侯方明审核。
```

## Topic Prediction

| topic_key | Name | Existing Path | Duplicate Policy |
|-----------|------|---------------|-----------------|
| `competition_aild` | AILD 智能设计大赛 | `projects/competition-consulting-qa/aild/` | reuse_existing |
| `competition_emergency_safety` | 全国青少年应急与安全科普创新大赛 | `projects/competition-consulting-qa/emergency-safety/` | reuse_existing |
| `chuangqingchun` | 创青春大赛 | `待确认` | require_user_confirmation |
| `unknown` | N/A | N/A — requires human confirmation | N/A |

## Topic Confidence Patterns

### `competition_aild`
- **HIGH**: `AILD`, `aild.caa.org.cn`, `智能设计大赛`
- **MEDIUM**: `aild`

### `competition_emergency_safety`
- **HIGH**: `nyseic.cn`, `全国青少年应急与安全科普创新大赛`, `应急安全`
- **MEDIUM**: `应急与安全`

### `chuangqingchun`
- **HIGH**: `创青春`, `中银杯`
- **MEDIUM**: `创业大赛`, `中国青年创青春`, `天津青年创青春`

## Staging Directory Structure
```
projects/_staging/materials/<topic>/<timestamp>_<sender>_<message_id>/
├── manifest.md          # List of all received items
├── original/           # Original files (images, PDFs, Word docs)
├── converted/          # Converted Markdown
├── candidate_source.md # Generated candidate source metadata
└── ingestion_report.md # Processing report
```

## Step-by-Step Processing

### Step 1: Extract Message
```
- chat_id: oc_a19b4f58f14f7bea48a67610eb0bcb33
- message_id: <from event>
- sender: <from event>
- timestamp: <from event>
- has_attachment: <from event metadata>
- ingestion_action: "preview_only" or "confirmed_ingestion"
- ingestion_trigger_reason: <from event>
- ingestion_predicted_topic: <from event>
- ingestion_confidence: <from event>
```

### Step 2a: If `preview_only` — Send Preview
- Identify material type from attachment presence, URL presence, or text content
- Predict topic using `ingestion_predicted_topic`
- Build staging path suggestion: `projects/_staging/materials/<topic>/`
- Reply with the preview template above

### Step 2b: If `confirmed_ingestion` — Execute Full Pipeline
1. **Convert content** — If attachment (PDF/Word/Excel/PPT), download and convert to Markdown
2. **Write staging files** — write_file to `projects/_staging/materials/<topic>/...`
3. **Generate `candidate_source.md`** with metadata (source chat, sender, timestamp, topic, confidence, file list)
4. **Generate `ingestion_report.md`** with processing summary
5. Reply in group confirming ingestion with staging path

### Step 3: For `unknown` topics
- Do NOT write to wiki
- Ask user to confirm topic before staging
- For `chuangqingchun` until path confirmed, always ask

## Rules
1. **Never write directly to official wiki** (`projects/competition-consulting-qa/*/official/`)
2. **Always stage first** → GPT/人工审核 → 才能 move to official
3. **For `unknown` topics**: always ask in group, never guess
4. **For `chuangqingchun`**: flag as needing path confirmation
5. **Duplicate files**: `duplicate_policy: reuse_existing` — do not overwrite
6. **Logging**: Never print raw_text. Only log: chat_id, message_id, has_attachment, has_url, trigger_reason, predicted_topic, confidence, action
