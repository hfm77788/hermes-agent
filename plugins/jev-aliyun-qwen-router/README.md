# Jev Aliyun Qwen Router

Bundled Hermes Agent plugin for per-turn model and reasoning-effort routing on an
Alibaba Cloud OpenAI-compatible endpoint that can serve both qwen3.8-flash and
qwen3.8-max-0902.

Policy:
- One Jev decision is made for the first LLM request of a Hermes turn.
- The decision is cached by turn_id and reused for tool-loop follow-ups.
- Jev selects low, medium, or xhigh reasoning effort.
- A separate Flash-vs-Max choice must select Max with both high probability and
  high choice confidence before Flash may escalate.
- Manual Max selections are not downgraded by default.
- Provider/Jev/parse failures return no rewrite, so Hermes sends the original request.

Privacy boundary:
- Only the latest user text is sent to TypeSafe.
- It is force-redacted with Hermes secret redaction and truncated before egress.
- System prompts, prior history, tool definitions, tool results, credentials, and
  provider configuration are not sent.

Default production settings:
- provider: custom:aliyun_qwen
- flash_model: qwen3.8-flash
- max_model: qwen3.8-max-0902
- jev_model: jev-latest
- timeout_seconds: 1.5
- min_effort_confidence: 0.50
- max_escalation_probability: 0.90
- min_max_choice_confidence: 0.80
- max_excerpt_chars: 2400
- respect_existing_reasoning: false (Jev owns per-turn effort; set true to preserve an operator-pinned request effort)
- respect_manual_max: true

The plugin is opt-in through plugins.enabled and requires TYPESAFE_API_KEY.
