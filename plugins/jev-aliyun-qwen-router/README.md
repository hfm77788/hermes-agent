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
- provider: custom:aliyun_ws
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

## Centralized profile policy

Set `centralized_profile_policy: true` in the **default/root** plugin settings to make the
root config the single routing-policy authority for every Hermes profile. Named profiles still
own their operational credentials and provider definitions, but local router settings no longer
change model policy.

Example:

```yaml
plugins:
  entries:
    jev-aliyun-qwen-router:
      settings:
        centralized_profile_policy: true
        providers: [custom:aliyun_ws]
        flash_model: qwen3.8-flash
        max_model: qwen3.8-max-0902
        max_escalation_probability: 0.90
        min_max_choice_confidence: 0.80
        default_policy: standard
        profile_policies:
          chief-engineer: deep
          hema-teacher: deep
          office-director: deep
        policies:
          standard:
            min_reasoning_effort: low
          deep:
            min_reasoning_effort: medium
```

Only policy keys are overridable by a role policy; provider/model identity remains global.
The cache is scoped by profile + turn id, and routing logs include `profile` and `policy`.

### Central Jev credential

When centralized profile policy is enabled, named profiles do **not** need their own copy of
`TYPESAFE_API_KEY`. The router resolves that credential from the default Hermes root using a
context-local home override and never persists it into the named profile. Legacy/non-centralized
mode keeps the old profile-local environment behavior.
