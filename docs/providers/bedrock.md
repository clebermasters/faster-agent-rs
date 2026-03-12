# AWS Bedrock Provider

Skill Agent supports Amazon Bedrock as an LLM provider, giving you access to
Claude, Amazon Nova, MiniMax M2.1, ZAI GLM-4.7, and any other model available
in your Bedrock account — all through the same agentic loop.

Set `LLM_PROVIDER=bedrock` (or `--llm-provider bedrock`) to activate it.

---

## Supported Models

| Model | ID | System Prompt | Tool Use | Max Output | Notes |
|---|---|---|---|---|---|
| Claude 3.5 Sonnet v2 | `anthropic.claude-3-5-sonnet-20241022-v2:0` | ✅ | ✅ Full | — | Recommended default |
| Claude 3.5 Haiku | `anthropic.claude-3-5-haiku-20241022-v1:0` | ✅ | ✅ Full | — | Fast & cheap |
| Claude 3 Opus | `anthropic.claude-3-opus-20240229-v1:0` | ✅ | ✅ Full | — | Highest reasoning |
| Amazon Nova 2 Lite | `amazon.nova-2-lite-v1:0` | ✅ | ✅ Full | — | AWS-native, multimodal |
| MiniMax M2.1 | `minimax.minimax-m2.1` | ⚠️ | ✅ Client-side | 8 192 | 1 M-token context |
| ZAI GLM-4.7 | `zai.glm-4.7` | ⚠️ | ⚠️ Unreliable | 4 096 | May skip tool calls |

**Legend:**
- ✅ — Natively supported via the Converse API
- ⚠️ — Handled with a workaround (see [Model-specific behaviour](#model-specific-behaviour))

> Use `--llm-model <MODEL_ID>` or `DEFAULT_MODEL=<MODEL_ID>` to select the model.

---

## Authentication

Choose one of five authentication methods via `BEDROCK_AUTH`:

| Value | Description | Required variables |
|---|---|---|
| `default` | AWS default credential chain | None (uses env vars / `~/.aws/credentials` / instance profile) |
| `static` | Long-term IAM access key | `BEDROCK_ACCESS_KEY_ID`, `BEDROCK_SECRET_ACCESS_KEY` |
| `sts-token` | Pre-obtained STS session token | `BEDROCK_ACCESS_KEY_ID`, `BEDROCK_SECRET_ACCESS_KEY`, `BEDROCK_SESSION_TOKEN` |
| `sts-role` | Assume an IAM role at startup | `BEDROCK_ROLE_ARN` (+ optional `BEDROCK_EXTERNAL_ID`) |
| `api-key` | Bedrock API Key (OpenAI-compat endpoint) | `BEDROCK_API_KEY` |

> **Tip for MiniMax & ZAI models:** Use `BEDROCK_AUTH=api-key`. The
> `bedrock-mantle` endpoint they route to supports system prompts and tool
> calling in standard OpenAI format, bypassing the limitations of those models
> on the native Converse API.

### `default` — AWS credential chain

No extra configuration needed. The standard AWS resolution order applies:

1. `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables
2. `~/.aws/credentials` profile
3. EC2 / ECS / Lambda instance profile

```bash
LLM_PROVIDER=bedrock \
DEFAULT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0 \
skill-agent agent "explain this codebase"
```

---

### `static` — Long-term IAM credentials

```bash
BEDROCK_AUTH=static
BEDROCK_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
BEDROCK_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
BEDROCK_REGION=us-east-1
```

The IAM user or role needs the `bedrock:InvokeModel` permission (plus
`bedrock:InvokeModelWithResponseStream` for streaming):

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": "arn:aws:bedrock:*::foundation-model/*"
}
```

---

### `sts-token` — Pre-obtained STS session token

Use this when your CI/CD system or orchestrator already hands you short-lived
STS credentials (e.g. GitHub Actions OIDC, AWS Vault exports):

```bash
BEDROCK_AUTH=sts-token
BEDROCK_ACCESS_KEY_ID=ASIAIOSFODNN7EXAMPLE
BEDROCK_SECRET_ACCESS_KEY=<secret>
BEDROCK_SESSION_TOKEN=AQoDYXdz...
BEDROCK_REGION=us-east-1
```

---

### `sts-role` — Assume an IAM role

The agent calls STS `AssumeRole` at startup using the **default credential
chain** as the source credentials, then uses the resulting temporary
credentials for all Bedrock calls.

```bash
BEDROCK_AUTH=sts-role
BEDROCK_ROLE_ARN=arn:aws:iam::123456789012:role/BedrockAgentRole
BEDROCK_ROLE_SESSION_NAME=skill-agent          # optional, default: skill-agent
BEDROCK_EXTERNAL_ID=shared-secret              # optional, for cross-account
BEDROCK_REGION=us-east-1
```

The source principal needs `sts:AssumeRole` on the target role, and the target
role needs `bedrock:InvokeModel`.

---

### `api-key` — Bedrock API Key (ABSK format)

Uses a Bedrock API Key to authenticate directly against the standard
`bedrock-runtime.{region}.amazonaws.com` Converse API endpoint using
`Authorization: Bearer {absk_token}` — no AWS Signature V4 required.

Generate an API key from the Bedrock console under **API Keys**. The key is
provided in **ABSK format**: `ABSK<base64(BedrockAPIKey-{id}:{secret})>`.
Pass the full ABSK string as-is — no decoding needed.

```bash
BEDROCK_AUTH=api-key
BEDROCK_API_KEY=ABSK<your-bedrock-api-key>
BEDROCK_REGION=us-east-1
```

This is the **recommended auth method for MiniMax M2.1 and ZAI GLM-4.7**
because it works with all Bedrock model families through the native Converse
API, including full system-prompt and tool-calling support.

---

## Environment Variable Reference

| Variable | CLI flag | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `--llm-provider` | `minimax` | Set to `bedrock` |
| `DEFAULT_MODEL` | `--llm-model` | `MiniMax-Text-01` | Bedrock model ID |
| `BEDROCK_AUTH` | `--bedrock-auth` | `default` | Auth mode |
| `BEDROCK_REGION` | `--bedrock-region` | `us-east-1` | AWS region |
| `BEDROCK_ACCESS_KEY_ID` | `--bedrock-access-key-id` | — | IAM access key |
| `BEDROCK_SECRET_ACCESS_KEY` | `--bedrock-secret-access-key` | — | IAM secret key |
| `BEDROCK_SESSION_TOKEN` | `--bedrock-session-token` | — | STS session token |
| `BEDROCK_ROLE_ARN` | `--bedrock-role-arn` | — | Role to assume |
| `BEDROCK_ROLE_SESSION_NAME` | `--bedrock-role-session-name` | `skill-agent` | Session tag |
| `BEDROCK_EXTERNAL_ID` | `--bedrock-external-id` | — | Cross-account external ID |
| `BEDROCK_API_KEY` | `--bedrock-api-key` | — | Bedrock API key (ABSK format) |

---

## Quick-Start Examples

### Claude 3.5 Sonnet — default credential chain

```bash
LLM_PROVIDER=bedrock \
DEFAULT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0 \
skill-agent agent "review my Rust code for performance issues"
```

### Amazon Nova 2 Lite — static credentials, streaming

```bash
LLM_PROVIDER=bedrock \
DEFAULT_MODEL=amazon.nova-2-lite-v1:0 \
BEDROCK_AUTH=static \
BEDROCK_ACCESS_KEY_ID=AKIA... \
BEDROCK_SECRET_ACCESS_KEY=... \
skill-agent agent --streaming "summarise the architecture of this project"
```

### MiniMax M2.1 — Bedrock API Key (recommended for MiniMax)

```bash
# BEDROCK_API_KEY must be the full ABSK token from the Bedrock console
export BEDROCK_API_KEY="ABSK<your-absk-token>"

LLM_PROVIDER=bedrock \
DEFAULT_MODEL=minimax.minimax-m2.1 \
BEDROCK_AUTH=api-key \
BEDROCK_REGION=us-east-1 \
skill-agent agent "build a CLI tool in Python that converts CSV to JSON"
```

### ZAI GLM-4.7 — assumed role, cross-account

```bash
LLM_PROVIDER=bedrock \
DEFAULT_MODEL=zai.glm-4.7 \
BEDROCK_AUTH=sts-role \
BEDROCK_ROLE_ARN=arn:aws:iam::987654321098:role/CrossAccountBedrock \
BEDROCK_EXTERNAL_ID=my-external-id \
BEDROCK_REGION=us-east-1 \
skill-agent agent "explain this SQL schema"
```

### Via `.env` file

```env
LLM_PROVIDER=bedrock
DEFAULT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_REGION=us-east-1
BEDROCK_AUTH=sts-role
BEDROCK_ROLE_ARN=arn:aws:iam::123456789012:role/BedrockAgentRole
```

---

## Model-Specific Behaviour

### Amazon Nova 2 Lite (`amazon.nova-2-lite-v1:0`)

Full Converse API support. The `topK` sampling parameter is passed via
`additionalModelRequestFields` (a Nova-specific requirement):

```json
{ "inferenceConfig": { "topK": 50 } }
```

This is handled automatically — no action needed on your side.

You can also use the regional inference profile prefix for cross-region routing:

```
us.amazon.nova-2-lite-v1:0
```

---

### MiniMax M2.1 (`minimax.minimax-m2.1`)

**Context window:** 1 M tokens — excellent for large codebases or long documents.
**Max output:** 8 192 tokens.

**System prompt:** The Converse API's native `system` field is not documented
for MiniMax. When using `BEDROCK_AUTH=static|sts-token|sts-role|default`, the
agent automatically injects the system prompt as the first user message:

```
[System Instructions]
<your system prompt here>
```

**Recommendation:** Use `BEDROCK_AUTH=api-key` (mantle endpoint) for full,
native system-prompt and tool-calling support with no workarounds.

---

### ZAI GLM-4.7 (`zai.glm-4.7`)

**Context window:** 128 K tokens.
**Max output:** 4 096 tokens.

**System prompt:** Same injection workaround as MiniMax (see above).

**Tool calling:** Supported via the Converse API `toolConfig`, but the model
occasionally answers directly instead of invoking a requested tool. The agent
logs a warning when tools are sent to this model:

```
WARN Bedrock: model zai.glm-4.7 has unreliable tool use — sending toolConfig anyway
```

If tool reliability matters, use `BEDROCK_AUTH=api-key` (mantle endpoint)
or choose a different model.

---

## Architecture

Two backend clients handle requests depending on the auth mode:

```
BEDROCK_AUTH=api-key
    └─► BedrockBearerClient (reqwest, HTTP/1.1)
            URL: https://bedrock-runtime.{region}.amazonaws.com/model/{id}/converse
            Auth: Authorization: Bearer <absk-token>
            Format: Bedrock Converse API (JSON)
            System prompts: native for Claude/Nova; injected for MiniMax/ZAI
            Tool calling: Bedrock toolSpec format

BEDROCK_AUTH=static|sts-token|sts-role|default
    └─► BedrockConverseClient (aws-sdk-bedrockruntime)
            URL: https://bedrock-runtime.{region}.amazonaws.com
            Auth: AWS Signature V4 (managed by aws-sdk-bedrockruntime)
            Format: Bedrock Converse API
            System prompts: native for Claude/Nova; injected for MiniMax/ZAI
            Tool calling: Bedrock toolSpec format
```

The per-model capability registry (`bedrock/models.rs`) drives all
model-specific adaptations automatically — add a new model pattern there to
extend support.

---

## Troubleshooting

**`AccessDeniedException`**
The principal lacks `bedrock:InvokeModel`. Attach the policy shown in the
[static credentials](#static--long-term-iam-credentials) section.

**`ValidationException: model not found`**
The model is not enabled in your account. Enable it from the Bedrock console
under **Model access**.

**`Could not find credentials`** (with `BEDROCK_AUTH=default`)
No credentials discovered in the default chain. Run `aws configure` or export
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.

**`STS AssumeRole failed`**
The source principal lacks `sts:AssumeRole` on the target role, or the trust
policy on the role does not allow the source principal.

**MiniMax / ZAI tool calls not working**
Switch to `BEDROCK_AUTH=api-key` which routes through the `bedrock-mantle`
endpoint for full tool-calling support.

**ZAI GLM-4.7 answers directly instead of using a tool**
This is a known model behaviour. Consider switching to Claude or Nova for
tool-heavy agentic tasks, or use `BEDROCK_AUTH=api-key` which may improve
reliability.
