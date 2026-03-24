# Running skill-agent via Cron

Run the skill-agent on a schedule to perform automated tasks — generate reports, check statuses, fetch news, etc.

## Key Concept: Working Directory

The agent resolves relative paths and generates output files based on **the directory you `cd` into before calling it**. Always `cd` to the target directory first.

## Basic Pattern

```bash
cd /path/to/workdir && skill-agent [OPTIONS] agent "your prompt"
```

## Examples

### Generate a daily AI news report (Opus 4.6 via wrapper)

```bash
# Every day at 8am
0 8 * * * cd /home/cleber_rodrigues/reports && skill-agent --streaming --llm-provider openai --llm-model claude-sonnet-4-6 agent "Search for the latest AI news and write an HTML report to ai-news-$(date +\%Y-\%m-\%d).html"
```

### Check USCIS status weekly (MiniMax)

```bash
# Every Monday at 9am
0 9 * * 1 cd /home/cleber_rodrigues/kiro-bot && skill-agent --streaming --llm-provider minimax --llm-model MiniMax-M2.7 agent "Check my USCIS application status"
```

### Daily code review (Bedrock Nova)

```bash
# Every day at 6pm
0 18 * * * cd /home/cleber_rodrigues/myproject && skill-agent --streaming --llm-provider bedrock --llm-model amazon.nova-pro-v1:0 --bedrock-auth default agent "Review recent git changes and write a summary to review-$(date +\%Y-\%m-\%d).md"
```

## Using the `ask` Script

The `ask` script always runs from the skill-agent directory and uses Opus 4.6. If you need output in a specific directory, tell the agent explicitly:

```bash
0 8 * * * /home/cleber_rodrigues/kiro-bot/skill-agent/ask "Generate a weekly summary and save it to /home/cleber_rodrigues/reports/weekly-$(date +\%Y-\%m-\%d).md"
```

## Wrapper Script for Cron

For complex setups, create a wrapper script:

```bash
#!/usr/bin/env bash
# /home/cleber_rodrigues/bin/agent-task.sh
set -euo pipefail

WORKDIR="${1:-.}"
PROVIDER="${2:-openai}"
MODEL="${3:-claude-opus-4-6}"
PROMPT="${4:?Usage: agent-task.sh <workdir> <provider> <model> \"prompt\"}"

cd "$WORKDIR"

# Load env vars
source /home/cleber_rodrigues/kiro-bot/skill-agent/.env 2>/dev/null || true

exec skill-agent \
  --streaming \
  --llm-provider "$PROVIDER" \
  --llm-model "$MODEL" \
  agent "$PROMPT"
```

Then in crontab:

```bash
# Daily report with Opus
0 8 * * * /home/cleber_rodrigues/bin/agent-task.sh /home/cleber_rodrigues/reports openai claude-opus-4-6 "Write today's AI news report as ai-news.html"

# Weekly review with Nova
0 9 * * 1 /home/cleber_rodrigues/bin/agent-task.sh /home/cleber_rodrigues/myproject bedrock amazon.nova-pro-v1:0 "Review this week's commits and write summary.md"
```

## Provider / Model Quick Reference

| Provider | Model | Auth |
|----------|-------|------|
| `openai` | `claude-opus-4-6` | Wrapper on localhost:8000 |
| `openai` | `claude-sonnet-4-6` | Wrapper on localhost:8000 |
| `minimax` | `MiniMax-M2.5` | `MINIMAX_API_KEY` env var |
| `bedrock` | `amazon.nova-pro-v1:0` | `--bedrock-auth default` (AWS creds) |
| `bedrock` | `amazon.nova-lite-v1:0` | `--bedrock-auth default` |
| `bedrock` | `anthropic.claude-sonnet-4-20250514-v1:0` | `--bedrock-auth default` |
| `bedrock` | `zai.zai-glm-4-9b-0414-v1` | `--bedrock-auth api-key` |
| `ollama` | any local model | `OLLAMA_URL` env var |

## Tips

- **Logging**: Redirect output to capture results: `... agent "prompt" > /tmp/agent-$(date +\%s).log 2>&1`
- **Env vars**: Cron has a minimal environment. Either `source` the `.env` file or set vars in the crontab: `MINIMAX_API_KEY=sk-... LLM_PROVIDER=minimax`
- **PATH**: Ensure `skill-agent` is in PATH or use the full path to the binary
- **Non-streaming for cron**: Use `--no-stream` (or omit `--streaming`) for cleaner log output since there's no terminal to display real-time tokens
