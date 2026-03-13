# skill-agent

The main CLI entry point for the [Skill Agent](https://github.com/clebermasters/faster-agent-rs) framework.

An autonomous AI agent that loads skills from the filesystem, presents them as a lightweight catalog to the LLM, and executes them on demand — no vector embeddings required for the agent to run.

## Install

```bash
cargo install skill-agent
```

## Quick Start

```bash
skill-agent agent "scrape https://example.com and save the content"
skill-agent list
skill-agent run my-skill --input "hello"
```

See the [main repository](https://github.com/clebermasters/faster-agent-rs) for full documentation.
