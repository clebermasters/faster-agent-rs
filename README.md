# Skill Agent

An autonomous AI agent that discovers and executes skills using semantic search and LLM tool chaining. Built with Rust.

## Overview

Skill Agent is a system that:
- **Discovers skills** from the filesystem using vector embeddings
- **Executes skills** autonomously via an LLM agent loop
- **Chains tools** together to complete multi-step tasks
- **Supports multiple LLM providers** (MiniMax, Ollama)

## Architecture

```
skill-agent/
├── crates/
│   ├── skill-core/         # Core types (Skill, Config)
│   ├── skill-registry/     # Loads/parses SKILL.md files
│   ├── skill-embeddings/   # Vector embeddings via Ollama
│   ├── skill-discovery/    # Semantic skill search engine
│   ├── skill-executor/     # Executes skill scripts
│   ├── skill-tools/        # Tool abstractions (bash, read, write, skills)
│   ├── skill-llm/          # LLM client + Agent loop
│   └── skill-agent/        # CLI entry point
└── skills/                 # Skill definitions
```

## Features

- **Semantic Skill Discovery**: Find skills using natural language queries
- **Tool Chaining**: LLM automatically calls multiple tools in sequence
- **Memory**: Conversation history maintained within each session
- **Multiple LLM Providers**: MiniMax (default), Ollama
- **Built-in Tools**: bash, read, write, and custom skills

## Installation

```bash
cd skill-agent
cargo build --release
```

## Configuration

Create a `.env` file:

```bash
# MiniMax (primary LLM)
MINIMAX_API_KEY=your-api-key-here

# Ollama (embeddings + optional LLM)
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

## Usage

### Interactive Agent Mode

```bash
cargo run -- agent
```

### Single Command

```bash
cargo run -- agent "scrape https://example.com save to output.html"
```

### CLI Options

```bash
cargo run -- --help

# Key options:
#   --llm-provider minmax|ollama   LLM provider (default: minimax)
#   --llm-model MODEL              LLM model name
#   --minimax-api-key KEY         MiniMax API key
#   --ollama-url URL               Ollama URL
#   -v, --verbose                  Enable verbose logging
```

## Available Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands |
| `read` | Read files from disk |
| `write` | Write/save content to files |
| `skill_*` | Custom skills (web-scraper, rss-fetcher, etc.) |

## Example Commands

### Web Scraping

```bash
cargo run -- agent "scrape https://example.com save to example.html"
cargo run -- agent "fetch https://httpbin.org/html save to page.html"
```

### File Operations

```bash
cargo run -- agent "read skills/README.md and save to copy.md"
cargo run -- agent "list all files in skills directory"
```

### Shell Commands

```bash
cargo run -- agent "run bash: echo hello world"
cargo run -- agent "check current directory"
cargo run -- agent "what is 10+5?"
```

### Complex Tasks

```bash
# Multi-step: scrape, read, list
cargo run -- agent "scrape example.com, read README.md, list files"

# Get current time and save
cargo run -- agent "get current date and save to timestamp.txt"
```

## Creating Custom Skills

Skills are defined in directories with a `SKILL.md` file:

```bash
skills/my-skill/
├── SKILL.md           # Skill definition
└── scripts/
    └── run.sh         # Execution script (optional)
```

### SKILL.md Format

```markdown
# My Skill

Description of what the skill does.

## Triggers
- do something
- execute task

## Parameters
- input: What to process (required)
```

### Example: Web Scraper Skill

```
skills/web-scraper/
├── SKILL.md
└── scripts/
    └── scrape.sh
```

## Memory & Sessions

- **Session-based**: Memory persists within a single agent session
- **Conversation history**: All messages sent to LLM for context
- **No persistence**: Memory is cleared when agent exits

To add persistent memory, implement conversation storage to disk.

## LLM Provider Setup

### MiniMax (Recommended)

```bash
# Set in .env
MINIMAX_API_KEY=sk-cp-...
```

### Ollama

```bash
# Run local Ollama
ollama serve
ollama pull llama3.2

# Use with skill-agent
cargo run -- --llm-provider ollama --llm-model llama3.2 agent "your task"
```

## Development

```bash
# Build
cargo build

# Run with debug logging
cargo run -- -v agent "your task"

# Run tests
cargo test
```

## License

MIT
