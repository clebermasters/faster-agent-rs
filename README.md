<div align="center">
  <h1>🚀 Skill Agent</h1>
  <p><strong>An autonomous AI agent that discovers and executes skills using LLM tool chaining. Built with Rust.</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Rust](https://github.com/clebermasters/faster-agent-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/clebermasters/faster-agent-rs/actions/workflows/ci.yml)
</div>

<br/>

Skill Agent empowers your workflows by autonomously determining what needs to be done. It loads capabilities (skills) from the filesystem and makes them available to the LLM via a lightweight metadata catalog — no vector embeddings required for the agent to run.

## 🌟 Key Features

- **📋 Skill Catalog in System Prompt**: All skills are described to the LLM via a lightweight metadata catalog (id, name, description, triggers). The agent picks the right skill at any point during a task — no upfront filtering.
- **🔗 Autonomous Tool Chaining**: An intelligent LLM loop automatically calls multiple tools sequentially to resolve complex tasks.
- **💬 Conversational Memory**: Retains conversation history within each session for continuous context.
- **🤖 Multi-LLM Support**: Works out-of-the-box with MiniMax, Ollama, and **AWS Bedrock** (Claude, Nova, MiniMax M2.1, ZAI GLM-4.7, and more).
- **🛠️ Extensible Tooling**: Ships with built-in tools (`bash`, `read`, `write`) and supports custom file-based skills and the Model Context Protocol (MCP).
- **⚡ Zero Embedding Overhead**: The `agent` command needs only the SKILL.md frontmatter — no Ollama, no SQLite DB, no indexing step.

---

## 🏗️ Architecture

The project is structured as a modular Rust workspace:

```text
skill-agent/
├── crates/
│   ├── skill-core/         # Core data types and configuration
│   ├── skill-registry/     # Loads and parses file-based skills (SKILL.md)
│   ├── skill-embeddings/   # Vector embeddings via Ollama (index/discover only)
│   ├── skill-discovery/    # Semantic and keyword skill search (index/discover only)
│   ├── skill-executor/     # Secure execution of skill scripts
│   ├── skill-tools/        # Tool abstraction layer (bash, read, write, run_skill)
│   ├── skill-llm/          # LLM client integrations & Agent execution loops
│   ├── skill-mcp/          # Model Context Protocol integration
│   └── skill-agent/        # The main CLI entry point
└── skills/                 # Directory containing custom skill definitions
```

### How Skills Reach the LLM

```
Startup:  SKILL.md files → parse frontmatter → in-memory registry
                                                        │
                                          ┌─────────────▼──────────────┐
                                          │  System Prompt              │
                                          │  SKILL CATALOG:             │
                                          │  - web-search: Search ...   │
                                          │  - summarize: Summarize ... │
                                          └─────────────┬──────────────┘
                                                        │
                              LLM sees catalog, picks skill_id
                                                        │
                                          run_skill(skill_id, input)
                                                        │
                                          Execute script on demand
```

---

## ⚡ Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (edition 2021)
- An LLM API key (MiniMax, AWS Bedrock, or a local Ollama model)
- [Ollama](https://ollama.com/) — **only needed** for the `index` and `discover` subcommands

### Installation

Clone the repository and build the project:

```bash
git clone https://github.com/clebermasters/faster-agent-rs.git
cd faster-agent-rs
cargo build --release
```

### Configuration

Create a `.env` file in the root directory:

```bash
# LLM Provider (choose one)
MINIMAX_API_KEY=your-minimax-api-key

# Ollama — only needed for 'index' and 'discover' subcommands
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

---

## 🎮 Usage

### Subcommands

| Subcommand | Needs Ollama? | Description |
|---|---|---|
| `agent` | No | Run the LLM agent with skill catalog |
| `list` | No | List all available skills |
| `run` | No | Execute a specific skill directly |
| `index` | **Yes** | Build the vector embedding database |
| `discover` | **Yes** | Semantic search over indexed skills |

### Agent Mode

Start an interactive session:

```bash
cargo run --release -- agent -i
```

Run a one-off task:

```bash
cargo run --release -- agent "scrape https://example.com and save the content to output.html"
```

### List Skills

```bash
cargo run --release -- list
```

### Run a Skill Directly

```bash
cargo run --release -- run web-search --input "latest Rust releases"
```

### Index Skills (Optional — for `discover` only)

Only needed if you want to use the `discover` subcommand for semantic search:

```bash
# Ensure Ollama is running and the model is pulled
ollama pull nomic-embed-text

cargo run --release -- index
```

### Discover Skills (Optional)

Semantic search over your indexed skill library:

```bash
cargo run --release -- discover "summarize a PDF document"
```

### Advanced CLI Options

```bash
cargo run --release -- --help

# Key flags:
#   --llm-provider <PROVIDER>  'minimax', 'ollama', or 'bedrock' (default: minimax)
#   --llm-model <MODEL>        LLM model name
#   --skills-dir <PATH>        Skills directory (default: ./skills)
#   -v, --verbose              Debug-level logging
#   --streaming                Streaming output
#
# Bedrock-specific:
#   --bedrock-auth <MODE>      default|static|sts-token|sts-role|api-key
#   --bedrock-region <REGION>  AWS region (default: us-east-1)
#   --bedrock-api-key <KEY>    Bedrock API Key (api-key auth)
```

---

## 🧰 Built-in Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands safely within the environment. |
| `read` | Read contents of files from the disk. |
| `write` | Write or append content to files. |
| `run_skill` | Execute any skill by ID. The agent picks from the catalog in its system prompt. |

### Example Scenarios

**Web Scraping & File Operations:**
```bash
cargo run -- agent "fetch https://httpbin.org/html, extract the text, and save to page.md"
```

**Complex Multi-Step Tasks:**
```bash
cargo run -- agent "read README.md, summarize the architecture section, and save it to arch-summary.txt"
```

---

## 🔌 Creating Custom Skills

Extend the agent's capabilities by adding declarative skills. Create a new directory under `skills/` containing a `SKILL.md` file.

```bash
skills/my-new-skill/
├── SKILL.md           # Frontmatter = metadata catalog; body = execution instructions
└── scripts/
    └── run.sh         # (Optional) Executable logic for the skill
```

### Example `SKILL.md`

```markdown
---
name: Web Scraper
description: Extract readable content from web pages
trigger:
  - scrape
  - extract html
capabilities:
  - Fetch web pages
  - Parse HTML structures
---

# Web Scraper Skill

This skill fetches a URL and returns the page content as plain text.
Pass the target URL as the input.
```

The **frontmatter** (`name`, `description`, `trigger`) is what appears in the skill catalog shown to the LLM. The **body** contains the full instructions and is only read when the skill is actually executed via `run_skill`.

---

## 🤖 LLM Providers

| Provider | `--llm-provider` | Description |
|---|---|---|
| **MiniMax** | `minimax` | Default. Cloud API, strong reasoning and coding. |
| **Ollama** | `ollama` | Local inference. Also used for vector embeddings (index/discover). |
| **AWS Bedrock** | `bedrock` | Access Claude, Nova, MiniMax M2.1, ZAI GLM-4.7 and more. Supports IAM, STS, and Bedrock API Key auth. |

For full AWS Bedrock setup instructions see:

**[docs/providers/bedrock.md](docs/providers/bedrock.md)**

### Tested Models from Bedrock:
- `global.anthropic.claude-haiku-4-5-20251001-v1:0`
- `global.amazon.nova-2-lite-v1:0`

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) to get started.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
