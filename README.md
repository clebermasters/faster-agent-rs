<div align="center">
  <h1>🚀 Skill Agent</h1>
  <p><strong>An autonomous AI agent that discovers and executes skills using semantic search and LLM tool chaining. Built with Rust.</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Rust](https://github.com/clebermasters/faster-agent-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/clebermasters/faster-agent-rs/actions/workflows/ci.yml)
</div>

<br/>

Skill Agent empowers your workflows by autonomously determining what needs to be done. It discovers capabilities (skills) from the filesystem using vector embeddings and seamlessly chains tools together to complete multi-step tasks.

## 🌟 Key Features

- **🧠 Semantic Skill Discovery**: Finds the right skills for the job using natural language queries and vector search.
- **🔗 Autonomous Tool Chaining**: An intelligent LLM loop automatically calls multiple tools sequentially to resolve complex tasks.
- **💬 Conversational Memory**: Retains conversation history within each session for continuous context.
- **🤖 Multi-LLM Support**: Works out-of-the-box with MiniMax and Ollama.
- **🛠️ Extensible Tooling**: Ships with built-in tools (`bash`, `read`, `write`) and supports custom file-based skills and the Model Context Protocol (MCP).

---

## 🏗️ Architecture

The project is structured as a modular Rust workspace:

```text
skill-agent/
├── crates/
│   ├── skill-core/         # Core data types and configuration
│   ├── skill-registry/     # Loads and parses file-based skills (SKILL.md)
│   ├── skill-embeddings/   # Local vector embeddings generation (via Ollama)
│   ├── skill-discovery/    # Semantic and keyword skill search engine
│   ├── skill-executor/     # Secure execution of skill scripts
│   ├── skill-tools/        # Tool abstraction layer (bash, read, write)
│   ├── skill-llm/          # LLM client integrations & Agent execution loops
│   ├── skill-mcp/          # Model Context Protocol integration
│   └── skill-agent/        # The main CLI entry point
└── skills/                 # Directory containing custom skill definitions
```

---

## ⚡ Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (edition 2021)
- [Ollama](https://ollama.com/) (For local embeddings and optional local LLM)

### Installation

Clone the repository and build the project:

```bash
git clone https://github.com/clebermasters/faster-agent-rs.git
cd faster-agent-rs
cargo build --release
```

### Configuration

Create a `.env` file in the root directory to configure your providers:

```bash
# MiniMax (Recommended primary LLM for optimal reasoning)
MINIMAX_API_KEY=your-minimax-api-key

# Ollama (Required for vector embeddings; optional for LLM)
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

*Note: Ensure Ollama is running (`ollama serve`) and the embedding model is pulled (`ollama pull nomic-embed-text`).*

---

## 🎮 Usage

### Interactive Agent Mode

Start a continuous conversational session:

```bash
cargo run --release -- agent
```

### Single Command Execution

Run a specific, one-off task:

```bash
cargo run --release -- agent "scrape https://example.com and save the content to output.html"
```

### Advanced CLI Options

```bash
cargo run --release -- --help

# Useful flags:
#   --llm-provider <PROVIDER>  Set to 'minimax' or 'ollama' (default: ollama)
#   --llm-model <MODEL>        Specify the LLM model name
#   -v, --verbose              Enable debug-level logging
#   --streaming                Enable streaming output for the agent
```

---

## 🧰 Built-in Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands safely within the environment. |
| `read` | Read contents of files from the disk. |
| `write` | Write or append content to files. |
| `skill_*` | Dynamically discovered custom skills defined in your workspace. |

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
├── SKILL.md           # Defines the skill's identity, triggers, and parameters
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
This skill extracts text from websites...
```

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
