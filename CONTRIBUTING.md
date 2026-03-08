# Contributing to Skill Agent

Thank you for your interest in contributing to the **Skill Agent** project! We appreciate your help to make this autonomous agent framework even better.

## 🚀 Getting Started

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork locally:
    ```bash
    git clone https://github.com/your-username/faster-agent-rs.git
    cd faster-agent-rs
    ```
3.  **Add upstream** repository to pull latest changes:
    ```bash
    git remote add upstream https://github.com/clebermasters/faster-agent-rs.git
    ```

## 🛠️ Development Setup

You will need the following installed:
-   **Rust** (edition 2021) - Install via [rustup](https://rustup.rs/)
-   **Ollama** - Required for vector embeddings locally. Ensure you pull the `nomic-embed-text` model:
    ```bash
    ollama run nomic-embed-text
    ```

Build the project:
```bash
cargo build
```

Run tests to ensure everything is working correctly:
```bash
cargo test --workspace
```

Run the linter to ensure code style:
```bash
cargo clippy --workspace -- -D warnings
```

Format your code:
```bash
cargo fmt --all
```

## 🏗️ Project Architecture

The project is split into a Rust workspace under the `crates/` directory:
-   `skill-core`: Foundational structs (`Skill`, `Config`).
-   `skill-discovery`: The engine that handles searching vector databases and finding skills.
-   `skill-embeddings`: Generates vector embeddings using the configured LLM provider (Ollama).
-   `skill-executor`: Executes code block scripts associated with skills.
-   `skill-llm`: LLM integration (`Ollama`, `MiniMax`).
-   `skill-mcp`: Integrates the Model Context Protocol.
-   `skill-registry`: Handles filesystem-based skill loading and caching.
-   `skill-tools`: Foundational actions (`bash`, `read`, `write`, `skill`).
-   `skill-agent`: The CLI entry point bringing it all together.

When adding a feature, identify which crate your logic belongs in. If it's a new capability or action, consider if it should be an external `skill/` (SKILL.md) or a native `ToolBox` extension.

## 📝 Pull Request Process

1.  Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2.  Update the `README.md` with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3.  Ensure your code is well-tested. Include unit tests for pure functions and integration tests where appropriate in the `tests/` directory.
4.  Run `cargo fmt` and `cargo clippy` before submitting.
5.  Your pull request must pass all Continuous Integration (CI) checks.

## 🐛 Bug Reports

When opening a bug report, please provide:
1.  Your operating system and Rust version.
2.  The exact command you ran.
3.  The complete output or error stack trace.
4.  Steps to reproduce the issue.

We look forward to reviewing your PRs!
