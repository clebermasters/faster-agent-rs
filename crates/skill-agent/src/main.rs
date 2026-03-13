mod agents;

use clap::{Parser, Subcommand};
use colored::Colorize;
use skill_core::{Config, SkillQuery};
use skill_discovery::SkillDiscoveryEngine;
use skill_embeddings::EmbeddingService;
use skill_executor::{ExecutionContext, SkillExecutor};
use skill_llm::{Agent, BedrockAuth, MiniMaxClient, OllamaClient, StreamingAgent};
use skill_mcp::McpRegistry;
use skill_registry::SkillRegistry;
use skill_tools::{BashTool, ReadTool, SkillTool, ToolBox, ToolRegistry, WriteTool};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "skill-agent")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Autonomous skill discovery agent", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, default_value = "./skills")]
    skills_dir: PathBuf,

    #[arg(long, default_value = "ollama")]
    provider: String,

    #[arg(long, env = "OLLAMA_URL", default_value = "http://localhost:11434")]
    ollama_url: String,

    #[arg(long, env = "EMBEDDING_MODEL", default_value = "nomic-embed-text")]
    ollama_model: String,

    #[arg(long, env = "DEFAULT_MODEL", default_value = "MiniMax-Text-01")]
    llm_model: String,

    #[arg(long, env = "LLM_PROVIDER", default_value = "minimax")]
    llm_provider: String,

    #[arg(long, env = "MAX_ITERATIONS", default_value = "10")]
    max_iterations: i32,

    #[arg(long)]
    minimax_url: Option<String>,

    #[arg(long)]
    minimax_api_key: Option<String>,

    #[arg(short, long, default_value = "./skill-agent.db")]
    db_path: PathBuf,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long, default_value = "false")]
    streaming: bool,

    #[arg(long, default_value = "./mcp.json")]
    mcp_config: PathBuf,

    #[arg(long, default_value = "30")]
    mcp_timeout: u64,

    /// Path to AGENTS.md file (default: ./AGENTS.md or ./CLAUDE.md)
    #[arg(long)]
    agents_file: Option<String>,

    /// Additional system prompt to append to the agent's prompt
    #[arg(long)]
    system_prompt: Option<String>,

    // ------------------------------------------------------------------
    // AWS Bedrock (--llm-provider bedrock)
    // ------------------------------------------------------------------

    /// Bedrock auth method: static | sts-token | sts-role | api-key | default
    #[arg(long, env = "BEDROCK_AUTH", default_value = "default")]
    bedrock_auth: String,

    /// AWS region for Bedrock (default: us-east-1)
    #[arg(long, env = "BEDROCK_REGION", default_value = "us-east-1")]
    bedrock_region: String,

    /// AWS Access Key ID (for static or sts-token auth)
    #[arg(long, env = "BEDROCK_ACCESS_KEY_ID")]
    bedrock_access_key_id: Option<String>,

    /// AWS Secret Access Key (for static or sts-token auth)
    #[arg(long, env = "BEDROCK_SECRET_ACCESS_KEY")]
    bedrock_secret_access_key: Option<String>,

    /// AWS Session Token (for sts-token auth — pre-obtained STS token)
    #[arg(long, env = "BEDROCK_SESSION_TOKEN")]
    bedrock_session_token: Option<String>,

    /// IAM Role ARN to assume via STS (for sts-role auth)
    #[arg(long, env = "BEDROCK_ROLE_ARN")]
    bedrock_role_arn: Option<String>,

    /// Session name tag for the assumed role (default: skill-agent)
    #[arg(long, env = "BEDROCK_ROLE_SESSION_NAME", default_value = "skill-agent")]
    bedrock_role_session_name: String,

    /// External ID for cross-account role assumption (optional)
    #[arg(long, env = "BEDROCK_EXTERNAL_ID")]
    bedrock_external_id: Option<String>,

    /// Bedrock API Key for the bedrock-mantle OpenAI-compatible endpoint
    /// (for api-key auth — best for MiniMax and ZAI models)
    #[arg(long, env = "BEDROCK_API_KEY")]
    bedrock_api_key: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Index all skills in the skills directory
    Index,

    /// Discover skills for a task
    Discover {
        /// The task description
        task: String,

        /// Maximum number of skills to return
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Minimum match threshold (0.0-1.0)
        #[arg(short, long, default_value = "0.1")]
        threshold: f64,
    },

    /// Execute a specific skill
    Run {
        /// The skill ID to run
        skill_id: String,

        /// Input to pass to the skill
        #[arg(short, long)]
        input: Option<String>,
    },

    /// List all available skills
    List,

    /// Start agent mode
    Agent {
        /// Initial task
        #[arg(default_value = "")]
        task: String,

        /// Keep the session open for continuous interaction
        #[arg(short, long, default_value_t = false)]
        interactive: bool,
    },

    /// Show system prompt for skill discovery
    SystemPrompt,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    colored::control::set_override(true);
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let subscriber = FmtSubscriber::builder()
        .with_max_level(if cli.verbose {
            Level::DEBUG
        } else {
            Level::WARN
        })
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let config = Config {
        skills_dir: cli.skills_dir.clone(),
        embedding_model: cli.ollama_model.clone(),
        ollama_url: cli.ollama_url.clone(),
        db_path: cli.db_path.clone(),
        ..Default::default()
    };

    let embeddings = match cli.provider.as_str() {
        "ollama" => {
            info!("Using Ollama provider");
            EmbeddingService::new_ollama(
                config.ollama_url.clone(),
                config.embedding_model.clone(),
                config.db_path.clone(),
            )
        }
        other => {
            anyhow::bail!("Unknown provider: {}. Only 'ollama' is supported.", other)
        }
    };

    let registry = SkillRegistry::new(config.skills_dir.clone());
    let mut engine = SkillDiscoveryEngine::new(registry, embeddings);

    match cli.command {
        Commands::Index => {
            info!("Indexing skills from {:?}", config.skills_dir);

            engine.registry_mut().load().await?;
            engine.index_all().await?;

            info!("Indexed {} skills", engine.registry().count());
        }

        Commands::Discover {
            task,
            limit,
            threshold,
        } => {
            engine.registry_mut().load().await?;
            engine.index_all().await?;

            let query = SkillQuery {
                task,
                limit,
                threshold,
                ..Default::default()
            };

            let results = engine.discover(query).await?;

            println!("\n=== Discovered Skills ===\n");
            for (i, discovered) in results.iter().enumerate() {
                println!(
                    "{}. {} (score: {:.2})",
                    i + 1,
                    discovered.skill.name,
                    discovered.score
                );
                println!("   {}", discovered.skill.description);
                println!();
            }
        }

        Commands::Run { skill_id, input } => {
            engine.registry_mut().load().await?;

            let skill = engine
                .registry()
                .get(&skill_id)
                .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", skill_id))?
                .clone();

            let executor = SkillExecutor::new(config.skills_dir);
            let context = ExecutionContext::default();

            let result = executor
                .execute_skill(&skill, input.as_deref(), &context)
                .await?;

            if result.success {
                println!("{}", result.output);
            } else {
                eprintln!("Error: {}", result.error.unwrap_or_default());
            }
        }

        Commands::List => {
            engine.registry_mut().load().await?;

            let skills = engine.registry().get_all();

            println!("\n=== Available Skills ({}) ===\n", skills.len());
            for skill in skills {
                println!("- {}: {}", skill.id, skill.name);
                if !skill.description.is_empty() {
                    println!("  {}", skill.description);
                }
                if !skill.triggers.is_empty() {
                    println!("  Triggers: {}", skill.triggers.join(", "));
                }
                println!();
            }
        }

        Commands::Agent { task, interactive } => {
            // Load skill metadata only (no embedding indexing needed)
            engine.registry_mut().load().await?;

            // Load MCP servers (lazy - will connect on first use)
            let mut mcp_registry = McpRegistry::new(Duration::from_secs(cli.mcp_timeout));
            if cli.mcp_config.exists() {
                match mcp_registry.load_from_file(&cli.mcp_config).await {
                    Ok(_) => {
                        info!(
                            "Loaded MCP registry with {} tools from {} servers",
                            mcp_registry.tool_count(),
                            mcp_registry.server_count()
                        );
                        if !mcp_registry.list_names().is_empty() {
                            info!("MCP tools available: {:?}", mcp_registry.list_names());
                        }
                    }
                    Err(e) => {
                        warn!("Failed to load MCP config: {}", e);
                    }
                }
            } else {
                info!("No MCP config found at {:?}", cli.mcp_config);
            }

            // Load AGENTS.md / custom system prompt
            let agents_config = agents::AgentsConfig::load(cli.agents_file.as_deref());
            if let Some(source) = &agents_config.source {
                info!("Loaded AGENTS.md from: {:?}", source);
            }

            // Build extra system prompt: CLI --system-prompt + AGENTS.md
            let mut extra_prompt_parts: Vec<String> = Vec::new();
            if let Some(cli_prompt) = &cli.system_prompt {
                extra_prompt_parts.push(cli_prompt.clone());
            }
            if let Some(agents_content) = agents_config.content() {
                extra_prompt_parts.push(agents_content.to_string());
            }
            let extra_system_prompt = if extra_prompt_parts.is_empty() {
                None
            } else {
                Some(extra_prompt_parts.join("\n\n"))
            };

            let mut tools = ToolRegistry::new();
            tools.register(ToolBox::Bash(BashTool::new()));
            tools.register(ToolBox::Read(ReadTool::new(
                config.skills_dir.clone().to_string_lossy().to_string(),
            )));
            tools.register(ToolBox::Write(WriteTool::new(
                config.skills_dir.clone().to_string_lossy().to_string(),
            )));

            // Register a single SkillTool holding all skills (metadata in system
            // prompt, full instructions read on demand during execution)
            let all_skills = engine.registry().get_all().into_iter().cloned().collect();
            let skill_tool = SkillTool::new(all_skills, config.skills_dir.clone());
            info!("Registered {} skills via run_skill tool", skill_tool.skill_count());
            tools.register(ToolBox::Skill(skill_tool));

            let llm: Box<dyn skill_llm::LLMClient> = match cli.llm_provider.as_str() {
                "minimax" => {
                    let url = cli
                        .minimax_url
                        .unwrap_or_else(|| "https://api.minimax.io".to_string());
                    let api_key = cli
                        .minimax_api_key
                        .unwrap_or_else(|| std::env::var("MINIMAX_API_KEY").unwrap_or_default());
                    if api_key.is_empty() {
                        anyhow::bail!("MiniMax API key not provided. Set MINIMAX_API_KEY env var or --minimax-api-key flag.");
                    }
                    info!("Using MiniMax provider: {}", url);
                    Box::new(MiniMaxClient::new(url, api_key, cli.llm_model.clone()))
                }
                "ollama" => {
                    info!("Using Ollama provider: {}", cli.ollama_url);
                    Box::new(OllamaClient::new(
                        cli.ollama_url.clone(),
                        cli.llm_model.clone(),
                    ))
                }
                "bedrock" => {
                    let auth = match cli.bedrock_auth.as_str() {
                        "static" => BedrockAuth::Static {
                            access_key_id: cli.bedrock_access_key_id.clone()
                                .ok_or_else(|| anyhow::anyhow!(
                                    "BEDROCK_ACCESS_KEY_ID required for --bedrock-auth static"
                                ))?,
                            secret_access_key: cli.bedrock_secret_access_key.clone()
                                .ok_or_else(|| anyhow::anyhow!(
                                    "BEDROCK_SECRET_ACCESS_KEY required for --bedrock-auth static"
                                ))?,
                        },
                        "sts-token" => BedrockAuth::StsToken {
                            access_key_id: cli.bedrock_access_key_id.clone()
                                .ok_or_else(|| anyhow::anyhow!(
                                    "BEDROCK_ACCESS_KEY_ID required for --bedrock-auth sts-token"
                                ))?,
                            secret_access_key: cli.bedrock_secret_access_key.clone()
                                .ok_or_else(|| anyhow::anyhow!(
                                    "BEDROCK_SECRET_ACCESS_KEY required for --bedrock-auth sts-token"
                                ))?,
                            session_token: cli.bedrock_session_token.clone()
                                .ok_or_else(|| anyhow::anyhow!(
                                    "BEDROCK_SESSION_TOKEN required for --bedrock-auth sts-token"
                                ))?,
                        },
                        "sts-role" => BedrockAuth::StsAssumeRole {
                            role_arn: cli.bedrock_role_arn.clone()
                                .ok_or_else(|| anyhow::anyhow!(
                                    "BEDROCK_ROLE_ARN required for --bedrock-auth sts-role"
                                ))?,
                            session_name: cli.bedrock_role_session_name.clone(),
                            external_id: cli.bedrock_external_id.clone(),
                            duration_secs: 3600,
                        },
                        "api-key" => BedrockAuth::BedrockApiKey {
                            api_key: cli.bedrock_api_key.clone()
                                .ok_or_else(|| anyhow::anyhow!(
                                    "BEDROCK_API_KEY required for --bedrock-auth api-key"
                                ))?,
                            region: cli.bedrock_region.clone(),
                        },
                        _ => BedrockAuth::DefaultChain,
                    };
                    info!(
                        "Using AWS Bedrock provider: auth={} region={} model={}",
                        cli.bedrock_auth, cli.bedrock_region, cli.llm_model
                    );
                    skill_llm::create_bedrock_client(
                        auth,
                        cli.bedrock_region.clone(),
                        cli.llm_model.clone(),
                    )
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to initialise Bedrock client: {}", e))?
                }
                other => {
                    anyhow::bail!(
                        "Unknown LLM provider: {}. Use 'minimax', 'ollama', or 'bedrock'.",
                        other
                    )
                }
            };

            // Convert mcp_registry to Arc for sharing
            let mcp_registry = std::sync::Arc::new(mcp_registry);

            // Add extra system prompt if provided
            let extra_prompt = extra_system_prompt.clone();

            if cli.streaming {
                let max_iters = if cli.max_iterations < 0 { usize::MAX } else { cli.max_iterations as usize };
                let mut agent = StreamingAgent::new(llm)
                    .with_tools(tools)
                    .with_mcp_registry(mcp_registry.clone())
                    .with_max_iterations(max_iters);

                if let Some(prompt) = extra_prompt {
                    agent = agent.with_extra_system_prompt(prompt);
                }

                let agent = agent;

                println!(
                    "\n{}",
                    "🤖 Skill Agent (Streaming Mode) initialized.".bold().cyan()
                );
                println!("{}\n", "Type 'quit' or 'exit' to end the session.".dimmed());

                if !task.is_empty() {
                    let result = agent.run(&task).await?;
                    println!("\n{}\n{}", "✨ Final Result:".bold().green(), result);
                }

                if interactive {
                    use std::io::{self, Write};
                    loop {
                        print!("\n{} ", ">".bold().cyan());
                        io::stdout().flush()?;

                        let mut input = String::new();
                        let bytes = io::stdin().read_line(&mut input)?;
                        if bytes == 0 {
                            break;
                        }

                        let input = input.trim();
                        if input == "quit" || input == "exit" {
                            break;
                        }

                        if !input.is_empty() {
                            let result = agent.run(input).await?;
                            println!("\n{}\n{}", "✨ Final Result:".bold().green(), result);
                        }
                    }
                }
            } else {
                let max_iters = if cli.max_iterations < 0 { usize::MAX } else { cli.max_iterations as usize };
                let mut agent_builder = Agent::new(llm)
                    .with_tools(tools)
                    .with_mcp_registry(mcp_registry.clone())
                    .with_max_iterations(max_iters);

                if let Some(prompt) = extra_system_prompt {
                    agent_builder = agent_builder.with_extra_system_prompt(prompt);
                }

                let agent = agent_builder;

                println!("\n{}", "🤖 Skill Agent initialized.".bold().cyan());
                println!("{}\n", "Type 'quit' or 'exit' to end the session.".dimmed());

                if !task.is_empty() {
                    let result = agent.run(&task).await?;
                    println!("\n{}\n{}", "✨ Final Result:".bold().green(), result);
                }

                if interactive {
                    use std::io::{self, Write};
                    loop {
                        print!("\n{} ", ">".bold().cyan());
                        io::stdout().flush()?;

                        let mut input = String::new();
                        let bytes = io::stdin().read_line(&mut input)?;
                        if bytes == 0 {
                            break;
                        }

                        let input = input.trim();
                        if input == "quit" || input == "exit" {
                            break;
                        }

                        if !input.is_empty() {
                            let result = agent.run(input).await?;
                            println!("\n{}\n{}", "✨ Final Result:".bold().green(), result);
                        }
                    }
                }
            }
        }

        Commands::SystemPrompt => {
            println!("{}", engine.get_system_prompt());
        }
    }

    Ok(())
}
