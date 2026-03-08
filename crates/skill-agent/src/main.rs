use clap::{Parser, Subcommand};
use skill_core::{Config, SkillQuery};
use skill_discovery::SkillDiscoveryEngine;
use skill_embeddings::EmbeddingService;
use skill_executor::{ExecutionContext, SkillExecutor};
use skill_llm::{Agent, MiniMaxClient, OllamaClient, StreamingAgent};
use skill_registry::SkillRegistry;
use skill_tools::{BashTool, ReadTool, SkillTool, ToolBox, ToolRegistry, WriteTool};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "skill-agent")]
#[command(about = "Autonomous skill discovery agent", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, default_value = "./skills")]
    skills_dir: PathBuf,

    #[arg(long, default_value = "ollama")]
    provider: String,

    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    #[arg(long, default_value = "nomic-embed-text")]
    ollama_model: String,

    #[arg(long, default_value = "qwen3-coder-next:cloud")]
    llm_model: String,

    #[arg(long, default_value = "ollama")]
    llm_provider: String,

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

    /// Start interactive agent mode
    Agent {
        /// Initial task
        #[arg(default_value = "")]
        task: String,
    },

    /// Show system prompt for skill discovery
    SystemPrompt,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let subscriber = FmtSubscriber::builder()
        .with_max_level(if cli.verbose {
            Level::DEBUG
        } else {
            Level::INFO
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

        Commands::Discover { task, limit, threshold } => {
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
                println!("{}. {} (score: {:.2})", i + 1, discovered.skill.name, discovered.score);
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

            let result = executor.execute_skill(&skill, input.as_deref(), &context).await?;

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

        Commands::Agent { task } => {
            engine.registry_mut().load().await?;
            engine.index_all().await?;

            let mut tools = ToolRegistry::new();
            tools.register(ToolBox::Bash(BashTool::new()));
            tools.register(ToolBox::Read(ReadTool::new(config.skills_dir.clone().to_string_lossy().to_string())));
            tools.register(ToolBox::Write(WriteTool::new(config.skills_dir.clone().to_string_lossy().to_string())));

            for skill in engine.registry().get_all() {
                tools.register(ToolBox::Skill(SkillTool::new(skill.clone(), config.skills_dir.clone())));
            }

            let llm: Box<dyn skill_llm::LLMClient> = match cli.llm_provider.as_str() {
                "minimax" => {
                    let url = cli.minimax_url.unwrap_or_else(|| "https://api.minimax.io".to_string());
                    let api_key = cli.minimax_api_key.unwrap_or_else(|| std::env::var("MINIMAX_API_KEY").unwrap_or_default());
                    if api_key.is_empty() {
                        anyhow::bail!("MiniMax API key not provided. Set MINIMAX_API_KEY env var or --minimax-api-key flag.");
                    }
                    info!("Using MiniMax provider: {}", url);
                    Box::new(MiniMaxClient::new(url, api_key, cli.llm_model.clone()))
                }
                "ollama" => {
                    info!("Using Ollama provider: {}", cli.ollama_url);
                    Box::new(OllamaClient::new(cli.ollama_url.clone(), cli.llm_model.clone()))
                }
                other => {
                    anyhow::bail!("Unknown LLM provider: {}. Use 'ollama' or 'minimax'.", other)
                }
            };
            
            if cli.streaming {
                let agent = StreamingAgent::new(llm).with_tools(tools).with_max_iterations(10);

                println!("=== Skill Agent (Streaming Mode) ===");
                println!("Type 'quit' to exit\n");

                if !task.is_empty() {
                    let result = agent.run(&task).await?;
                    println!("\n=== Final Result ===\n{}", result);
                }

                use std::io::{self, Write};
                loop {
                    print!("\n> ");
                    io::stdout().flush()?;
                    
                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;
                    
                    let input = input.trim();
                    if input == "quit" || input == "exit" {
                        break;
                    }

                    if !input.is_empty() {
                        let result = agent.run(input).await?;
                        println!("\n=== Final Result ===\n{}", result);
                    }
                }
            } else {
                let agent = Agent::new(llm).with_tools(tools).with_max_iterations(10);

                println!("=== Skill Agent ===");
                println!("Type 'quit' to exit\n");

                if !task.is_empty() {
                    let result = agent.run(&task).await?;
                    println!("\n=== Result ===\n{}", result);
                }

                use std::io::{self, Write};
                loop {
                    print!("\n> ");
                    io::stdout().flush()?;
                    
                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;
                    
                    let input = input.trim();
                    if input == "quit" || input == "exit" {
                        break;
                    }

                    if !input.is_empty() {
                        let result = agent.run(input).await?;
                        println!("\n=== Result ===\n{}", result);
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
