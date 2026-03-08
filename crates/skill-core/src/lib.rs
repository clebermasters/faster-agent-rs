use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub id: String,
    pub name: String,
    pub description: String,
    pub triggers: Vec<String>,
    pub instructions: String,
    pub capabilities: Vec<String>,
    pub resources: Vec<SkillResource>,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillResource {
    pub name: String,
    pub path: PathBuf,
    pub resource_type: ResourceType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceType {
    Script,
    Reference,
    Asset,
    Config,
}

impl Skill {
    pub fn new(
        id: String,
        name: String,
        description: String,
        instructions: String,
        path: PathBuf,
    ) -> Self {
        Self {
            id,
            name,
            description,
            triggers: Vec::new(),
            instructions,
            capabilities: Vec::new(),
            resources: Vec::new(),
            path,
        }
    }

    pub fn with_triggers(mut self, triggers: Vec<String>) -> Self {
        self.triggers = triggers;
        self
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_resources(mut self, resources: Vec<SkillResource>) -> Self {
        self.resources = resources;
        self
    }

    pub fn search_text(&self) -> String {
        let mut parts = vec![
            self.name.clone(),
            self.description.clone(),
            self.instructions.clone(),
        ];
        parts.extend(self.triggers.clone());
        parts.extend(self.capabilities.clone());
        parts.join(" ")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredSkill {
    pub skill: Skill,
    pub score: f64,
    pub match_type: MatchType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MatchType {
    Semantic,
    Keyword,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    pub current_dir: PathBuf,
    pub files: Vec<PathBuf>,
    pub env_vars: std::collections::HashMap<String, String>,
    pub history: Vec<String>,
}

impl Default for AgentContext {
    fn default() -> Self {
        Self {
            current_dir: std::env::current_dir().unwrap_or_default(),
            files: Vec::new(),
            env_vars: std::env::vars().collect(),
            history: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillQuery {
    pub task: String,
    pub context: AgentContext,
    pub limit: usize,
    pub threshold: f64,
}

impl Default for SkillQuery {
    fn default() -> Self {
        Self {
            task: String::new(),
            context: AgentContext::default(),
            limit: 5,
            threshold: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillResult {
    pub skill_id: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub skills_dir: PathBuf,
    pub embedding_model: String,
    pub ollama_url: String,
    pub vector_dim: usize,
    pub db_path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            skills_dir: PathBuf::from("./skills"),
            embedding_model: "nomic-embed-text".to_string(),
            ollama_url: "http://localhost:11434".to_string(),
            vector_dim: 768,
            db_path: PathBuf::from("./skill-agent.db"),
        }
    }
}
