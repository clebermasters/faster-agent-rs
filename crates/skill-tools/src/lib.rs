use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub mod bash;
pub mod read;
pub mod skill;
pub mod write;

pub use bash::BashTool;
pub use read::ReadTool;
pub use skill::SkillTool;
pub use write::WriteTool;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Tool not found: {0}")]
    NotFound(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Clone)]
pub enum ToolBox {
    Bash(BashTool),
    Read(ReadTool),
    Write(WriteTool),
    Skill(SkillTool),
}

impl ToolBox {
    pub fn definition(&self) -> ToolDefinition {
        match self {
            ToolBox::Bash(t) => t.definition(),
            ToolBox::Read(t) => t.definition(),
            ToolBox::Write(t) => t.definition(),
            ToolBox::Skill(t) => t.definition(),
        }
    }

    pub async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        match self {
            ToolBox::Bash(t) => t.execute(params).await,
            ToolBox::Read(t) => t.execute(params).await,
            ToolBox::Write(t) => t.execute(params).await,
            ToolBox::Skill(t) => t.execute(params).await,
        }
    }
}

pub struct ToolRegistry {
    tools: HashMap<String, ToolBox>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: ToolBox) {
        let def = tool.definition();
        self.tools.insert(def.name.clone(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&ToolBox> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
