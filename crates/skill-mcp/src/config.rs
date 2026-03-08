use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

use crate::error::{McpError, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct McpConfig {
    #[serde(rename = "mcpServers", default)]
    pub servers: HashMap<String, McpServerConfig>,

    #[serde(default)]
    pub imports: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum McpServerConfig {
    Stdio(StdioServerConfig),
    Sse(SseServerConfig),
}

#[derive(Debug, Clone, Deserialize)]
pub struct StdioServerConfig {
    pub command: String,

    #[serde(default)]
    pub args: Vec<String>,

    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SseServerConfig {
    pub url: String,

    #[serde(default)]
    pub headers: HashMap<String, String>,
}

impl McpConfig {
    pub async fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            warn!("MCP config file not found: {:?}", path);
            return Ok(Self::default());
        }

        info!("Loading MCP config from: {:?}", path);

        let content = tokio::fs::read_to_string(path).await?;

        let config: McpConfig = serde_json::from_str(&content)
            .map_err(|e| McpError::Parse(format!("Failed to parse mcp.json: {}", e)))?;

        debug!("Loaded {} MCP servers", config.servers.len());

        Ok(config)
    }

    pub fn server_configs(&self) -> Vec<(String, McpServerConfig)> {
        self.servers
            .iter()
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect()
    }
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            servers: HashMap::new(),
            imports: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimax_config() {
        let json = r#"{
            "mcpServers": {
                "MiniMax": {
                    "command": "uvx",
                    "args": ["minimax-coding-plan-mcp"],
                    "env": {
                        "MINIMAX_API_KEY": "sk-cp-xxx"
                    }
                }
            }
        }"#;

        let config: McpConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.servers.len(), 1);

        let server = config.servers.get("MiniMax").unwrap();
        match server {
            McpServerConfig::Stdio(stdio) => {
                assert_eq!(stdio.command, "uvx");
                assert_eq!(stdio.args, vec!["minimax-coding-plan-mcp"]);
                assert!(stdio.env.contains_key("MINIMAX_API_KEY"));
            }
            _ => panic!("Expected Stdio config"),
        }
    }
}
