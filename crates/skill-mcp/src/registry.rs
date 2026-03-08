use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use tracing::{debug, info, warn};

use crate::config::McpConfig;
use crate::error::Result;

pub use crate::client::McpToolDefinition;

pub struct McpRegistry {
    clients: HashMap<String, tokio::sync::RwLock<crate::client::McpClient>>,
    tools: HashMap<String, crate::client::McpToolDefinition>,
    server_names: HashMap<String, String>,
    timeout: Duration,
}

impl McpRegistry {
    pub fn new(timeout: Duration) -> Self {
        Self {
            clients: HashMap::new(),
            tools: HashMap::new(),
            server_names: HashMap::new(),
            timeout,
        }
    }

    pub async fn load_from_config(&mut self, config: &McpConfig) -> Result<()> {
        info!("Loading MCP servers from config");

        for (name, server_config) in config.servers.iter() {
            let (command, args, env) = match server_config {
                crate::config::McpServerConfig::Stdio(s) => {
                    (s.command.clone(), s.args.clone(), s.env.clone())
                }
                crate::config::McpServerConfig::Sse(_) => {
                    warn!(
                        "SSE transport not supported yet, skipping server '{}'",
                        name
                    );
                    continue;
                }
            };

            match crate::client::McpClient::connect_stdio(
                name.clone(),
                command,
                args,
                env,
                self.timeout,
            )
            .await
            {
                Ok(mut client) => {
                    let client_tools = client.list_tools().await?;

                    for tool in client_tools {
                        let full_name = format!("{}_{}", name, tool.name);
                        self.server_names.insert(full_name.clone(), name.clone());
                        self.tools.insert(full_name, tool);
                    }

                    self.clients
                        .insert(name.clone(), tokio::sync::RwLock::new(client));
                }
                Err(e) => {
                    warn!("Failed to connect to MCP server '{}': {}", name, e);
                }
            }
        }

        info!(
            "MCP registry loaded {} tools from {} servers",
            self.tools.len(),
            self.clients.len()
        );

        Ok(())
    }

    pub async fn load_from_file(&mut self, path: &Path) -> Result<()> {
        let config = McpConfig::load(path).await?;
        self.load_from_config(&config).await
    }

    pub fn get(&self, name: &str) -> Option<&crate::client::McpToolDefinition> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<crate::client::McpToolDefinition> {
        self.tools.values().cloned().collect()
    }

    pub fn list_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| crate::error::McpError::ToolNotFound(name.to_string()))?;

        let server_name = self.server_names.get(name).ok_or_else(|| {
            crate::error::McpError::Connection(format!("Server not found for tool: {}", name))
        })?;

        let client = self.clients.get(server_name).ok_or_else(|| {
            crate::error::McpError::Connection(format!("Client not found: {}", server_name))
        })?;

        let mut client = client.write().await;

        client.call_tool(&tool.name, arguments).await
    }

    pub fn is_loaded(&self) -> bool {
        !self.clients.is_empty()
    }

    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    pub fn server_count(&self) -> usize {
        self.clients.len()
    }

    pub async fn shutdown(&mut self) {
        info!("Shutting down MCP registry");

        for (name, client) in self.clients.drain() {
            let mut c = client.write().await;
            c.disconnect().await;
            debug!("Disconnected MCP server: {}", name);
        }

        self.tools.clear();
        self.server_names.clear();
    }
}

impl Default for McpRegistry {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}
