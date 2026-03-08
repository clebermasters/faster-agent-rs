use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tracing::{debug, info, warn};

use crate::error::{McpError, Result};

macro_rules! json {
    ($($json:tt)*) => { serde_json::json!($($json)*) };
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct JsonRpcResponse {
    #[serde(default)]
    jsonrpc: String,
    #[serde(default)]
    id: u64,
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Tool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "inputSchema", default)]
    input_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ListToolsResult {
    tools: Vec<Tool>,
}

pub struct McpClient {
    name: String,
    command: String,
    args: Vec<String>,
    env: HashMap<String, String>,
    process: Option<Child>,
    stdin: Option<tokio::sync::mpsc::Sender<String>>,
    reader: tokio::sync::mpsc::Receiver<String>,
    request_id: u64,
}

impl McpClient {
    pub async fn connect_stdio(
        name: String,
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
        timeout: Duration,
    ) -> Result<Self> {
        info!("Starting MCP server '{}': {} {:?}", name, command, args);

        let mut child = Command::new(&command)
            .args(&args)
            .envs(&env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| McpError::Connection(format!("Failed to spawn MCP server: {}", e)))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Connection("Failed to capture stdout".to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Connection("Failed to capture stdin".to_string()))?;

        let (response_tx, response_rx) = tokio::sync::mpsc::channel::<String>(100);
        let (request_tx, mut request_rx) = tokio::sync::mpsc::channel::<String>(100);

        // Spawn a task to read from stdout
        let mut reader = BufReader::new(stdout).lines();
        let response_tx_clone = response_tx.clone();

        tokio::spawn(async move {
            while let Ok(Some(line)) = reader.next_line().await {
                debug!("MCP raw response: {}", line);
                if response_tx_clone.send(line).await.is_err() {
                    break;
                }
            }
        });

        // Spawn a task to write to stdin
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let mut stdin = stdin;
            while let Some(request) = request_rx.recv().await {
                debug!("MCP request: {}", request);
                if stdin
                    .write_all(format!("{}\n", request).as_bytes())
                    .await
                    .is_err()
                {
                    break;
                }
                if stdin.flush().await.is_err() {
                    break;
                }
            }
        });

        // Wait for the server to be ready
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Send initialize request
        let mut client = Self {
            name: name.clone(),
            command,
            args,
            env,
            process: Some(child),
            stdin: Some(request_tx),
            reader: response_rx,
            request_id: 0,
        };

        // Initialize
        let _ = client
            .call_method(
                "initialize",
                Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "skill-agent",
                        "version": "0.1.0"
                    }
                })),
            )
            .await;

        // Send initialized notification
        let _ = client.call_method("notifications/initialized", None).await;

        // List tools
        let tools = client.list_tools().await?;

        info!(
            "Connected to MCP server '{}' with {} tools",
            name,
            tools.len()
        );

        Ok(client)
    }

    pub async fn list_tools(&mut self) -> Result<Vec<McpToolDefinition>> {
        let response = self.call_method("tools/list", None).await?;

        let result = response
            .result
            .ok_or_else(|| McpError::Server("No result in response".to_string()))?;

        let tools_result: ListToolsResult = serde_json::from_value(result)
            .map_err(|e| McpError::Parse(format!("Failed to parse tools list: {}", e)))?;

        Ok(tools_result
            .tools
            .into_iter()
            .map(|t| McpToolDefinition {
                name: t.name,
                description: t.description.unwrap_or_default(),
                input_schema: t.input_schema.unwrap_or(json!({"type": "object"})),
            })
            .collect())
    }

    pub async fn call_tool(&mut self, name: &str, arguments: serde_json::Value) -> Result<String> {
        let response = self
            .call_method(
                "tools/call",
                Some(json!({
                    "name": name,
                    "arguments": arguments
                })),
            )
            .await?;

        let result = response
            .result
            .ok_or_else(|| McpError::Server("No result in response".to_string()))?;

        // Extract content from the result
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| result.to_string());

        Ok(content)
    }

    async fn call_method(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse> {
        self.request_id += 1;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: self.request_id,
            method: method.to_string(),
            params,
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| McpError::Parse(format!("Failed to serialize request: {}", e)))?;

        // Send request via stdin
        if let Some(ref stdin) = self.stdin {
            stdin
                .send(request_json)
                .await
                .map_err(|e| McpError::Connection(format!("Failed to send request: {}", e)))?;
        } else {
            return Err(McpError::Connection("Stdin not available".to_string()));
        }

        // Wait for response with timeout
        let response_str =
            match tokio::time::timeout(Duration::from_secs(30), self.reader.recv()).await {
                Ok(Some(response)) => response,
                Ok(None) => {
                    return Err(McpError::Connection(
                        "MCP server closed connection".to_string(),
                    ))
                }
                Err(_) => return Err(McpError::Timeout("MCP server response timeout".to_string())),
            };

        let response: JsonRpcResponse = serde_json::from_str(&response_str)
            .map_err(|e| McpError::Parse(format!("Failed to parse response: {}", e)))?;

        if let Some(error) = response.error {
            return Err(McpError::Server(format!(
                "JSON-RPC error {}: {}",
                error.code, error.message
            )));
        }

        Ok(response)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn disconnect(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill().await;
            info!("Disconnected MCP server '{}'", self.name);
        }
    }
}

pub struct McpRegistry {
    clients: HashMap<String, tokio::sync::RwLock<McpClient>>,
    tools: HashMap<String, McpToolDefinition>,
    server_names: HashMap<String, String>, // tool_name -> server_name
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

    pub async fn load_from_config(&mut self, config: &crate::config::McpConfig) -> Result<()> {
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

            match McpClient::connect_stdio(name.clone(), command, args, env, self.timeout).await {
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
        let config = crate::config::McpConfig::load(path).await?;
        self.load_from_config(&config).await
    }

    pub fn get(&self, name: &str) -> Option<&McpToolDefinition> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<McpToolDefinition> {
        self.tools.values().cloned().collect()
    }

    pub fn list_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| McpError::ToolNotFound(name.to_string()))?;

        let server_name = self
            .server_names
            .get(name)
            .ok_or_else(|| McpError::Connection(format!("Server not found for tool: {}", name)))?;

        let client = self
            .clients
            .get(server_name)
            .ok_or_else(|| McpError::Connection(format!("Client not found: {}", server_name)))?;

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
