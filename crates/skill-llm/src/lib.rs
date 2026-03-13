use anyhow::Result;
use colored::*;
use futures::Stream;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use skill_tools::{ToolDefinition, ToolRegistry};
use std::collections::HashMap;
use std::io::Write;
use std::pin::Pin;
use tracing::{debug, error, info, warn};

#[cfg(feature = "bedrock")]
pub mod bedrock;
#[cfg(feature = "bedrock")]
pub use bedrock::{BedrockAuth, create_bedrock_client};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub done: bool,
    pub done_reason: Option<String>,
}

pub trait LLMClient: Send + Sync {
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>>;
    fn chat_streaming(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send + '_>>;
}

pub struct MiniMaxClient {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl MiniMaxClient {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
            model,
        }
    }
}

impl LLMClient for MiniMaxClient {
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        Box::pin(async move {
            // Convert system messages to user messages (MiniMax doesn't support system role)
            let messages: Vec<serde_json::Value> = messages
                .into_iter()
                .map(|m| {
                    if m.role == "system" {
                        // Convert system to user
                        json!({
                            "role": "user",
                            "content": m.content
                        })
                    } else if m.role == "tool" {
                        // MiniMax doesn't properly support tool_call_id - send as regular message
                        json!({
                            "role": "user",
                            "content": format!("[Tool Result for {}]: {}", m.tool_call_id.unwrap_or_default(), m.content)
                        })
                    } else {
                        json!({
                            "role": m.role,
                            "content": m.content
                        })
                    }
                })
                .collect();

            let mut body = json!({
                "model": model,
                "messages": messages,
                "stream": false,
            });

            if let Some(tools) = tools {
                let openai_tools: Vec<serde_json::Value> = tools
                    .into_iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters
                            }
                        })
                    })
                    .collect();
                body["tools"] = serde_json::Value::Array(openai_tools);
            }

            debug!("MiniMax request body: {}", body);

            debug!("Sending chat request to MiniMax: {}", base_url);

            let response = client
                .post(format!("{}/v1/chat/completions", base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                anyhow::bail!("MiniMax API error: {} - {}", status, error_text);
            }

            let chat_resp: serde_json::Value = response.json().await?;

            let message = chat_resp["choices"][0]["message"].clone();
            let role = message["role"].as_str().unwrap_or("assistant").to_string();
            let content = message["content"].as_str().unwrap_or("").to_string();

            let tool_calls = if message.get("tool_calls").is_some() {
                let calls: Vec<ToolCall> = message["tool_calls"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|c| {
                        let func = c.get("function")?;
                        Some(ToolCall {
                            id: c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
                            name: func["name"].as_str()?.to_string(),
                            arguments: func["arguments"].clone(),
                        })
                    })
                    .collect();
                if calls.is_empty() {
                    None
                } else {
                    Some(calls)
                }
            } else {
                None
            };

            Ok(ChatResponse {
                message: Message {
                    role,
                    content,
                    tool_call_id: None,
                },
                tool_calls,
                done: true,
            })
        })
    }

    fn chat_streaming(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send + '_>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        Box::pin(async_stream::try_stream! {
            let messages: Vec<serde_json::Value> = messages
                .into_iter()
                .map(|m| {
                    if m.role == "system" {
                        json!({
                            "role": "user",
                            "content": m.content
                        })
                    } else if m.role == "tool" {
                        json!({
                            "role": "user",
                            "content": format!("[Tool Result for {}]: {}", m.tool_call_id.unwrap_or_default(), m.content)
                        })
                    } else {
                        json!({
                            "role": m.role,
                            "content": m.content
                        })
                    }
                })
                .collect();

            let mut body = json!({
                "model": model,
                "messages": messages,
                "stream": true,
            });

            if let Some(tools) = tools {
                let openai_tools: Vec<serde_json::Value> = tools
                    .into_iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters
                            }
                        })
                    })
                    .collect();
                body["tools"] = serde_json::Value::Array(openai_tools);
            }

            debug!("[MINIMAX] Sending streaming chat request");

            let response = client
                .post(format!("{}/v1/chat/completions", base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    error!("[MINIMAX] Request error: {}", e);
                    anyhow::anyhow!("Request error: {}", e)
                })?;

            // Check status - get status code before consuming response
            let status = response.status();
            if !status.is_success() {
                // Can't call text() here because it would consume response
                // Just use status code in error
                error!("[MINIMAX] API error: {}", status);
                Err(anyhow::anyhow!("MiniMax API error: {}", status))?;
            }

            info!("[MINIMAX] Streaming response started");
            let mut stream = response.bytes_stream();
            let mut bytes_received = 0;

            while let Some(chunk_result) = stream.next().await {
                    let chunk_bytes = match chunk_result {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            error!("[MINIMAX] Stream read error: {}", e);
                            Err(anyhow::anyhow!("Stream error: {}", e))?;
                            unreachable!()
                        }
                    };

                bytes_received += chunk_bytes.len();
                debug!("[MINIMAX] Received {} bytes (total: {})", chunk_bytes.len(), bytes_received);

                if chunk_bytes.is_empty() {
                    continue;
                }

                let chunk_str = String::from_utf8_lossy(&chunk_bytes);

                for line in chunk_str.lines() {
                    let line = line.trim();
                    if !line.starts_with("data:") {
                        continue;
                    }

                    let data = line[5..].trim();
                    if data.is_empty() {
                        continue;
                    }

                    if data == "[DONE]" {
                        info!("[MINIMAX] Received [DONE] signal");
                        yield ChatChunk {
                            content: String::new(),
                            tool_calls: None,
                            done: true,
                            done_reason: Some("stop".to_string()),
                        };
                        return;
                    }

                        match serde_json::from_str::<serde_json::Value>(data) {
                            Ok(chat_resp) => {
                                let delta = chat_resp.get("choices")
                                .and_then(|c| c.as_array())
                                .and_then(|c| c.first())
                                .and_then(|c| c.get("delta"));

                            let content = delta
                                .and_then(|d| d.get("content"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();

                            let tool_calls = delta.and_then(|d| d.get("tool_calls")).and_then(|tc| {
                                // Accumulate tool_calls across chunks
                                // Note: MiniMax may send partial tool_calls in subsequent chunks (without name/id)
                                let calls: Vec<ToolCall> = tc
                                    .as_array()
                                    .unwrap_or(&vec![])
                                    .iter()
                                    .filter_map(|c| {
                                        let func = c.get("function")?;
                                        let args = func.get("arguments")?;
                                        // Arguments may be fragmented across chunks - need to get the string value
                                        let args_str = args.as_str().unwrap_or("");
                                        // For partial tool_calls (continuations), name/id may be missing
                                        let name = func["name"].as_str().map(|s| s.to_string()).unwrap_or_default();
                                        let id = c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                        let arguments = serde_json::Value::String(args_str.to_string());
                                        Some(ToolCall {
                                            id,
                                            name,
                                            arguments,
                                        })
                                    })
                                    .collect();
                                if calls.is_empty() {
                                    None
                                } else {
                                    Some(calls)
                                }
                            });

                            let finish_reason = chat_resp.get("choices")
                                .and_then(|c| c.as_array())
                                .and_then(|c| c.first())
                                .and_then(|c| c.get("finish_reason"))
                                .and_then(|f| f.as_str());

                            let done = finish_reason.is_some();
                            let done_reason = finish_reason.map(|s| s.to_string());

                            info!("[MINIMAX] Yielding chunk: content_len={}, done={}", content.len(), done);
                            yield ChatChunk {
                                content,
                                tool_calls,
                                done,
                                done_reason,
                            };
                        }
                        Err(_) => continue,
                    }
                }
            }

            info!("[MINIMAX] Stream ended normally, total bytes: {}", bytes_received);
        })
    }
}

pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
}

impl OllamaClient {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model,
        }
    }

    pub async fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<ChatResponse> {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });

        if let Some(tools) = tools {
            let ollama_tools: Vec<serde_json::Value> = tools
                .into_iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(ollama_tools);
        }

        debug!("Sending chat request to Ollama");

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama API error: {} - {}", status, error_text);
        }

        let chat_resp: serde_json::Value = response.json().await?;

        let message = chat_resp["message"].clone();
        let role = message["role"].as_str().unwrap_or("assistant").to_string();
        let content = message["content"].as_str().unwrap_or("").to_string();

        let tool_calls = if message.get("tool_calls").is_some() {
            let calls: Vec<ToolCall> = message["tool_calls"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|c| {
                    let func = c.get("function")?;
                    Some(ToolCall {
                        id: c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        name: func["name"].as_str()?.to_string(),
                        arguments: func["arguments"].clone(),
                    })
                })
                .collect();
            if calls.is_empty() {
                None
            } else {
                Some(calls)
            }
        } else {
            None
        };

        Ok(ChatResponse {
            message: Message {
                role,
                content,
                tool_call_id: None,
            },
            tool_calls,
            done: chat_resp["done"].as_bool().unwrap_or(true),
        })
    }

    pub async fn generate(&self, prompt: String) -> Result<String> {
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
        });

        let response = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama API error: {} - {}", status, error_text);
        }

        let resp: serde_json::Value = response.json().await?;
        Ok(resp["response"].as_str().unwrap_or("").to_string())
    }

    pub fn chat_streaming(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send + '_>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let model = self.model.clone();

        Box::pin(async_stream::try_stream! {
            let mut body = json!({
                "model": model,
                "messages": messages,
                "stream": true,
            });

            if let Some(tools) = tools {
                let ollama_tools: Vec<serde_json::Value> = tools
                    .into_iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters
                            }
                        })
                    })
                    .collect();
                body["tools"] = serde_json::Value::Array(ollama_tools);
            }

            debug!("[OLLAMA] Sending streaming chat request");

            let response = client
                .post(format!("{}/api/chat", base_url))
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    error!("[OLLAMA] Request error: {}", e);
                    anyhow::anyhow!("Request error: {}", e)
                })?;

            // Check status - get status code before consuming response
            let status = response.status();
            if !status.is_success() {
                error!("[OLLAMA] API error: {}", status);
                Err(anyhow::anyhow!("Ollama API error: {}", status))?;
            }

            info!("[OLLAMA] Streaming response started");
            let mut stream = response.bytes_stream();
            let mut bytes_received = 0;

            while let Some(chunk_result) = stream.next().await {
                    let chunk_bytes = match chunk_result {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            error!("[OLLAMA] Stream read error: {}", e);
                            Err(anyhow::anyhow!("Stream error: {}", e))?;
                            unreachable!()
                        }
                    };

                bytes_received += chunk_bytes.len();
                debug!("[OLLAMA] Received {} bytes (total: {})", chunk_bytes.len(), bytes_received);

                if chunk_bytes.is_empty() {
                    continue;
                }

                let chunk_str = String::from_utf8_lossy(&chunk_bytes);

                for line in chunk_str.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }

                    match serde_json::from_str::<serde_json::Value>(line) {
                        Ok(chat_resp) => {
                            let message = chat_resp.get("message");
                            let content = message
                                .and_then(|m| m.get("content"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();

                            let tool_calls = message.and_then(|m| m.get("tool_calls")).and_then(|tc| {
                                let calls: Vec<ToolCall> = tc
                                    .as_array()
                                    .unwrap_or(&vec![])
                                    .iter()
                                    .filter_map(|c| {
                                        let func = c.get("function")?;
                                        let args = func.get("arguments")?;
                                        // Some providers return arguments as a JSON string, need to parse it
                                        let arguments = if let Some(args_str) = args.as_str() {
                                            serde_json::from_str(args_str).unwrap_or(args.clone())
                                        } else {
                                            args.clone()
                                        };
                                        Some(ToolCall {
                                            id: c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
                                            name: func["name"].as_str()?.to_string(),
                                            arguments,
                                        })
                                    })
                                    .collect();
                                if calls.is_empty() {
                                    None
                                } else {
                                    Some(calls)
                                }
                            });

                            let done = chat_resp["done"].as_bool().unwrap_or(false);
                            let done_reason = chat_resp["done_reason"].as_str().map(|s| s.to_string());

                            info!("[OLLAMA] Yielding chunk: content_len={}, done={}", content.len(), done);
                            yield ChatChunk {
                                content,
                                tool_calls,
                                done,
                                done_reason,
                            };
                        }
                        Err(_) => continue,
                    }
                }
            }

            info!("[OLLAMA] Stream ended normally, total bytes: {}", bytes_received);
        })
    }
}

impl LLMClient for OllamaClient {
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let model = self.model.clone();

        Box::pin(async move {
            let mut body = json!({
                "model": model,
                "messages": messages,
                "stream": false,
            });

            if let Some(tools) = tools {
                let ollama_tools: Vec<serde_json::Value> = tools
                    .into_iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters
                            }
                        })
                    })
                    .collect();
                body["tools"] = serde_json::Value::Array(ollama_tools);
            }

            debug!("Sending chat request to Ollama");

            let response = client
                .post(format!("{}/api/chat", base_url))
                .json(&body)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                anyhow::bail!("Ollama API error: {} - {}", status, error_text);
            }

            let chat_resp: serde_json::Value = response.json().await?;

            let message = chat_resp["message"].clone();
            let role = message["role"].as_str().unwrap_or("assistant").to_string();
            let content = message["content"].as_str().unwrap_or("").to_string();

            let tool_calls = if message.get("tool_calls").is_some() {
                let calls: Vec<ToolCall> = message["tool_calls"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|c| {
                        let func = c.get("function")?;
                        Some(ToolCall {
                            id: c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
                            name: func["name"].as_str()?.to_string(),
                            arguments: func["arguments"].clone(),
                        })
                    })
                    .collect();
                if calls.is_empty() {
                    None
                } else {
                    Some(calls)
                }
            } else {
                None
            };

            Ok(ChatResponse {
                message: Message {
                    role,
                    content,
                    tool_call_id: None,
                },
                tool_calls,
                done: chat_resp["done"].as_bool().unwrap_or(true),
            })
        })
    }

    fn chat_streaming(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send + '_>> {
        self.chat_streaming(messages, tools)
    }
}

fn create_spinner(msg: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ")
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    spinner.set_message(msg.to_string());
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));
    spinner
}

pub struct Agent {
    llm: Box<dyn LLMClient>,
    tool_registry: ToolRegistry,
    mcp_registry: Option<std::sync::Arc<skill_mcp::McpRegistry>>,
    max_iterations: usize,
    tool_call_history: HashMap<String, usize>,
    extra_system_prompt: Option<String>,
}

impl Agent {
    pub fn new(llm: Box<dyn LLMClient>) -> Self {
        Self {
            llm,
            tool_registry: ToolRegistry::new(),
            mcp_registry: None,
            max_iterations: 10,
            tool_call_history: HashMap::new(),
            extra_system_prompt: None,
        }
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tool_registry = tools;
        self
    }

    pub fn with_extra_system_prompt(mut self, prompt: String) -> Self {
        self.extra_system_prompt = Some(prompt);
        self
    }

    pub fn with_mcp_registry(mut self, registry: std::sync::Arc<skill_mcp::McpRegistry>) -> Self {
        self.mcp_registry = Some(registry);
        self
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    fn build_system_prompt(&self) -> String {
        let mut tools_list = Vec::new();

        // Add regular tools
        for tool in self.tool_registry.list() {
            tools_list.push(format!("- {}: {}", tool.name, tool.description));
        }

        // Add MCP tools
        if let Some(ref mcp) = self.mcp_registry {
            for tool in mcp.list() {
                tools_list.push(format!("- {}: {}", tool.name, tool.description));
            }
        }

        let tools_str = if tools_list.is_empty() {
            "No tools available".to_string()
        } else {
            tools_list.join("\n")
        };

        // Build skill catalog section (metadata only — no full instructions)
        let skill_catalog_section = match self.tool_registry.skill_catalog() {
            Some(catalog) => format!(
                "\n\nSKILL CATALOG:\n\
                 The following skills are available via the 'run_skill' tool.\n\
                 To use a skill, call run_skill with the skill_id and your input.\n\
                 {}", catalog
            ),
            None => String::new(),
        };

        let mut prompt = format!(
            r#"You are an autonomous agent that MUST use tools to complete tasks.

AVAILABLE TOOLS:
{}{}

STRICT RULES:
1. You MUST use tools to gather information or execute actions when needed.
2. ALWAYS format your final responses to the user in clean Markdown.
3. NEVER create or write files using the 'write' tool UNLESS the user explicitly asks you to save, write, or create a file.
4. If the user asks a question, use tools to find the answer and print the summary directly.

When you finish a task, provide a clear, formatted summary of what was done."#,
            tools_str, skill_catalog_section
        );

        // Add extra system prompt (from AGENTS.md or CLI)
        if let Some(ref extra) = self.extra_system_prompt {
            prompt.push_str("\n\n");
            prompt.push_str(extra);
        }

        prompt
    }

    pub async fn run(&self, task: &str) -> Result<String> {
        let system_prompt = self.build_system_prompt();

        let mut messages = vec![
            Message {
                role: "system".to_string(),
                content: system_prompt.to_string(),
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: task.to_string(),
                tool_call_id: None,
            },
        ];

        let mut tool_defs = self.tool_registry.list();

        // Add MCP tools to the list
        if let Some(ref mcp) = self.mcp_registry {
            let mcp_tools: Vec<ToolDefinition> = mcp
                .list()
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.input_schema,
                })
                .collect();
            info!("Adding {} MCP tools to available tools", mcp_tools.len());
            tool_defs.extend(mcp_tools);
            info!("MCP tools: {:?}", mcp.list_names());
        }

        let mut tool_history: HashMap<String, usize> = HashMap::new();

        info!("Starting agent loop for task: {}", task);
        info!("Available tools: {:?}", self.tool_registry.names());
        debug!("Total messages at start: {}", messages.len());

        for iteration in 0..self.max_iterations {
            info!(
                "=== Iteration {}/{} ===",
                iteration + 1,
                self.max_iterations
            );
            debug!("Messages before LLM call: {}", messages.len());

            // Log last few messages for debugging
            if iteration > 0 {
                debug!(
                    "Last 3 messages roles: {:?}",
                    messages
                        .iter()
                        .rev()
                        .take(3)
                        .map(|m| &m.role)
                        .collect::<Vec<_>>()
                );
            }

            let spinner = create_spinner("🤔 Thinking...");
            let response_result = self
                .llm
                .chat(messages.clone(), Some(tool_defs.clone()))
                .await;
            spinner.finish_and_clear();
            
            let response = response_result?;

            debug!(
                "LLM response - has_tool_calls: {}, content_len: {}",
                response.tool_calls.is_some(),
                response.message.content.len()
            );

            // Debug: log raw response if empty
            if response.message.content.is_empty() && response.tool_calls.is_none() {
                warn!(
                    "LLM returned empty response (no content, no tool_calls)! Message: {:?}",
                    response.message
                );
            }

            if let Some(tool_calls) = response.tool_calls {
                // Execute tool calls ONE AT A TIME and ask LLM for next step after each
                // This enables chaining - LLM sees result before deciding next action
                for (i, call) in tool_calls.iter().enumerate() {
                    let formatted_args = serde_json::to_string_pretty(&call.arguments).unwrap_or_else(|_| format!("{:?}", call.arguments));
                    println!("\n{} {}", "⚙️  Action:".bold().yellow(), call.name.bold().white());
                    println!("{}", formatted_args.dimmed());

                    let call_key = format!("{}:{}", call.name, call.arguments);
                    let count = tool_history.entry(call_key.clone()).or_insert(0);
                    *count += 1;
                    debug!("Tool '{}' call count: {}", call.name, count);

                    if *count > 2 {
                        println!("{} {} {}", "⚠️  Warning:".bold().yellow(), call.name.bold(), "called multiple times. Forcing a different approach.".dimmed());
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: format!(
                                "__tool_use__:{}",
                                serde_json::to_string(&json!({
                                    "id": call.id,
                                    "name": call.name,
                                    "input": call.arguments
                                })).unwrap_or_default()
                            ),
                            tool_call_id: None,
                        });
                        messages.push(Message {
                            role: "tool".to_string(),
                            content: format!(
                                "ERROR: Tool '{}' has failed multiple times ({} attempts). \
                                Try a DIFFERENT approach.",
                                call.name, count
                            ),
                            tool_call_id: call.id.clone(),
                        });
                        continue;
                    }

                    // Check if it's a regular tool
                    if let Some(tool) = self.tool_registry.get(&call.name) {
                        debug!("Executing tool: {}", call.name);

                        // Handle arguments - MiniMax sends them as a string, need to parse
                        let args = if let Some(s) = call.arguments.as_str() {
                            serde_json::from_str(s).unwrap_or(call.arguments.clone())
                        } else {
                            call.arguments.clone()
                        };

                        let tool_spinner = create_spinner(&format!("Executing {}...", call.name).yellow().to_string());
                        let result = tool.execute(args).await;
                        tool_spinner.finish_and_clear();
                        let result = result?;

                        if result.success {
                            println!("{} {} returned {} characters.", "✅ Success:".bold().green(), call.name.bold(), result.output.len());
                        } else {
                            if let Some(err) = &result.error {
                                println!("{} {} failed: {}", "❌ Error:".bold().red(), call.name.bold(), err);
                            } else {
                                println!("{} {} failed.", "❌ Error:".bold().red(), call.name.bold());
                            }
                        }

                        // If write tool succeeded, task is complete
                        if call.name == "write" && result.success {
                            debug!("Write tool succeeded - task complete!");
                            return Ok(format!(
                                "Task completed successfully! Saved content to: {}",
                                call.arguments
                                    .get("path")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("file")
                            ));
                        }

                        let result_message = Self::format_tool_result(&call.name, &result);
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: format!(
                                "__tool_use__:{}",
                                serde_json::to_string(&json!({
                                    "id": call.id,
                                    "name": call.name,
                                    "input": call.arguments
                                })).unwrap_or_default()
                            ),
                            tool_call_id: None,
                        });
                        messages.push(Message {
                            role: "tool".to_string(),
                            content: result_message,
                            tool_call_id: call.id.clone(),
                        });

                        // Ask LLM what to do next
                        messages.push(Message {
                            role: "system".to_string(),
                            content: format!(
                                "You just called '{}' and got a result. \n\
                                Your task is: {}\n\
                                What is the NEXT step? Do you need to call another tool? \
                                If you got content from a skill, you MUST save it with 'write'.",
                                call.name, task
                            ),
                            tool_call_id: None,
                        });

                        debug!("Asking LLM what to do next after tool: {}", call.name);
                        break; // Exit the for loop to ask LLM for next step
                    }
                    // Check if it's an MCP tool (try both short name and prefixed name)
                    else if let Some(ref mcp) = self.mcp_registry {
                        // Try short name first, then try with MCP prefix
                        let mcp_tool_name = if mcp.get(&call.name).is_some() {
                            call.name.clone()
                        } else {
                            // Try common MCP prefixes
                            let prefixed = format!("MiniMax_{}", call.name);
                            if mcp.get(&prefixed).is_some() {
                                prefixed
                            } else {
                                continue; // Not found
                            }
                        };

                        debug!(
                            "Executing MCP tool: {} (matched from {})",
                            mcp_tool_name, call.name
                        );

                        let args = if let Some(s) = call.arguments.as_str() {
                            serde_json::from_str(s).unwrap_or(call.arguments.clone())
                        } else {
                            call.arguments.clone()
                        };

                        let tool_spinner = create_spinner(&format!("Executing MCP tool {}...", mcp_tool_name).yellow().to_string());
                        let mcp_result = mcp.call_tool(&mcp_tool_name, args).await;
                        tool_spinner.finish_and_clear();

                        match mcp_result {
                            Ok(result) => {
                                println!("{} {} returned {} characters.", "✅ Success:".bold().green(), mcp_tool_name.bold(), result.len());
                                messages.push(Message {
                                    role: "assistant".to_string(),
                                    content: format!(
                                        "__tool_use__:{}",
                                        serde_json::to_string(&json!({
                                            "id": call.id,
                                            "name": call.name,
                                            "input": call.arguments
                                        })).unwrap_or_default()
                                    ),
                                    tool_call_id: None,
                                });
                                messages.push(Message {
                                    role: "tool".to_string(),
                                    content: result,
                                    tool_call_id: call.id.clone(),
                                });

                                messages.push(Message {
                                    role: "system".to_string(),
                                    content: format!(
                                        "You just called MCP tool '{}' and got a result. \n\
                                            Your task is: {}\n\
                                            What is the NEXT step?",
                                        call.name, task
                                    ),
                                    tool_call_id: None,
                                });

                                break;
                            }
                            Err(e) => {
                                println!("{} {} failed: {}", "❌ Error:".bold().red(), mcp_tool_name.bold(), e);
                                messages.push(Message {
                                    role: "assistant".to_string(),
                                    content: format!(
                                        "__tool_use__:{}",
                                        serde_json::to_string(&json!({
                                            "id": call.id,
                                            "name": call.name,
                                            "input": call.arguments
                                        })).unwrap_or_default()
                                    ),
                                    tool_call_id: None,
                                });
                                messages.push(Message {
                                    role: "tool".to_string(),
                                    content: format!(
                                        "ERROR: MCP tool '{}' failed: {}",
                                        mcp_tool_name, e
                                    ),
                                    tool_call_id: call.id.clone(),
                                });
                                break;
                            }
                        }
                    } else {
                        println!("{} {} not found.", "❌ Error:".bold().red(), call.name.bold());
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: format!(
                                "__tool_use__:{}",
                                serde_json::to_string(&json!({
                                    "id": call.id,
                                    "name": call.name,
                                    "input": call.arguments
                                })).unwrap_or_default()
                            ),
                            tool_call_id: None,
                        });
                        messages.push(Message {
                            role: "tool".to_string(),
                            content: format!("ERROR: Tool '{}' not found.", call.name),
                            tool_call_id: call.id.clone(),
                        });
                    }
                }

                continue;
            } else {
                let final_response = response.message.content.clone();
                debug!(
                    "No tool calls, LLM responded directly. Response: {:?}",
                    &final_response[..final_response.len().min(200)]
                );

                if !final_response.trim().is_empty() {
                    println!("\n{}", final_response.trim());
                }

                if final_response.trim().is_empty() {
                    // Empty response - prompt the LLM to try again
                    debug!("LLM returned empty response, prompting to continue...");
                    messages.push(Message {
                        role: "system".to_string(),
                        content: "You MUST use a tool to complete the task. If you got content from a skill, use the 'write' tool to save it. What tool will you call next?".to_string(),
                        tool_call_id: None,
                    });
                    continue;
                }

                // Check if this looks like a final answer (not asking for more tools)
                let lower_response = final_response.to_lowercase();
                let is_final = !lower_response.contains("tool")
                    && !lower_response.contains("call")
                    && !lower_response.contains("need to")
                    && !lower_response.contains("should i")
                    && !lower_response.contains("would you");

                if is_final || lower_response.len() > 50 {
                    // This looks like a final answer - return it
                    info!("Detected final answer from LLM");
                    return Ok(final_response);
                }

                // LLM is still trying to use tools or asking questions
                // Try one more time with a stronger prompt
                info!("LLM returned text but no tool calls. Prompting to use tools...");
                messages.push(Message {
                    role: "system".to_string(),
                    content: format!(
                        "Your previous response '{}' didn't use any tools. The task '{}' requires using tools. What is the next tool you need to call?",
                        final_response.chars().take(100).collect::<String>(), task
                    ),
                    tool_call_id: None,
                });
            }
        }

        warn!("Max iterations reached without completing task");
        Ok("Max iterations reached. Task may not be complete.".to_string())
    }

    fn format_tool_result(tool_name: &str, result: &skill_tools::ToolResult) -> String {
        if result.success {
            if result.output.is_empty() {
                format!(
                    "Tool '{}' completed successfully but produced no output.",
                    tool_name
                )
            } else {
                format!("Tool '{}' succeeded:\n{}", tool_name, result.output)
            }
        } else {
            let error_msg = result.error.as_deref().unwrap_or("Unknown error");
            format!(
                "Tool '{}' FAILED with error: {}\n\nPrevious output: {}",
                tool_name, error_msg, result.output
            )
        }
    }
}

pub struct StreamingAgent {
    llm: Box<dyn LLMClient>,
    tool_registry: ToolRegistry,
    mcp_registry: Option<std::sync::Arc<skill_mcp::McpRegistry>>,
    max_iterations: usize,
    show_thinking: bool,
    extra_system_prompt: Option<String>,
}

impl StreamingAgent {
    pub fn new(llm: Box<dyn LLMClient>) -> Self {
        Self {
            llm,
            tool_registry: ToolRegistry::new(),
            mcp_registry: None,
            max_iterations: 10,
            show_thinking: true,
            extra_system_prompt: None,
        }
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tool_registry = tools;
        self
    }

    pub fn with_mcp_registry(mut self, registry: std::sync::Arc<skill_mcp::McpRegistry>) -> Self {
        self.mcp_registry = Some(registry);
        self
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn with_thinking(mut self, show: bool) -> Self {
        self.show_thinking = show;
        self
    }

    pub fn with_extra_system_prompt(mut self, prompt: String) -> Self {
        self.extra_system_prompt = Some(prompt);
        self
    }

    fn build_system_prompt(&self) -> String {
        let mut tools_list = Vec::new();

        // Add regular tools
        for tool in self.tool_registry.list() {
            tools_list.push(format!("- {}: {}", tool.name, tool.description));
        }

        // Add MCP tools
        if let Some(ref mcp) = self.mcp_registry {
            for tool in mcp.list() {
                tools_list.push(format!("- {}: {}", tool.name, tool.description));
            }
        }

        let tools_str = if tools_list.is_empty() {
            "No tools available".to_string()
        } else {
            tools_list.join("\n")
        };

        // Build skill catalog section (metadata only — no full instructions)
        let skill_catalog_section = match self.tool_registry.skill_catalog() {
            Some(catalog) => format!(
                "\n\nSKILL CATALOG:\n\
                 The following skills are available via the 'run_skill' tool.\n\
                 To use a skill, call run_skill with the skill_id and your input.\n\
                 {}", catalog
            ),
            None => String::new(),
        };

        let mut prompt = format!(
            r#"You are an autonomous agent that MUST use tools to complete tasks.

AVAILABLE TOOLS:
{}{}

STRICT RULES:
1. You MUST use tools to gather information or execute actions when needed.
2. ALWAYS format your final responses to the user in clean Markdown.
3. NEVER create or write files using the 'write' tool UNLESS the user explicitly asks you to save, write, or create a file.
4. If the user asks a question, use tools to find the answer and print the summary directly.

When you finish a task, provide a clear, formatted summary of what was done."#,
            tools_str, skill_catalog_section
        );

        // Add extra system prompt (from AGENTS.md or CLI)
        if let Some(ref extra) = self.extra_system_prompt {
            prompt.push_str("\n\n");
            prompt.push_str(extra);
        }

        prompt
    }

    pub async fn run(&self, task: &str) -> Result<String> {
        println!("\n{}", Self::header("AGENT STARTED"));
        println!("{} {}\n", Self::icon("task"), task);

        let system_prompt = self.build_system_prompt();

        let mut messages = vec![
            Message {
                role: "system".to_string(),
                content: system_prompt.to_string(),
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: task.to_string(),
                tool_call_id: None,
            },
        ];

        let mut tool_defs = self.tool_registry.list();

        // Add MCP tools to the list
        if let Some(ref mcp) = self.mcp_registry {
            let mcp_tools: Vec<ToolDefinition> = mcp
                .list()
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.input_schema,
                })
                .collect();
            println!("{} Adding {} MCP tools", Self::icon("mcp"), mcp_tools.len());
            tool_defs.extend(mcp_tools);
            println!("{} MCP tools: {:?}", Self::icon("mcp"), mcp.list_names());
        }

        let mut tool_history: HashMap<String, usize> = HashMap::new();

        println!(
            "{} Tools: {:?}\n",
            Self::icon("tools"),
            self.tool_registry.names()
        );

        for iteration in 0..self.max_iterations {
            println!(
                "{}",
                Self::iteration_header(iteration + 1, self.max_iterations)
            );

            let mut spinner = Some(create_spinner("🤔 Thinking..."));
            let response_stream = self
                .llm
                .chat_streaming(messages.clone(), Some(tool_defs.clone()));

            let mut accumulated_content = String::new();
            let mut final_tool_calls: Option<Vec<ToolCall>> = None;
            let mut is_done = false;

            tokio::pin!(response_stream);

            info!("[STREAM] Starting to consume response stream");
            let mut chunk_count = 0;

            while let Some(chunk_result) = response_stream.next().await {
                if let Some(s) = spinner.take() {
                    s.finish_and_clear();
                }
                chunk_count += 1;
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        error!("[STREAM] Stream error: {}", e);
                        anyhow::bail!("Stream error: {}", e);
                    }
                };

                info!("[STREAM] Chunk #{}: content_len={}, tool_calls={:?}, done={}, done_reason={:?}", 
                    chunk_count, chunk.content.len(), chunk.tool_calls.is_some(), chunk.done, chunk.done_reason);

                if !chunk.content.is_empty() {
                    accumulated_content.push_str(&chunk.content);
                    info!(
                        "[STREAM] Accumulated content now: {} chars",
                        accumulated_content.len()
                    );
                    if self.show_thinking {
                        print!("{}", Self::thinking(&chunk.content));
                    }
                }

                if let Some(tool_calls) = chunk.tool_calls {
                    info!(
                        "[STREAM] Got tool calls in chunk #{}: {:?}",
                        chunk_count, tool_calls
                    );

                    // FIX: Accumulate tool call arguments across chunks
                    // If we already have tool_calls, we need to merge them (append arguments)
                    // Note: MiniMax may send partial tool_calls without function name in subsequent chunks
                    match &final_tool_calls {
                        Some(existing) => {
                            let mut combined = existing.clone();

                            // Check if new tool_calls have function names (full) or just partial args
                            let new_has_names: Vec<bool> = tool_calls
                                .iter()
                                .map(|c| c.name.is_empty() == false)
                                .collect();

                            // If new chunks have partial args (no name), append to existing
                            // Otherwise, add as new tool calls
                            for (i, new_call) in tool_calls.iter().enumerate() {
                                if i < combined.len() && !new_has_names[i] {
                                    // Merge: append arguments strings to existing
                                    if let (Some(existing_args), Some(new_args)) = (
                                        combined[i].arguments.as_str(),
                                        new_call.arguments.as_str(),
                                    ) {
                                        let merged_args = format!("{}{}", existing_args, new_args);
                                        info!("[STREAM] Merged partial args: {:?}", merged_args);
                                        combined[i].arguments =
                                            serde_json::Value::String(merged_args);
                                    }
                                } else {
                                    combined.push(new_call.clone());
                                }
                            }
                            final_tool_calls = Some(combined);
                            info!("[STREAM] Combined tool calls: {:?}", final_tool_calls);
                        }
                        None => {
                            final_tool_calls = Some(tool_calls);
                        }
                    }
                }

                is_done = chunk.done;
                if is_done {
                    info!("[STREAM] Done flag received at chunk #{}", chunk_count);
                    break;
                }
            }
            if let Some(s) = spinner.take() {
                s.finish_and_clear();
            }

            info!("[STREAM] Stream consumption complete. Total chunks: {}, accumulated_content: {} chars, final_tool_calls: {:?}, is_done: {}", 
                chunk_count, accumulated_content.len(), final_tool_calls, is_done);

            if !accumulated_content.is_empty() {
                println!(
                    "{} {}",
                    Self::icon("response"),
                    Self::response(&accumulated_content)
                );
            }

            if let Some(tool_calls) = final_tool_calls {
                if !tool_calls.is_empty() {
                    println!("{} {} tool call(s)", Self::icon("tools"), tool_calls.len());

                    for (i, call) in tool_calls.iter().enumerate() {
                        println!(
                            "{} Tool {}: {} {}",
                            Self::indent(2),
                            i + 1,
                            Self::tool_name(call.name.as_str()),
                            Self::tool_args(&call.arguments)
                        );

                        let call_key = format!("{}:{}", call.name, call.arguments);
                        let count = tool_history.entry(call_key.clone()).or_insert(0);
                        *count += 1;

                        if *count > 2 {
                            println!(
                                "{} {}",
                                Self::icon("warn"),
                                Self::warn(&format!(
                                    "Doom loop detected: {} called {} times",
                                    call.name, count
                                ))
                            );
                            messages.push(Message {
                                role: "assistant".to_string(),
                                content: format!(
                                    "__tool_use__:{}",
                                    serde_json::to_string(&json!({
                                        "id": call.id,
                                        "name": call.name,
                                        "input": call.arguments
                                    })).unwrap_or_default()
                                ),
                                tool_call_id: None,
                            });
                            messages.push(Message {
                                role: "tool".to_string(),
                                content: format!(
                                    "ERROR: Tool '{}' has failed multiple times ({} attempts). Try a DIFFERENT approach.",
                                    call.name, count
                                ),
                                tool_call_id: call.id.clone(),
                            });
                            continue;
                        }

                        // Check if it's a regular tool
                        if let Some(tool) = self.tool_registry.get(&call.name) {
                            println!(
                                "{} Executing: {}",
                                Self::icon("exec"),
                                Self::exec(call.name.as_str())
                            );

                            let args = if let Some(s) = call.arguments.as_str() {
                                serde_json::from_str(s).unwrap_or(call.arguments.clone())
                            } else {
                                call.arguments.clone()
                            };

                            let tool_spinner = create_spinner(&format!("Executing {}...", call.name).yellow().to_string());
                            let result = tool.execute(args).await;
                            tool_spinner.finish_and_clear();
                            let result = result?;

                            if result.success {
                                println!(
                                    "{} {}",
                                    Self::icon("success"),
                                    Self::success(&format!(
                                        "Tool '{}' executed successfully",
                                        call.name
                                    ))
                                );
                            } else {
                                println!(
                                    "{} {}",
                                    Self::icon("error"),
                                    Self::error(&format!(
                                        "Tool '{}' failed: {}",
                                        call.name,
                                        result.error.as_deref().unwrap_or("Unknown")
                                    ))
                                );
                            }

                            if call.name == "write" && result.success {
                                println!("\n{}", Self::success_header("TASK COMPLETED"));
                                return Ok(format!(
                                    "Task completed successfully! Saved content to: {}",
                                    call.arguments
                                        .get("path")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("file")
                                ));
                            }

                            let result_message = Agent::format_tool_result(&call.name, &result);
                            messages.push(Message {
                                role: "assistant".to_string(),
                                content: format!(
                                    "__tool_use__:{}",
                                    serde_json::to_string(&json!({
                                        "id": call.id,
                                        "name": call.name,
                                        "input": call.arguments
                                    })).unwrap_or_default()
                                ),
                                tool_call_id: None,
                            });
                            messages.push(Message {
                                role: "tool".to_string(),
                                content: result_message,
                                tool_call_id: call.id.clone(),
                            });

                            messages.push(Message {
                                role: "system".to_string(),
                                content: format!(
                                    "You just called '{}' and got a result. \n\
                                    Your task is: {}\n\
                                    What is the NEXT step? Do you need to call another tool? \
                                    If you got content from a skill, you MUST save it with 'write'.",
                                    call.name, task
                                ),
                                tool_call_id: None,
                            });

                            break;
                        }
                        // Check if it's an MCP tool (try both short name and prefixed name)
                        else if let Some(ref mcp) = self.mcp_registry {
                            // Try short name first, then try with MCP prefix
                            let mcp_tool_name = if mcp.get(&call.name).is_some() {
                                call.name.clone()
                            } else {
                                // Try common MCP prefixes
                                let prefixed = format!("MiniMax_{}", call.name);
                                if mcp.get(&prefixed).is_some() {
                                    prefixed
                                } else {
                                    println!(
                                        "{} Tool not found: {}",
                                        Self::icon("error"),
                                        call.name
                                    );
                                    messages.push(Message {
                                        role: "assistant".to_string(),
                                        content: format!(
                                            "__tool_use__:{}",
                                            serde_json::to_string(&json!({
                                                "id": call.id,
                                                "name": call.name,
                                                "input": call.arguments
                                            })).unwrap_or_default()
                                        ),
                                        tool_call_id: None,
                                    });
                                    messages.push(Message {
                                        role: "tool".to_string(),
                                        content: format!("ERROR: Tool '{}' not found.", call.name),
                                        tool_call_id: call.id.clone(),
                                    });
                                    break;
                                }
                            };

                            println!(
                                "{} Executing MCP tool: {} (matched from {})",
                                Self::icon("exec"),
                                Self::exec(mcp_tool_name.as_str()),
                                call.name
                            );

                            let args = if let Some(s) = call.arguments.as_str() {
                                serde_json::from_str(s).unwrap_or(call.arguments.clone())
                            } else {
                                call.arguments.clone()
                            };

                            let tool_spinner = create_spinner(&format!("Executing MCP tool {}...", mcp_tool_name).yellow().to_string());
                            let mcp_result = mcp.call_tool(&mcp_tool_name, args).await;
                            tool_spinner.finish_and_clear();

                            match mcp_result {
                                Ok(result) => {
                                    println!(
                                        "{} {}",
                                        Self::icon("success"),
                                        Self::success(&format!(
                                            "MCP tool '{}' executed",
                                            mcp_tool_name
                                        ))
                                    );
                                    messages.push(Message {
                                        role: "assistant".to_string(),
                                        content: format!(
                                            "__tool_use__:{}",
                                            serde_json::to_string(&json!({
                                                "id": call.id,
                                                "name": call.name,
                                                "input": call.arguments
                                            })).unwrap_or_default()
                                        ),
                                        tool_call_id: None,
                                    });
                                    messages.push(Message {
                                        role: "tool".to_string(),
                                        content: result,
                                        tool_call_id: call.id.clone(),
                                    });

                                    messages.push(Message {
                                        role: "system".to_string(),
                                        content: format!(
                                            "You just called MCP tool '{}' and got a result. \n\
                                                Your task is: {}\n\
                                                What is the NEXT step?",
                                            mcp_tool_name, task
                                        ),
                                        tool_call_id: None,
                                    });

                                    break;
                                }
                                Err(e) => {
                                    println!(
                                        "{} {}",
                                        Self::icon("error"),
                                        Self::error(&format!(
                                            "MCP tool '{}' failed: {}",
                                            mcp_tool_name, e
                                        ))
                                    );
                                    messages.push(Message {
                                        role: "assistant".to_string(),
                                        content: format!(
                                            "__tool_use__:{}",
                                            serde_json::to_string(&json!({
                                                "id": call.id,
                                                "name": call.name,
                                                "input": call.arguments
                                            })).unwrap_or_default()
                                        ),
                                        tool_call_id: None,
                                    });
                                    messages.push(Message {
                                        role: "tool".to_string(),
                                        content: format!(
                                            "ERROR: MCP tool '{}' failed: {}",
                                            mcp_tool_name, e
                                        ),
                                        tool_call_id: call.id.clone(),
                                    });
                                    break;
                                }
                            }
                        }
                    }

                    continue;
                }
            }

            if accumulated_content.trim().is_empty() && !is_done {
                println!(
                    "{} {}",
                    Self::icon("warn"),
                    Self::warn("LLM returned empty response")
                );
                messages.push(Message {
                    role: "system".to_string(),
                    content: "You MUST use a tool to complete the task. If you got content from a skill, use the 'write' tool to save it. What tool will you call next?".to_string(),
                    tool_call_id: None,
                });
                continue;
            }

            let lower_response = accumulated_content.to_lowercase();
            let is_final = !lower_response.contains("tool")
                && !lower_response.contains("call")
                && !lower_response.contains("need to")
                && !lower_response.contains("should i")
                && !lower_response.contains("would you");

            if is_final || accumulated_content.len() > 50 {
                println!("\n{}", Self::final_header("FINAL ANSWER"));
                return Ok(accumulated_content);
            }

            println!("{} Prompting to use tools...", Self::icon("hint"));
            messages.push(Message {
                role: "system".to_string(),
                content: format!(
                    "Your previous response '{}' didn't use any tools. The task '{}' requires using tools. What is the next tool you need to call?",
                    accumulated_content.chars().take(100).collect::<String>(), task
                ),
                tool_call_id: None,
            });
        }

        println!("{}", Self::warn_header("MAX ITERATIONS REACHED"));
        Ok("Max iterations reached. Task may not be complete.".to_string())
    }

    fn header(s: &str) -> String {
        format!("\n╔══════════════════════════════════════════════════════════════╗\n║  {}  ║\n╚══════════════════════════════════════════════════════════════╝", Self::center(s, 62))
    }

    fn iteration_header(current: usize, total: usize) -> String {
        format!("\n┌──────────────────────────────────────────────────────────────┐\n│  🧠 {} / {}                                                      │\n└──────────────────────────────────────────────────────────────┘", current, total)
    }

    fn success_header(s: &str) -> String {
        format!("\n✅ ════════════════════════════════════════════════════════════ ✅\n   {}\n✅ ════════════════════════════════════════════════════════════ ✅", s)
    }

    fn final_header(s: &str) -> String {
        format!("\n📤 ═══════════════════════════════════════════════════════════ 📤\n   {}\n📤 ═══════════════════════════════════════════════════════════ 📤", s)
    }

    fn warn_header(s: &str) -> String {
        format!("\n⚠️  ═══════════════════════════════════════════════════════════ ⚠️\n   {}\n⚠️  ═══════════════════════════════════════════════════════════ ⚠️", s)
    }

    fn icon(s: &str) -> &str {
        match s {
            "task" => "🎯",
            "tools" => "🔧",
            "response" => "💬",
            "exec" => "⚡",
            "success" => "✅",
            "error" => "❌",
            "warn" => "⚠️",
            "hint" => "💡",
            "thinking" => "🤔",
            _ => "•",
        }
    }

    fn thinking(s: &str) -> String {
        print!("{}", s);
        std::io::stdout().flush().ok();
        String::new()
    }

    fn response(s: &str) -> String {
        let truncated = if s.len() > 200 {
            format!("{}...", &s[..200])
        } else {
            s.to_string()
        };
        truncated.lines().next().unwrap_or(&truncated).to_string()
    }

    fn tool_name(s: &str) -> String {
        format!("[{}]", s)
    }

    fn tool_args(args: &serde_json::Value) -> String {
        if let Some(s) = args.as_str() {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                return format!("{:?}", parsed);
            }
            s.chars().take(50).collect::<String>()
        } else {
            format!("{:?}", args).chars().take(50).collect::<String>()
        }
    }

    fn exec(s: &str) -> String {
        s.to_string()
    }

    fn success(s: &str) -> String {
        s.to_string()
    }

    fn error(s: &str) -> String {
        s.to_string()
    }

    fn warn(s: &str) -> String {
        s.to_string()
    }

    fn indent(spaces: usize) -> String {
        " ".repeat(spaces)
    }

    fn center(s: &str, width: usize) -> String {
        if s.len() >= width {
            s.to_string()
        } else {
            let padding = width - s.len();
            let left = padding / 2;
            let right = padding - left;
            format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
        }
    }
}
