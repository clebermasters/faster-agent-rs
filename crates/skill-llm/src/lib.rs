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
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;
use tracing::{debug, error, info, warn};

#[cfg(feature = "bedrock")]
pub mod bedrock;
#[cfg(feature = "bedrock")]
pub use bedrock::{create_bedrock_client, BedrockAuth};

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
    #[serde(default)]
    pub thinking: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseEvent {
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultEvent {
    pub name: String,
    #[serde(alias = "output")]
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub done: bool,
    pub done_reason: Option<String>,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub tool_use: Option<ToolUseEvent>,
    #[serde(default)]
    pub tool_result: Option<ToolResultEvent>,
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
                thinking: None,
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
                            thinking: None,
                            tool_use: None,
                            tool_result: None,
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
                                thinking: None,
                                tool_use: None,
                                tool_result: None,
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

// ---------------------------------------------------------------------------
// OpenAI-compatible client (works with any provider exposing /v1/chat/completions)
// ---------------------------------------------------------------------------

pub struct OpenAIClient {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAIClient {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
            model,
        }
    }

    /// Build a tool-calling instruction to PREPEND to the system prompt.
    /// The wrapper doesn't support OpenAI function calling, so we instruct
    /// the model to output tool calls in a parseable text format.
    fn tool_instruction(_tools: &[ToolDefinition]) -> String {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        format!(
            "IMPORTANT: The user's project directory is: {}\n\
             When reading files or running commands, use absolute paths based on this directory.\n\
             For example, to read Cargo.toml use: {}/Cargo.toml\n\n",
            cwd, cwd
        )
    }

    /// Build the request body for the wrapper.
    /// When tools are provided, enable CLI tools so the model can execute actions.
    fn build_request_body(
        model: &str,
        messages: &[serde_json::Value],
        stream: bool,
        has_tools: bool,
    ) -> serde_json::Value {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": stream,
            "include_thinking": true,
        });
        if has_tools {
            // Enable the wrapper's CLI tools (Bash, Read, Write, etc.)
            // so the model can execute actions via the Claude CLI.
            body["enable_tools"] = json!(true);
        }
        body
    }

    /// Extract tool calls from `<tool_call>...</tool_call>` blocks in text.
    /// Returns (clean_content, tool_calls).
    fn extract_tool_calls(content: &str) -> (String, Option<Vec<ToolCall>>) {
        let mut calls = Vec::new();
        let mut clean = String::new();
        let mut rest = content;

        while let Some(start) = rest.find("<tool_call>") {
            clean.push_str(&rest[..start]);
            let after_tag = &rest[start + "<tool_call>".len()..];
            if let Some(end) = after_tag.find("</tool_call>") {
                let json_str = after_tag[..end].trim();
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    let name = val["name"].as_str().unwrap_or("").to_string();
                    let arguments = val.get("arguments").cloned().unwrap_or(json!({}));
                    if !name.is_empty() {
                        calls.push(ToolCall {
                            id: Some(format!("call_{}", calls.len())),
                            name,
                            arguments,
                        });
                    }
                }
                rest = &after_tag[end + "</tool_call>".len()..];
            } else {
                // Unclosed tag — keep as content
                clean.push_str(&rest[start..]);
                rest = "";
                break;
            }
        }
        clean.push_str(rest);
        let clean = clean.trim().to_string();

        if calls.is_empty() {
            (clean, None)
        } else {
            (clean, Some(calls))
        }
    }
}

impl LLMClient for OpenAIClient {
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
            let has_tools = tools.as_ref().map_or(false, |t| !t.is_empty());

            let mut messages: Vec<serde_json::Value> = messages
                .into_iter()
                .map(|m| {
                    json!({
                        "role": m.role,
                        "content": m.content
                    })
                })
                .collect();

            // Inject CWD hint so the CLI model uses absolute paths from the project dir
            if has_tools {
                let cwd_hint = Self::tool_instruction(&[]);
                if let Some(sys) = messages.iter_mut().find(|m| m["role"] == "system") {
                    let existing = sys["content"].as_str().unwrap_or("").to_string();
                    sys["content"] = json!(format!("{}{}", cwd_hint, existing));
                }
            }

            let body = Self::build_request_body(&model, &messages, false, has_tools);
            let cwd = std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            debug!("[OPENAI] Sending chat request to: {}", base_url);

            let mut req = client
                .post(format!("{}/v1/chat/completions", base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json");
            if has_tools && !cwd.is_empty() {
                req = req.header("X-Claude-Add-Dir", &cwd);
            }
            let response = req.json(&body).send().await?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                anyhow::bail!("OpenAI API error: {} - {}", status, error_text);
            }

            let chat_resp: serde_json::Value = response.json().await?;

            let message = chat_resp["choices"][0]["message"].clone();
            let role = message["role"].as_str().unwrap_or("assistant").to_string();
            let raw_content = message["content"].as_str().unwrap_or("").to_string();
            let thinking = message
                .get("thinking")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            // Extract tool calls from text content
            let (content, text_tool_calls) = Self::extract_tool_calls(&raw_content);

            // Prefer native tool_calls, fall back to text-based
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
                    text_tool_calls
                } else {
                    Some(calls)
                }
            } else {
                text_tool_calls
            };

            Ok(ChatResponse {
                message: Message {
                    role,
                    content,
                    tool_call_id: None,
                },
                tool_calls,
                done: true,
                thinking,
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
            let has_tools = tools.as_ref().map_or(false, |t| !t.is_empty());

            let mut messages: Vec<serde_json::Value> = messages
                .into_iter()
                .map(|m| {
                    json!({
                        "role": m.role,
                        "content": m.content
                    })
                })
                .collect();

            // Inject CWD hint so the CLI model uses absolute paths from the project dir
            if has_tools {
                let cwd_hint = OpenAIClient::tool_instruction(&[]);
                if let Some(sys) = messages.iter_mut().find(|m| m["role"] == "system") {
                    let existing = sys["content"].as_str().unwrap_or("").to_string();
                    sys["content"] = json!(format!("{}{}", cwd_hint, existing));
                }
            }

            let body = OpenAIClient::build_request_body(&model, &messages, true, has_tools);
            let cwd = std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            debug!("[OPENAI] Sending streaming chat request");

            let mut req = client
                .post(format!("{}/v1/chat/completions", base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json");
            if has_tools && !cwd.is_empty() {
                req = req.header("X-Claude-Add-Dir", &cwd);
            }
            let response = req
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    error!("[OPENAI] Request error: {}", e);
                    anyhow::anyhow!("Request error: {}", e)
                })?;

            let status = response.status();
            if !status.is_success() {
                error!("[OPENAI] API error: {}", status);
                Err(anyhow::anyhow!("OpenAI API error: {}", status))?;
            }

            info!("[OPENAI] Streaming response started");
            let mut stream = response.bytes_stream();
            let mut bytes_received = 0;

            while let Some(chunk_result) = stream.next().await {
                let chunk_bytes = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        error!("[OPENAI] Stream read error: {}", e);
                        Err(anyhow::anyhow!("Stream error: {}", e))?;
                        unreachable!()
                    }
                };

                bytes_received += chunk_bytes.len();
                debug!("[OPENAI] Received {} bytes (total: {})", chunk_bytes.len(), bytes_received);

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
                        info!("[OPENAI] Received [DONE] signal");
                        yield ChatChunk {
                            content: String::new(),
                            tool_calls: None,
                            done: true,
                            done_reason: Some("stop".to_string()),
                            thinking: None,
                            tool_use: None,
                            tool_result: None,
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

                            let thinking = delta
                                .and_then(|d| d.get("thinking"))
                                .and_then(|t| t.as_str())
                                .map(|s| s.to_string());

                            let tool_calls = delta.and_then(|d| d.get("tool_calls")).and_then(|tc| {
                                let calls: Vec<ToolCall> = tc
                                    .as_array()
                                    .unwrap_or(&vec![])
                                    .iter()
                                    .filter_map(|c| {
                                        let func = c.get("function")?;
                                        let args = func.get("arguments")?;
                                        let args_str = args.as_str().unwrap_or("");
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

                            let tool_use = delta
                                .and_then(|d| d.get("tool_use"))
                                .and_then(|tu| {
                                    Some(ToolUseEvent {
                                        name: tu.get("name").and_then(|v| v.as_str())?.to_string(),
                                        input: tu.get("input").cloned().unwrap_or(json!({})),
                                    })
                                });

                            let tool_result = delta
                                .and_then(|d| d.get("tool_result"))
                                .and_then(|tr| {
                                    Some(ToolResultEvent {
                                        name: tr.get("name").and_then(|v| v.as_str()).unwrap_or("tool").to_string(),
                                        content: tr.get("output").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                        is_error: tr.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false),
                                    })
                                });

                            let finish_reason = chat_resp.get("choices")
                                .and_then(|c| c.as_array())
                                .and_then(|c| c.first())
                                .and_then(|c| c.get("finish_reason"))
                                .and_then(|f| f.as_str());

                            let done = finish_reason.is_some();
                            let done_reason = finish_reason.map(|s| s.to_string());

                            yield ChatChunk {
                                content,
                                tool_calls,
                                done,
                                done_reason,
                                thinking,
                                tool_use,
                                tool_result,
                            };
                        }
                        Err(_) => continue,
                    }
                }
            }

            info!("[OPENAI] Stream ended normally, total bytes: {}", bytes_received);
        })
    }
}

// ---------------------------------------------------------------------------
// OpenAI Codex subscription-backed client (ChatGPT-authenticated Responses API)
// ---------------------------------------------------------------------------

pub struct OpenAICodexClient {
    client: Client,
    base_url: String,
    auth_path: PathBuf,
    model: String,
    client_version: String,
    max_retries: usize,
    retry_delay: Duration,
}

impl OpenAICodexClient {
    const DEFAULT_BASE_URL: &'static str = "https://chatgpt.com/backend-api/codex";
    const REFRESH_URL: &'static str = "https://auth.openai.com/oauth/token";
    const CLIENT_ID: &'static str = "app_EMoamEEZ73f0CkXaXp7hrann";

    pub fn new(
        base_url: Option<String>,
        auth_path: Option<String>,
        client_version: Option<String>,
        model: String,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url
                .unwrap_or_else(|| Self::DEFAULT_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            auth_path: Self::expand_tilde(
                &auth_path.unwrap_or_else(|| "~/.codex/auth.json".to_string()),
            ),
            model,
            client_version: client_version.unwrap_or_else(Self::detect_client_version),
            max_retries: 2,
            retry_delay: Duration::from_secs(1),
        }
    }

    fn expand_tilde(path: &str) -> PathBuf {
        if let Some(stripped) = path.strip_prefix("~/") {
            if let Ok(home) = std::env::var("HOME") {
                return PathBuf::from(home).join(stripped);
            }
        }
        PathBuf::from(path)
    }

    fn detect_client_version() -> String {
        std::process::Command::new("codex")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|stdout| stdout.split_whitespace().nth(1).map(str::to_string))
            .unwrap_or_else(|| "0.120.0".to_string())
    }

    fn load_auth_payload(&self) -> Result<serde_json::Value> {
        if !self.auth_path.exists() {
            anyhow::bail!(
                "Codex auth file not found at {}. Run `codex login` first.",
                self.auth_path.display()
            );
        }

        let raw = fs::read_to_string(&self.auth_path)?;
        let payload: serde_json::Value = serde_json::from_str(&raw)?;

        let access_token = payload
            .get("tokens")
            .and_then(|tokens| tokens.get("access_token"))
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if access_token.is_empty() {
            anyhow::bail!(
                "Codex auth file at {} does not contain an access token.",
                self.auth_path.display()
            );
        }

        Ok(payload)
    }

    fn access_token(auth_payload: &serde_json::Value) -> Result<String> {
        auth_payload
            .get("tokens")
            .and_then(|tokens| tokens.get("access_token"))
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("Codex auth payload is missing an access token"))
    }

    fn refresh_token(auth_payload: &serde_json::Value) -> Result<String> {
        auth_payload
            .get("tokens")
            .and_then(|tokens| tokens.get("refresh_token"))
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("Codex auth payload is missing a refresh token"))
    }

    fn persist_auth_tokens(
        &self,
        auth_payload: &mut serde_json::Value,
        refreshed_tokens: &serde_json::Value,
    ) -> Result<()> {
        let token_map = auth_payload
            .get_mut("tokens")
            .and_then(|tokens| tokens.as_object_mut())
            .ok_or_else(|| anyhow::anyhow!("Codex auth payload is missing the token object"))?;

        for key in ["id_token", "access_token", "refresh_token"] {
            if let Some(value) = refreshed_tokens.get(key).cloned() {
                token_map.insert(key.to_string(), value);
            }
        }

        let serialized = serde_json::to_string_pretty(auth_payload)?;
        fs::write(&self.auth_path, serialized)?;
        Ok(())
    }

    async fn refresh_access_token(&self, auth_payload: &mut serde_json::Value) -> Result<()> {
        let refresh_token = Self::refresh_token(auth_payload)?;

        let response = self
            .client
            .post(Self::REFRESH_URL)
            .header("Content-Type", "application/json")
            .json(&json!({
                "client_id": Self::CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Failed to refresh Codex OAuth token: {} - {}",
                status,
                Self::extract_error_message(&error_text)
            );
        }

        let refreshed: serde_json::Value = response.json().await?;
        self.persist_auth_tokens(auth_payload, &refreshed)?;
        Ok(())
    }

    fn extract_error_message(body: &str) -> String {
        serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|payload| {
                payload
                    .get("detail")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                    .or_else(|| {
                        payload
                            .get("error")
                            .and_then(|value| value.get("message"))
                            .and_then(|value| value.as_str())
                            .map(str::to_string)
                    })
                    .or_else(|| {
                        payload
                            .get("error")
                            .and_then(|value| value.as_str())
                            .map(str::to_string)
                    })
            })
            .filter(|message| !message.is_empty())
            .unwrap_or_else(|| body.to_string())
    }

    fn convert_tools(tools: Vec<ToolDefinition>) -> Vec<serde_json::Value> {
        tools
            .into_iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect()
    }

    fn convert_messages(messages: Vec<Message>) -> Result<(String, Vec<serde_json::Value>)> {
        let mut instructions = Vec::new();
        let mut input = Vec::new();

        for message in messages {
            match message.role.as_str() {
                "system" => {
                    if !message.content.trim().is_empty() {
                        instructions.push(message.content);
                    }
                }
                "user" => input.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": message.content,
                    }]
                })),
                "assistant" => {
                    if message.content.starts_with("__tool_use__:") {
                        let value: serde_json::Value =
                            serde_json::from_str(&message.content["__tool_use__:".len()..])?;
                        let arguments_value =
                            value.get("input").cloned().unwrap_or_else(|| json!({}));
                        let arguments = serde_json::to_string(&arguments_value)
                            .unwrap_or_else(|_| "{}".to_string());
                        let call_id = value
                            .get("id")
                            .and_then(|v| v.as_str())
                            .filter(|v| !v.is_empty())
                            .ok_or_else(|| {
                                anyhow::anyhow!("Tool use marker is missing a call id")
                            })?;
                        let name = value
                            .get("name")
                            .and_then(|v| v.as_str())
                            .filter(|v| !v.is_empty())
                            .ok_or_else(|| {
                                anyhow::anyhow!("Tool use marker is missing a tool name")
                            })?;

                        input.push(json!({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        }));
                    } else if !message.content.trim().is_empty() {
                        input.push(json!({
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": message.content,
                            }]
                        }));
                    }
                }
                "tool" => {
                    let call_id = message
                        .tool_call_id
                        .as_deref()
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| anyhow::anyhow!("Tool result is missing a tool_call_id"))?;
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": message.content,
                    }));
                }
                other => input.push(json!({
                    "role": other,
                    "content": [{
                        "type": "input_text",
                        "text": message.content,
                    }]
                })),
            }
        }

        if input.is_empty() {
            input.push(json!({
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "",
                }]
            }));
        }

        let instructions = if instructions.is_empty() {
            "You are a helpful assistant.".to_string()
        } else {
            instructions.join("\n\n")
        };

        Ok((instructions, input))
    }

    fn build_request_body(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<serde_json::Value> {
        let (mut instructions, input) = Self::convert_messages(messages)?;
        let has_tools = tools
            .as_ref()
            .map(|tools| !tools.is_empty())
            .unwrap_or(false);
        if has_tools {
            instructions = format!("{}{}", OpenAIClient::tool_instruction(&[]), instructions);
        }
        let mut body = json!({
            "model": self.model,
            "instructions": instructions,
            "input": input,
            "parallel_tool_calls": false,
            "store": false,
            "stream": true,
            "text": {
                "format": {
                    "type": "text"
                }
            }
        });

        if let Some(tools) = tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::Value::Array(Self::convert_tools(tools));
            }
        }

        Ok(body)
    }

    async fn send_responses_request(&self, body: &serde_json::Value) -> Result<reqwest::Response> {
        let mut auth_payload = self.load_auth_payload()?;
        let url = format!("{}/responses", self.base_url);

        for attempt in 0..=self.max_retries {
            let access_token = Self::access_token(&auth_payload)?;
            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", access_token))
                .header("Content-Type", "application/json")
                .query(&[("client_version", self.client_version.as_str())])
                .json(body)
                .send()
                .await;

            let response = match response {
                Ok(response) => response,
                Err(err) => {
                    if attempt < self.max_retries {
                        tokio::time::sleep(self.retry_delay * (attempt as u32 + 1)).await;
                        continue;
                    }
                    return Err(err.into());
                }
            };

            if response.status() == reqwest::StatusCode::UNAUTHORIZED {
                self.refresh_access_token(&mut auth_payload).await?;
                if attempt < self.max_retries {
                    continue;
                }
            }

            if response.status().is_success() {
                return Ok(response);
            }

            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);
            if retryable && attempt < self.max_retries {
                tokio::time::sleep(self.retry_delay * (attempt as u32 + 1)).await;
                continue;
            }

            anyhow::bail!(
                "OpenAI Codex API error: {} - {}",
                status,
                Self::extract_error_message(&error_text)
            );
        }

        anyhow::bail!("OpenAI Codex request failed without a response")
    }

    fn parse_tool_arguments(arguments: &str) -> serde_json::Value {
        serde_json::from_str(arguments)
            .unwrap_or_else(|_| serde_json::Value::String(arguments.to_string()))
    }

    fn parse_sse_response(body: &str) -> Result<ChatResponse> {
        let mut content = String::new();
        let mut saw_text_delta = false;
        let mut pending_calls: HashMap<String, (String, String, String)> = HashMap::new();
        let mut tool_calls = Vec::new();

        for raw_line in body.lines() {
            let line = raw_line.trim();
            if !line.starts_with("data:") {
                continue;
            }

            let data = line[5..].trim();
            if data.is_empty() {
                continue;
            }

            let event: serde_json::Value = match serde_json::from_str(data) {
                Ok(event) => event,
                Err(_) => continue,
            };

            match event
                .get("type")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
            {
                "response.output_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|value| value.as_str()) {
                        content.push_str(delta);
                        saw_text_delta = true;
                    }
                }
                "response.output_text.done" => {
                    if !saw_text_delta {
                        if let Some(text) = event.get("text").and_then(|value| value.as_str()) {
                            content.push_str(text);
                        }
                    }
                }
                "response.output_item.added" => {
                    if event
                        .get("item")
                        .and_then(|item| item.get("type"))
                        .and_then(|value| value.as_str())
                        == Some("function_call")
                    {
                        let item = &event["item"];
                        let item_id = item["id"].as_str().unwrap_or_default().to_string();
                        if !item_id.is_empty() {
                            pending_calls.insert(
                                item_id,
                                (
                                    item["call_id"].as_str().unwrap_or_default().to_string(),
                                    item["name"].as_str().unwrap_or_default().to_string(),
                                    String::new(),
                                ),
                            );
                        }
                    }
                }
                "response.function_call_arguments.delta" => {
                    let item_id = event["item_id"].as_str().unwrap_or_default();
                    let delta = event["delta"].as_str().unwrap_or_default();
                    if let Some((_, _, arguments)) = pending_calls.get_mut(item_id) {
                        arguments.push_str(delta);
                    }
                }
                "response.function_call_arguments.done" => {
                    let item_id = event["item_id"].as_str().unwrap_or_default();
                    if let Some((_, _, arguments)) = pending_calls.get_mut(item_id) {
                        *arguments = event["arguments"].as_str().unwrap_or_default().to_string();
                    }
                }
                "response.output_item.done" => {
                    if event
                        .get("item")
                        .and_then(|item| item.get("type"))
                        .and_then(|value| value.as_str())
                        == Some("function_call")
                    {
                        let item = &event["item"];
                        let item_id = item["id"].as_str().unwrap_or_default();
                        let (call_id, name, stored_arguments) =
                            pending_calls.remove(item_id).unwrap_or_else(|| {
                                (
                                    item["call_id"].as_str().unwrap_or_default().to_string(),
                                    item["name"].as_str().unwrap_or_default().to_string(),
                                    String::new(),
                                )
                            });
                        let arguments = item["arguments"]
                            .as_str()
                            .filter(|value| !value.is_empty())
                            .unwrap_or(&stored_arguments);
                        tool_calls.push(ToolCall {
                            id: Some(call_id),
                            name,
                            arguments: Self::parse_tool_arguments(arguments),
                        });
                    }
                }
                "response.failed" => {
                    let message = event
                        .get("response")
                        .and_then(|response| response.get("error"))
                        .and_then(|error| error.get("message"))
                        .and_then(|value| value.as_str())
                        .unwrap_or("OpenAI Codex response failed");
                    anyhow::bail!(message.to_string());
                }
                _ => {}
            }
        }

        if content.trim().is_empty() && tool_calls.is_empty() {
            anyhow::bail!("OpenAI Codex returned an empty response")
        }

        Ok(ChatResponse {
            message: Message {
                role: "assistant".to_string(),
                content: content.trim().to_string(),
                tool_call_id: None,
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            done: true,
            thinking: None,
        })
    }
}

impl LLMClient for OpenAICodexClient {
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>> {
        Box::pin(async move {
            let body = self.build_request_body(messages, tools)?;
            debug!("[OPENAI-CODEX] Sending chat request to: {}", self.base_url);
            let response = self.send_responses_request(&body).await?;
            let sse_body = response.text().await?;
            Self::parse_sse_response(&sse_body)
        })
    }

    fn chat_streaming(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {
            let body = self.build_request_body(messages, tools)?;
            debug!("[OPENAI-CODEX] Sending streaming chat request to: {}", self.base_url);
            let response = self.send_responses_request(&body).await?;
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut saw_text_delta = false;
            let mut pending_calls: HashMap<String, (String, String, String)> = HashMap::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk_bytes = chunk_result?;
                if chunk_bytes.is_empty() {
                    continue;
                }

                buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));

                while let Some(newline_idx) = buffer.find('\n') {
                    let line: String = buffer.drain(..=newline_idx).collect();
                    let line = line.trim();
                    if !line.starts_with("data:") {
                        continue;
                    }

                    let data = line[5..].trim();
                    if data.is_empty() {
                        continue;
                    }

                    let event: serde_json::Value = match serde_json::from_str(data) {
                        Ok(event) => event,
                        Err(_) => continue,
                    };

                    match event.get("type").and_then(|value| value.as_str()).unwrap_or_default() {
                        "response.output_text.delta" => {
                            if let Some(delta) = event.get("delta").and_then(|value| value.as_str()) {
                                saw_text_delta = true;
                                yield ChatChunk {
                                    content: delta.to_string(),
                                    tool_calls: None,
                                    done: false,
                                    done_reason: None,
                                    thinking: None,
                                    tool_use: None,
                                    tool_result: None,
                                };
                            }
                        }
                        "response.output_text.done" => {
                            if !saw_text_delta {
                                if let Some(text) = event.get("text").and_then(|value| value.as_str()) {
                                    yield ChatChunk {
                                        content: text.to_string(),
                                        tool_calls: None,
                                        done: false,
                                        done_reason: None,
                                        thinking: None,
                                        tool_use: None,
                                        tool_result: None,
                                    };
                                }
                            }
                        }
                        "response.output_item.added" => {
                            if event
                                .get("item")
                                .and_then(|item| item.get("type"))
                                .and_then(|value| value.as_str())
                                == Some("function_call")
                            {
                                let item = &event["item"];
                                let item_id = item["id"].as_str().unwrap_or_default().to_string();
                                if !item_id.is_empty() {
                                    pending_calls.insert(
                                        item_id,
                                        (
                                            item["call_id"].as_str().unwrap_or_default().to_string(),
                                            item["name"].as_str().unwrap_or_default().to_string(),
                                            String::new(),
                                        ),
                                    );
                                }
                            }
                        }
                        "response.function_call_arguments.delta" => {
                            let item_id = event["item_id"].as_str().unwrap_or_default();
                            let delta = event["delta"].as_str().unwrap_or_default();
                            if let Some((_, _, arguments)) = pending_calls.get_mut(item_id) {
                                arguments.push_str(delta);
                            }
                        }
                        "response.function_call_arguments.done" => {
                            let item_id = event["item_id"].as_str().unwrap_or_default();
                            if let Some((_, _, arguments)) = pending_calls.get_mut(item_id) {
                                *arguments = event["arguments"].as_str().unwrap_or_default().to_string();
                            }
                        }
                        "response.output_item.done" => {
                            if event
                                .get("item")
                                .and_then(|item| item.get("type"))
                                .and_then(|value| value.as_str())
                                == Some("function_call")
                            {
                                let item = &event["item"];
                                let item_id = item["id"].as_str().unwrap_or_default();
                                let (call_id, name, stored_arguments) = pending_calls
                                    .remove(item_id)
                                    .unwrap_or_else(|| {
                                        (
                                            item["call_id"].as_str().unwrap_or_default().to_string(),
                                            item["name"].as_str().unwrap_or_default().to_string(),
                                            String::new(),
                                        )
                                    });
                                let arguments = item["arguments"]
                                    .as_str()
                                    .filter(|value| !value.is_empty())
                                    .unwrap_or(&stored_arguments);

                                yield ChatChunk {
                                    content: String::new(),
                                    tool_calls: Some(vec![ToolCall {
                                        id: Some(call_id),
                                        name,
                                        arguments: Self::parse_tool_arguments(arguments),
                                    }]),
                                    done: false,
                                    done_reason: None,
                                    thinking: None,
                                    tool_use: None,
                                    tool_result: None,
                                };
                            }
                        }
                        "response.completed" => {
                            yield ChatChunk {
                                content: String::new(),
                                tool_calls: None,
                                done: true,
                                done_reason: Some("stop".to_string()),
                                thinking: None,
                                tool_use: None,
                                tool_result: None,
                            };
                            return;
                        }
                        "response.failed" => {
                            let message = event
                                .get("response")
                                .and_then(|response| response.get("error"))
                                .and_then(|error| error.get("message"))
                                .and_then(|value| value.as_str())
                                .unwrap_or("OpenAI Codex response failed");
                            Err(anyhow::anyhow!(message.to_string()))?;
                        }
                        _ => {}
                    }
                }
            }

            Err(anyhow::anyhow!("OpenAI Codex stream ended before response.completed"))?;
        })
    }
}

#[cfg(test)]
mod openai_codex_tests {
    use super::*;

    fn sample_tool() -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        }
    }

    #[test]
    fn convert_messages_preserves_function_call_history() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "System prompt".to_string(),
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: "Open README".to_string(),
                tool_call_id: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "__tool_use__:{\"id\":\"call_123\",\"name\":\"read_file\",\"input\":{\"path\":\"/tmp/README.md\"}}".to_string(),
                tool_call_id: None,
            },
            Message {
                role: "tool".to_string(),
                content: "# Heading".to_string(),
                tool_call_id: Some("call_123".to_string()),
            },
        ];

        let (instructions, input) = OpenAICodexClient::convert_messages(messages).unwrap();

        assert_eq!(instructions, "System prompt");
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_123");
        assert_eq!(input[1]["name"], "read_file");
        assert_eq!(input[1]["arguments"], "{\"path\":\"/tmp/README.md\"}");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_123");
        assert_eq!(input[2]["output"], "# Heading");
    }

    #[test]
    fn build_request_body_adds_tool_instruction_when_tools_are_present() {
        let client = OpenAICodexClient::new(
            Some("https://chatgpt.com/backend-api/codex".to_string()),
            Some("~/.codex/auth.json".to_string()),
            Some("0.120.0".to_string()),
            "gpt-5.4".to_string(),
        );

        let body = client
            .build_request_body(
                vec![Message {
                    role: "user".to_string(),
                    content: "Read the file".to_string(),
                    tool_call_id: None,
                }],
                Some(vec![sample_tool()]),
            )
            .unwrap();

        let instructions = body["instructions"].as_str().unwrap_or_default();
        assert!(instructions.contains("IMPORTANT: The user's project directory is:"));
        assert_eq!(body["tools"].as_array().map(|v| v.len()), Some(1));
    }

    #[test]
    fn parse_sse_response_reads_text_output() {
        let body = r#"
event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","status":"in_progress","content":[],"role":"assistant"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"hello"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":" world"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}
"#;

        let response = OpenAICodexClient::parse_sse_response(body).unwrap();
        assert_eq!(response.message.content, "hello world");
        assert!(response.tool_calls.is_none());
    }

    #[test]
    fn parse_sse_response_reads_function_calls() {
        let body = r#"
event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_abc","name":"read_file"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","delta":"{\"path\":\"/tmp/README.md\"}","item_id":"fc_1"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","arguments":"{\"path\":\"/tmp/README.md\"}","item_id":"fc_1"}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\"path\":\"/tmp/README.md\"}","call_id":"call_abc","name":"read_file"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}
"#;

        let response = OpenAICodexClient::parse_sse_response(body).unwrap();
        let tool_calls = response.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_abc"));
        assert_eq!(tool_calls[0].name, "read_file");
        assert_eq!(tool_calls[0].arguments["path"], "/tmp/README.md");
    }
}

// ---------------------------------------------------------------------------
// Anthropic-compatible client (works with z.ai and any Anthropic-format API)
// ---------------------------------------------------------------------------

pub struct AnthropicClient {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl AnthropicClient {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
            model,
        }
    }

    /// Convert internal messages to Anthropic format.
    /// Anthropic uses a separate `system` top-level field instead of a system message.
    fn convert_messages(messages: Vec<Message>) -> (Option<String>, Vec<serde_json::Value>) {
        let mut system_prompt = None;
        let mut converted = Vec::new();

        for m in messages {
            if m.role == "system" {
                system_prompt = Some(m.content);
            } else if m.role == "tool" {
                // Tool result → Anthropic tool_result content block
                converted.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.unwrap_or_default(),
                        "content": m.content
                    }]
                }));
            } else if m.role == "assistant" {
                // If this is a __tool_use__ marker, extract and convert
                if m.content.starts_with("__tool_use__:") {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(
                        &m.content["__tool_use__:".len()..],
                    ) {
                        converted.push(json!({
                            "role": "assistant",
                            "content": [{
                                "type": "tool_use",
                                "id": val["id"],
                                "name": val["name"],
                                "input": val["input"]
                            }]
                        }));
                    }
                } else {
                    converted.push(json!({
                        "role": "assistant",
                        "content": m.content
                    }));
                }
            } else {
                converted.push(json!({
                    "role": m.role,
                    "content": m.content
                }));
            }
        }

        (system_prompt, converted)
    }

    /// Convert internal tool definitions to Anthropic format.
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                })
            })
            .collect()
    }
}

impl LLMClient for AnthropicClient {
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
            let (system_prompt, anthropic_messages) = Self::convert_messages(messages);

            let mut body = json!({
                "model": model,
                "max_tokens": 4096,
                "messages": anthropic_messages,
            });

            if let Some(sys) = system_prompt {
                body["system"] = json!(sys);
            }

            if let Some(ref tools) = tools {
                if !tools.is_empty() {
                    body["tools"] = serde_json::Value::Array(Self::convert_tools(tools));
                }
            }

            debug!("[ANTHROPIC] Sending chat request to: {}", base_url);

            let response = client
                .post(format!("{}/v1/messages", base_url))
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                anyhow::bail!("Anthropic API error: {} - {}", status, error_text);
            }

            let resp: serde_json::Value = response.json().await?;

            // Parse content blocks
            let empty_blocks = vec![];
            let content_blocks = resp["content"].as_array().unwrap_or(&empty_blocks);
            let mut text_content = String::new();
            let mut tool_calls = Vec::new();

            for block in content_blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        text_content.push_str(block["text"].as_str().unwrap_or(""));
                    }
                    Some("tool_use") => {
                        tool_calls.push(ToolCall {
                            id: block["id"].as_str().map(|s| s.to_string()),
                            name: block["name"].as_str().unwrap_or_default().to_string(),
                            arguments: block["input"].clone(),
                        });
                    }
                    _ => {}
                }
            }

            let thinking = resp
                .get("thinking")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            Ok(ChatResponse {
                message: Message {
                    role: "assistant".to_string(),
                    content: text_content,
                    tool_call_id: None,
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                done: true,
                thinking,
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
            let (system_prompt, anthropic_messages) = Self::convert_messages(messages);

            let mut body = json!({
                "model": model,
                "max_tokens": 4096,
                "messages": anthropic_messages,
                "stream": true,
            });

            if let Some(sys) = system_prompt {
                body["system"] = json!(sys);
            }

            if let Some(ref tools) = tools {
                if !tools.is_empty() {
                    body["tools"] = serde_json::Value::Array(Self::convert_tools(tools));
                }
            }

            debug!("[ANTHROPIC] Sending streaming chat request");

            let response = client
                .post(format!("{}/v1/messages", base_url))
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    error!("[ANTHROPIC] Request error: {}", e);
                    anyhow::anyhow!("Request error: {}", e)
                })?;

            let status = response.status();
            if !status.is_success() {
                error!("[ANTHROPIC] API error: {}", status);
                Err(anyhow::anyhow!("Anthropic API error: {}", status))?;
            }

            info!("[ANTHROPIC] Streaming response started");
            let mut stream = response.bytes_stream();
            let mut bytes_received = 0;

            // Track tool_use blocks: (id, name) from content_block_start, accumulated JSON input
            let mut tool_blocks: HashMap<usize, (String, String, String)> = HashMap::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk_bytes = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        error!("[ANTHROPIC] Stream read error: {}", e);
                        Err(anyhow::anyhow!("Stream error: {}", e))?;
                        unreachable!()
                    }
                };

                bytes_received += chunk_bytes.len();
                debug!("[ANTHROPIC] Received {} bytes (total: {})", chunk_bytes.len(), bytes_received);

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

                    match serde_json::from_str::<serde_json::Value>(data) {
                        Ok(event) => {
                            let event_type = event["type"].as_str().unwrap_or("");

                            match event_type {
                                "content_block_delta" => {
                                    let delta = &event["delta"];
                                    let delta_type = delta["type"].as_str().unwrap_or("");

                                    match delta_type {
                                        "text_delta" => {
                                            let text = delta["text"].as_str().unwrap_or("");
                                            if !text.is_empty() {
                                                yield ChatChunk {
                                                    content: text.to_string(),
                                                    tool_calls: None,
                                                    done: false,
                                                    done_reason: None,
                                                    thinking: None,
                                                    tool_use: None,
                                                    tool_result: None,
                                                };
                                            }
                                        }
                                        "input_json_delta" => {
                                            let index = event["index"].as_u64().unwrap_or(0) as usize;
                                            let partial = delta["partial_json"].as_str().unwrap_or("");
                                            tool_blocks.entry(index)
                                                .and_modify(|(_, _, buf)| buf.push_str(partial));
                                        }
                                        "thinking_delta" => {
                                            let thinking = delta["thinking"].as_str().unwrap_or("");
                                            if !thinking.is_empty() {
                                                yield ChatChunk {
                                                    content: String::new(),
                                                    tool_calls: None,
                                                    done: false,
                                                    done_reason: None,
                                                    thinking: Some(thinking.to_string()),
                                                    tool_use: None,
                                                    tool_result: None,
                                                };
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                "content_block_start" => {
                                    let content_block = &event["content_block"];
                                    if content_block["type"].as_str() == Some("tool_use") {
                                        let index = event["index"].as_u64().unwrap_or(0) as usize;
                                        let id = content_block["id"].as_str().unwrap_or_default().to_string();
                                        let name = content_block["name"].as_str().unwrap_or_default().to_string();
                                        tool_blocks.insert(index, (id, name, String::new()));
                                    }
                                }
                                "message_delta" => {
                                    let stop_reason = event["delta"]["stop_reason"].as_str();

                                    // On tool_use stop, emit all accumulated tool calls
                                    if stop_reason == Some("tool_use") && !tool_blocks.is_empty() {
                                        let calls: Vec<ToolCall> = tool_blocks.iter()
                                            .map(|(_, (id, name, buf))| {
                                                let arguments = serde_json::from_str::<serde_json::Value>(buf)
                                                    .unwrap_or(json!({}));
                                                ToolCall {
                                                    id: Some(id.clone()),
                                                    name: name.clone(),
                                                    arguments,
                                                }
                                            })
                                            .collect();
                                        yield ChatChunk {
                                            content: String::new(),
                                            tool_calls: Some(calls),
                                            done: false,
                                            done_reason: None,
                                            thinking: None,
                                            tool_use: None,
                                            tool_result: None,
                                        };
                                    }

                                    if stop_reason.is_some() {
                                        info!("[ANTHROPIC] Received stop_reason: {:?}", stop_reason);
                                        yield ChatChunk {
                                            content: String::new(),
                                            tool_calls: None,
                                            done: true,
                                            done_reason: stop_reason.map(|s| s.to_string()),
                                            thinking: None,
                                            tool_use: None,
                                            tool_result: None,
                                        };
                                        return;
                                    }
                                }
                                "message_stop" => {
                                    info!("[ANTHROPIC] Received message_stop");
                                    yield ChatChunk {
                                        content: String::new(),
                                        tool_calls: None,
                                        done: true,
                                        done_reason: Some("stop".to_string()),
                                        thinking: None,
                                        tool_use: None,
                                        tool_result: None,
                                    };
                                    return;
                                }
                                "ping" | "message_start" | "content_block_stop" => {}
                                _ => {
                                    debug!("[ANTHROPIC] Unknown event type: {}", event_type);
                                }
                            }
                        }
                        Err(_) => continue,
                    }
                }
            }

            info!("[ANTHROPIC] Stream ended normally, total bytes: {}", bytes_received);
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
            thinking: None,
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
                                thinking: None,
                                tool_use: None,
                                tool_result: None,
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
                thinking: None,
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
    extra_system_prompt: Option<String>,
}

impl Agent {
    pub fn new(llm: Box<dyn LLMClient>) -> Self {
        Self {
            llm,
            tool_registry: ToolRegistry::new(),
            mcp_registry: None,
            max_iterations: 10,
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
                 {}",
                catalog
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
                for call in &tool_calls {
                    let formatted_args = serde_json::to_string_pretty(&call.arguments)
                        .unwrap_or_else(|_| format!("{:?}", call.arguments));
                    println!(
                        "\n{} {}",
                        "⚙️  Action:".bold().yellow(),
                        call.name.bold().white()
                    );
                    println!("{}", formatted_args.dimmed());

                    let call_key = format!("{}:{}", call.name, call.arguments);
                    let count = tool_history.entry(call_key.clone()).or_insert(0);
                    *count += 1;
                    debug!("Tool '{}' call count: {}", call.name, count);

                    if *count > 2 {
                        println!(
                            "{} {} {}",
                            "⚠️  Warning:".bold().yellow(),
                            call.name.bold(),
                            "called multiple times. Forcing a different approach.".dimmed()
                        );
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: format!(
                                "__tool_use__:{}",
                                serde_json::to_string(&json!({
                                    "id": call.id,
                                    "name": call.name,
                                    "input": call.arguments
                                }))
                                .unwrap_or_default()
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

                        let tool_spinner = create_spinner(
                            &format!("Executing {}...", call.name).yellow().to_string(),
                        );
                        let result = tool.execute(args).await;
                        tool_spinner.finish_and_clear();
                        let result = result?;

                        if result.success {
                            println!(
                                "{} {} returned {} characters.",
                                "✅ Success:".bold().green(),
                                call.name.bold(),
                                result.output.len()
                            );
                        } else {
                            if let Some(err) = &result.error {
                                println!(
                                    "{} {} failed: {}",
                                    "❌ Error:".bold().red(),
                                    call.name.bold(),
                                    err
                                );
                            } else {
                                println!(
                                    "{} {} failed.",
                                    "❌ Error:".bold().red(),
                                    call.name.bold()
                                );
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
                                }))
                                .unwrap_or_default()
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

                        let tool_spinner = create_spinner(
                            &format!("Executing MCP tool {}...", mcp_tool_name)
                                .yellow()
                                .to_string(),
                        );
                        let mcp_result = mcp.call_tool(&mcp_tool_name, args).await;
                        tool_spinner.finish_and_clear();

                        match mcp_result {
                            Ok(result) => {
                                println!(
                                    "{} {} returned {} characters.",
                                    "✅ Success:".bold().green(),
                                    mcp_tool_name.bold(),
                                    result.len()
                                );
                                messages.push(Message {
                                    role: "assistant".to_string(),
                                    content: format!(
                                        "__tool_use__:{}",
                                        serde_json::to_string(&json!({
                                            "id": call.id,
                                            "name": call.name,
                                            "input": call.arguments
                                        }))
                                        .unwrap_or_default()
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
                                println!(
                                    "{} {} failed: {}",
                                    "❌ Error:".bold().red(),
                                    mcp_tool_name.bold(),
                                    e
                                );
                                messages.push(Message {
                                    role: "assistant".to_string(),
                                    content: format!(
                                        "__tool_use__:{}",
                                        serde_json::to_string(&json!({
                                            "id": call.id,
                                            "name": call.name,
                                            "input": call.arguments
                                        }))
                                        .unwrap_or_default()
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
                        println!(
                            "{} {} not found.",
                            "❌ Error:".bold().red(),
                            call.name.bold()
                        );
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: format!(
                                "__tool_use__:{}",
                                serde_json::to_string(&json!({
                                    "id": call.id,
                                    "name": call.name,
                                    "input": call.arguments
                                }))
                                .unwrap_or_default()
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
                 {}",
                catalog
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

                info!("[STREAM] Chunk #{}: content_len={}, thinking={}, tool_calls={:?}, done={}, done_reason={:?}",
                    chunk_count, chunk.content.len(), chunk.thinking.is_some(), chunk.tool_calls.is_some(), chunk.done, chunk.done_reason);

                // Display thinking tokens (dimmed) before content
                if let Some(ref thinking) = chunk.thinking {
                    if self.show_thinking && !thinking.is_empty() {
                        print!("{}", thinking.dimmed());
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                }

                // Display tool execution events in real-time
                if let Some(ref tu) = chunk.tool_use {
                    let args_preview = serde_json::to_string(&tu.input).unwrap_or_default();
                    let args_short = if args_preview.len() > 120 {
                        format!("{}...", &args_preview[..120])
                    } else {
                        args_preview
                    };
                    println!(
                        "\n{} {} {}",
                        Self::icon("exec"),
                        Self::tool_name(&tu.name),
                        args_short.dimmed()
                    );
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }

                if let Some(ref tr) = chunk.tool_result {
                    let preview = if tr.content.len() > 150 {
                        format!("{}...", &tr.content[..150])
                    } else {
                        tr.content.clone()
                    };
                    let first_line = preview.lines().next().unwrap_or(&preview);
                    if tr.is_error {
                        println!(
                            "{} {} {}",
                            Self::icon("error"),
                            format!("{} failed:", tr.name).bold().red(),
                            first_line.dimmed()
                        );
                    } else {
                        println!(
                            "{} {} {}",
                            Self::icon("success"),
                            format!("{} →", tr.name).bold().green(),
                            first_line.dimmed()
                        );
                    }
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }

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

            // Extract text-based tool calls from accumulated content (for providers
            // that don't support native function calling, e.g. OpenAI wrapper)
            if final_tool_calls.is_none() && accumulated_content.contains("<tool_call>") {
                let (clean, text_calls) = OpenAIClient::extract_tool_calls(&accumulated_content);
                if text_calls.is_some() {
                    accumulated_content = clean;
                    final_tool_calls = text_calls;
                    info!(
                        "[STREAM] Extracted text-based tool calls: {:?}",
                        final_tool_calls
                    );
                }
            }

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
                                    }))
                                    .unwrap_or_default()
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

                            let tool_spinner = create_spinner(
                                &format!("Executing {}...", call.name).yellow().to_string(),
                            );
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
                                    }))
                                    .unwrap_or_default()
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
                                            }))
                                            .unwrap_or_default()
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

                            let tool_spinner = create_spinner(
                                &format!("Executing MCP tool {}...", mcp_tool_name)
                                    .yellow()
                                    .to_string(),
                            );
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
                                            }))
                                            .unwrap_or_default()
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
                                            }))
                                            .unwrap_or_default()
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
