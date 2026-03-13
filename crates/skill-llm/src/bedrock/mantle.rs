use anyhow::Result;
use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::json;
use std::pin::Pin;
use tracing::{debug, error, info};

use crate::{ChatChunk, ChatResponse, LLMClient, Message, ToolCall};
use skill_tools::ToolDefinition;

// ---------------------------------------------------------------------------
// BedrockMantleClient
// ---------------------------------------------------------------------------

/// LLM client that talks to Amazon Bedrock via the **bedrock-mantle**
/// OpenAI-compatible endpoint, authenticated with a Bedrock API Key.
///
/// Endpoint: `https://bedrock-mantle.{region}.api.aws/v1`
///
/// This is the preferred path for MiniMax M2.1 and ZAI GLM-4.7 because the
/// mantle endpoint natively supports:
/// - System prompts (standard `{"role": "system", ...}` messages)
/// - Tool / function calling in OpenAI format
/// - Streaming via SSE `data:` lines
///
/// For models that work fine with the Converse API (Claude, Nova) you can
/// also use this client — it is functionally equivalent but uses Bearer-token
/// auth instead of AWS Signature V4.
pub struct BedrockMantleClient {
    client: Client,
    /// `https://bedrock-mantle.{region}.api.aws/v1`
    base_url: String,
    api_key: String,
    model_id: String,
}

impl BedrockMantleClient {
    pub fn new(api_key: String, region: String, model_id: String) -> Self {
        let base_url = format!("https://bedrock-mantle.{}.api.aws/v1", region);
        // AWS CLI / SDK tools expose the key as  ABSK<base64(BedrockAPIKey-...:...)>
        // The mantle endpoint expects the decoded inner value as the Bearer token.
        let resolved_key = decode_absk_token(&api_key);
        info!(
            "BedrockMantleClient ready: model={} endpoint={}",
            model_id, base_url
        );
        Self {
            client: Client::new(),
            base_url,
            api_key: resolved_key,
            model_id,
        }
    }
}

/// If `token` starts with `ABSK`, base64-decode the remainder to recover the
/// actual `BedrockAPIKey-{id}:{secret}` string that the mantle endpoint accepts.
/// Any other format is returned unchanged.
fn decode_absk_token(token: &str) -> String {
    if let Some(b64) = token.strip_prefix("ABSK") {
        // Use standard base64 alphabet (the suffix is standard base64)
        if let Ok(bytes) = base64_decode(b64) {
            if let Ok(s) = String::from_utf8(bytes) {
                return s;
            }
        }
    }
    token.to_string()
}

/// Minimal base64 decoder (standard alphabet) without an extra dependency.
/// Falls back gracefully on any decode error.
fn base64_decode(input: &str) -> Result<Vec<u8>, ()> {
    // Pad to a multiple of 4
    let padded = match input.len() % 4 {
        2 => format!("{}==", input),
        3 => format!("{}=", input),
        _ => input.to_string(),
    };

    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lookup = [0u8; 256];
    for (i, &c) in TABLE.iter().enumerate() {
        lookup[c as usize] = i as u8;
    }

    let bytes = padded.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(());
    }

    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        if chunk.iter().any(|&b| b != b'=' && !TABLE.contains(&b)) {
            return Err(());
        }
        let b0 = lookup[chunk[0] as usize];
        let b1 = lookup[chunk[1] as usize];
        let b2 = if chunk[2] == b'=' {
            0
        } else {
            lookup[chunk[2] as usize]
        };
        let b3 = if chunk[3] == b'=' {
            0
        } else {
            lookup[chunk[3] as usize]
        };
        out.push((b0 << 2) | (b1 >> 4));
        if chunk[2] != b'=' {
            out.push((b1 << 4) | (b2 >> 2));
        }
        if chunk[3] != b'=' {
            out.push((b2 << 6) | b3);
        }
    }
    Ok(out)
}

impl LLMClient for BedrockMantleClient {
    // -----------------------------------------------------------------------
    // Non-streaming
    // -----------------------------------------------------------------------
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model_id.clone();

        Box::pin(async move {
            let openai_messages = convert_messages_to_openai(&messages);

            let mut body = json!({
                "model": model,
                "messages": openai_messages,
                "stream": false,
            });

            if let Some(tools) = tools {
                body["tools"] = serde_json::Value::Array(convert_tools_to_openai(tools));
            }

            debug!("BedrockMantle request body: {}", body);

            let response = client
                .post(format!("{}/chat/completions", base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                anyhow::bail!("Bedrock Mantle API error: {} — {}", status, text);
            }

            let chat_resp: serde_json::Value = response.json().await?;
            parse_openai_response(chat_resp)
        })
    }

    // -----------------------------------------------------------------------
    // Streaming
    // -----------------------------------------------------------------------
    fn chat_streaming(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send + '_>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model_id.clone();

        Box::pin(try_stream! {
            let openai_messages = convert_messages_to_openai(&messages);

            let mut body = json!({
                "model": model,
                "messages": openai_messages,
                "stream": true,
            });

            if let Some(tools) = tools {
                body["tools"] = serde_json::Value::Array(convert_tools_to_openai(tools));
            }

            let response = client
                .post(format!("{}/chat/completions", base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Bedrock Mantle request error: {}", e))?;

            let status = response.status();
            if !status.is_success() {
                error!("Bedrock Mantle API error: {}", status);
                Err(anyhow::anyhow!("Bedrock Mantle API error: {}", status))?;
            }

            info!("Bedrock Mantle streaming response started");
            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                let chunk_bytes = chunk_result
                    .map_err(|e| anyhow::anyhow!("Bedrock Mantle stream error: {}", e))?;

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
                        info!("Bedrock Mantle stream: [DONE]");
                        yield ChatChunk {
                            content: String::new(),
                            tool_calls: None,
                            done: true,
                            done_reason: Some("stop".to_string()),
                        };
                        return;
                    }

                    match serde_json::from_str::<serde_json::Value>(data) {
                        Ok(chunk_json) => {
                            let delta = chunk_json
                                .get("choices")
                                .and_then(|c| c.as_array())
                                .and_then(|c| c.first())
                                .and_then(|c| c.get("delta"));

                            let content = delta
                                .and_then(|d| d.get("content"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();

                            let tool_calls =
                                delta.and_then(|d| d.get("tool_calls")).and_then(|tc| {
                                    let calls: Vec<ToolCall> = tc
                                        .as_array()
                                        .unwrap_or(&vec![])
                                        .iter()
                                        .filter_map(|c| {
                                            let func = c.get("function")?;
                                            let args_str = func
                                                .get("arguments")
                                                .and_then(|a| a.as_str())
                                                .unwrap_or("");
                                            let name = func["name"]
                                                .as_str()
                                                .map(|s| s.to_string())
                                                .unwrap_or_default();
                                            let id = c
                                                .get("id")
                                                .and_then(|v| v.as_str())
                                                .map(|s| s.to_string());
                                            Some(ToolCall {
                                                id,
                                                name,
                                                arguments: serde_json::Value::String(
                                                    args_str.to_string(),
                                                ),
                                            })
                                        })
                                        .collect();
                                    if calls.is_empty() { None } else { Some(calls) }
                                });

                            let finish_reason = chunk_json
                                .get("choices")
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
                            };
                        }
                        Err(_) => continue,
                    }
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert our internal messages to the OpenAI `messages` array format.
/// The mantle endpoint supports `system` natively so we pass it through as-is.
fn convert_messages_to_openai(messages: &[Message]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            if m.role == "tool" {
                // OpenAI tool result format
                json!({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id.clone().unwrap_or_default(),
                    "content": m.content
                })
            } else {
                json!({
                    "role": m.role,
                    "content": m.content
                })
            }
        })
        .collect()
}

/// Convert our `ToolDefinition` list to OpenAI function-calling format.
fn convert_tools_to_openai(tools: Vec<ToolDefinition>) -> Vec<serde_json::Value> {
    tools
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
        .collect()
}

/// Parse a non-streaming OpenAI-format response into our `ChatResponse`.
fn parse_openai_response(chat_resp: serde_json::Value) -> Result<ChatResponse> {
    let message = &chat_resp["choices"][0]["message"];
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
}
