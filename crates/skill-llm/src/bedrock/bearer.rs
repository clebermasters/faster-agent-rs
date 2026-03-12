use anyhow::Result;
use async_stream::try_stream;
use futures::Stream;
use reqwest::Client;
use serde_json::{json, Value};
use std::pin::Pin;
use tracing::{debug, info, warn};

use crate::{ChatChunk, ChatResponse, LLMClient, Message, ToolCall};
use skill_tools::ToolDefinition;

use super::models::{ModelCapabilities, ToolUseSupport};

// ---------------------------------------------------------------------------
// BedrockBearerClient
// ---------------------------------------------------------------------------

/// LLM client that talks to the **standard Amazon Bedrock Runtime**
/// (`bedrock-runtime.{region}.amazonaws.com`) using a Bedrock API Key
/// passed as an HTTP Bearer token — no AWS Signature V4 required.
///
/// This is the path used when `BEDROCK_AUTH=api-key` (or `--bedrock-auth
/// api-key`) and the supplied token is in the ABSK format:
///
/// ```text
/// ABSK<base64(BedrockAPIKey-{id}:{secret})>
/// ```
///
/// The ABSK token is passed **as-is** — the `bedrock-runtime` endpoint
/// accepts the encoded form directly in the `Authorization: Bearer` header.
///
/// Endpoint used: `POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/converse`
///
/// Per-model quirks (system-prompt injection, `topK` via
/// `additionalModelRequestFields`, max-output caps) are handled via the
/// same [`ModelCapabilities`] registry used by `BedrockConverseClient`.
pub struct BedrockBearerClient {
    client: Client,
    region: String,
    /// Raw ABSK token — passed as the Bearer value without modification.
    token: String,
    model_id: String,
    caps: ModelCapabilities,
}

impl BedrockBearerClient {
    pub fn new(token: String, region: String, model_id: String) -> Self {
        let caps = ModelCapabilities::for_model(&model_id);
        info!(
            "BedrockBearerClient ready: model={} region={} system_native={} tool_use={:?}",
            model_id, region, caps.system_prompt_native, caps.tool_use
        );
        // Force HTTP/1.1: the bedrock-runtime bearer-token endpoint appears to
        // reject HTTP/2 requests with IncompleteSignatureException. HTTP/1.1
        // works correctly (verified with curl and Python requests).
        let client = Client::builder()
            .http1_only()
            .build()
            .expect("reqwest Client builder");
        Self {
            client,
            region,
            token,
            model_id,
            caps,
        }
    }

    fn converse_url(&self) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse",
            self.region, self.model_id
        )
    }

    fn build_body(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Value {
        let (system_parts, converse_messages) = split_messages_json(&messages, &self.caps);

        let mut body = json!({ "messages": converse_messages });

        if !system_parts.is_empty() {
            body["system"] = Value::Array(system_parts);
        }

        // Cap max output tokens where the model requires it
        if let Some(max_tokens) = self.caps.max_output_tokens {
            body["inferenceConfig"] = json!({ "maxTokens": max_tokens });
        }

        // Nova: topK inside additionalModelRequestFields
        if self.caps.topk_via_additional_fields {
            body["additionalModelRequestFields"] =
                json!({ "inferenceConfig": { "topK": 50 } });
        }

        if let Some(tools) = tools {
            match self.caps.tool_use {
                ToolUseSupport::None => {
                    warn!(
                        "Bedrock: model {} does not support tool use — omitting toolConfig",
                        self.model_id
                    );
                }
                ToolUseSupport::Unreliable => {
                    warn!(
                        "Bedrock: model {} has unreliable tool use — sending toolConfig anyway",
                        self.model_id
                    );
                    body["toolConfig"] = build_tool_config_json(tools);
                }
                _ => {
                    body["toolConfig"] = build_tool_config_json(tools);
                }
            }
        }

        body
    }
}

impl LLMClient for BedrockBearerClient {
    // -----------------------------------------------------------------------
    // Non-streaming
    // -----------------------------------------------------------------------
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>> {
        Box::pin(async move {
            let url = self.converse_url();
            let body = self.build_body(messages, tools);

            debug!("BedrockBearer POST {} body={}", url, body);

            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.token))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("BedrockBearer request error: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                anyhow::bail!("BedrockBearer API error: {} — {}", status, text);
            }

            let resp_json: Value = response.json().await?;
            debug!("BedrockBearer response: {}", resp_json);
            parse_converse_json(&resp_json)
        })
    }

    // -----------------------------------------------------------------------
    // Streaming (non-streaming fallback — emits full response as one chunk)
    // -----------------------------------------------------------------------
    fn chat_streaming(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send + '_>> {
        Box::pin(try_stream! {
            let chat_response = self.chat(messages, tools).await?;

            // Emit text content as one chunk
            if !chat_response.message.content.is_empty() {
                yield ChatChunk {
                    content: chat_response.message.content,
                    tool_calls: None,
                    done: false,
                    done_reason: None,
                };
            }

            // Emit tool calls if present
            if let Some(tool_calls) = chat_response.tool_calls {
                yield ChatChunk {
                    content: String::new(),
                    tool_calls: Some(tool_calls),
                    done: false,
                    done_reason: None,
                };
            }

            // Final done chunk
            yield ChatChunk {
                content: String::new(),
                tool_calls: None,
                done: true,
                done_reason: Some("end_turn".to_string()),
            };
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert our internal `Vec<Message>` to Converse API JSON format.
///
/// Returns `(system_blocks, messages)` — system blocks go into the top-level
/// `"system"` field; everything else becomes a Converse `messages` entry.
fn split_messages_json(
    messages: &[Message],
    caps: &ModelCapabilities,
) -> (Vec<Value>, Vec<Value>) {
    let mut system_parts: Vec<Value> = Vec::new();
    let mut converse_messages: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if caps.system_prompt_native {
                    system_parts.push(json!({ "text": msg.content }));
                } else {
                    // Inject as first user message with a clear marker
                    let injected = format!("[System Instructions]\n{}", msg.content);
                    converse_messages.push(json!({
                        "role": "user",
                        "content": [{ "text": injected }]
                    }));
                }
            }

            "tool" => {
                // For Unreliable / None models that don't support toolResult blocks,
                // encode as plain text so the model can still follow along.
                if matches!(caps.tool_use, ToolUseSupport::Unreliable | ToolUseSupport::None) {
                    converse_messages.push(json!({
                        "role": "user",
                        "content": [{ "text": format!("[Tool Result]: {}", msg.content) }]
                    }));
                } else {
                    let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                    converse_messages.push(json!({
                        "role": "user",
                        "content": [{
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{ "text": msg.content }]
                            }
                        }]
                    }));
                }
            }

            "user" => {
                converse_messages.push(json!({
                    "role": "user",
                    "content": [{ "text": msg.content }]
                }));
            }

            "assistant" => {
                if let Some(rest) = msg.content.strip_prefix("__tool_use__:") {
                    // Reconstruct the toolUse block that the agent loop encoded as a marker.
                    // Bedrock Converse requires: assistant[toolUse] → user[toolResult].
                    // For Unreliable / None models, fall back to plain text.
                    let use_native_block = matches!(
                        caps.tool_use,
                        ToolUseSupport::Full | ToolUseSupport::ClientSide
                    );

                    match serde_json::from_str::<Value>(rest) {
                        Ok(tu_json) => {
                            if use_native_block {
                                let tool_use_id =
                                    tu_json["id"].as_str().unwrap_or("").to_string();
                                let name =
                                    tu_json["name"].as_str().unwrap_or("").to_string();
                                let input = tu_json["input"].clone();
                                converse_messages.push(json!({
                                    "role": "assistant",
                                    "content": [{
                                        "toolUse": {
                                            "toolUseId": tool_use_id,
                                            "name": name,
                                            "input": input
                                        }
                                    }]
                                }));
                            } else {
                                // Text fallback for Unreliable / None models
                                let name = tu_json["name"].as_str().unwrap_or("unknown");
                                converse_messages.push(json!({
                                    "role": "assistant",
                                    "content": [{ "text": format!(
                                        "[Calling tool: {} with args: {}]",
                                        name, tu_json["input"]
                                    )}]
                                }));
                            }
                        }
                        Err(e) => {
                            warn!(
                                "BedrockBearer: failed to parse __tool_use__ JSON: {} — \
                                 emitting as text",
                                e
                            );
                            converse_messages.push(json!({
                                "role": "assistant",
                                "content": [{ "text": msg.content }]
                            }));
                        }
                    }
                } else {
                    converse_messages.push(json!({
                        "role": "assistant",
                        "content": [{ "text": msg.content }]
                    }));
                }
            }

            other => {
                warn!("BedrockBearer: unknown message role '{}' — skipping", other);
            }
        }
    }

    (system_parts, converse_messages)
}

/// Convert `ToolDefinition` list to the Converse API `toolConfig` JSON object.
fn build_tool_config_json(tools: Vec<ToolDefinition>) -> Value {
    let tool_specs: Vec<Value> = tools
        .into_iter()
        .map(|t| {
            json!({
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": { "json": t.parameters }
                }
            })
        })
        .collect();

    json!({ "tools": tool_specs })
}

/// Parse a Converse API JSON response into our `ChatResponse`.
///
/// Handles `text`, `toolUse`, and silently skips unknown blocks such as
/// `reasoningContent` (emitted by MiniMax M2.1).
fn parse_converse_json(resp: &Value) -> Result<ChatResponse> {
    let content_blocks = resp["output"]["message"]["content"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let mut text_content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in content_blocks {
        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
            text_content.push_str(text);
        } else if let Some(tu) = block.get("toolUse") {
            let id = tu["toolUseId"].as_str().map(|s| s.to_string());
            let name = tu["name"].as_str().unwrap_or("").to_string();
            let arguments = tu["input"].clone();
            tool_calls.push(ToolCall { id, name, arguments });
        }
        // reasoningContent and other blocks are intentionally ignored
    }

    Ok(ChatResponse {
        message: Message {
            role: "assistant".to_string(),
            content: text_content,
            tool_call_id: None,
        },
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        done: true,
    })
}
