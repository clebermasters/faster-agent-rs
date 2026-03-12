use anyhow::{Context, Result};
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, Message as BedrockMessage, SystemContentBlock,
    Tool, ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock,
    ToolSpecification, ToolUseBlock,
};
use aws_smithy_types::Document;
use serde_json::Value;
use tracing::warn;

use crate::{ChatResponse, Message, ToolCall};
use skill_tools::ToolDefinition;

use super::models::ModelCapabilities;

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Split our internal `Vec<Message>` into the two parameters expected by the
/// Bedrock Converse API:
///
/// - `system_blocks`: fed into the `system` field (only when
///   `caps.system_prompt_native` is true; otherwise injected as a user msg).
/// - `converse_messages`: the `messages` list.
pub fn split_messages(
    messages: Vec<Message>,
    caps: &ModelCapabilities,
) -> (Vec<SystemContentBlock>, Vec<BedrockMessage>) {
    let mut system_blocks: Vec<SystemContentBlock> = Vec::new();
    let mut converse_messages: Vec<BedrockMessage> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if caps.system_prompt_native {
                    system_blocks.push(SystemContentBlock::Text(msg.content));
                } else {
                    // Inject as the first user message with a clear marker so
                    // the model still receives the system instructions.
                    let injected = format!("[System Instructions]\n{}", msg.content);
                    converse_messages.push(
                        BedrockMessage::builder()
                            .role(ConversationRole::User)
                            .content(ContentBlock::Text(injected))
                            .build()
                            .expect("BedrockMessage builder: role + content are set"),
                    );
                }
            }

            "tool" => {
                // Tool results must become a user message that contains a
                // `toolResult` content block referencing the original tool-use ID.
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                let result_block = ToolResultBlock::builder()
                    .tool_use_id(&tool_use_id)
                    .content(ToolResultContentBlock::Text(msg.content))
                    .build()
                    .expect("ToolResultBlock builder");

                converse_messages.push(
                    BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::ToolResult(result_block))
                        .build()
                        .expect("BedrockMessage builder"),
                );
            }

            "user" => {
                converse_messages.push(
                    BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(msg.content))
                        .build()
                        .expect("BedrockMessage builder"),
                );
            }

            "assistant" => {
                converse_messages.push(
                    BedrockMessage::builder()
                        .role(ConversationRole::Assistant)
                        .content(ContentBlock::Text(msg.content))
                        .build()
                        .expect("BedrockMessage builder"),
                );
            }

            other => {
                warn!("Bedrock convert: unknown message role '{}' — skipping", other);
            }
        }
    }

    (system_blocks, converse_messages)
}

// ---------------------------------------------------------------------------
// Tool-definition conversion
// ---------------------------------------------------------------------------

/// Convert our `ToolDefinition` list into a Bedrock `ToolConfiguration`.
///
/// Bedrock toolSpec uses `inputSchema.json` (a `Document`) instead of
/// OpenAI's flat `parameters` object.
pub fn convert_tools(tools: Vec<ToolDefinition>) -> ToolConfiguration {
    let bedrock_tools: Vec<Tool> = tools
        .into_iter()
        .map(|t| {
            let schema_doc = json_value_to_document(t.parameters);
            let spec = ToolSpecification::builder()
                .name(&t.name)
                .description(&t.description)
                .input_schema(ToolInputSchema::Json(schema_doc))
                .build()
                .expect("ToolSpecification builder");
            Tool::ToolSpec(spec)
        })
        .collect();

    ToolConfiguration::builder()
        .set_tools(Some(bedrock_tools))
        .build()
        .expect("ToolConfiguration builder")
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/// Parse a non-streaming Converse response into our `ChatResponse`.
pub fn parse_converse_output(
    output: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
) -> Result<ChatResponse> {
    let output_msg = output
        .output()
        .context("Bedrock Converse: missing output")?
        .as_message()
        .map_err(|_| anyhow::anyhow!("Bedrock Converse: output is not a message"))?;

    let mut text_content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in output_msg.content() {
        match block {
            ContentBlock::Text(t) => text_content.push_str(t),
            ContentBlock::ToolUse(tu) => {
                if let Some(tc) = parse_tool_use_block(tu) {
                    tool_calls.push(tc);
                }
            }
            _ => {} // ignore images, documents, etc.
        }
    }

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
    })
}

/// Convert a single `ToolUseBlock` into our `ToolCall`.
pub fn parse_tool_use_block(tu: &ToolUseBlock) -> Option<ToolCall> {
    let args = document_to_json_value(tu.input().clone());

    Some(ToolCall {
        id: Some(tu.tool_use_id().to_string()),
        name: tu.name().to_string(),
        arguments: args,
    })
}

// ---------------------------------------------------------------------------
// aws_smithy_types::Document ↔ serde_json::Value helpers
// ---------------------------------------------------------------------------

/// Recursively convert a `serde_json::Value` into an AWS `Document`.
pub fn json_value_to_document(value: Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Document::Number(aws_smithy_types::Number::NegInt(i))
            } else if let Some(u) = n.as_u64() {
                Document::Number(aws_smithy_types::Number::PosInt(u))
            } else {
                Document::Number(aws_smithy_types::Number::Float(
                    n.as_f64().unwrap_or(0.0),
                ))
            }
        }
        Value::String(s) => Document::String(s),
        Value::Array(arr) => {
            Document::Array(arr.into_iter().map(json_value_to_document).collect())
        }
        Value::Object(map) => Document::Object(
            map.into_iter()
                .map(|(k, v)| (k, json_value_to_document(v)))
                .collect(),
        ),
    }
}

/// Recursively convert an AWS `Document` into a `serde_json::Value`.
pub fn document_to_json_value(doc: Document) -> Value {
    match doc {
        Document::Null => Value::Null,
        Document::Bool(b) => Value::Bool(b),
        Document::Number(n) => match n {
            aws_smithy_types::Number::PosInt(u) => Value::Number(u.into()),
            aws_smithy_types::Number::NegInt(i) => Value::Number(i.into()),
            aws_smithy_types::Number::Float(f) => {
                serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            }
        },
        Document::String(s) => Value::String(s),
        Document::Array(arr) => {
            Value::Array(arr.into_iter().map(document_to_json_value).collect())
        }
        Document::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, document_to_json_value(v)))
                .collect(),
        ),
    }
}
