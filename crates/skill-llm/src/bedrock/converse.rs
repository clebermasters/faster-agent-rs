use anyhow::{Context, Result};
use async_stream::stream;
use aws_sdk_bedrockruntime::types::{ContentBlockDelta, ConverseStreamOutput};
use futures::Stream;
use serde_json::json;
use std::pin::Pin;
use tracing::{debug, info, warn};

use crate::{ChatChunk, ChatResponse, LLMClient, Message, ToolCall};
use skill_tools::ToolDefinition;

use super::{
    auth::{build_aws_config, BedrockAuth},
    convert::{convert_tools, json_value_to_document, parse_converse_output, split_messages},
    models::{ModelCapabilities, ToolUseSupport},
};

// ---------------------------------------------------------------------------
// BedrockConverseClient
// ---------------------------------------------------------------------------

/// LLM client that talks to Amazon Bedrock via the **Converse / ConverseStream**
/// API using AWS SDK credentials (static IAM, STS token, assumed role, or the
/// default credential chain).
///
/// Handles per-model quirks transparently:
/// - System-prompt injection for MiniMax / ZAI models
/// - `topK` placement in `additionalModelRequestFields` for Nova models
/// - Tool-use warning for ZAI GLM-4.7 (unreliable tool invocation)
pub struct BedrockConverseClient {
    client: aws_sdk_bedrockruntime::Client,
    model_id: String,
    caps: ModelCapabilities,
}

impl BedrockConverseClient {
    pub async fn new(auth: BedrockAuth, region: String, model_id: String) -> Result<Self> {
        let sdk_config = build_aws_config(auth, &region).await?;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);
        let caps = ModelCapabilities::for_model(&model_id);
        info!(
            "BedrockConverseClient ready: model={} system_native={} tool_use={:?}",
            model_id, caps.system_prompt_native, caps.tool_use
        );
        Ok(Self {
            client,
            model_id,
            caps,
        })
    }
}

impl LLMClient for BedrockConverseClient {
    // -----------------------------------------------------------------------
    // Non-streaming
    // -----------------------------------------------------------------------
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>> {
        Box::pin(async move {
            let (system_blocks, converse_messages) = split_messages(messages, &self.caps);

            debug!(
                "Bedrock Converse: model={} system_blocks={} messages={}",
                self.model_id,
                system_blocks.len(),
                converse_messages.len()
            );

            let mut req = self
                .client
                .converse()
                .model_id(&self.model_id)
                .set_messages(Some(converse_messages));

            if !system_blocks.is_empty() {
                req = req.set_system(Some(system_blocks));
            }

            // Nova: topK lives inside additionalModelRequestFields
            if self.caps.topk_via_additional_fields {
                let extra = json_value_to_document(json!({
                    "inferenceConfig": { "topK": 50 }
                }));
                req = req.additional_model_request_fields(extra);
            }

            if let Some(tools) = tools {
                match self.caps.tool_use {
                    ToolUseSupport::None => {
                        warn!(
                            "Model {} does not support tool use — tool definitions omitted",
                            self.model_id
                        );
                    }
                    ToolUseSupport::Unreliable => {
                        warn!(
                            "Model {} has unreliable tool use (may answer directly) — \
                             sending toolConfig anyway",
                            self.model_id
                        );
                        req = req.tool_config(convert_tools(tools));
                    }
                    _ => {
                        req = req.tool_config(convert_tools(tools));
                    }
                }
            }

            let output = req
                .send()
                .await
                .with_context(|| format!("Bedrock Converse failed for model {}", self.model_id))?;

            parse_converse_output(output)
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
        Box::pin(stream! {
            let (system_blocks, converse_messages) = split_messages(messages, &self.caps);

            debug!(
                "Bedrock ConverseStream: model={} system_blocks={} messages={}",
                self.model_id,
                system_blocks.len(),
                converse_messages.len()
            );

            let mut req = self
                .client
                .converse_stream()
                .model_id(&self.model_id)
                .set_messages(Some(converse_messages));

            if !system_blocks.is_empty() {
                req = req.set_system(Some(system_blocks));
            }

            if self.caps.topk_via_additional_fields {
                let extra = json_value_to_document(json!({
                    "inferenceConfig": { "topK": 50 }
                }));
                req = req.additional_model_request_fields(extra);
            }

            if let Some(tools) = tools {
                match self.caps.tool_use {
                    ToolUseSupport::None => {
                        warn!(
                            "Model {} does not support tool use — tool definitions omitted",
                            self.model_id
                        );
                    }
                    ToolUseSupport::Unreliable => {
                        warn!(
                            "Model {} has unreliable tool use — sending toolConfig anyway",
                            self.model_id
                        );
                        req = req.tool_config(convert_tools(tools));
                    }
                    _ => {
                        req = req.tool_config(convert_tools(tools));
                    }
                }
            }

            let response = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!(
                        "Bedrock ConverseStream failed for model {}: {}",
                        self.model_id, e
                    ));
                    return;
                }
            };

            // Access the `stream` field directly to take ownership — calling
            // `.stream()` returns `&mut EventReceiver` which can't be held
            // across yield points inside the generator.
            let mut event_stream = response.stream;

            // Accumulate streaming tool-call fields across content blocks
            let mut tool_id_buf: Option<String> = None;
            let mut tool_name_buf: Option<String> = None;
            let mut tool_input_buf = String::new();

            loop {
                match event_stream.recv().await {
                    Ok(Some(event)) => {
                        match event {
                            // Incremental text or tool-input delta
                            ConverseStreamOutput::ContentBlockDelta(d) => {
                                match d.delta() {
                                    Some(ContentBlockDelta::Text(t)) => {
                                        yield Ok(ChatChunk {
                                            content: t.to_string(),
                                            tool_calls: None,
                                            done: false,
                                            done_reason: None,
                                        });
                                    }
                                    Some(ContentBlockDelta::ToolUse(tu)) => {
                                        tool_input_buf.push_str(tu.input());
                                    }
                                    _ => {}
                                }
                            }

                            // Start of a new content block — capture tool metadata
                            ConverseStreamOutput::ContentBlockStart(s) => {
                                use aws_sdk_bedrockruntime::types::ContentBlockStart;
                                if let Some(ContentBlockStart::ToolUse(tu)) = s.start() {
                                    tool_id_buf = Some(tu.tool_use_id().to_string());
                                    tool_name_buf = Some(tu.name().to_string());
                                    tool_input_buf.clear();
                                }
                            }

                            // End of a content block — emit complete ToolCall if pending
                            ConverseStreamOutput::ContentBlockStop(_) => {
                                if let (Some(id), Some(name)) =
                                    (tool_id_buf.take(), tool_name_buf.take())
                                {
                                    let args: serde_json::Value =
                                        serde_json::from_str(&tool_input_buf)
                                            .unwrap_or(serde_json::Value::Object(
                                                Default::default(),
                                            ));
                                    yield Ok(ChatChunk {
                                        content: String::new(),
                                        tool_calls: Some(vec![ToolCall {
                                            id: Some(id),
                                            name,
                                            arguments: args,
                                        }]),
                                        done: false,
                                        done_reason: None,
                                    });
                                    tool_input_buf.clear();
                                }
                            }

                            // Stream finished
                            ConverseStreamOutput::MessageStop(stop) => {
                                let reason = Some(stop.stop_reason().as_str().to_string());
                                info!("Bedrock ConverseStream done: reason={:?}", reason);
                                yield Ok(ChatChunk {
                                    content: String::new(),
                                    tool_calls: None,
                                    done: true,
                                    done_reason: reason,
                                });
                                return;
                            }

                            // metadata, messageStart — nothing to yield
                            _ => {}
                        }
                    }
                    Ok(None) => return,
                    Err(e) => {
                        yield Err(anyhow::anyhow!("Bedrock stream error: {}", e));
                        return;
                    }
                }
            }
        })
    }
}
