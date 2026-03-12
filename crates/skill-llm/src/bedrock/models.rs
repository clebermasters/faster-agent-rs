/// Per-model capability flags for the Bedrock Converse API.
///
/// Different providers expose different features — this registry lets the
/// client adapt its request/response handling per model without if-chains
/// scattered across the codebase.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// True when the Converse API `system` field is honoured by the model.
    /// False for MiniMax and ZAI/GLM — system prompt is injected as the
    /// first user message with a `[System Instructions]` marker instead.
    pub system_prompt_native: bool,

    /// Tool-use support level for this model via the Converse API.
    pub tool_use: ToolUseSupport,

    /// Amazon Nova models require `topK` inside
    /// `additionalModelRequestFields.inferenceConfig` rather than the
    /// top-level `inferenceConfig` block.
    pub topk_via_additional_fields: bool,

    /// Hard output-token cap imposed by the model (8 192 for MiniMax M2.1,
    /// 4 096 for ZAI GLM-4.7). `None` means "use whatever the caller sets".
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolUseSupport {
    /// Standard Converse `toolConfig` is sent and the model reliably returns
    /// `toolUse` content blocks (Claude, Nova).
    Full,
    /// `toolConfig` is sent and the model returns tool calls correctly, but
    /// AWS Agents server-side execution is not available (MiniMax M2.1).
    /// For our client-driven loop this behaves identically to `Full`.
    ClientSide,
    /// `toolConfig` is sent but the model occasionally answers directly
    /// instead of invoking a tool (ZAI GLM-4.7). A warning is logged.
    Unreliable,
    /// The model does not support tool use at all; `toolConfig` is omitted.
    None,
}

impl ModelCapabilities {
    /// Look up capabilities for a given Bedrock model ID.
    ///
    /// Regional inference-profile prefixes (`us.`, `eu.`, `ap.`) are
    /// stripped before matching so that e.g. `us.amazon.nova-lite-v1:0`
    /// resolves the same as `amazon.nova-lite-v1:0`.
    pub fn for_model(model_id: &str) -> Self {
        let id = model_id
            .trim_start_matches("us.")
            .trim_start_matches("eu.")
            .trim_start_matches("ap.");

        if id.starts_with("amazon.nova") {
            // Amazon Nova v1 and v2 — full Converse API support.
            // topK must go in additionalModelRequestFields.
            return Self {
                system_prompt_native: true,
                tool_use: ToolUseSupport::Full,
                topk_via_additional_fields: true,
                max_output_tokens: None,
            };
        }

        if id.starts_with("anthropic.claude") {
            return Self {
                system_prompt_native: true,
                tool_use: ToolUseSupport::Full,
                topk_via_additional_fields: false,
                max_output_tokens: None,
            };
        }

        if id.starts_with("minimax.") {
            // MiniMax M2.1: 1 M-token context, 8 K output cap.
            // System prompt is NOT documented for bedrock-runtime Converse;
            // we inject it as the first user message.
            // Client-side tool calling confirmed via Converse toolConfig.
            return Self {
                system_prompt_native: false,
                tool_use: ToolUseSupport::ClientSide,
                topk_via_additional_fields: false,
                max_output_tokens: Some(8192),
            };
        }

        if id.starts_with("zai.") {
            // ZAI GLM-4.7: 128 K context, 4 K output cap.
            // Same system-prompt limitation as MiniMax.
            // Tool calling supported but occasionally unreliable.
            return Self {
                system_prompt_native: false,
                tool_use: ToolUseSupport::Unreliable,
                topk_via_additional_fields: false,
                max_output_tokens: Some(4096),
            };
        }

        // Meta Llama, Mistral, Cohere, AI21, etc. — safe defaults.
        Self {
            system_prompt_native: true,
            tool_use: ToolUseSupport::ClientSide,
            topk_via_additional_fields: false,
            max_output_tokens: None,
        }
    }
}
