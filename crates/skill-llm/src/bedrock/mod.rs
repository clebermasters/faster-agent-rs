pub mod auth;
pub mod bearer;
pub mod converse;
pub mod convert;
pub mod mantle;
pub mod models;

pub use auth::BedrockAuth;
pub use bearer::BedrockBearerClient;
pub use converse::BedrockConverseClient;
pub use mantle::BedrockMantleClient;

use crate::LLMClient;
use anyhow::Result;

/// Create the appropriate Bedrock client based on the chosen auth mode.
///
/// All variants route to [`BedrockConverseClient`] (AWS SDK + Converse API).
///
/// - [`BedrockAuth::BedrockApiKey`]: sets `AWS_BEARER_TOKEN_BEDROCK` so the
///   SDK uses `Authorization: Bearer {absk_token}` instead of SigV4. This
///   works with all model families including MiniMax M2.1 and ZAI GLM-4.7.
///
/// - All other variants: standard SigV4 auth via the appropriate credential
///   provider (static, STS token, assumed role, or default chain).
///
/// Per-model quirks (system-prompt injection for MiniMax/ZAI, `topK` via
/// `additionalModelRequestFields` for Nova, tool-use warnings for ZAI) are
/// handled transparently by the [`models::ModelCapabilities`] registry.
///
/// [`BedrockMantleClient`] and [`BedrockBearerClient`] are retained for
/// potential future use with non-ABSK API key formats.
pub async fn create_bedrock_client(
    auth: BedrockAuth,
    region: String,
    model_id: String,
) -> Result<Box<dyn LLMClient>> {
    match auth {
        BedrockAuth::BedrockApiKey { api_key, region } => {
            // Use BedrockBearerClient (reqwest + HTTP/1.1) for ABSK tokens.
            // The standard bedrock-runtime endpoint supports
            // `Authorization: Bearer {absk_token}` auth over HTTP/1.1.
            Ok(Box::new(BedrockBearerClient::new(
                api_key, region, model_id,
            )))
        }
        other => Ok(Box::new(
            BedrockConverseClient::new(other, region, model_id).await?,
        )),
    }
}
