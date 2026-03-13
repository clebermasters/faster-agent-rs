use anyhow::{Context, Result};
use aws_config::{BehaviorVersion, Region};
use aws_credential_types::Credentials;
use tracing::info;

/// The three authentication modes exposed to the CLI / environment.
#[derive(Debug, Clone)]
pub enum BedrockAuth {
    /// Long-term static IAM credentials (access key + secret key).
    ///
    /// Env vars: `BEDROCK_ACCESS_KEY_ID`, `BEDROCK_SECRET_ACCESS_KEY`
    Static {
        access_key_id: String,
        secret_access_key: String,
    },

    /// Pre-obtained STS session token (short-lived, already assumed).
    ///
    /// Env vars: `BEDROCK_ACCESS_KEY_ID`, `BEDROCK_SECRET_ACCESS_KEY`,
    ///           `BEDROCK_SESSION_TOKEN`
    StsToken {
        access_key_id: String,
        secret_access_key: String,
        session_token: String,
    },

    /// Call STS `AssumeRole` at startup and use the resulting temp creds.
    /// The source credentials for the STS call come from the default chain.
    ///
    /// Env vars: `BEDROCK_ROLE_ARN`, `BEDROCK_ROLE_SESSION_NAME` (optional),
    ///           `BEDROCK_EXTERNAL_ID` (optional)
    StsAssumeRole {
        role_arn: String,
        /// Session name tag visible in CloudTrail (default: `skill-agent`).
        session_name: String,
        /// Required for cross-account roles that mandate an ExternalId.
        external_id: Option<String>,
        /// Credential lifetime in seconds (default: 3600).
        duration_secs: u32,
    },

    /// Fall through to the AWS default credential chain:
    /// env vars → `~/.aws/credentials` → EC2/ECS/Lambda instance profile.
    DefaultChain,

    /// Bedrock API Key (ABSK format: `ABSK<base64(BedrockAPIKey-{id}:{secret})>`).
    ///
    /// Handled at the factory level in `mod.rs` — routes to
    /// [`BedrockBearerClient`] which calls the standard `bedrock-runtime`
    /// Converse API with `Authorization: Bearer {absk_token}` (HTTP/1.1,
    /// no SigV4 required). Works with all Bedrock model families.
    ///
    /// Env var: `BEDROCK_API_KEY`
    BedrockApiKey { api_key: String, region: String },
}

/// Build an `aws_config::SdkConfig` from the given auth variant.
///
/// `BedrockApiKey` must **not** be passed here — the factory in `mod.rs`
/// intercepts it before calling this function.
pub async fn build_aws_config(auth: BedrockAuth, region: &str) -> Result<aws_config::SdkConfig> {
    let region_val = Region::new(region.to_string());

    match auth {
        BedrockAuth::Static {
            access_key_id,
            secret_access_key,
        } => {
            info!("Bedrock auth: static IAM credentials");
            let creds = Credentials::new(
                access_key_id,
                secret_access_key,
                None,
                None,
                "skill-agent-static",
            );
            Ok(aws_config::defaults(BehaviorVersion::latest())
                .region(region_val)
                .credentials_provider(creds)
                .load()
                .await)
        }

        BedrockAuth::StsToken {
            access_key_id,
            secret_access_key,
            session_token,
        } => {
            info!("Bedrock auth: STS session token");
            let creds = Credentials::new(
                access_key_id,
                secret_access_key,
                Some(session_token),
                None,
                "skill-agent-sts-token",
            );
            Ok(aws_config::defaults(BehaviorVersion::latest())
                .region(region_val)
                .credentials_provider(creds)
                .load()
                .await)
        }

        BedrockAuth::StsAssumeRole {
            role_arn,
            session_name,
            external_id,
            duration_secs,
        } => {
            info!("Bedrock auth: STS AssumeRole ({})", role_arn);

            // 1. Load base credentials from the default chain.
            let base_config = aws_config::defaults(BehaviorVersion::latest())
                .region(Region::new(region.to_string()))
                .load()
                .await;

            // 2. Create an STS client to call AssumeRole.
            let sts = aws_sdk_sts::Client::new(&base_config);

            // 3. Build and send the AssumeRole request.
            let mut req = sts
                .assume_role()
                .role_arn(&role_arn)
                .role_session_name(&session_name)
                .duration_seconds(duration_secs as i32);

            if let Some(ext_id) = &external_id {
                req = req.external_id(ext_id);
            }

            let resp = req.send().await.context("STS AssumeRole failed")?;

            // 4. Extract the temporary credentials.
            let c = resp
                .credentials()
                .context("STS AssumeRole returned no credentials")?;

            let creds = Credentials::new(
                c.access_key_id(),
                c.secret_access_key(),
                Some(c.session_token().to_string()),
                None,
                "skill-agent-assumed-role",
            );

            Ok(aws_config::defaults(BehaviorVersion::latest())
                .region(Region::new(region.to_string()))
                .credentials_provider(creds)
                .load()
                .await)
        }

        BedrockAuth::DefaultChain => {
            info!("Bedrock auth: default credential chain");
            Ok(aws_config::defaults(BehaviorVersion::latest())
                .region(region_val)
                .load()
                .await)
        }

        BedrockAuth::BedrockApiKey { .. } => {
            anyhow::bail!(
                "BedrockApiKey must not be passed to build_aws_config — \
                 it routes to BedrockBearerClient in the factory"
            )
        }
    }
}
