use crate::{ToolDefinition, ToolError, ToolResult};
use serde::Deserialize;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{debug, info, warn};

#[derive(Debug, Deserialize)]
pub struct BashParams {
    pub command: String,
    #[serde(default)]
    pub timeout: Option<String>,
    #[serde(default)]
    pub workdir: Option<String>,
}

impl BashParams {
    pub fn timeout_ms(&self) -> Option<u64> {
        self.timeout.as_ref().and_then(|s| s.parse().ok())
    }
}

#[derive(Debug, Clone)]
pub struct BashTool;

impl BashTool {
    pub fn new() -> Self {
        Self
    }

    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "Execute a shell command".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in milliseconds (optional)"
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory (optional)"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    pub async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        info!("Bash tool executing with params: {:?}", params);
        
        let params: BashParams = serde_json::from_value(params)
            .map_err(|e| {
                warn!("Bash tool failed to parse params: {}", e);
                ToolError::InvalidParameters(e.to_string())
            })?;

        debug!("Parsed bash command: {}, timeout: {:?}, workdir: {:?}", 
            params.command, params.timeout, params.workdir);

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(&params.command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(ref workdir) = params.workdir {
            debug!("Setting workdir to: {}", workdir);
            cmd.current_dir(workdir);
        }

        let output = if let Some(timeout) = params.timeout_ms() {
            debug!("Executing with timeout: {}ms", timeout);
            match tokio::time::timeout(std::time::Duration::from_millis(timeout), cmd.output()).await {
                Ok(Ok(output)) => {
                    debug!("Bash command completed successfully");
                    output
                }
                Ok(Err(e)) => {
                    warn!("Bash command execution error: {}", e);
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Command failed: {}", e)),
                    });
                }
                Err(_) => {
                    warn!("Bash command timed out after {}ms", timeout);
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Command timed out after {}ms", timeout)),
                    });
                }
            }
        } else {
            debug!("Executing without timeout");
            match cmd.output().await {
                Ok(output) => {
                    debug!("Bash command completed");
                    output
                }
                Err(e) => {
                    warn!("Bash command spawn error: {}", e);
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to execute: {}", e)),
                    });
                }
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let success = output.status.success();
        debug!("Bash exit status: {}, stdout_len: {}, stderr_len: {}", 
            output.status, stdout.len(), stderr.len());
        
        let output_str = if stderr.is_empty() {
            stdout
        } else {
            format!("{}\n--- stderr ---\n{}", stdout, stderr)
        };

        let result = ToolResult {
            success,
            output: output_str,
            error: if success { None } else { Some(stderr) },
        };
        
        info!("Bash tool result: success={}, output_len={}", result.success, result.output.len());
        Ok(result)
    }
}
