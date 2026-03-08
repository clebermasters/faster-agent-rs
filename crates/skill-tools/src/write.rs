use crate::{ToolDefinition, ToolError, ToolResult};
use serde::Deserialize;
use tokio::fs;
use tracing::{debug, info, warn};

#[derive(Debug, Deserialize)]
pub struct WriteParams {
    pub path: String,
    pub content: String,
    #[serde(default)]
    pub append: Option<String>,
}

impl WriteParams {
    pub fn append_bool(&self) -> Option<bool> {
        self.append.as_ref().and_then(|s| {
            if s == "true" || s == "false" {
                Some(s == "true")
            } else {
                None
            }
        })
    }
}

#[derive(Clone)]
pub struct WriteTool {
    base_dir: String,
}

impl WriteTool {
    pub fn new(base_dir: String) -> Self {
        Self { base_dir }
    }

    fn resolve_path(&self, path: &str) -> std::path::PathBuf {
        let path = std::path::Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::path::Path::new(&self.base_dir).join(path)
        }
    }

    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write".to_string(),
            description: "Write content to a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Append to file instead of overwriting (default: false)"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    pub async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        info!("Write tool executing with params: {:?}", params);

        let params: WriteParams = serde_json::from_value(params).map_err(|e| {
            warn!("Write tool failed to parse params: {}", e);
            ToolError::InvalidParameters(e.to_string())
        })?;

        let path = self.resolve_path(&params.path);
        debug!(
            "Resolved path: {:?}, append: {:?}",
            path,
            params.append_bool()
        );
        debug!("Content length: {} chars", params.content.len());

        if let Some(parent) = path.parent() {
            if !parent.exists() {
                debug!("Creating parent directory: {:?}", parent);
                if let Err(e) = fs::create_dir_all(parent).await {
                    warn!("Failed to create directory: {}", e);
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to create directory: {}", e)),
                    });
                }
            }
        }

        let result = if params.append_bool() == Some(true) {
            debug!("Opening file for append: {:?}", path);
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .await
        } else {
            debug!("Creating new file: {:?}", path);
            fs::File::create(&path).await
        };

        match result {
            Ok(mut file) => {
                use tokio::io::AsyncWriteExt;
                debug!("Writing {} bytes to file", params.content.len());
                if let Err(e) = file.write_all(params.content.as_bytes()).await {
                    warn!("Write failed: {}", e);
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to write: {}", e)),
                    });
                }
                info!("Successfully wrote to file: {:?}", path);
                Ok(ToolResult {
                    success: true,
                    output: format!("Written to {}", path.display()),
                    error: None,
                })
            }
            Err(e) => {
                warn!("Failed to create/open file: {}", e);
                Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to create file: {}", e)),
                })
            }
        }
    }
}
