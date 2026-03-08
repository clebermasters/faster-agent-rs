use crate::{ToolDefinition, ToolError, ToolResult};
use serde::Deserialize;
use tokio::fs;

#[derive(Debug, Deserialize)]
pub struct ReadParams {
    pub path: String,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Clone)]
pub struct ReadTool {
    base_dir: String,
}

impl ReadTool {
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
            name: "read".to_string(),
            description: "Read contents of a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "offset": {
                        "type": "number",
                        "description": "Line number to start reading from (1-indexed, optional)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of lines to read (optional)"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    pub async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        let params: ReadParams = serde_json::from_value(params)
            .map_err(|e| ToolError::InvalidParameters(e.to_string()))?;

        let path = self.resolve_path(&params.path);

        match fs::read_to_string(&path).await {
            Ok(content) => {
                let lines: Vec<&str> = content.lines().collect();
                let total_lines = lines.len();

                let start = params
                    .offset
                    .unwrap_or(1)
                    .saturating_sub(1)
                    .min(total_lines);
                let end = params
                    .limit
                    .map(|l| (start + l).min(total_lines))
                    .unwrap_or(total_lines);

                let selected: Vec<&str> = lines[start..end].to_vec();
                let output = selected.join("\n");

                Ok(ToolResult {
                    success: true,
                    output: if params.offset.is_some() || params.limit.is_some() {
                        format!(
                            "{} lines {}-{} of {}:\n\n{}",
                            path.display(),
                            start + 1,
                            end,
                            total_lines,
                            output
                        )
                    } else {
                        output
                    },
                    error: None,
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to read {}: {}", path.display(), e)),
            }),
        }
    }
}
