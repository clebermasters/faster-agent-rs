use crate::{ToolDefinition, ToolError, ToolResult};
use skill_core::Skill;
use skill_executor::{ExecutionContext, SkillExecutor};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

#[derive(Clone)]
pub struct SkillTool {
    skill: Skill,
    executor: Arc<SkillExecutor>,
}

impl SkillTool {
    pub fn new(skill: Skill, skills_base_dir: PathBuf) -> Self {
        let executor = SkillExecutor::new(skills_base_dir);
        Self {
            skill,
            executor: Arc::new(executor),
        }
    }

    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: format!("skill_{}", self.skill.id),
            description: format!(
                "{}\n\nInstructions: {}",
                self.skill.description, self.skill.instructions
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input to pass to the skill"
                    }
                },
                "required": ["input"]
            }),
        }
    }

    pub async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        info!("=== SkillTool '{}' executing ===", self.skill.id);
        debug!("Skill tool params raw: {:?}", params);

        let input = Self::extract_input(&params);
        debug!("Extracted input for skill '{}': {:?}", self.skill.id, input);

        if input.is_none() {
            warn!("No input extracted for skill '{}'!", self.skill.id);
        }

        let context = ExecutionContext::default();

        info!(
            "Executing skill '{}' with input: {:?}",
            self.skill.id, input
        );
        let result = self
            .executor
            .execute_skill(&self.skill, input.as_deref(), &context)
            .await
            .map_err(|e| {
                error!("Skill '{}' execution error: {}", self.skill.id, e);
                ToolError::ExecutionError(e.to_string())
            })?;

        info!(
            "Skill '{}' result: success={}, output_len={}, error={:?}",
            self.skill.id,
            result.success,
            result.output.len(),
            result.error
        );
        debug!(
            "Skill '{}' output (first 300 chars): {:?}",
            self.skill.id,
            &result.output[..result.output.len().min(300)]
        );

        Ok(ToolResult {
            success: result.success,
            output: result.output,
            error: result.error,
        })
    }

    fn extract_input(params: &serde_json::Value) -> Option<String> {
        debug!("extract_input called with: {:?}", params);

        if let Some(obj) = params.as_object() {
            // Try top-level keys first
            for key in &["input", "url", "query", "value"] {
                if let Some(v) = obj.get(*key) {
                    if let Some(s) = v.as_str() {
                        if !s.is_empty()
                            && !s.contains("string")
                            && !s.contains("Input to pass")
                            && !s.contains("description")
                        {
                            debug!("Found input at top-level key '{}': {}", key, s);
                            return Some(s.to_string());
                        }
                    }
                }
            }

            // Try nested in "input" object
            if let Some(input_obj) = obj.get("input").or_else(|| obj.get("query")) {
                if let Some(s) = input_obj.as_str() {
                    if !s.is_empty() && !s.contains("string") && !s.contains("Input to pass") {
                        debug!("Found input in nested 'input': {}", s);
                        return Some(s.to_string());
                    }
                }
                if let Some(nested) = input_obj.as_object() {
                    for key in &["value", "url", "query", "description"] {
                        if let Some(v) = nested.get(*key) {
                            if let Some(s) = v.as_str() {
                                if !s.is_empty()
                                    && !s.contains("string")
                                    && !s.contains("Input to pass")
                                {
                                    debug!("Found input in nested 'input.{}': {}", key, s);
                                    return Some(s.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Direct string
        if let Some(s) = params.as_str() {
            if !s.is_empty() {
                debug!("Found input as direct string: {}", s);
                return Some(s.to_string());
            }
        }

        warn!("No input found in params: {:?}", params);
        None
    }
}
