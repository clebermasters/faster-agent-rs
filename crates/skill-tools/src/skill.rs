use crate::{ToolDefinition, ToolError, ToolResult};
use skill_core::Skill;
use skill_executor::{ExecutionContext, SkillExecutor};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

#[derive(Clone)]
pub struct SkillTool {
    skills: HashMap<String, Skill>,
    executor: Arc<SkillExecutor>,
}

impl SkillTool {
    pub fn new(skills: Vec<Skill>, skills_base_dir: PathBuf) -> Self {
        let executor = SkillExecutor::new(skills_base_dir);
        let map = skills.into_iter().map(|s| (s.id.clone(), s)).collect();
        Self {
            skills: map,
            executor: Arc::new(executor),
        }
    }

    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "run_skill".to_string(),
            description: "Execute a skill by its ID. Use the skill catalog in the system prompt \
                          to find the right skill_id for the task. Pass the skill_id and the input \
                          to execute it."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "The ID of the skill to execute (from the skill catalog)"
                    },
                    "input": {
                        "type": "string",
                        "description": "Input to pass to the skill"
                    }
                },
                "required": ["skill_id", "input"]
            }),
        }
    }

    /// Return a lightweight metadata catalog for the system prompt.
    /// One line per skill: id, name, description, triggers.
    pub fn skill_catalog(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }

        let mut lines: Vec<String> = self
            .skills
            .values()
            .map(|s| {
                let triggers = if s.triggers.is_empty() {
                    String::new()
                } else {
                    format!(" [triggers: {}]", s.triggers.join(", "))
                };
                format!("- {} ({}): {}{}", s.id, s.name, s.description, triggers)
            })
            .collect();
        lines.sort(); // deterministic order
        lines.join("\n")
    }

    pub fn skill_count(&self) -> usize {
        self.skills.len()
    }

    pub async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        let skill_id = params
            .get("skill_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParameters("Missing 'skill_id' parameter".into()))?;

        info!("=== SkillTool executing skill '{}' ===", skill_id);
        debug!("Skill tool params raw: {:?}", params);

        let skill = self.skills.get(skill_id).ok_or_else(|| {
            let available: Vec<&str> = self.skills.keys().map(|k| k.as_str()).collect();
            ToolError::NotFound(format!(
                "Skill '{}' not found. Available skills: {:?}",
                skill_id, available
            ))
        })?;

        let input = Self::extract_input(&params);
        debug!("Extracted input for skill '{}': {:?}", skill_id, input);

        if input.is_none() {
            warn!("No input extracted for skill '{}'!", skill_id);
        }

        let context = ExecutionContext::default();

        info!("Executing skill '{}' with input: {:?}", skill_id, input);
        let result = self
            .executor
            .execute_skill(skill, input.as_deref(), &context)
            .await
            .map_err(|e| {
                error!("Skill '{}' execution error: {}", skill_id, e);
                ToolError::ExecutionError(e.to_string())
            })?;

        info!(
            "Skill '{}' result: success={}, output_len={}, error={:?}",
            skill_id,
            result.success,
            result.output.len(),
            result.error
        );
        debug!(
            "Skill '{}' output (first 300 chars): {:?}",
            skill_id,
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
