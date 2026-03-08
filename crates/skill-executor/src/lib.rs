use crate::error::ExecutorError;
use skill_core::{ResourceType, Skill, SkillResult};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;
use tracing::{debug, info};

pub mod error;
pub mod context;

pub use context::ExecutionContext;

pub struct SkillExecutor {
    skills_base_dir: PathBuf,
    default_timeout_secs: u64,
}

impl SkillExecutor {
    pub fn new(skills_base_dir: PathBuf) -> Self {
        Self {
            skills_base_dir,
            default_timeout_secs: 300,
        }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.default_timeout_secs = secs;
        self
    }

    pub async fn execute_skill(
        &self,
        skill: &Skill,
        input: Option<&str>,
        context: &ExecutionContext,
    ) -> Result<SkillResult, ExecutorError> {
        let start = Instant::now();

        info!("Executing skill: {}", skill.name);
        debug!("Skill resources: {:?}", skill.resources);

        let mut output = String::new();
        let mut errors = Vec::new();

        for resource in &skill.resources {
            if resource.resource_type == ResourceType::Script {
                match self.run_script(&resource.path, input, context).await {
                    Ok(result) => {
                        output.push_str(&format!("\n--- {} ---\n", resource.name));
                        output.push_str(&result);
                    }
                    Err(e) => {
                        errors.push(format!("{}: {}", resource.name, e));
                    }
                }
            }
        }

        if !skill.instructions.is_empty() {
            output.push_str("\n--- Instructions ---\n");
            output.push_str(&skill.instructions);
        }

        let execution_time_ms = start.elapsed().as_millis() as u64;
        let success = errors.is_empty();

        let error_msg = if errors.is_empty() {
            None
        } else {
            Some(errors.join("; "))
        };

        Ok(SkillResult {
            skill_id: skill.id.clone(),
            success,
            output,
            error: error_msg,
            execution_time_ms,
        })
    }

    async fn run_script(
        &self,
        script_path: &PathBuf,
        input: Option<&str>,
        context: &ExecutionContext,
    ) -> Result<String, ExecutorError> {
        debug!("Running script: {:?} with input: {:?}", script_path, input);

        let mut cmd = Command::new("bash");
        cmd.arg(script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&context.working_dir);

        for (key, value) in &context.env_vars {
            cmd.env(key, value);
        }

        if let Some(inp) = input {
            cmd.arg(inp);
        }

        let output = cmd
            .output()
            .await
            .map_err(|e| ExecutorError::ExecutionError(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            return Err(ExecutorError::ScriptError(format!(
                "Script failed: {}",
                stderr
            )));
        }

        Ok(stdout)
    }

    pub async fn get_skill_instructions(&self, skill: &Skill) -> Result<String, ExecutorError> {
        Ok(skill.instructions.clone())
    }

    pub async fn get_resource_content(
        &self,
        resource_path: &PathBuf,
    ) -> Result<String, ExecutorError> {
        let content = tokio::fs::read_to_string(resource_path)
            .await
            .map_err(|e| ExecutorError::IoError(e.to_string()))?;
        Ok(content)
    }
}
