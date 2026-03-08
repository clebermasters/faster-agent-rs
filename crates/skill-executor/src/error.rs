use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("IO error: {0}")]
    IoError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Script error: {0}")]
    ScriptError(String),

    #[error("Skill not found: {0}")]
    SkillNotFound(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}
