use thiserror::Error;

#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Skill not found: {0}")]
    NotFound(String),

    #[error("Skill already exists: {0}")]
    AlreadyExists(String),
}
