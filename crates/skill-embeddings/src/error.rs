use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Request error: {0}")]
    RequestError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Ollama not running: {0}")]
    OllamaNotRunning(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),
}
