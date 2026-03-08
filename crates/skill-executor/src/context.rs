use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub working_dir: PathBuf,
    pub env_vars: HashMap<String, String>,
    pub user_input: Option<String>,
}

impl ExecutionContext {
    pub fn new(working_dir: PathBuf) -> Self {
        Self {
            working_dir,
            env_vars: std::env::vars().collect(),
            user_input: None,
        }
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    pub fn with_user_input(mut self, input: impl Into<String>) -> Self {
        self.user_input = Some(input.into());
        self
    }

    pub fn default() -> Self {
        Self::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::default()
    }
}
