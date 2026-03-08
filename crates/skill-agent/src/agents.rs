use std::fs;
use std::path::PathBuf;
use tracing::{debug, warn};

pub struct AgentsConfig {
    pub content: Option<String>,
    pub source: Option<PathBuf>,
}

impl AgentsConfig {
    pub fn load(agents_file: Option<&str>) -> Self {
        let path = if let Some(path) = agents_file {
            Some(PathBuf::from(path))
        } else {
            Self::find_agents_file()
        };

        if let Some(ref path) = path {
            if path.exists() {
                match fs::read_to_string(path) {
                    Ok(content) => {
                        let content = content.trim().to_string();
                        if !content.is_empty() {
                            debug!("Loaded AGENTS.md from: {:?}", path);
                            return Self {
                                content: Some(content),
                                source: Some(path.clone()),
                            };
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read AGENTS.md {:?}: {}", path, e);
                    }
                }
            }
        }

        Self {
            content: None,
            source: None,
        }
    }

    fn find_agents_file() -> Option<PathBuf> {
        let candidates = vec!["AGENTS.md", "CLAUDE.md"];

        let cwd = std::env::current_dir().ok()?;

        let mut dir = cwd.as_path();
        while dir != dir.parent().unwrap_or(dir) {
            for candidate in &candidates {
                let path = dir.join(candidate);
                if path.exists() {
                    debug!("Found {:?} in {:?}", candidate, dir);
                    return Some(path);
                }
            }
            dir = dir.parent().unwrap_or(dir);
        }

        if let Ok(home) = std::env::var("HOME") {
            let global_path = PathBuf::from(&home).join(".config/skill-agent/AGENTS.md");
            if global_path.exists() {
                debug!("Found global AGENTS.md: {:?}", global_path);
                return Some(global_path);
            }

            let claude_global = PathBuf::from(&home).join(".claude/CLAUDE.md");
            if claude_global.exists() {
                debug!("Found global CLAUDE.md: {:?}", claude_global);
                return Some(claude_global);
            }
        }

        None
    }

    pub fn content(&self) -> Option<&str> {
        self.content.as_deref()
    }
}
