use crate::error::RegistryError;
use skill_core::{ResourceType, Skill, SkillResource};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use walkdir::WalkDir;

pub mod error;

#[derive(Debug)]
pub struct SkillRegistry {
    skills: HashMap<String, Skill>,
    skills_dir: PathBuf,
}

impl SkillRegistry {
    pub fn new(skills_dir: PathBuf) -> Self {
        Self {
            skills: HashMap::new(),
            skills_dir,
        }
    }

    pub async fn load(&mut self) -> Result<(), RegistryError> {
        self.skills.clear();

        if !self.skills_dir.exists() {
            warn!("Skills directory does not exist: {:?}", self.skills_dir);
            return Ok(());
        }

        info!("Loading skills from: {:?}", self.skills_dir);

        for entry in WalkDir::new(&self.skills_dir)
            .max_depth(2)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.file_name().map(|n| n == "SKILL.md").unwrap_or(false) {
                if let Some(skill) = self.load_skill(path).await? {
                    let id = skill.id.clone();
                    debug!("Loaded skill: {} ({})", skill.name, id);
                    self.skills.insert(id, skill);
                }
            }
        }

        info!("Loaded {} skills", self.skills.len());
        Ok(())
    }

    async fn load_skill(&self, skill_md_path: &Path) -> Result<Option<Skill>, RegistryError> {
        let skill_dir = skill_md_path
            .parent()
            .ok_or_else(|| RegistryError::InvalidPath("SKILL.md has no parent".into()))?;

        let content = tokio::fs::read_to_string(skill_md_path).await?;

        let (frontmatter, body) = parse_skill_md(&content)?;

        let name = frontmatter
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unnamed")
            .to_string();

        let description = frontmatter
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let triggers: Vec<String> = get_string_array(&frontmatter, "trigger");

        let capabilities: Vec<String> = get_string_array(&frontmatter, "capabilities");

        let id = slugify(&name);

        let resources = self.load_resources(skill_dir).await?;

        let skill = Skill::new(id, name, description, body, skill_dir.to_path_buf())
            .with_triggers(triggers)
            .with_capabilities(capabilities)
            .with_resources(resources);

        Ok(Some(skill))
    }

    async fn load_resources(&self, skill_dir: &Path) -> Result<Vec<SkillResource>, RegistryError> {
        let mut resources = Vec::new();

        let scripts_dir = skill_dir.join("scripts");
        if scripts_dir.is_dir() {
            for entry in WalkDir::new(&scripts_dir)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                resources.push(SkillResource {
                    name: entry.file_name().to_string_lossy().to_string(),
                    path: entry.path().to_path_buf(),
                    resource_type: ResourceType::Script,
                });
            }
        }

        let references_dir = skill_dir.join("references");
        if references_dir.is_dir() {
            for entry in WalkDir::new(&references_dir)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                resources.push(SkillResource {
                    name: entry.file_name().to_string_lossy().to_string(),
                    path: entry.path().to_path_buf(),
                    resource_type: ResourceType::Reference,
                });
            }
        }

        Ok(resources)
    }

    pub fn get(&self, id: &str) -> Option<&Skill> {
        self.skills.get(id)
    }

    pub fn get_all(&self) -> Vec<&Skill> {
        self.skills.values().collect()
    }

    pub fn count(&self) -> usize {
        self.skills.len()
    }

    pub fn skills_dir(&self) -> &Path {
        &self.skills_dir
    }
}

fn parse_skill_md(content: &str) -> Result<(serde_yaml::Value, String), RegistryError> {
    let trimmed = content.trim();

    if !trimmed.starts_with("---") {
        return Ok((serde_yaml::Value::Null, trimmed.to_string()));
    }

    let end_marker = trimmed[3..]
        .find("---")
        .ok_or_else(|| RegistryError::ParseError("Missing closing ---".into()))?;

    let frontmatter_str = &trimmed[3..3 + end_marker];
    let body = trimmed[3 + end_marker + 3..].trim().to_string();

    let frontmatter: serde_yaml::Value = serde_yaml::from_str(frontmatter_str)
        .map_err(|e| RegistryError::ParseError(e.to_string()))?;

    Ok((frontmatter, body))
}

fn get_string_array(value: &serde_yaml::Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(|v| {
            if let Some(arr) = v.as_sequence() {
                Some(
                    arr.iter()
                        .filter_map(|item| item.as_str().map(String::from))
                        .collect(),
                )
            } else if let Some(s) = v.as_str() {
                Some(vec![s.to_string()])
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("Hello World"), "hello-world");
        assert_eq!(slugify("RSS Fetcher"), "rss-fetcher");
    }

    #[test]
    fn test_parse_skill_md_without_frontmatter() {
        let content = "# Hello\n\nThis is the body.";
        let (fm, body) = parse_skill_md(content).unwrap();
        assert_eq!(fm, serde_yaml::Value::Null);
        assert!(body.contains("Hello"));
    }
}
