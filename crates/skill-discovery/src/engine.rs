use skill_core::{DiscoveredSkill, MatchType, Skill, SkillQuery};
use skill_embeddings::EmbeddingService;
use skill_registry::SkillRegistry;
use thiserror::Error;
use tracing::{debug, info};

pub struct SkillDiscoveryEngine {
    registry: SkillRegistry,
    embeddings: EmbeddingService,
    semantic_weight: f64,
    keyword_weight: f64,
}

#[derive(Error, Debug)]
pub enum DiscoveryError {
    #[error("Registry error: {0}")]
    RegistryError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("No skills available")]
    NoSkillsAvailable,

    #[error("Skill not found: {0}")]
    SkillNotFound(String),
}

pub struct Prompts;

impl Prompts {
    pub const SYSTEM_PROMPT: &'static str = r#"You are an AI agent with access to a skill system. 

## Available Skills

When you encounter a task, you can use skills to help complete it. Skills are specialized capabilities that can be discovered and invoked.

## How to Use Skills

1. **Analyze the task**: Understand what the user is asking for
2. **Discover relevant skills**: Query the skill discovery system with a description of what you need
3. **Select the best skill**: Review the returned skills and their match scores
4. **Execute the skill**: Use the skill's instructions and resources
5. **Continue with results**: Use the skill's output to help the user

## Skill Discovery

When you need to find a skill, think about:
- What specific capability do I need?
- What keywords describe this capability?
- What tools or actions would help?

Then use the skill discovery to find relevant skills. Skills are ranked by relevance score.

## Important Notes

- Not every task requires a skill - use your judgment
- If no skill matches well, proceed without one
- Skills provide capabilities but you remain in control
- Always explain what you're doing when using skills

## Skills Structure

Each skill has:
- **name**: What it's called
- **description**: What it does
- **instructions**: How to use it
- **capabilities**: What it can do
- **resources**: Scripts and files available"#;
}

impl SkillDiscoveryEngine {
    pub fn new(registry: SkillRegistry, embeddings: EmbeddingService) -> Self {
        Self {
            registry,
            embeddings,
            semantic_weight: 0.7,
            keyword_weight: 0.3,
        }
    }

    pub fn with_weights(mut self, semantic: f64, keyword: f64) -> Self {
        self.semantic_weight = semantic;
        self.keyword_weight = keyword;
        self
    }

    pub async fn index_all(&mut self) -> Result<(), DiscoveryError> {
        self.embeddings
            .init_db()
            .await
            .map_err(|e| DiscoveryError::EmbeddingError(e.to_string()))?;

        let skills = self.registry.get_all();
        for skill in skills {
            if self
                .embeddings
                .get_embedding(&skill.id)
                .await
                .map_err(|e| DiscoveryError::EmbeddingError(e.to_string()))?
                .is_none()
            {
                self.embeddings
                    .index_skill(skill)
                    .await
                    .map_err(|e| DiscoveryError::EmbeddingError(e.to_string()))?;
            }
        }

        info!("Indexed all skills");
        Ok(())
    }

    pub async fn discover(
        &mut self,
        query: SkillQuery,
    ) -> Result<Vec<DiscoveredSkill>, DiscoveryError> {
        let limit = query.limit;
        let threshold = query.threshold;

        let semantic_matches: Vec<(String, f64)> = self.semantic_search(&query.task, limit).await?;

        let keyword_matches = self.keyword_search(&query.task, limit);

        let mut combined = Vec::new();

        for (skill_id, semantic_score) in &semantic_matches {
            let keyword_score = keyword_matches
                .iter()
                .find(|(id, _)| id == skill_id)
                .map(|(_, s)| *s)
                .unwrap_or(0.0);

            let hybrid_score =
                (semantic_score * self.semantic_weight) + (keyword_score * self.keyword_weight);

            let match_type = if keyword_score > 0.0 {
                MatchType::Hybrid
            } else {
                MatchType::Semantic
            };

            if hybrid_score >= threshold {
                if let Some(skill) = self.registry.get(skill_id) {
                    combined.push(DiscoveredSkill {
                        skill: skill.clone(),
                        score: hybrid_score,
                        match_type,
                    });
                }
            }
        }

        for (skill_id, keyword_score) in keyword_matches {
            if !semantic_matches.iter().any(|(id, _)| id == &skill_id) {
                if keyword_score >= threshold {
                    if let Some(skill) = self.registry.get(&skill_id) {
                        combined.push(DiscoveredSkill {
                            skill: skill.clone(),
                            score: keyword_score * self.keyword_weight,
                            match_type: MatchType::Keyword,
                        });
                    }
                }
            }
        }

        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        combined.truncate(limit);

        debug!(
            "Discovered {} skills for query: {}",
            combined.len(),
            query.task
        );
        Ok(combined)
    }

    async fn semantic_search(
        &mut self,
        task: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, DiscoveryError> {
        let query_embedding = self
            .embeddings
            .embed_text(task)
            .await
            .map_err(|e| DiscoveryError::EmbeddingError(e.to_string()))?;

        let skill_ids: Vec<String> = self
            .registry
            .get_all()
            .iter()
            .map(|s| s.id.clone())
            .collect();

        let results = self
            .embeddings
            .search_similar(&query_embedding, &skill_ids, limit)
            .await
            .map_err(|e| DiscoveryError::EmbeddingError(e.to_string()))?;

        Ok(results)
    }

    fn keyword_search(&self, task: &str, limit: usize) -> Vec<(String, f64)> {
        let task_lower = task.to_lowercase();
        let task_words: Vec<&str> = task_lower.split_whitespace().collect();

        let mut scores = Vec::new();

        for skill in self.registry.get_all() {
            let mut score = 0.0f64;

            for trigger in &skill.triggers {
                let trigger_lower = trigger.to_lowercase();
                if task_lower.contains(&trigger_lower) {
                    score += 1.0;
                }
                for word in &task_words {
                    if trigger_lower.contains(word) || word.contains(&trigger_lower) {
                        score += 0.5;
                    }
                }
            }

            for capability in &skill.capabilities {
                let cap_lower = capability.to_lowercase();
                if task_lower.contains(&cap_lower) {
                    score += 0.8;
                }
                for word in &task_words {
                    if cap_lower.contains(word) || word.contains(&cap_lower) {
                        score += 0.3;
                    }
                }
            }

            let desc_lower = skill.description.to_lowercase();
            for word in &task_words {
                if desc_lower.contains(word) {
                    score += 0.2;
                }
            }

            if score > 0.0 {
                let normalized_score = (score / 10.0).min(1.0);
                scores.push((skill.id.clone(), normalized_score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);

        scores
    }

    pub async fn reindex_skill(&mut self, skill: &Skill) -> Result<(), DiscoveryError> {
        self.embeddings
            .index_skill(skill)
            .await
            .map_err(|e| DiscoveryError::EmbeddingError(e.to_string()))?;
        Ok(())
    }

    pub fn get_system_prompt(&self) -> &str {
        Prompts::SYSTEM_PROMPT
    }

    pub fn registry(&self) -> &SkillRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut SkillRegistry {
        &mut self.registry
    }
}
