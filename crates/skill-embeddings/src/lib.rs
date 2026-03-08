use crate::error::EmbeddingError;
use rusqlite::{params, Connection};
use serde::Deserialize;
use skill_core::Skill;
use std::path::PathBuf;
use tracing::{debug, info};

pub mod error;

#[derive(Clone)]
pub struct EmbeddingService {
    provider: Provider,
    db_path: PathBuf,
}

#[derive(Clone)]
enum Provider {
    Ollama { url: String, model: String },
}

impl EmbeddingService {
    pub fn new_ollama(url: String, model: String, db_path: PathBuf) -> Self {
        Self {
            provider: Provider::Ollama { url, model },
            db_path,
        }
    }

    pub async fn init_db(&self) -> Result<(), EmbeddingError> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                skill_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?;

        info!("Initialized embeddings database at {:?}", self.db_path);
        Ok(())
    }

    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        match &self.provider {
            Provider::Ollama { url, model } => self.embed_ollama(url, model, text).await,
        }
    }

    async fn embed_ollama(
        &self,
        url: &str,
        model: &str,
        text: &str,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let client = reqwest::Client::new();

        let request = serde_json::json!({
            "model": model,
            "input": text,
        });

        let response = client
            .post(format!("{}/api/embeddings", url))
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::RequestError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::ApiError(format!(
                "Status: {}, Body: {}",
                status, body
            )));
        }

        let result: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        Ok(result.embedding)
    }

    pub async fn embed_skill(&self, skill: &Skill) -> Result<Vec<f32>, EmbeddingError> {
        let text = skill.search_text();
        debug!("Embedding skill: {} ({} chars)", skill.name, text.len());
        self.embed_text(&text).await
    }

    pub async fn store_embedding(
        &self,
        skill_id: &str,
        embedding: &[f32],
    ) -> Result<(), EmbeddingError> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?;

        let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "INSERT OR REPLACE INTO embeddings (skill_id, embedding, updated_at) VALUES (?1, ?2, ?3)",
            params![skill_id, embedding_bytes, now],
        )
        .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?;

        debug!("Stored embedding for skill: {}", skill_id);
        Ok(())
    }

    pub async fn get_embedding(&self, skill_id: &str) -> Result<Option<Vec<f32>>, EmbeddingError> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?;

        let mut stmt = conn
            .prepare("SELECT embedding FROM embeddings WHERE skill_id = ?1")
            .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?;

        let mut rows = stmt
            .query(params![skill_id])
            .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?
        {
            let embedding_bytes: Vec<u8> = row
                .get(0)
                .map_err(|e| EmbeddingError::DatabaseError(e.to_string()))?;
            let embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Ok(Some(embedding))
        } else {
            Ok(None)
        }
    }

    pub async fn index_skill(&self, skill: &Skill) -> Result<(), EmbeddingError> {
        let embedding = self.embed_skill(skill).await?;
        self.store_embedding(&skill.id, &embedding).await?;
        info!("Indexed skill: {}", skill.name);
        Ok(())
    }

    pub async fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        (dot_product / (mag_a * mag_b)) as f64
    }

    pub async fn search_similar(
        &self,
        query_embedding: &[f32],
        skill_ids: &[String],
        limit: usize,
    ) -> Result<Vec<(String, f64)>, EmbeddingError> {
        let mut scores = Vec::new();

        for skill_id in skill_ids {
            if let Some(embedding) = self.get_embedding(skill_id).await? {
                let similarity = self.cosine_similarity(query_embedding, &embedding).await;
                scores.push((skill_id.clone(), similarity));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);

        Ok(scores)
    }
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}
