use skill_core::Config;
use std::path::PathBuf;

fn main() {
    let config = Config {
        skills_dir: PathBuf::from("./skills"),
        ..Default::default()
    };

    println!("Skill Agent Basic Example");
    println!("Skills directory: {:?}", config.skills_dir);
    println!("Embedding model: {}", config.embedding_model);
}
