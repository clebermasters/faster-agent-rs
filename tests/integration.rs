use skill_core::Config;

#[tokio::test]
async fn test_basic_setup() {
    let config = Config::default();
    assert_eq!(config.vector_dim, 768);
    // Add real integration tests here
}
