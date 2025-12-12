use thiserror::Error;

/// Suprascalar crate-specific Result type alias
pub type Result<T> = std::result::Result<T, SuprascalarError>;

#[derive(Error, Debug)]
pub enum SuprascalarError {
    #[error("Candle operation error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Failed to load model file: {0}")]
    Io(#[from] std::io::Error),

    // HfHub is optional depending on if you bundle the model or download it
    #[error("HF Hub API error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    #[error("Config parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Model not found: {name}")]
    ModelNotFound { name: String },

    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    // Tokenizer error wrapper
    #[error("Tokenization error: {0}")]
    Tokenizer(String),

    #[error("Context length exceeded: limit {limit}, current {current}")]
    ContextLimitExceeded { limit: usize, current: usize },

    #[error("Unknown error: {0}")]
    Unknown(String),
}
