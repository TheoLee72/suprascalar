pub mod agent;
pub mod candle_transformers_patched;
pub mod error;
pub mod models;

pub use agent::Agent;
pub use error::{Result, SuprascalarError};
pub use models::LLMBackend;
pub use models::qqwen3::CandleQwen;
