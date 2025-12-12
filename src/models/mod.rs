use crate::error::Result;
pub mod qqwen3;

/// The core trait that any Model backend must implement.
pub trait LLMBackend {
    /// Generate a response based on the provided prompt string.
    fn generate(&mut self, prompt: &str) -> Result<String>;
}
