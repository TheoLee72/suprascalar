use suprascalar::{Agent, CandleQwen, SuprascalarError};

fn main() -> Result<(), SuprascalarError> {
    // 1. Setup paths
    let repo = "unsloth/Qwen3-14B-GGUF";
    let model_file = "Qwen3-14B-Q4_K_M.gguf";
    let tokenizer_repo = "Qwen/Qwen3-14B";

    println!(">>> Initializing Suprascalar...");

    // 2. Load backend with custom error handling
    let backend = match CandleQwen::new(repo, model_file, tokenizer_repo) {
        Ok(b) => Box::new(b),
        Err(SuprascalarError::Io(e)) => {
            eprintln!("File not found: {}", e);
            return Err(SuprascalarError::Io(e));
        }
        Err(e) => return Err(e),
    };

    // 3. Create Agent
    let mut agent = Agent::new("Suprascalar", backend, "You are a helpful AI assistant.");

    // 4. Chat
    match agent.chat("Hello! /no_think") {
        Ok(response) => println!("Agent: {}", response),
        Err(SuprascalarError::ContextLimitExceeded { limit, current }) => {
            eprintln!("Context too long! {}/{}", current, limit);
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}
