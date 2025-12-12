use crate::error::Result;
use crate::models::LLMBackend;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct Agent {
    #[allow(dead_code)]
    name: String,
    model: Box<dyn LLMBackend>,
    history: Vec<Message>,
}

impl Agent {
    pub fn new(name: &str, model: Box<dyn LLMBackend>, system_prompt: &str) -> Self {
        let mut agent = Self {
            name: name.to_string(),
            model,
            history: Vec::new(),
        };

        agent.history.push(Message {
            role: "system".to_string(),
            content: system_prompt.to_string(),
        });

        agent
    }

    // Returns our custom Result
    pub fn chat(&mut self, user_input: &str) -> Result<String> {
        self.history.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        let prompt = self.build_prompt();

        // Error propagation works seamlessly with '?'
        let response = self.model.generate(&prompt)?;

        self.history.push(Message {
            role: "assistant".to_string(),
            content: response.clone(),
        });

        Ok(response)
    }

    fn build_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.history {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
}
