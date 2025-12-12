use super::LLMBackend;
use crate::error::{Result, SuprascalarError};

use crate::candle_transformers_patched::quantized_qwen3::ModelWeights as Qwen3;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub struct CandleQwen {
    model: Qwen3,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    device: Device,
}

impl CandleQwen {
    pub fn new(repo: &str, model_file: &str, tokenizer_repo: &str) -> Result<Self> {
        let device = Device::new_cuda(0)?;
        //huggingface api
        let api = Api::new()?;

        //tokenizer
        let tokenizer_path = api
            .model(tokenizer_repo.to_string())
            .get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| SuprascalarError::Tokenizer(e.to_string()))?;

        //model
        let model_path = api.model(repo.to_string()).get(model_file)?;
        let mut file = std::fs::File::open(&model_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let model = Qwen3::from_gguf(content, &mut file, &device)?;

        let logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.95));

        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            device,
        })
    }
}

impl LLMBackend for CandleQwen {
    fn generate(&mut self, prompt: &str) -> Result<String> {
        self.model.clear_kv_cache();

        // Tokenizer errors need manual mapping to SuprascalarError::Tokenizer
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| SuprascalarError::Tokenizer(e.to_string()))?;

        let tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = Vec::new();

        // Check context limit (Example of using the custom error)
        if tokens.len() > 32000 {
            return Err(SuprascalarError::ContextLimitExceeded {
                limit: 32000,
                current: tokens.len(),
            });
        }

        let mut input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut pos = 0;

        // Simplified loop
        for _ in 0..1000 {
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            let next_token = self.logits_processor.sample(&logits)?;

            // tokens.push(next_token);
            generated_tokens.push(next_token);

            // Break on EOS (Simplified)
            if next_token == self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(0)
                || next_token == self.tokenizer.token_to_id("<|im_end|>").unwrap_or(0)
            {
                break;
            }
            let (_b, seq_len) = input.dims2()?;
            pos += seq_len;
            input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }

        let result = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| SuprascalarError::Tokenizer(e.to_string()))?;

        Ok(result)
    }
}
