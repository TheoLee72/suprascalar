use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

// [수정] 사용자가 요청한 Qwen 전용 모듈 사용 (Llama 집착 버림)
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
// use candle_transformers::models::qwen3_moe::ModelWeights as
// use suprascalar::candle_patched::quantized_qwen3::ModelWeights as Qwen3;

use hf_hub::api::sync::Api;
use std::io::Write;
use tokenizers::Tokenizer;

// 두 모델을 아우르는 Enum 정의
enum Model {
    Qwen2(Qwen2),
    Qwen3(Qwen3),
}

impl Model {
    fn forward(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        match self {
            Model::Qwen2(m) => m.forward(x, pos).map_err(E::from),
            Model::Qwen3(m) => m.forward(x, pos).map_err(E::from),
        }
    }
}

// 모델 로드 시 어떤 타입인지 구분하기 위한 플래그
enum ModelType {
    Qwen2,
    Qwen3,
}

struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    name: String,
}

impl Engine {
    fn new(
        name: &str,
        repo: &str,
        model_file: &str,
        tokenizer_repo: &str,
        device: &Device,
        model_type: ModelType, // 모델 타입 지정
    ) -> Result<Self> {
        println!("⏳ Loading [{}]...", name);
        let api = Api::new()?;

        // 1. 토크나이저 로드
        let tokenizer_path = api
            .model(tokenizer_repo.to_string())
            .get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        // 2. 모델 로드 (GGUF)
        let model_path = api.model(repo.to_string()).get(model_file)?;
        let mut file = std::fs::File::open(&model_path)?;
        // 첨부해주신 파일(qwen2.rs, qwen3.rs)의 로직을 그대로 따름
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

        // 3. 타입에 따라 적절한 모듈 사용
        let model = match model_type {
            ModelType::Qwen2 => {
                let m = Qwen2::from_gguf(content, &mut file, device)?;
                Model::Qwen2(m)
            }
            ModelType::Qwen3 => {
                let m = Qwen3::from_gguf(content, &mut file, device)?;
                Model::Qwen3(m)
            }
        };

        println!("✅ [{}] Loaded!", name);
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            name: name.to_string(),
        })
    }

    fn generate_one(&mut self, prompt: &str) -> Result<()> {
        println!("\n🤖 Generating with [{}]:", self.name);

        // Qwen 채팅 템플릿 적용
        let formatted_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );
        print!("{}", formatted_prompt);
        std::io::stdout().flush()?;

        let tokens = self
            .tokenizer
            .encode(formatted_prompt, true)
            .map_err(E::msg)?;
        let mut tokens = tokens.get_ids().to_vec();
        let mut input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut pos = 0;
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), None);

        for _ in 0..1000 {
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            let decoded = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            print!("{}", decoded);
            std::io::stdout().flush()?;

            let (_b, seq_len) = input.dims2()?;
            pos += seq_len;
            input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }
        println!("\n... (stopped)");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔥 Day 4: Dual Brain Loading (Qwen3 0.6B & Qwen3 14B)");
    println!("--------------------------------------------------");

    let device = Device::new_cuda(0)?;

    // 1. Verifier (Main): Qwen3-14B (Using quantized_qwen3)
    // [수정] qwen3.rs 참조하여 14B 모델 경로 설정
    let mut verifier = Engine::new(
        "Verifier (Qwen3-14B)",
        "unsloth/Qwen3-14B-GGUF", // Repo ID
        // "Qwen3-14B-Q6_K.gguf",    // File Name
        "Qwen3-14B-Q4_K_M.gguf",
        "Qwen/Qwen3-14B", // Tokenizer Repo
        &device,
        ModelType::Qwen3, // Qwen3 모듈 사용
    )?;
    // let mut verifier = Engine::new(
    //     "Draft (DeepSeek-R1-Distill-Qwen)",
    //     "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    //     "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    //     "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    //     &device,
    //     ModelType::Qwen2,
    // )?;

    // let mut verifier = Engine::new(
    //     "Verifier (Qwen3-30B-A3B)",
    //     "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    //     "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
    //     "Qwen/Qwen3-30B-A3B-Instruct-2507",
    //     &device,
    //     ModelType::Qwen3,
    // )?;

    // 2. Draft (Fast): Qwen3-0.6B (Using quantized_qwen3)
    // // 이건 qwen3.rs 파일에 있던 경로 그대로 유지
    // let mut draft = Engine::new(
    //     "Draft (Qwen3-0.6B)",
    //     "unsloth/Qwen3-0.6B-GGUF",
    //     "Qwen3-0.6B-Q4_K_M.gguf",
    //     "Qwen/Qwen3-0.6B",
    //     &device,
    //     ModelType::Qwen3,
    // )?;

    println!("--------------------------------------------------");
    println!("🎉 Success! Qwen3-0.6B (Draft) & Qwen3-32B (Verifier) loaded.");
    println!("--------------------------------------------------");

    let prompt = "Explain the difference between Mutex and RwLock in Rust.";

    // // Draft 모델 속도 측정
    // let start = std::time::Instant::now();
    // draft.generate_one(prompt)?;
    // println!("Draft Latency: {:.2?}", start.elapsed());

    // Verifier 모델 속도 측정
    let start = std::time::Instant::now();
    verifier.generate_one(prompt)?;
    println!("Verifier Latency: {:.2?}", start.elapsed());

    Ok(())
}
