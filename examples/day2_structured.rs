use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;
use hf_hub::api::sync::Api;
use serde::{Deserialize, Serialize};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Deserialize)]
struct AgentAction {
    tool: String,
    args: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔥 Day 2: Structured Output (Smart Mode)");

    // --- 1. 모델 로딩 (동일) ---
    let api = Api::new()?;
    let tokenizer_path = api
        .model("microsoft/Phi-3-mini-4k-instruct".to_string())
        .get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    let model_path = api
        .model("bartowski/Phi-3-mini-4k-instruct-GGUF".to_string())
        .get("Phi-3-mini-4k-instruct-Q4_K_M.gguf")?;

    let device = Device::new_cuda(0)?;
    let mut file = std::fs::File::open(&model_path)?;
    let model_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mut model = Phi3::from_gguf(false, model_content, &mut file, &device)?;

    // Temp를 0으로 하면 창의성이 죽어서 대화가 딱딱해질 수 있으니 살짝 올림 (0.1)
    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.1), None);

    // --- 2. 시스템 프롬프트 수정 (빠져나갈 구멍 제공) ---
    let system_prompt = r#"You are an intelligent assistant.
You can call tools to answer questions, or just chat normally.

Tools available:
- get_weather(location: str): Use this ONLY when asked about weather.
- calculator(expression: str): Use this ONLY for math.

RESPONSE FORMAT:
1. If you need a tool, output JSON: {"tool": "tool_name", "args": "arguments"}
2. If you just want to chat, output JSON: {"tool": "final_answer", "args": "Your chat response here"}

Do not output anything else."#;

    // 테스트 1: 도구가 필요한 질문
    // let user_msg = "What is 25 * 4?";
    // 테스트 2: 그냥 인사 (주석 풀고 테스트 해보세요)
    let user_msg = "Hello! Who are you?";

    let prompt = format!(
        "<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>",
        system_prompt, user_msg
    );

    println!("Query: {}", user_msg);
    print!("Thinking... ");
    std::io::stdout().flush()?;

    // --- 3. 추론 ---
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let mut tokens = tokens.get_ids().to_vec();
    let prompt_len = tokens.len();
    let mut pos = 0;
    let mut input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;

    let mut full_response = String::new();

    for _ in 0..200 {
        let logits = model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?;

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);

        let decoded = tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(E::msg)?;
        full_response = decoded.clone();

        // [핵심 수정] JSON이 닫히면('}') 즉시 멈춤.
        // Phi-3는 가끔 JSON 뒤에 설명을 덧붙이는 버릇이 있어서 여기서 끊어야 함.
        if full_response.trim().ends_with('}') {
            break;
        }

        if next_token == 32000 || next_token == 32007 {
            break;
        }

        let (_b, seq_len) = input.dims2()?;
        pos += seq_len;
        input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
    }

    println!("\nRaw LLM Output: {}", full_response);

    // --- 4. 파싱 및 분기 처리 ---
    let clean_json = full_response.trim();
    // 가끔 ```json ... ``` 으로 감싸서 줄 때가 있어서 제거
    let clean_json = clean_json
        .trim_start_matches("```json")
        .trim_end_matches("```")
        .trim();

    match serde_json::from_str::<AgentAction>(clean_json) {
        Ok(action) => {
            if action.tool == "final_answer" {
                println!("\n💬 Chat: {}", action.args);
            } else {
                println!("\n🛠️ Tool Call: {} with args {}", action.tool, action.args);
                if action.tool == "calculator" {
                    println!("   -> Calling calculator logic...");
                }
            }
        }
        Err(e) => {
            println!("\n❌ Failed to parse JSON: {}", e);
            println!("Response was: {}", clean_json);
        }
    }

    Ok(())
}
