use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
// 중요: 속도를 위해 'quantized_phi3' 모듈을 사용합니다.
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;
use hf_hub::api::sync::Api;
use std::io::Write;
use tokenizers::Tokenizer;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔥 Day 1: Suprascalar Inference Engine (Direct Logic)");

    // =========================================================================
    // 1. 모델 & 토크나이저 준비 (HF Hub 사용)
    // =========================================================================
    let api = Api::new()?;

    // (1) Tokenizer: 사용자가 지정한 Microsoft 공식 Repo 사용
    let tokenizer_repo_id = "microsoft/Phi-3-mini-4k-instruct";
    println!("📥 Fetching tokenizer from: {}", tokenizer_repo_id);
    let tokenizer_path = api
        .model(tokenizer_repo_id.to_string())
        .get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    // (2) Model Weights:
    // Microsoft 공식 Repo에는 'safetensors(7GB)'만 있고 'GGUF(양자화)' 파일이 없습니다.
    // 로컬 CPU에서 빠르게 돌리려면 GGUF가 필수이므로,
    // 동일한 모델의 GGUF 변환 버전(Bartowski)을 가져옵니다.
    let model_repo_id = "bartowski/Phi-3-mini-4k-instruct-GGUF";
    let model_filename = "Phi-3-mini-4k-instruct-Q4_K_M.gguf";

    println!(
        "📥 Fetching model weights from: {}/{}",
        model_repo_id, model_filename
    );
    let model_path = api.model(model_repo_id.to_string()).get(model_filename)?;

    // =========================================================================
    // 2. 엔진 초기화 (Boilerplate 없이 라이브러리 기능 직접 사용)
    // =========================================================================
    // let device = Device::Cpu;
    let device = Device::new_cuda(0)?;

    println!("⚙️ Loading GGUF model...");
    let mut file = std::fs::File::open(&model_path)?;
    let model_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mut model = Phi3::from_gguf(false, model_content, &mut file, &device)?; // disable flash attention (CPU-friendly)

    // Candle 내장 LogitsProcessor (Temperature, Top-P, Seed 설정)
    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.95));

    println!("✅ Engine Ready!");

    // =========================================================================
    // 3. 추론 실행 (Reference의 run() 함수 로직을 main으로 가져옴)
    // =========================================================================
    let prompt = "<|user|>\nHow to make cake?.<|end|>\n<|assistant|>";
    println!("\nGenerating response for: \n{}", prompt);
    println!("---");

    // (1) Encode
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let mut tokens = tokens.get_ids().to_vec();
    let prompt_len = tokens.len();
    let mut generated_tokens = 0usize;
    let sample_len = 200; // 최대 생성 길이

    print!("{}", prompt);
    std::io::stdout().flush()?;
    let mut last_printed = 0usize;

    let start_gen = std::time::Instant::now();

    // [수정 1] 위치(Position) 추적 변수 선언
    let mut pos = 0;

    // [수정 2] 첫 입력은 '프롬프트 전체'입니다.
    // Tensor::new(tokens.as_slice()...) -> 프롬프트 전체 토큰
    let mut input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;

    // (2) Generation Loop
    for _ in 0..sample_len {
        // [수정 3] model.forward에 'pos' 인자 추가
        // input: 이번에 처리할 토큰들 (첫 턴엔 프롬프트 전체, 그 뒤론 토큰 1개)
        // pos: 이 토큰들이 전체 문장에서 시작되는 위치
        let logits = model.forward(&input, pos)?;

        // 로짓 추출 (마지막 토큰의 예측값)
        let logits = logits.squeeze(0)?; // 모델이 배치 차원만 남기므로 seq 차원 제거

        // Sampling
        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;

        // 누적 디코딩을 통해 공백을 포함한 원문 형태를 복원한다.
        let decoded = tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(E::msg)?;
        let new_text = &decoded[last_printed..];
        print!("{}", new_text);
        std::io::stdout().flush()?;
        last_printed = decoded.len();

        if next_token == 32000 || next_token == 32007 {
            break;
        }

        // [수정 4] 다음 턴 준비
        // pos 업데이트: 방금 처리한 입력 길이만큼 더해줌 (첫 턴: 프롬프트 길이, 이후: 1)
        let (_b, seq_len) = input.dims2()?;
        pos += seq_len;

        // input 업데이트: 이제부터는 '방금 만든 토큰 하나'만 모델에 넣습니다. (KV Cache 활용)
        input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
    }

    let dt = start_gen.elapsed();
    println!(
        "\n\n---\n⚡ {} tokens generated ({:.2} token/s)",
        generated_tokens,
        generated_tokens as f64 / dt.as_secs_f64(),
    );

    Ok(())
}
