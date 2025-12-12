use anyhow::{Error as E, Result};
use candle_core::backend::BackendDevice;
use candle_core::{Device, IndexOp, Tensor};

// Assuming you have the patched Qwen3 or wrapper with forward_speculative
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;
use suprascalar::candle_transformers_patched::quantized_qwen3::ModelWeights as Qwen3;

use hf_hub::api::sync::Api;
use std::io::Write;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

enum Model {
    Qwen2(Qwen2),
    Qwen3(Qwen3),
}

impl Model {
    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        match self {
            Model::Qwen2(m) => m.forward(x, offset).map_err(E::from),
            Model::Qwen3(m) => m.forward(x, offset).map_err(E::from),
        }
    }
    // Assumes this returns [Batch, Seq_Len, Vocab]
    fn forward_speculative(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        match self {
            Model::Qwen2(m) => m.forward(x, offset).map_err(E::from),
            // Model::Qwen3(m) => m.forward_speculative(x, offset).map_err(E::from),
            Model::Qwen3(m) => m.forward(x, offset).map_err(E::from),
        }
    }
}

// ... [ModelType, Engine struct, Engine::new implementations are same as before] ...
enum ModelType {
    Qwen2,
    Qwen3,
}
struct Engine {
    model: Model,
    device: Device,
}
impl Engine {
    fn new(repo: &str, model_file: &str, device: &Device, model_type: ModelType) -> Result<Self> {
        let api = Api::new()?;
        let model_path = api.model(repo.to_string()).get(model_file)?;
        let mut file = std::fs::File::open(&model_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let model = match model_type {
            ModelType::Qwen2 => Model::Qwen2(Qwen2::from_gguf(content, &mut file, device)?),
            ModelType::Qwen3 => Model::Qwen3(Qwen3::from_gguf(content, &mut file, device)?),
        };
        Ok(Self {
            model,
            device: device.clone(),
        })
    }
}

#[derive(Default)]
struct PerfStats {
    draft_forward: Duration,
    verifier_chunk: Duration,
    // 전체 Step 4(보너스/싱크 포함) 타이밍
    verifier_resync: Duration,
    // verifier.model.forward(...) + verifier device sync 만 측정한 타이밍
    verifier_resync_verifier_only: Duration,
}

fn sync_device(device: &Device) -> Result<()> {
    match device {
        Device::Cuda(dev) => dev.synchronize().map_err(E::from),
        _ => Ok(()),
    }
}

fn run_speculative(
    draft: &mut Engine,
    verifier: &mut Engine,
    tokenizer: &Tokenizer,
    prompt: &str,
    n_tokens: usize,
    k_draft: usize,
) -> Result<()> {
    println!("\n🚀 Speculative Decoding (GPU-Resident Optimization)");
    println!("Prompt: {}\n---", prompt);

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut generated_cnt = 0;
    let mut draft_pos = 0;
    let mut verifier_pos = 0;
    let mut last_printed = 0;
    let mut total_drafted = 0;
    let mut total_draft_accepted = 0;
    let mut total_positions_accepted = 0;
    let mut total_bonus = 0;
    let mut stats = PerfStats::default();
    // Count verifier forwards (including speculative)
    let mut verifier_forward_count_total: usize = 0;
    let mut verifier_forward_speculative_count: usize = 0;

    let mut current_k = k_draft.max(1);
    const MAX_K: usize = 8;
    const MIN_K: usize = 1;
    const ADJUST_WINDOW: usize = 12;
    let mut adjust_acc_sum = 0f32;
    let mut adjust_cnt = 0usize;

    print!("{}", prompt);
    std::io::stdout().flush()?;

    // 1. Initial Prompt Processing (Prefill)
    // We treat this normally to get the KV cache ready
    let input = Tensor::new(tokens.as_slice(), &verifier.device)?.unsqueeze(0)?;

    // Draft Prefill
    let t_pre = Instant::now();
    let draft_prefill_logits = draft.model.forward(&input, 0)?;
    sync_device(&draft.device)?;
    stats.draft_forward += t_pre.elapsed();

    let mut last_draft_logits = draft_prefill_logits.squeeze(0)?;
    // [Optimized] Keep token on GPU to avoid Sync
    let mut draft_init_token_tensor = last_draft_logits.argmax(0)?.reshape((1, 1))?;

    // 🔥 중요: 첫 턴의 Verifier 결과(Logits)를 저장해둬야 함 (첫 Draft 검증용)
    // let t_pre_v = Instant::now();
    // Extract the last token from `input` as a [1, 1] tensor on the verifier device
    let mut bonus_token_tensor = input
        .narrow(1, tokens.len().saturating_sub(1), 1)?
        .reshape((1, 1))?;
    let _ = verifier.model.forward(&input, 0)?;
    verifier_forward_count_total += 1;
    // sync_device(&verifier.device)?;
    // stats.verifier_chunk += t_pre_v.elapsed();

    // let mut last_verifier_logits = logits.squeeze(0)?; // [vocab]

    draft_pos += tokens.len();
    verifier_pos += tokens.len() - 1;

    while generated_cnt < n_tokens {
        let remaining = n_tokens - generated_cnt;
        let step_k = remaining.min(current_k).max(1);

        // ================================================================
        // Step 1: Sequential Drafting (🔥 GPU-Resident Loop Optimized)
        // ================================================================
        // CPU 대기(Sync) 없이 GPU 안에서만 텐서를 돌립니다.
        let t_draft = Instant::now();

        // 1. 초기 토큰 설정 (GPU Resident)
        // draft_init_token_tensor is already [1, 1] on GPU
        let mut current_input = draft_init_token_tensor.clone(); //굳이 필요없을 수도

        // [Optimized] Pre-allocate verify_input_gpu
        // We need to store [init, draft_1, draft_2, ...]
        // Total length = step_k
        // We can use a pre-allocated tensor and update it.
        // Note: DType must match. Tokenizer produces u32.
        // But draft_init_token_tensor is u32?
        // Let's check dtype.
        let dtype = current_input.dtype();
        let mut verify_input_gpu = Tensor::zeros((1, step_k + 1), dtype, &draft.device)?;

        // Set first token
        verify_input_gpu = verify_input_gpu.slice_assign(&[0..1, 0..1], &bonus_token_tensor)?;
        verify_input_gpu = verify_input_gpu.slice_assign(&[0..1, 1..2], &current_input)?;

        for i in 1..step_k {
            // A. Forward (Async Kernel Launch)
            let logits = draft.model.forward(&current_input, draft_pos)?;

            // B. Argmax (GPU Operation)
            let next_token_tensor = logits.squeeze(0)?.argmax(0)?.reshape((1, 1))?;

            // C. 저장 (In-place update)
            verify_input_gpu =
                verify_input_gpu.slice_assign(&[0..1, i + 1..i + 2], &next_token_tensor)?;

            // D. 다음 입력 준비
            current_input = next_token_tensor;

            draft_pos += 1;
        }
        sync_device(&draft.device)?;
        stats.draft_forward += t_draft.elapsed();

        // ================================================================
        // Step 2: Verifier 병렬 검증
        // ================================================================
        let t_verify = Instant::now();

        // verify_input_gpu is already ready!

        // 2. 현재 pos에서 forward
        let verifier_logits = verifier
            .model
            .forward_speculative(&verify_input_gpu, verifier_pos)?;
        verifier_forward_count_total += 1;
        verifier_forward_speculative_count += 1;

        sync_device(&verifier.device)?;
        stats.verifier_chunk += t_verify.elapsed();

        let verifier_logits = verifier_logits.squeeze(0)?; // [step_k, vocab]

        // ================================================================
        // Step 3: Comparison Loop (Vectorized Logic)
        // ================================================================

        // [Optimized] 이제 여기서 한 번에 CPU로 가져옵니다 (Batch Sync)
        let draft_tokens = verify_input_gpu
            .squeeze(0)?
            .narrow(0, 1, step_k)?
            .to_vec1::<u32>()?;

        // ref_logits : [step_k, vocab]
        let ref_logits = verifier_logits.narrow(0, 0, step_k)?;

        // pred_tokens : [step_k]
        let pred_tokens = ref_logits.argmax(1)?;
        let pred_tokens = pred_tokens.to_vec1::<u32>()?;

        // 최초 불일치 지점 찾기
        let mismatch_idx = draft_tokens
            .iter()
            .zip(pred_tokens.iter())
            .position(|(draft_tok, pred_tok)| draft_tok != pred_tok);

        let mut accepted_from_draft = 0usize;
        let mut positions_advanced;
        let mut final_token: Option<u32> = None;

        match mismatch_idx {
            Some(idx) => {
                // 앞부분은 그대로 수락
                if idx > 0 {
                    tokens.extend_from_slice(&draft_tokens[..idx]);
                    accepted_from_draft += idx;
                }
                // 불일치 지점에서는 Verifier 토큰으로 교체
                let replace_tok = pred_tokens[idx];
                tokens.push(replace_tok);
                final_token = Some(replace_tok);
            }
            None => {
                // 전부 일치 → 모두 수락
                tokens.extend_from_slice(&draft_tokens);
                accepted_from_draft += draft_tokens.len();
            }
        }
        positions_advanced = accepted_from_draft + usize::from(final_token.is_some());

        // ================================================================
        // Step 4: Bonus Token or Sync
        // - We measure two timings:
        //   1) `t_resync` : total time for the Step 4 block (legacy)
        //   2) `verifier-only` : only the verifier.model.forward + its device sync
        // ================================================================
        let t_resync = Instant::now();
        if final_token.is_none() {
            // All Accepted! -> Bonus Token
            let bonus_logits = verifier_logits.i(step_k)?;
            let bonus_token = bonus_logits.argmax(0)?.to_scalar::<u32>()?;
            bonus_token_tensor = bonus_logits.argmax(0)?.reshape((1, 1))?;

            tokens.push(bonus_token);
            positions_advanced += 1;
            total_bonus += 1;

            // 다음 턴을 위해 Logit 저장 (verifier-only timing 측정)
            // let input = Tensor::new(&[bonus_token], &verifier.device)?.unsqueeze(0)?;
            // let t_verifier_only = Instant::now();
            // let logits = verifier.model.forward(&input, verifier_pos + step_k)?; //여기~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            // sync_device(&verifier.device)?;
            // stats.verifier_resync_verifier_only += t_verifier_only.elapsed();
            // last_verifier_logits = logits.squeeze(0)?;

            // Draft 모델 싱크 맞추기
            let len = tokens.len();
            let last_two = &tokens[len - 2..len];
            let input = Tensor::new(last_two, &draft.device)?.unsqueeze(0)?;
            last_draft_logits = draft.model.forward(&input, draft_pos)?;
            sync_device(&draft.device)?;
            last_draft_logits = last_draft_logits.squeeze(0)?;
            // [Optimized] Keep as Tensor
            draft_init_token_tensor = last_draft_logits.argmax(0)?.reshape((1, 1))?;
            draft_pos += 2;
        } else {
            // Rejected -> Correction & Sync
            let accepted_idx = accepted_from_draft;

            // Draft 모델 싱크 맞추기 & 다음 턴 검증용 Logit 계산
            let correct_token = final_token.unwrap();
            let input = Tensor::new(&[correct_token], &verifier.device)?.unsqueeze(0)?;

            // Sync Draft Model State (not included in verifier-only timing)
            let draft_input = input.clone();
            last_draft_logits = draft
                .model
                .forward(&draft_input, verifier_pos + accepted_idx)?;
            sync_device(&draft.device)?;
            last_draft_logits = last_draft_logits.squeeze(0)?;
            // [Optimized] Keep as Tensor
            draft_init_token_tensor = last_draft_logits.argmax(0)?.reshape((1, 1))?;

            // Reset Draft Pos to correct position
            draft_pos = verifier_pos + accepted_idx + 1;

            // Verifier: 다음 턴 검증용 Logit 계산 (verifier-only timing)
            bonus_token_tensor = input;
            // let t_verifier_only = Instant::now();
            // let logits = verifier
            // .model
            // .forward(&input, verifier_pos + accepted_idx)?; //여기~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            // sync_device(&verifier.device)?;
            // stats.verifier_resync_verifier_only += t_verifier_only.elapsed();
            // last_verifier_logits = logits.squeeze(0)?;
        }
        // 기존의 total Step 4 타이밍도 남겨둡니다.
        stats.verifier_resync += t_resync.elapsed();

        total_drafted += step_k;
        total_draft_accepted += accepted_from_draft;
        total_positions_accepted += positions_advanced;

        verifier_pos += positions_advanced;
        generated_cnt += positions_advanced;

        // Print
        let text = tokenizer
            .decode(&tokens[last_printed..], true)
            .map_err(E::msg)?;
        if !text.is_empty() {
            print!("{}", text);
            std::io::stdout().flush()?;
            last_printed = tokens.len();
        }

        let acc_ratio = accepted_from_draft as f32 / step_k as f32;
        adjust_acc_sum += acc_ratio;
        adjust_cnt += 1;
        if adjust_cnt == ADJUST_WINDOW {
            let avg = adjust_acc_sum / ADJUST_WINDOW as f32;
            if avg > 0.6 && current_k < MAX_K {
                current_k += 1;
                println!(
                    "\n⬆️ Increasing speculative window to {} (avg acceptance {:.0}%)",
                    current_k,
                    avg * 100.0
                );
            } else if avg < 0.4 && current_k > MIN_K {
                current_k -= 1;
                println!(
                    "\n⬇️ Decreasing speculative window to {} (avg acceptance {:.0}%)",
                    current_k,
                    avg * 100.0
                );
            }
            adjust_acc_sum = 0.0;
            adjust_cnt = 0;
        }
    }

    println!("\n\nDone.");
    let rate = (total_draft_accepted as f32 / total_drafted as f32) * 100.0;
    println!(
        "Acceptance Rate (draft only): {:.2}% | Bonus tokens: {} | Total advanced: {}",
        rate, total_bonus, total_positions_accepted
    );
    println!(
        "⏱️ draft {:.2?} | verifier-batch {:.2?} | verifier-sync-total {:.2?} | verifier-sync-verifier_only {:.2?}",
        stats.draft_forward,
        stats.verifier_chunk,
        stats.verifier_resync,
        stats.verifier_resync_verifier_only
    );
    println!(
        "Verifier forward calls: total={} (speculative={})",
        verifier_forward_count_total, verifier_forward_speculative_count
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔥 Speculative Decoding (Batch Verification + GPU Resident)");

    let device = Device::new_cuda(0)?;
    let api = Api::new()?;

    let tokenizer_path = api
        .model("Qwen/Qwen3-14B".to_string())
        .get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    let mut verifier = Engine::new(
        "unsloth/Qwen3-14B-GGUF",
        "Qwen3-14B-Q4_K_M.gguf",
        &device,
        ModelType::Qwen3,
    )?;

    let mut draft = Engine::new(
        "unsloth/Qwen3-0.6B-GGUF",
        "Qwen3-0.6B-Q4_K_M.gguf",
        &device,
        ModelType::Qwen3,
    )?;

    let prompt = "Explain the difference between Mutex and RwLock in Rust.";
    let start = std::time::Instant::now();

    // k_draft는 초기값일 뿐이며 루프 내부에서 수용률에 따라 자동 조정됩니다.
    run_speculative(&mut draft, &mut verifier, &tokenizer, prompt, 1000, 3)?;

    println!("\n✅ Total time: {:.2?}", start.elapsed());
    Ok(())
}
