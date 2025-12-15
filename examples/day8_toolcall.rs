use std::io::{self, Write};
use suprascalar::{Agent, CandleQwen, SuprascalarError};

fn main() -> Result<(), SuprascalarError> {
    // 1. 모델 설정 (day6_simple_agent와 동일)
    let repo = "unsloth/Qwen3-14B-GGUF";
    let model_file = "Qwen3-14B-Q4_K_M.gguf";
    let tokenizer_repo = "Qwen/Qwen3-14B";

    println!(">>> Loading Model (This may take a while)...");

    // 2. 백엔드 초기화
    let backend = match CandleQwen::new(repo, model_file, tokenizer_repo) {
        Ok(b) => Box::new(b),
        Err(SuprascalarError::Io(e)) => {
            eprintln!("Failed to load model files: {}", e);
            return Err(SuprascalarError::Io(e));
        }
        Err(e) => return Err(e),
    };

    // 3. 에이전트 생성
    // Agent 내부의 'history' 벡터가 대화 내용을 저장합니다.
    let mut agent = Agent::new(
        "Suprascalar",
        backend,
        "You are Suprascalar, an intelligent and helpful AI assistant running locally.",
    );
    agent.register_tool(suprascalar::tools::ls::ListDirectory::new());

    println!(">>> Suprascalar is ready! (Type '/exit' or 'quit' to stop)");
    println!("------------------------------------------------------------");

    // 4. 대화 루프 (REPL: Read-Eval-Print Loop)
    loop {
        // 프롬프트 출력 (줄바꿈 없이)
        print!("User: ");
        io::stdout().flush().map_err(SuprascalarError::Io)?;

        // 사용자 입력 받기
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(SuprascalarError::Io)?;
        let input = input.trim();

        // 종료 명령어 처리
        if input.eq_ignore_ascii_case("/exit") || input.eq_ignore_ascii_case("quit") {
            println!(">>> Saving context and exiting... Goodbye!");
            break;
        }

        if input.is_empty() {
            continue;
        }

        // 에이전트와 대화
        // agent.chat() 내부에서 history에 user 메시지와 assistant 메시지가 누적됩니다.
        print!("Agent: ");
        io::stdout().flush().map_err(SuprascalarError::Io)?;

        match agent.chat(input) {
            Ok(response) => {
                // response는 이미 출력되었거나 여기서 출력할 수 있습니다.
                // 현재 agent.chat 구현상 내부에서 스트리밍을 안 한다면 여기서 출력해야 합니다.
                // 만약 agent.chat이 단순히 String을 반환한다면:
                println!("{}", response);
            }
            Err(SuprascalarError::ContextLimitExceeded { limit, current }) => {
                eprintln!("\n[Error] Context limit reached! ({}/{})", current, limit);
                eprintln!("Consider summarizing the history or restarting.");
                // 여기서 오래된 기억을 지우는 로직(Memory Management)을 추가할 수 있습니다.
                break;
            }
            Err(e) => eprintln!("\n[Error] {}", e),
        }

        println!("------------------------------------------------------------");
    }

    Ok(())
}
