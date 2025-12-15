use super::Tool;
use crate::error::{Result, SuprascalarError};
use serde_json::{Value, json};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// 터미널 세션을 유지하며 쉘 명령어를 실행하는 도구
/// 'cd' 명령어를 통한 디렉토리 변경 상태를 기억합니다.
pub struct TerminalSession {
    // Tool은 Sync+Send여야 하므로 내부 가변성을 위해 Mutex 사용
    cwd: Mutex<PathBuf>,
}

impl TerminalSession {
    pub fn new() -> Self {
        Self {
            // 초기 시작 위치: 현재 프로세스의 작업 디렉토리
            cwd: Mutex::new(env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))),
        }
    }

    /// LLM 컨텍스트 보호를 위한 출력 제한 (앞 1000자 + ... + 뒤 1000자)
    fn truncate_output(output: String) -> String {
        const MAX_CHARS: usize = 2000;
        if output.len() > MAX_CHARS {
            let half = MAX_CHARS / 2;
            let start = &output[..half];
            let end = &output[output.len() - half..];
            format!(
                "{}\n... [Output truncated. Total: {} chars. Use 'head'/'tail' or specific file tools.] ...\n{}",
                start,
                output.len(),
                end
            )
        } else {
            output
        }
    }
}

impl Tool for TerminalSession {
    fn name(&self) -> &str {
        "run_shell_command"
    }

    fn description(&self) -> &str {
        "Executes a shell command on the system. Maintains directory state for 'cd'. \
        Use this for navigation (ls, cd, pwd) and system operations. \
        Note: Output is truncated if too long."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (e.g., 'ls -la', 'cd src', 'cargo test')"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        // 1. 명령어 파싱
        let command_str = args["command"]
            .as_str()
            .ok_or_else(|| SuprascalarError::Unknown("Missing 'command' parameter".to_string()))?;

        // 2. 'cd' 명령어 가로채기 (상태 업데이트 로직)
        // 실제 쉘에서 'cd'를 실행하면 자식 프로세스만 이동하고 끝나버리기 때문에 여기서 직접 처리해야 함
        let trimmed_cmd = command_str.trim();
        if trimmed_cmd.starts_with("cd ") || trimmed_cmd == "cd" {
            let target = if trimmed_cmd == "cd" {
                "~" // 홈 디렉토리로 이동 의도
            } else {
                trimmed_cmd.strip_prefix("cd ").unwrap().trim()
            };

            // 상태 변경은 짧게 잠금 후 처리
            let mut cwd_guard = self.cwd.lock().map_err(|_| {
                SuprascalarError::Unknown("Failed to lock terminal state".to_string())
            })?;

            let resolved = resolve_cd_target(&cwd_guard, target)?;
            let canonical = resolved.canonicalize().map_err(SuprascalarError::Io)?;
            *cwd_guard = canonical;
            return Ok(format!("Changed directory to: {}", cwd_guard.display()));
        }

        // 3. 일반 명령어 실행 (subprocess)
        // 실행 디렉토리는 잠금 상태에서 복사하고, 실행 중에는 잠금을 해제한다.
        let run_dir = {
            let cwd_guard = self.cwd.lock().map_err(|_| {
                SuprascalarError::Unknown("Failed to lock terminal state".to_string())
            })?;
            cwd_guard.clone()
        };

        let output_result = if cfg!(target_os = "windows") {
            Command::new("cmd")
                .args(["/C", command_str])
                .current_dir(run_dir)
                .output()
        } else {
            Command::new("sh")
                .arg("-c")
                .arg(command_str)
                .current_dir(run_dir)
                .output()
        };

        // 4. 결과 처리 및 포맷팅
        match output_result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                let combined = if output.status.success() {
                    if stdout.trim().is_empty() {
                        "(Command executed successfully with no output)".to_string()
                    } else {
                        stdout.into_owned()
                    }
                } else {
                    format!(
                        "Command failed (Exit Code: {}):\n{}",
                        output.status.code().unwrap_or(-1),
                        stderr
                    )
                };

                Ok(Self::truncate_output(combined))
            }
            Err(e) => Err(SuprascalarError::Io(e)),
        }
    }
}

fn resolve_cd_target(current_cwd: &Path, target: &str) -> Result<PathBuf> {
    // '~' 또는 '~/subdir' 처리
    if target == "~" || target.starts_with("~/") {
        let home = env::var("HOME")
            .map(PathBuf::from)
            .map_err(|_| SuprascalarError::Unknown("HOME is not set".to_string()))?;

        if target == "~" {
            return Ok(home);
        }

        let remainder = target.trim_start_matches("~/");
        return Ok(home.join(remainder));
    }

    // 절대 경로는 그대로 사용, 상대 경로는 현재 cwd에 결합
    let candidate = PathBuf::from(target);
    if candidate.is_absolute() {
        Ok(candidate)
    } else {
        Ok(current_cwd.join(candidate))
    }
}
