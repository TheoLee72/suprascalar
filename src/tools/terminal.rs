use super::Tool;
use crate::error::{Result, SuprascalarError};
use regex::Regex;
use serde_json::{Value, json};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// 터미널 세션을 유지하며 쉘 명령어를 실행하는 도구
/// Safety Layer 포함: 위험 명령어 차단 및 Git 자동 스냅샷 기능
pub struct TerminalSession {
    cwd: Mutex<PathBuf>,
    safety_enabled: bool,
}

impl TerminalSession {
    pub fn new() -> Self {
        Self {
            // 초기 시작 위치: 현재 프로세스의 작업 디렉토리
            cwd: Mutex::new(env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))),
            safety_enabled: true, // 기본적으로 안전 모드 켜짐
        }
    }

    /// LLM 컨텍스트 보호를 위한 출력 제한
    fn truncate_output(output: String) -> String {
        const MAX_CHARS: usize = 2000;
        if output.len() > MAX_CHARS {
            let half = MAX_CHARS / 2;
            let start = &output[..half];
            let end = &output[output.len() - half..];
            format!(
                "{}\n... [Output truncated. Total: {} chars] ...\n{}",
                start,
                output.len(),
                end
            )
        } else {
            output
        }
    }

    /// [Safety 1] 위험한 명령어 감지 (Blocklist)
    fn check_safety(&self, cmd: &str) -> Result<()> {
        if !self.safety_enabled {
            return Ok(());
        }

        // 위험한 명령어 패턴 정의
        let dangerous_patterns = [
            (r"rm\s+-[rRf]+", "Recursive deletion (rm -rf) is forbidden."),
            (r"mkfs", "Formatting filesystems is forbidden."),
            (r"dd\s+if=", "Low-level disk access (dd) is forbidden."),
            (r":\(\)\{\s*:\|:&", "Fork bombs are forbidden."),
            // 인터랙티브 도구는 에이전트를 멈추게 하므로 차단
            (
                r"(^|\s)vim?(\s|$)",
                "Interactive editors (vim) act blocking.",
            ),
            (
                r"(^|\s)nano(\s|$)",
                "Interactive editors (nano) act blocking.",
            ),
            (r"(^|\s)sudo(\s|$)", "Root privileges (sudo) are forbidden."),
        ];

        for (pattern, reason) in dangerous_patterns {
            // 정규식 컴파일 (실제로는 lazy_static 등으로 최적화 가능하지만 여기선 단순화)
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(cmd) {
                    return Err(SuprascalarError::CommandBlocked {
                        command: cmd.to_string(),
                        reason: reason.to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// [Safety 2] 실행 전 Git 자동 커밋 (Snapshot)
    /// 현재 작업 디렉토리가 Git 저장소이고 변경사항이 있다면 커밋을 생성합니다.
    fn create_git_snapshot(&self, dir: &Path, cmd_context: &str) {
        // 1. 해당 디렉토리가 Git 저장소인지 확인 (.git 폴더 존재 여부)
        // 간단한 체크: 현재 폴더에 .git이 있거나, git status가 성공하면 저장소임.
        let status_check = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(dir)
            .output();

        if let Ok(output) = status_check {
            // git 명령어가 실패했거나(저장소 아님), 변경사항이 없으면(빈 stdout) 리턴
            if !output.status.success() || output.stdout.is_empty() {
                return;
            }

            // 2. 변경사항이 확인되면 자동 커밋 진행
            // Stage all changes
            let _ = Command::new("git")
                .args(["add", "."])
                .current_dir(dir)
                .output();

            // Commit
            let commit_msg = format!("Suprascalar Auto-save: Before running '{}'", cmd_context);
            let _ = Command::new("git")
                .args(["commit", "-m", &commit_msg])
                .current_dir(dir)
                .output();

            // 디버깅용 출력 (필요시 주석 해제)
            // println!(">> [Safety] Auto-saved changes via Git.");
        }
    }
}

impl Tool for TerminalSession {
    fn name(&self) -> &str {
        "run_shell_command"
    }

    fn description(&self) -> &str {
        "Executes a shell command. Use for ls, cd, grep, etc. \
        Dangerous commands (rm -rf, sudo) are blocked. \
        Git snapshots are created automatically before file modifications."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        // 1. 명령어 파싱
        let command_str = args["command"].as_str().ok_or_else(|| {
            SuprascalarError::InvalidToolInput("Missing 'command' parameter".to_string())
        })?;

        // [Safety 1] 금지어 검사
        self.check_safety(command_str)?;

        // 2. 'cd' 명령어 가로채기
        let trimmed_cmd = command_str.trim();
        if trimmed_cmd.starts_with("cd ") || trimmed_cmd == "cd" {
            let target = if trimmed_cmd == "cd" {
                "~"
            } else {
                trimmed_cmd.strip_prefix("cd ").unwrap().trim()
            };

            let mut cwd_guard = self.cwd.lock().map_err(|_| {
                SuprascalarError::TerminalState("Failed to lock terminal state".to_string())
            })?;

            let resolved = resolve_cd_target(&cwd_guard, target)?;
            // 경로 존재 여부 확인 (canonicalize)
            let canonical = resolved.canonicalize().map_err(SuprascalarError::Io)?;

            *cwd_guard = canonical;
            return Ok(format!("Changed directory to: {}", cwd_guard.display()));
        }

        // 3. 일반 명령어 실행 준비
        // Mutex 락을 잠깐 잡아서 경로만 복사 (실행 중에는 락 해제)
        let run_dir = {
            let cwd_guard = self.cwd.lock().map_err(|_| {
                SuprascalarError::TerminalState("Failed to lock terminal state".to_string())
            })?;
            cwd_guard.clone()
        };

        // [Safety 2] Git 스냅샷 생성
        // 명령어를 실행하기 직전, 현재 작업 디렉토리(run_dir) 상태를 저장
        if self.safety_enabled {
            self.create_git_snapshot(&run_dir, trimmed_cmd);
        }

        // 4. 프로세스 실행
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

        // 5. 결과 처리
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

/// 'cd' 타겟 경로 해석 헬퍼 함수 (기존 로직 유지)
fn resolve_cd_target(current_cwd: &Path, target: &str) -> Result<PathBuf> {
    if target == "~" || target.starts_with("~/") {
        let home = env::var("HOME")
            .map(PathBuf::from)
            .map_err(|_| SuprascalarError::MissingEnvVar("HOME".to_string()))?;

        if target == "~" {
            return Ok(home);
        }
        let remainder = target.trim_start_matches("~/");
        return Ok(home.join(remainder));
    }

    let candidate = PathBuf::from(target);
    if candidate.is_absolute() {
        Ok(candidate)
    } else {
        Ok(current_cwd.join(candidate))
    }
}
