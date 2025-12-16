use super::Tool;
use crate::error::{Result, SuprascalarError};
use serde_json::{Value, json};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 파일 읽기/쓰기 도구 (Host-side I/O)
/// 보안 기능: Path Traversal 방지 (프로젝트 폴더 탈출 금지)
pub struct FileIO;

impl FileIO {
    pub fn new() -> Self {
        Self
    }

    /// [Security Patch] Symlink 공격 방지를 위한 물리적 경로 검증
    /// 1. 프로젝트 루트의 진짜 경로(Real Path)를 구합니다.
    /// 2. 요청한 경로의 진짜 경로를 구합니다.
    /// 3. 요청한 경로가 프로젝트 루트 안쪽에 있는지 확인합니다.
    fn validate_path(&self, path_str: &str) -> Result<PathBuf> {
        // 1. 프로젝트 루트의 물리적 경로 (Symlink 해제됨)
        let cwd = env::current_dir().map_err(SuprascalarError::Io)?;
        let canonical_root = cwd.canonicalize().map_err(SuprascalarError::Io)?;

        // 2. 타겟 경로 구성
        let target_path = cwd.join(path_str);

        // 3. 물리적 경로 확인 (Symlink Resolution)
        // 케이스 A: 파일/폴더가 이미 존재하는 경우
        if target_path.exists() {
            let real_path = target_path.canonicalize().map_err(|e| {
                SuprascalarError::Unknown(format!("Failed to resolve path '{}': {}", path_str, e))
            })?;

            if !real_path.starts_with(&canonical_root) {
                return Err(SuprascalarError::Unknown(format!(
                    "SECURITY BLOCK: Symlink detected! '{}' resolves to '{}', which is outside the project root.",
                    path_str,
                    real_path.display()
                )));
            }
            return Ok(real_path);
        }

        // 케이스 B: 파일이 존재하지 않는 경우 (새로 쓰기)
        // 존재하지 않는 파일은 canonicalize가 불가능하므로, "존재하는 가장 깊은 부모 디렉토리"를 검사해야 함.
        let mut current_check = target_path.parent();

        while let Some(p) = current_check {
            if p.exists() {
                // 존재하는 부모를 찾았다! 이 부모가 혹시 외부로 연결된 심볼릭 링크인지 확인
                let real_parent = p.canonicalize().map_err(SuprascalarError::Io)?;

                if !real_parent.starts_with(&canonical_root) {
                    return Err(SuprascalarError::Unknown(format!(
                        "SECURITY BLOCK: Parent directory symlink escape detected! '{}' resolves to outside.",
                        p.display()
                    )));
                }

                // 부모가 안전하다면, 루프 종료 (안전함)
                break;
            }
            // 더 상위 부모로 이동
            current_check = p.parent();
        }

        // 여기까지 오면 안전함 (부모들이 모두 Safe Zone 안에 있음)
        // 단, 리턴값은 canonicalize된 경로가 아니라 논리적 경로여야 함 (파일이 아직 없으므로)
        // 하지만 편의상 절대경로(target_path)를 반환
        Ok(target_path)
    }

    /// Git snapshot before mutating files for basic auditing/safety
    fn create_git_snapshot(&self, context: &str) {
        let Ok(cwd) = env::current_dir() else {
            return;
        };

        let status = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&cwd)
            .output();

        if let Ok(output) = status {
            if !output.status.success() || output.stdout.is_empty() {
                return;
            }

            let _ = Command::new("git")
                .args(["add", "."])
                .current_dir(&cwd)
                .output();

            let msg = format!("Suprascalar Auto-save: Before file_io '{}'", context);
            let _ = Command::new("git")
                .args(["commit", "-m", &msg])
                .current_dir(&cwd)
                .output();
        }
    }
}

impl Tool for FileIO {
    fn name(&self) -> &str {
        "read_write_file"
    }

    fn description(&self) -> &str {
        "Reads or writes a file on the host system. \
        Strictly sandboxed: Cannot access files outside the current project directory. \
        Use this to create/edit code files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "Action to perform"
                },
                "path": {
                    "type": "string",
                    "description": "Relative file path (e.g., 'src/main.rs')"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write')"
                },
                "line_start": { "type": "integer" },
                "line_end": { "type": "integer" }
            },
            "required": ["action", "path"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        let action = args["action"]
            .as_str()
            .ok_or_else(|| SuprascalarError::Unknown("Missing 'action'".to_string()))?;
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| SuprascalarError::Unknown("Missing 'path'".to_string()))?;

        // [Security] 여기서 Symlink까지 확인하는 강력한 검증 수행
        let path = self.validate_path(path_str)?;

        match action {
            "read" => {
                if !path.exists() {
                    return Err(SuprascalarError::Unknown(format!(
                        "File '{}' does not exist.",
                        path_str
                    )));
                }
                let content = fs::read_to_string(&path).map_err(SuprascalarError::Io)?;

                let start = args["line_start"].as_u64();
                let end = args["line_end"].as_u64();

                let sliced = if start.is_some() || end.is_some() {
                    let lines: Vec<&str> = content.lines().collect();
                    if lines.is_empty() {
                        String::new()
                    } else {
                        let start_idx = start.unwrap_or(1);
                        let end_idx = end.unwrap_or(lines.len() as u64);

                        if start_idx == 0 || end_idx == 0 || start_idx > end_idx {
                            return Err(SuprascalarError::Unknown(
                                "Invalid line range: ensure 1-based start <= end".to_string(),
                            ));
                        }

                        let start_pos = (start_idx.saturating_sub(1) as usize).min(lines.len());
                        let end_pos = (end_idx as usize).min(lines.len());

                        if start_pos >= end_pos {
                            String::new()
                        } else {
                            lines[start_pos..end_pos].join("\n")
                        }
                    }
                } else {
                    content
                };

                Ok(format!("File '{}':\n```\n{}\n```", path_str, sliced))
            }
            "write" => {
                let content = args["content"]
                    .as_str()
                    .ok_or_else(|| SuprascalarError::Unknown("Missing 'content'".to_string()))?;

                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).map_err(SuprascalarError::Io)?;
                }

                self.create_git_snapshot(path_str);

                fs::write(&path, content).map_err(SuprascalarError::Io)?;
                Ok(format!("Successfully wrote to '{}'.", path_str))
            }
            _ => Ok(format!("Unknown action: {}", action)),
        }
    }
}
