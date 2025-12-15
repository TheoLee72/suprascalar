use super::Tool;
use crate::error::{Result, SuprascalarError};
use serde_json::{Value, json};
use std::fs;
use std::path::Path;

pub struct ListDirectory;

impl ListDirectory {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for ListDirectory {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "List files and directories in a specific path. Useful for exploring the file system."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list (default is current directory '.')"
                }
            },
            "required": ["path"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        // 인자 파싱 (없으면 현재 디렉토리)
        let path_str = args["path"].as_str().unwrap_or(".");
        let path = Path::new(path_str);

        // 경로 존재 여부 확인
        if !path.exists() {
            return Ok(format!("Error: Path '{}' does not exist.", path_str));
        }

        // 디렉토리 읽기
        let entries = fs::read_dir(path).map_err(SuprascalarError::Io)?;

        let mut file_list = String::new();
        file_list.push_str(&format!("Files in '{}':\n", path_str));

        for entry in entries {
            let entry = entry.map_err(SuprascalarError::Io)?;
            let file_type = entry.file_type().map_err(SuprascalarError::Io)?;
            let file_name = entry.file_name().to_string_lossy().to_string();

            let prefix = if file_type.is_dir() {
                "[DIR] "
            } else {
                "[FILE]"
            };
            file_list.push_str(&format!("{} {}\n", prefix, file_name));
        }

        Ok(file_list)
    }
}
