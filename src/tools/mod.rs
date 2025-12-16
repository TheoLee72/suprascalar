// src/tools/mod.rs

use crate::error::Result;
use serde_json::Value;

// 서브 모듈(구현체) 등록
pub mod docker;
pub mod file_io;
pub mod ls;
pub mod terminal;

/// Suprascalar의 모든 도구가 구현해야 하는 인터페이스입니다.
/// MCP(Model Context Protocol) 표준과 호환되도록 설계되었습니다.
pub trait Tool: Send + Sync {
    /// 도구의 고유 이름 (예: "list_files", "calculator")
    fn name(&self) -> &str;

    /// 도구에 대한 설명 (System Prompt에 주입됨)
    fn description(&self) -> &str;

    /// 도구 인자의 JSON Schema
    fn parameters(&self) -> Value;

    /// 도구 실행 로직
    fn execute(&self, args: Value) -> Result<String>;
}
