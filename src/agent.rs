use crate::error::{Result, SuprascalarError};
use crate::models::LLMBackend;
use crate::tools::Tool; // Tool 트레이트 가져오기
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct Agent {
    #[allow(dead_code)]
    name: String,
    model: Box<dyn LLMBackend>,
    history: Vec<Message>,
    // 원본 시스템 프롬프트 (도구 설명이 붙기 전의 순수 페르소나)
    base_system_prompt: String,
    // 등록된 도구 저장소 (이름 -> 도구 객체)
    tools: HashMap<String, Box<dyn Tool>>,
}

impl Agent {
    /// 에이전트 생성
    pub fn new(name: &str, model: Box<dyn LLMBackend>, system_prompt: &str) -> Self {
        let mut agent = Self {
            name: name.to_string(),
            model,
            history: Vec::new(),
            base_system_prompt: system_prompt.to_string(),
            tools: HashMap::new(),
        };

        // 초기 시스템 메시지 설정 (도구가 없으면 기본 프롬프트만 들어감)
        agent.refresh_system_message();

        agent
    }

    /// 도구 등록 메서드
    /// 사용자가 `agent.register_tool(ListDirectory::new())` 형태로 호출
    pub fn register_tool(&mut self, tool: impl Tool + 'static) {
        let name = tool.name().to_string();
        self.tools.insert(name, Box::new(tool));

        // 도구가 추가되었으니 시스템 프롬프트를 갱신하여 LLM에게 알려줌
        self.refresh_system_message();
    }

    /// 시스템 메시지를 재구성하는 내부 메서드
    /// (기본 페르소나 + 도구 정의)
    fn refresh_system_message(&mut self) {
        let mut full_prompt = self.base_system_prompt.clone();

        // 도구가 있다면, 사용법과 목록을 프롬프트에 추가
        if !self.tools.is_empty() {
            full_prompt.push_str("\n\n## Available Tools\n");
            full_prompt.push_str("You have access to the following tools. To use a tool, you MUST respond with a JSON object strictly following this schema:\n");
            full_prompt.push_str("```json\n{ \"tool\": \"tool_name\", \"args\": { ... } }\n```\n");

            full_prompt.push_str("\n## Tool Definitions\n");
            for tool in self.tools.values() {
                // 각 도구의 이름, 설명, 파라미터 스키마를 JSON 형태로 주입
                let schema = serde_json::json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters()
                });
                // 예쁘게 포맷팅해서 가독성 높임
                if let Ok(schema_str) = serde_json::to_string_pretty(&schema) {
                    full_prompt.push_str(&format!("{}\n", schema_str));
                }
            }
        }

        // history의 첫 번째 메시지(System Prompt)를 교체하거나 추가
        if let Some(first_msg) = self.history.first_mut() {
            if first_msg.role == "system" {
                first_msg.content = full_prompt;
                return;
            }
        }

        // history가 비어있거나 첫 메시지가 시스템이 아니면 새로 추가 (초기화 시점)
        self.history.insert(
            0,
            Message {
                role: "system".to_string(),
                content: full_prompt,
            },
        );
    }

    /// ReAct 루프가 적용된 Chat 메서드
    pub fn chat(&mut self, user_input: &str) -> Result<String> {
        // 1. 사용자 입력 저장
        self.history.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // 최대 루프 횟수 (무한 루프 방지)
        let max_turns = 5;
        let mut current_turn = 0;

        loop {
            current_turn += 1;
            if current_turn > max_turns {
                return Err(SuprascalarError::Unknown(
                    "Max agent turns exceeded".to_string(),
                ));
            }

            // 2. 프롬프트 구성 및 LLM 생성
            let prompt = self.build_prompt();
            let response_text = self.model.generate(&prompt)?;

            // 3. LLM 응답 저장 (Assistant 메시지)
            self.history.push(Message {
                role: "assistant".to_string(),
                content: response_text.clone(),
            });

            // 4. 도구 호출 감지 (JSON 파싱)
            // ```json { ... } ``` 패턴을 찾습니다.
            if let Some(tool_call) = self.extract_tool_call(&response_text) {
                println!(
                    ">>> Tool Call Detected: {} ({})",
                    tool_call.tool_name, tool_call.args
                );

                // 5. 도구 실행
                let tool_output = self.execute_tool(&tool_call.tool_name, tool_call.args);

                // 6. 실행 결과를 시스템 관찰(Observation)로 저장
                // Qwen이나 ChatML에서는 보통 'user' 역할로 결과를 알려주거나,
                // 'tool' 역할이 있다면 그걸 씁니다. 여기서는 'user'로 컨텍스트를 줍니다.
                let observation_msg = format!(
                    "Tag: <tool_output>\nResult: {}\n</tool_output>\n(Please continue using this result.)",
                    tool_output
                );

                self.history.push(Message {
                    role: "user".to_string(), // 혹은 system
                    content: observation_msg,
                });

                // 루프를 계속 돕니다. (LLM이 결과를 보고 최종 답변을 할 때까지)
                continue;
            } else {
                // 도구 호출이 없으면 최종 답변으로 간주하고 루프 종료
                return Ok(response_text);
            }
        }
    }
    /// 도구 호출 정보를 담을 내부 구조체
    fn extract_tool_call(&self, text: &str) -> Option<ToolCallInfo> {
        // 1) 먼저 ```json { ... } ``` 패턴을 시도해 봅니다 (우선순위)
        if let Ok(re) = Regex::new(r"```json\s*(\{[\s\S]*?\})\s*```") {
            if let Some(caps) = re.captures(text) {
                if let Some(m) = caps.get(1) {
                    if let Ok(parsed) = serde_json::from_str::<Value>(m.as_str()) {
                        if let Some(tool_name) = parsed.get("tool").and_then(|v| v.as_str()) {
                            let args = parsed.get("args").cloned().unwrap_or(Value::Null);
                            return Some(ToolCallInfo {
                                tool_name: tool_name.to_string(),
                                args,
                            });
                        }
                    }
                }
            }
        }

        // 2) 코드펜스가 없을 때: 텍스트 내의 JSON 객체들을 탐색하여 파싱 가능한 것 찾기
        //    중첩 중괄호를 수동으로 추적하여 균형잡힌 JSON 블록을 추출합니다.
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            // '{' 를 찾음
            if bytes[i] == b'{' {
                let mut depth: i32 = 0;
                let mut j = i;
                while j < len {
                    if bytes[j] == b'{' {
                        depth += 1;
                    } else if bytes[j] == b'}' {
                        depth -= 1;
                        if depth == 0 {
                            // 후보 문자열
                            if let Ok(candidate) = std::str::from_utf8(&bytes[i..=j]) {
                                if let Ok(parsed) = serde_json::from_str::<Value>(candidate) {
                                    if let Some(tool_name) =
                                        parsed.get("tool").and_then(|v| v.as_str())
                                    {
                                        let args =
                                            parsed.get("args").cloned().unwrap_or(Value::Null);
                                        return Some(ToolCallInfo {
                                            tool_name: tool_name.to_string(),
                                            args,
                                        });
                                    }
                                }
                            }
                            break;
                        }
                    }
                    j += 1;
                }
                // 다음 위치로 이동
                i += 1;
            } else {
                i += 1;
            }
        }

        None
    }

    fn execute_tool(&self, name: &str, args: Value) -> String {
        match self.tools.get(name) {
            Some(tool) => match tool.execute(args) {
                Ok(output) => output,
                Err(e) => format!("Error executing tool: {}", e),
            },
            None => format!("Error: Tool '{}' not found.", name),
        }
    }

    fn build_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.history {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
}

// 내부적으로만 쓸 구조체
struct ToolCallInfo {
    tool_name: String,
    args: Value,
}
