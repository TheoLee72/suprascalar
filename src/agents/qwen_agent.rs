use crate::error::{Result, SuprascalarError};
use crate::models::LLMBackend;
use crate::tools::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::env;

// NousFnCallPrompt 포맷 상수
const FN_CALL_TEMPLATE: &str = r#"# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{\"name\": <function-name>, \"arguments\": <args-json-object>}}
</tool_call>"#;

const FN_CALL_TEMPLATE_WITH_CI: &str = r#"# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{\"name\": <function-name>, \"arguments\": <args-json-object>}}
</tool_call>
For code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:
<tool_call>
{{\"name\": <function-name>, \"arguments\": {{\"code\": \"\"}}}}
<code>
Here is the code.
</code>
</tool_call>"#;

const CODE_TOOL_PATTERN: &str = "code_interpreter";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Function,
}

impl Role {
    fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Function => "function",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ContentItem {
    Text(String),
}

impl ContentItem {
    fn text<T: Into<String>>(text: T) -> Self {
        ContentItem::Text(text.into())
    }

    fn get_type_and_value(&self) -> (&'static str, &str) {
        match self {
            ContentItem::Text(t) => ("text", t.as_str()),
        }
    }

    fn push_into(target: &mut Vec<ContentItem>, text: impl Into<String>) {
        target.push(ContentItem::Text(text.into()));
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentItem>,
    pub reasoning_content: Option<String>,
    pub function_call: Option<FunctionCall>,
    pub extra: Option<HashMap<String, String>>,
}

impl Message {
    fn new(role: Role, content: Vec<ContentItem>) -> Self {
        Self {
            role,
            content,
            reasoning_content: None,
            function_call: None,
            extra: None,
        }
    }

    fn system_text(text: impl Into<String>) -> Self {
        Message::new(Role::System, vec![ContentItem::text(text)])
    }

    fn user_text(text: impl Into<String>) -> Self {
        Message::new(Role::User, vec![ContentItem::text(text)])
    }

    fn assistant_text(text: impl Into<String>) -> Self {
        Message::new(Role::Assistant, vec![ContentItem::text(text)])
    }

    fn function_text(text: impl Into<String>) -> Self {
        Message::new(Role::Function, vec![ContentItem::text(text)])
    }

    fn content_as_string(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                ContentItem::Text(t) => Some(t.as_str()),
            })
            .collect::<Vec<&str>>()
            .join("")
    }
}

pub struct Agent {
    #[allow(dead_code)]
    name: String,
    model: Box<dyn LLMBackend>,
    history: Vec<Message>,
    base_system_prompt: String,
    tools: HashMap<String, Box<dyn Tool>>,
}

/// Builder for configuring an `Agent` before construction.
pub struct AgentBuilder {
    name: String,
    model: Box<dyn LLMBackend>,
    system_prompt: String,
    tools: Vec<Box<dyn Tool>>,
}

impl Agent {
    /// 에이전트 생성 (crate 내부 전용)
    fn new(name: &str, model: Box<dyn LLMBackend>, system_prompt: &str) -> Self {
        let mut agent = Self {
            name: name.to_string(),
            model,
            history: Vec::new(),
            base_system_prompt: system_prompt.to_string(),
            tools: HashMap::new(),
        };

        agent.refresh_system_message();
        agent
    }

    /// Builder entrypoint for fluent configuration.
    pub fn builder(name: &str, model: Box<dyn LLMBackend>, system_prompt: &str) -> AgentBuilder {
        AgentBuilder {
            name: name.to_string(),
            model,
            system_prompt: system_prompt.to_string(),
            tools: Vec::new(),
        }
    }

    /// 도구 등록 메서드 (빌드 이후 런타임에 추가할 때 사용)
    pub fn register_tool(&mut self, tool: impl Tool + 'static) -> &mut Self {
        self.register_tool_box(Box::new(tool))
    }

    fn register_tool_box(&mut self, tool: Box<dyn Tool>) -> &mut Self {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
        self.refresh_system_message();
        self
    }

    /// 시스템 메시지를 재구성하는 내부 메서드
    fn refresh_system_message(&mut self) {
        let mut full_prompt = self.base_system_prompt.clone();
        if let Some(tool_section) = self.render_tool_system_prompt() {
            full_prompt.push_str("\n\n");
            full_prompt.push_str(&tool_section);
        }

        if let Some(first_msg) = self.history.first_mut() {
            if first_msg.role == Role::System {
                first_msg.content = vec![ContentItem::text(full_prompt)];
                return;
            }
        }

        self.history.insert(0, Message::system_text(full_prompt));
    }

    /// Qwen 함수 호출 포맷을 따르는 시스템 프롬프트를 생성합니다.
    fn render_tool_system_prompt(&self) -> Option<String> {
        if self.tools.is_empty() {
            return None;
        }

        let mut tool_descs = Vec::new();
        let mut tool_names = Vec::new();

        for tool in self.tools.values() {
            let desc = FunctionDescriptor {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters(),
            };
            tool_names.push(desc.name.clone());
            tool_descs.push(ToolDescriptor {
                kind: "function",
                function: desc,
            });
        }

        let tool_descs = tool_descs
            .iter()
            .map(|td| serde_json::to_string(td).unwrap_or_else(|_| "{}".into()))
            .collect::<Vec<String>>()
            .join("\n");

        let section = if special_code_mode()
            && tool_names
                .iter()
                .any(|name| name.contains(CODE_TOOL_PATTERN))
        {
            FN_CALL_TEMPLATE_WITH_CI.replace("{tool_descs}", &tool_descs)
        } else {
            FN_CALL_TEMPLATE.replace("{tool_descs}", &tool_descs)
        };

        Some(section)
    }

    /// ReAct 루프가 적용된 Chat 메서드 (NousFnCallPrompt 스타일)
    pub fn chat(&mut self, user_input: &str) -> Result<String> {
        self.history.push(Message::user_text(user_input));

        let max_turns = 5;
        let mut current_turn = 0;

        loop {
            current_turn += 1;
            if current_turn > max_turns {
                return Err(SuprascalarError::Unknown(
                    "Max agent turns exceeded".to_string(),
                ));
            }

            let prompt = self.build_prompt()?;
            let response_text = self.model.generate(&prompt)?;
            //log
            // println!("{}", response_text);

            // 모델 응답을 우선 기록(원본 텍스트)
            let assistant_raw = Message::assistant_text(response_text.clone());
            let parsed = self.postprocess_fncall_messages(vec![assistant_raw.clone()])?;

            let mut function_calls: Vec<FunctionCall> = Vec::new();
            let mut answer_acc = String::new();

            for msg in parsed {
                match msg.role {
                    Role::Assistant => {
                        if let Some(fc) = msg.function_call.clone() {
                            function_calls.push(fc);
                            self.history.push(msg);
                        } else {
                            answer_acc.push_str(&msg.content_as_string());
                            self.history.push(msg);
                        }
                    }
                    _ => {
                        self.history.push(msg);
                    }
                }
            }

            if function_calls.is_empty() {
                if answer_acc.is_empty() {
                    // 도구 호출이 없는 순수 답변
                    return Ok(response_text);
                }
                return Ok(answer_acc);
            }

            for fc in function_calls {
                let args_value = serde_json::from_str::<Value>(&fc.arguments)
                    .unwrap_or_else(|_| Value::String(fc.arguments.clone()));
                let tool_output = self.execute_tool(&fc.name, args_value);

                let observation = Message::function_text(tool_output);
                self.history.push(observation);
            }
        }
    }

    /// NousFnCallPrompt: 입력 메시지를 함수 호출 가능 형태로 사전 처리
    fn preprocess_fncall_messages(&self, messages: &[Message]) -> Result<Vec<Message>> {
        let mut processed: Vec<Message> = Vec::new();

        for msg in messages.iter().cloned() {
            match msg.role {
                Role::System | Role::User => processed.push(msg),
                Role::Assistant => {
                    let mut content = msg.content.clone();
                    if let Some(fc) = msg.function_call.clone() {
                        if !special_code_mode() || !fc.name.contains(CODE_TOOL_PATTERN) {
                            let parsed_args: Value = json5::from_str(&fc.arguments)
                                .unwrap_or_else(|_| Value::String(fc.arguments.clone()));
                            let fc_obj = json!({"name": fc.name, "arguments": parsed_args});
                            let fc_text = format!(
                                "<tool_call>\n{}\n</tool_call>",
                                serde_json::to_string(&fc_obj).unwrap_or_else(|_| "{}".into())
                            );
                            ContentItem::push_into(&mut content, fc_text);
                        } else {
                            let mut parsed_args: Value = json5::from_str(&fc.arguments)
                                .unwrap_or_else(|_| Value::String(fc.arguments.clone()));
                            let code = parsed_args
                                .get("code")
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            if let Some(obj) = parsed_args.as_object_mut() {
                                obj.insert("code".to_string(), Value::String(String::new()));
                            }
                            let fc_obj = json!({"name": fc.name, "arguments": parsed_args});
                            let fc_text = format!(
                                "<tool_call>\n{}\n<code>\n{}\n</code>\n</tool_call>",
                                serde_json::to_string(&fc_obj).unwrap_or_else(|_| "{}".into()),
                                code
                            );
                            ContentItem::push_into(&mut content, fc_text);
                        }
                    }

                    if let Some(last) = processed.last_mut() {
                        if last.role == Role::Assistant {
                            if let Some(ContentItem::Text(t)) = last.content.last() {
                                if !t.ends_with('\n') {
                                    ContentItem::push_into(&mut last.content, "\n");
                                }
                            }
                            last.content.extend(content);
                            continue;
                        }
                    }

                    processed.push(Message {
                        role: Role::Assistant,
                        content,
                        reasoning_content: msg.reasoning_content,
                        function_call: None,
                        extra: msg.extra,
                    });
                }
                Role::Function => {
                    let mut content = msg.content.clone();
                    content.insert(0, ContentItem::text("<tool_response>\n"));
                    content.push(ContentItem::text("\n</tool_response>"));

                    if let Some(last) = processed.last_mut() {
                        if last.role == Role::User {
                            ContentItem::push_into(&mut last.content, "\n");
                            last.content.extend(content);
                            continue;
                        }
                    }

                    processed.push(Message {
                        role: Role::User,
                        content,
                        reasoning_content: None,
                        function_call: None,
                        extra: None,
                    });
                }
            }
        }

        if let Some(tool_system) = self.render_tool_system_prompt() {
            if let Some(first) = processed.first_mut() {
                if first.role == Role::System {
                    ContentItem::push_into(&mut first.content, format!("\n\n{}", tool_system));
                } else {
                    processed.insert(0, Message::system_text(tool_system));
                }
            } else {
                processed.push(Message::system_text(tool_system));
            }
        }

        Ok(processed)
    }

    /// NousFnCallPrompt: 모델 응답을 함수 호출 구조로 역변환
    fn postprocess_fncall_messages(&self, messages: Vec<Message>) -> Result<Vec<Message>> {
        let mut new_messages = Vec::new();
        let mut tool_id: usize = 1;

        for msg in messages.into_iter() {
            let role = msg.role;
            let content = msg.content;
            let reasoning_content = msg.reasoning_content;
            let extra = msg.extra.unwrap_or_default();

            match role {
                Role::System | Role::User => {
                    new_messages.push(Message {
                        role,
                        content,
                        reasoning_content,
                        function_call: None,
                        extra: if extra.is_empty() { None } else { Some(extra) },
                    });
                }
                Role::Assistant => {
                    if let Some(reason) = reasoning_content {
                        new_messages.push(Message {
                            role: Role::Assistant,
                            content: vec![],
                            reasoning_content: Some(reason),
                            function_call: None,
                            extra: None,
                        });
                    }

                    let mut new_content = Vec::new();

                    for item in content.into_iter() {
                        let (item_type, item_text) = item.get_type_and_value();

                        if item_type != "text" {
                            new_content.push(item);
                            continue;
                        }

                        let mut thought_in_content = false;
                        if item_text.contains("<think>") {
                            thought_in_content = true;
                        }

                        let mut remaining_text = item_text.to_string();
                        if thought_in_content && remaining_text.contains("</think>") {
                            let parts: Vec<&str> = remaining_text.split("</think>").collect();
                            let before = parts[..parts.len() - 1].join("</think>") + "</think>";
                            new_content.push(ContentItem::text(before));
                            remaining_text = parts.last().unwrap_or(&"").to_string();
                        }

                        if let Some(idx) = remaining_text.find("<tool_call>") {
                            let tool_call_list: Vec<&str> =
                                remaining_text.split("<tool_call>").collect();
                            let pre_thought = tool_call_list[0];
                            if !pre_thought.trim().is_empty() {
                                new_content.push(ContentItem::text(pre_thought));
                            }

                            for txt in tool_call_list.into_iter().skip(1) {
                                if txt.trim().is_empty() {
                                    continue;
                                }

                                if !txt.contains("</tool_call>") {
                                    let (fn_name, fn_args) = extract_fn(txt);
                                    if !fn_name.is_empty() {
                                        if !new_content.is_empty() {
                                            new_messages.push(Message {
                                                role: Role::Assistant,
                                                content: new_content.clone(),
                                                reasoning_content: None,
                                                function_call: None,
                                                extra: None,
                                            });
                                            new_content.clear();
                                        }

                                        let mut extra_map = extra.clone();
                                        extra_map.insert("function_id".into(), tool_id.to_string());
                                        tool_id += 1;

                                        new_messages.push(Message {
                                            role: Role::Assistant,
                                            content: Vec::new(),
                                            reasoning_content: None,
                                            function_call: Some(FunctionCall {
                                                name: fn_name,
                                                arguments: fn_args,
                                            }),
                                            extra: Some(extra_map),
                                        });
                                    }
                                    continue;
                                }

                                let parts: Vec<&str> = txt.split("</tool_call>").collect();
                                if !new_content.is_empty() {
                                    new_messages.push(Message {
                                        role: Role::Assistant,
                                        content: new_content.clone(),
                                        reasoning_content: None,
                                        function_call: None,
                                        extra: None,
                                    });
                                    new_content.clear();
                                }

                                let mut fn_obj: Option<Value> = None;

                                if special_code_mode()
                                    && parts[0].contains("<code>")
                                    && parts[0].contains("</code>")
                                {
                                    let mut code_sections = parts[0].split("<code>");
                                    if let Some(first) = code_sections.next() {
                                        if let Ok(v) = json5::from_str::<Value>(first) {
                                            fn_obj = Some(v);
                                        }
                                    }
                                    if let Some(last_section) = code_sections.next() {
                                        let code = last_section.replace("</code>", "");
                                        if let Some(Value::Object(ref mut obj)) = fn_obj {
                                            if let Some(args) = obj.get_mut("arguments") {
                                                if let Some(args_obj) = args.as_object_mut() {
                                                    args_obj
                                                        .insert("code".into(), Value::String(code));
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if let Ok(v) = json5::from_str::<Value>(parts[0].trim()) {
                                        fn_obj = Some(v);
                                    }
                                }

                                if let Some(fn_obj) = fn_obj {
                                    if let (Some(fn_name), Some(arguments)) = (
                                        fn_obj.get("name").and_then(|v| v.as_str()),
                                        fn_obj.get("arguments"),
                                    ) {
                                        let mut extra_map = extra.clone();
                                        extra_map.insert("function_id".into(), tool_id.to_string());
                                        tool_id += 1;

                                        new_messages.push(Message {
                                            role: Role::Assistant,
                                            content: Vec::new(),
                                            reasoning_content: None,
                                            function_call: Some(FunctionCall {
                                                name: fn_name.to_string(),
                                                arguments: serde_json::to_string(arguments)
                                                    .unwrap_or_else(|_| "{}".into()),
                                            }),
                                            extra: Some(extra_map),
                                        });
                                    }
                                } else {
                                    let (fn_name, fn_args) = extract_fn(parts[0].trim());
                                    if !fn_name.is_empty() {
                                        let mut extra_map = extra.clone();
                                        extra_map.insert("function_id".into(), tool_id.to_string());
                                        tool_id += 1;

                                        new_messages.push(Message {
                                            role: Role::Assistant,
                                            content: Vec::new(),
                                            reasoning_content: None,
                                            function_call: Some(FunctionCall {
                                                name: fn_name,
                                                arguments: fn_args,
                                            }),
                                            extra: Some(extra_map),
                                        });
                                    }
                                }
                            }
                        } else {
                            if !remaining_text.is_empty() {
                                new_content.push(ContentItem::text(remaining_text));
                            }
                        }
                    }

                    if !new_content.is_empty() {
                        new_messages.push(Message {
                            role: Role::Assistant,
                            content: new_content,
                            reasoning_content: None,
                            function_call: None,
                            extra: if extra.is_empty() { None } else { Some(extra) },
                        });
                    }
                }
                Role::Function => {
                    // Function 역할은 입력으로만 들어오지 않고, 실행 결과로만 추가될 예정
                }
            }
        }

        Ok(new_messages)
    }

    fn build_prompt(&self) -> Result<String> {
        let processed = self.preprocess_fncall_messages(&self.history)?;
        let mut prompt = String::new();

        for msg in processed {
            let role = msg.role.as_str();
            let content = msg.content_as_string();
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }

        prompt.push_str("<|im_start|>assistant\n");
        Ok(prompt)
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
}

#[derive(Serialize)]
struct ToolDescriptor {
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionDescriptor,
}

#[derive(Serialize)]
struct FunctionDescriptor {
    name: String,
    description: String,
    parameters: Value,
}

fn special_code_mode() -> bool {
    env::var("SPECIAL_CODE_MODE")
        .unwrap_or_else(|_| "false".to_string())
        .to_lowercase()
        == "true"
}

// Mainly for removing incomplete special tokens when streaming the output
fn remove_incomplete_special_tokens(text: &str) -> String {
    if text == "<tool_call>\n{\"name\": " {
        return String::new();
    }
    text.to_string()
}

fn extract_fn(text: &str) -> (String, String) {
    let mut fn_name = String::new();
    let mut fn_args = String::new();

    let fn_name_s = "\"name\": \"";
    let fn_name_e = "\", \"";
    let fn_args_s = "\"arguments\": ";

    if let Some(i) = text.find(fn_name_s) {
        let rest = &text[i + fn_name_s.len()..];
        if let Some(j) = rest.find(fn_name_e) {
            fn_name = rest[..j].to_string();
        }
    }

    if let Some(k) = text.find(fn_args_s) {
        let rest = &text[k + fn_args_s.len()..];
        fn_args = rest.trim().to_string();
        if fn_args.len() > 2 {
            fn_args = fn_args[..fn_args.len() - 1].to_string();
        } else {
            fn_args.clear();
        }
    }

    (fn_name, fn_args)
}

impl AgentBuilder {
    /// Add a tool before building the agent.
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Finalize and construct the agent.
    pub fn build(self) -> Result<Agent> {
        let mut agent = Agent::new(&self.name, self.model, &self.system_prompt);
        for tool in self.tools {
            agent.register_tool_box(tool);
        }
        Ok(agent)
    }
}
