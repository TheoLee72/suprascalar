# Suprascalar: Rust Agentic Framework

## 1\. Vision & Philosophy

**Suprascalar** is a high-performance, local-first Agentic AI framework built in Rust. It leverages the `candle` ML framework to run quantized Large Language Models (LLMs) efficiently on consumer hardware.

Unlike Python-based frameworks (LangChain, CrewAI) that often rely on API latency and dynamic typing, Suprascalar focuses on:

- **Local Sovereignty:** No data leaves the machine.
- **Type Safety:** leveraging Rust's `Result<T, E>` and `Enums` for robust error handling.
- **Explicit Control:** Direct management of the generation loop and memory context.

## 2\. Core Architecture

The framework is composed of three primary layers, adhering to the "Connected Problem-Solver" pattern (Level 1 Agent).

### A. The Brain: `LLMBackend`

_Ref: Suprascalar/src/models/mod.rs_

The `LLMBackend` trait abstracts the underlying inference engine.

- **Implementation:** `CandleQwen` (Quantized Qwen 2.5/3).
- **Execution Strategy:** "Full Context Refresh."
  - _Current Behavior:_ For every new turn in a conversation, the backend clears the Key-Value (KV) Cache via `clear_kv_cache()`.
  - _Reasoning:_ The Agent reconstructs the full prompt history on every turn. Clearing the cache ensures the model processes the updated context window as a coherent new sequence, preventing tensor shape mismatches.

### B. The State Manager: `Agent`

_Ref: Suprascalar/src/agent.rs_

The Agent is the persistent entity that maintains identity and memory.

- **Identity:** Defined by a `system_prompt` (e.g., "You are a coding assistant...").
- **Short-Term Memory:** Implemented as a `Vec<Message>`.
  - Stores `System`, `User`, and `Assistant` turns.
  - **Context Engineering:** The Agent is responsible for serializing this history into the specific ChatML format (`<|im_start|>...`) required by Qwen.
- **Lifecycle:**
  1.  Receive Input.
  2.  Append to History.
  3.  Build Full Prompt (Context Engineering).
  4.  Generate Response (Blocking).
  5.  Append Output to History.

### C. The Safety Layer: `SuprascalarError`

_Ref: Suprascalar/src/error.rs_

We utilize a custom error enum (`thiserror`) to handle failures gracefully.

- **`ContextLimitExceeded`:** Prevents the agent from crashing when the conversation grows too long for the model's window (e.g., \> 32k tokens).
- **`Candle` / `Tokenizer`:** Wraps lower-level inference errors for easier debugging.

## 3\. Design Patterns Implemented

Based on _Agentic Design Patterns_, Suprascalar currently implements:

### 1\. Prompt Chaining (The Conversation Loop)

- **Pattern:** The output of one step (the Assistant's reply) becomes the input context for the next step.
- **Implementation:** The `examples/multi_turn_chat.rs` REPL loop demonstrates this. The `Agent` struct automatically chains the conversation history, ensuring continuity.

### 2\. Context Engineering

- **Pattern:** Structuring input to maximize model performance.
- **Implementation:** The `build_prompt` method in `Agent` strictly enforces ChatML templates. This ensures the model distinguishes between instructions (System) and conversation (User).

### 3\. Exception Handling

- **Pattern:** Managing operational failures.
- **Implementation:** The `Result<T, SuprascalarError>` type allows the runtime to catch specific issues (like missing model files or context overflow) and decide whether to crash, retry, or summarize history.

## 4\. Technical Specifications (Current MVP)

| Component            | Specification                              |
| :------------------- | :----------------------------------------- |
| **Model Arch**       | Qwen 2.5 / 3 (GGUF Format)                 |
| **Inference**        | CPU/CUDA via `candle-core`                 |
| **Quantization**     | Q4_K_M (4-bit)                             |
| **Context Strategy** | Sliding Window (Manual Limit Check)        |
| **KV Cache**         | Stateless (Cleared on every generate call) |

## 5\. Roadmap

### Phase 1: Foundation (Current)

- [x] Abstract `LLMBackend` trait.
- [x] Implement `CandleQwen` with GGUF support.
- [x] Implement `Agent` with history.
- [x] Fix Multi-turn KV Cache issues (`clear_kv_cache`).
- [x] Custom Error Handling.

### Phase 2: Tool Use (Next Step)

_Ref: Chapter 5: Tool Use_

- [ ] Define the `Tool` trait (Rust function wrapper).
- [ ] Implement `ToolRegistry` in the Agent.
- [ ] Implement "Stop Token" detection to catch when the model wants to use a tool.
- [ ] Build a Parser to convert Model Output -\> Tool Execution.

### Phase 3: Optimization

- [ ] **KV Cache Reuse:** Optimize `generate` to only process _new_ tokens instead of re-processing the whole history (Performance boost).
- [ ] **Streaming:** Implement `TokenOutputStream` for real-time typewriter effects.

---

### How to Run

1.  **Download Model:**
    Ensure `Qwen3-14B-Q4_K_M.gguf` and `tokenizer.json` are in the project root.
2.  **Run Multi-turn Chat:**
    ```bash
    cargo run --release --example multi_turn_chat
    ```
