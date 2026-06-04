use std::path::PathBuf;
use std::sync::Arc;

use poly_agent_core::*;
use poly_agent_providers::{ChatRequest, ModelAdapter, ModelResponse, ProviderError};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{AgentRuntime, AgentTool, ToolContext, ToolRegistry};

// --- Shared test helpers ---

struct LoopingAdapter;

#[async_trait::async_trait]
impl ModelAdapter for LoopingAdapter {
    async fn chat(
        &self,
        _request: ChatRequest,
    ) -> Result<ModelResponse, ProviderError> {
        Ok(ModelResponse::ToolCalls(vec![ToolCall {
            id: "call_1".to_string(),
            name: "test_tool".to_string(),
            arguments: serde_json::json!({}),
        }]))
    }
}

struct TextAdapter(String);

#[async_trait::async_trait]
impl ModelAdapter for TextAdapter {
    async fn chat(
        &self,
        _request: ChatRequest,
    ) -> Result<ModelResponse, ProviderError> {
        Ok(ModelResponse::Text(self.0.clone()))
    }
}

struct TestTool;

#[async_trait::async_trait]
impl AgentTool for TestTool {
    fn name(&self) -> &'static str { "test_tool" }
    fn description(&self) -> &'static str { "A test tool" }
    fn risk(&self) -> ToolRisk { ToolRisk::Safe }
    fn parameters_schema(&self) -> serde_json::Value { serde_json::json!({}) }
    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        Ok(ToolResult { tool_call_id: String::new(), output: "test output".to_string(), is_error: false, cached: false })
    }
}

struct DangerousTool;

#[async_trait::async_trait]
impl AgentTool for DangerousTool {
    fn name(&self) -> &'static str { "dangerous_tool" }
    fn description(&self) -> &'static str { "A dangerous tool" }
    fn risk(&self) -> ToolRisk { ToolRisk::RequiresApproval }
    fn parameters_schema(&self) -> serde_json::Value { serde_json::json!({}) }
    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        Ok(ToolResult { tool_call_id: String::new(), output: "executed dangerous tool".to_string(), is_error: false, cached: false })
    }
}

struct FakeApplyPatch;

#[async_trait::async_trait]
impl AgentTool for FakeApplyPatch {
    fn name(&self) -> &'static str { "apply_patch" }
    fn description(&self) -> &'static str { "fake" }
    fn risk(&self) -> ToolRisk { ToolRisk::RequiresApproval }
    fn parameters_schema(&self) -> serde_json::Value { serde_json::json!({"type":"object"}) }
    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        Ok(ToolResult { tool_call_id: String::new(), output: "patched".to_string(), is_error: false, cached: false })
    }
}

fn test_input(max_steps: usize) -> AgentInput {
    AgentInput {
        prompt: "test".to_string(),
        workspace: PathBuf::from("/tmp/test"),
        model: ModelConfig {
            provider: ModelProvider::Ollama,
            model: "test".to_string(),
            base_url: None,
            api_key: None,
        },
        limits: RuntimeLimits { max_steps, ..RuntimeLimits::default() },
    }
}

// --- Tests ---

#[tokio::test]
async fn runtime_stops_at_max_steps() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    let runtime = AgentRuntime::new(tools, Arc::new(LoopingAdapter));
    let (tx, mut rx) = mpsc::channel(64);

    let result = runtime.run(test_input(3), tx).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::StepLimitReached));

    let mut got_limit = false;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::StepLimitReached { .. }) {
            got_limit = true;
        }
    }
    assert!(got_limit);
}

#[tokio::test]
async fn runtime_handles_unknown_tool() {
    let tools = ToolRegistry::new();

    struct UnknownThenTextAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for UnknownThenTextAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 { Ok(ModelResponse::ToolCalls(vec![ToolCall { id: "call_1".to_string(), name: "nonexistent_tool".to_string(), arguments: serde_json::json!({}) }])) }
            else { Ok(ModelResponse::Text("I see the tool was not found".to_string())) }
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(UnknownThenTextAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) }));
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(test_input(5), tx).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn runtime_returns_text_immediately() {
    let tools = ToolRegistry::new();
    let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Hello!".to_string())));
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(test_input(5), tx).await.unwrap();
    assert_eq!(result.text, "Hello!");
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn runtime_pauses_and_resumes_on_approval() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(DangerousTool));

    struct DangerousThenTextAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for DangerousThenTextAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 { Ok(ModelResponse::ToolCalls(vec![ToolCall { id: "call_1".to_string(), name: "dangerous_tool".to_string(), arguments: serde_json::json!({}) }])) }
            else { Ok(ModelResponse::Text("Done!".to_string())) }
        }
    }

    let runtime = Arc::new(AgentRuntime::new(tools, Arc::new(DangerousThenTextAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) })));
    let (tx, mut rx) = mpsc::channel(64);
    let runtime_clone = runtime.clone();
    let handle = tokio::spawn(async move { runtime_clone.run(test_input(5), tx).await });

    let mut run_id = None;
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ApprovalRequired { run_id: id, call } = event {
            assert_eq!(call.name, "dangerous_tool");
            run_id = Some(id);
            break;
        }
    }
    runtime.approve_tool(run_id.unwrap(), "call_1").await.unwrap();
    let result = handle.await.unwrap().unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn runtime_pauses_and_handles_rejection() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(DangerousTool));

    struct DangerousThenTextAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for DangerousThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 { Ok(ModelResponse::ToolCalls(vec![ToolCall { id: "call_1".to_string(), name: "dangerous_tool".to_string(), arguments: serde_json::json!({}) }])) }
            else {
                let last = request.messages.last().unwrap();
                assert_eq!(last.role, ChatRole::Tool);
                assert_eq!(last.content, "Tool execution denied by user.");
                Ok(ModelResponse::Text("Cancelled".to_string()))
            }
        }
    }

    let runtime = Arc::new(AgentRuntime::new(tools, Arc::new(DangerousThenTextAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) })));
    let (tx, mut rx) = mpsc::channel(64);
    let runtime_clone = runtime.clone();
    let handle = tokio::spawn(async move { runtime_clone.run(test_input(5), tx).await });

    let mut run_id = None;
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ApprovalRequired { run_id: id, .. } = event { run_id = Some(id); break; }
    }
    runtime.reject_tool(run_id.unwrap(), "call_1").await.unwrap();
    let result = handle.await.unwrap().unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn runtime_handles_unknown_approval_id() {
    let runtime = AgentRuntime::new(ToolRegistry::new(), Arc::new(TextAdapter("Hello".to_string())));
    let err = runtime.approve_tool(Uuid::new_v4(), "unknown_id").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("No pending approval found"));
}

#[test]
fn truncate_output_within_limit() {
    assert_eq!(AgentRuntime::truncate_output("hello world", 100), "hello world");
}

#[test]
fn truncate_output_over_limit() {
    let r = AgentRuntime::truncate_output("hello world, this is a long string", 11);
    assert!(r.starts_with("hello world"));
    assert!(r.contains("[output truncated]"));
}

#[tokio::test]
async fn readme_title_change_prompts_approval_for_apply_patch() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeApplyPatch));

    struct ApplyPatchThenTextAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for ApplyPatchThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                let sys = request.messages.first().unwrap();
                assert_eq!(sys.role, ChatRole::System);
                assert!(sys.content.contains("apply_patch"));
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(), name: "apply_patch".to_string(),
                    arguments: serde_json::json!({"path":"README.md","expected_old_text":"# poly-agent","replacement_text":"# Poly Agent","reason":"rename project"}),
                }]))
            } else { Ok(ModelResponse::Text("I have updated the title.".to_string())) }
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(ApplyPatchThenTextAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) }));
    let mut input = test_input(5);
    input.prompt = "Change README.md title from poly-agent to Poly Agent".to_string();
    let (tx, mut rx) = mpsc::channel(64);
    let handle = tokio::spawn(async move { runtime.run(input, tx).await });

    let mut approval_seen = false;
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ApprovalRequired { call, .. } = event { assert_eq!(call.name, "apply_patch"); approval_seen = true; break; }
    }
    assert!(approval_seen);
    handle.abort();
}

#[tokio::test]
async fn read_only_summary_still_works() {
    let runtime = AgentRuntime::new(ToolRegistry::new(), Arc::new(TextAdapter("It's a Rust agent runtime.".to_string())));
    let mut input = test_input(5);
    input.prompt = "Summarise what this project does".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx).await.unwrap();
    assert_eq!(result.text, "It's a Rust agent runtime.");
}

#[tokio::test]
async fn guard_overwrites_false_success_claim() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct ReadThenFalseClaimAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for ReadThenFalseClaimAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            Ok(ModelResponse::Text("I have updated the file. Done.".to_string()))
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(ReadThenFalseClaimAdapter));
    let mut input = test_input(5);
    input.prompt = "Change README.md title from poly-agent to Poly Agent".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
    assert!(!result.text.contains("I have updated"));
    assert!(result.text.contains("no edit was applied"));
}

#[tokio::test]
async fn codebase_intent_triggers_tool_use() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct TestToolAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for TestToolAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                assert!(request.messages.iter().any(|m| m.content.contains("MUST inspect the workspace")));
                Ok(ModelResponse::ToolCalls(vec![ToolCall { id: "call_1".to_string(), name: "test_tool".to_string(), arguments: serde_json::json!({}) }]))
            } else { Ok(ModelResponse::Text("I found the answer.".to_string())) }
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(TestToolAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) }));
    let mut input = test_input(5);
    input.prompt = "What is this codebase?".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx).await.unwrap();
    assert_eq!(result.text, "I found the answer.");
}

#[tokio::test]
async fn codebase_intent_with_zero_tools_returns_guard_message() {
    let tools = ToolRegistry::new();
    struct NoToolAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for NoToolAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            Ok(ModelResponse::Text("Some answer without tools.".to_string()))
        }
    }
    let runtime = AgentRuntime::new(tools, Arc::new(NoToolAdapter));
    let mut input = test_input(5);
    input.prompt = "What is this codebase?".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx).await.unwrap();
    assert!(result.text.contains("I need to inspect the workspace"));
}

#[tokio::test]
async fn malformed_output_triggers_retry() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct MalformedThenTextAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for MalformedThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 { Err(ProviderError::MalformedModelOutput) }
            else {
                assert!(request.messages.iter().any(|m| m.content.contains("invalid internal control tokens")));
                Ok(ModelResponse::Text("Corrected response.".to_string()))
            }
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(MalformedThenTextAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) }));
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(test_input(5), tx).await.unwrap();
    assert_eq!(result.text, "Corrected response.");
}

#[tokio::test]
async fn graceful_fallback_after_tool_success() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct ToolThenErrorAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for ToolThenErrorAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            static CALLED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let c = CALLED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 { Ok(ModelResponse::ToolCalls(vec![ToolCall { id: "call_1".to_string(), name: "test_tool".to_string(), arguments: serde_json::json!({}) }])) }
            else { Err(ProviderError::Api { status: 500, body: "Internal error".to_string() }) }
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(ToolThenErrorAdapter));
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(test_input(5), tx).await.unwrap();

    match result.finish_reason {
        FinishReason::PartialFailure { tool_name, last_tool_result } => {
            assert_eq!(tool_name, "test_tool");
            assert!(last_tool_result.contains("test output"));
        }
        _ => panic!("Expected PartialFailure, got {:?}", result.finish_reason),
    }
}

// --- New tests for step-limit synthesis, unknown-tool recovery, read cache ---

/// Adapter that returns tool calls when tools are available, text when tools are empty.
struct SynthesisAdapter {
    call_count: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl ModelAdapter for SynthesisAdapter {
    async fn chat(
        &self,
        request: ChatRequest,
    ) -> Result<ModelResponse, ProviderError> {
        let _ = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if request.tools.is_empty() {
            Ok(ModelResponse::Text("Synthesized answer.".to_string()))
        } else {
            Ok(ModelResponse::ToolCalls(vec![ToolCall {
                id: "call_1".to_string(),
                name: "test_tool".to_string(),
                arguments: serde_json::json!({}),
            }]))
        }
    }
}

#[tokio::test]
async fn max_steps_synthesis_produces_final_answer() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    let runtime = AgentRuntime::new(tools, Arc::new(SynthesisAdapter {
        call_count: std::sync::atomic::AtomicUsize::new(0),
    }));
    let (tx, mut rx) = mpsc::channel(64);

    let result = runtime.run(test_input(3), tx).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::StepLimitSynthesized));
    assert_eq!(result.text, "Synthesized answer.");

    // StepLimitReached event should still fire.
    let mut got_limit = false;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::StepLimitReached { .. }) {
            got_limit = true;
        }
    }
    assert!(got_limit);
}

#[tokio::test]
async fn same_unknown_tool_twice_fails_cleanly() {
    let tools = ToolRegistry::new();

    struct UnknownTwiceAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for UnknownTwiceAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ModelResponse::ToolCalls(vec![ToolCall {
                id: format!("call_{}", c),
                name: "bad_tool".to_string(),
                arguments: serde_json::json!({}),
            }]))
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(UnknownTwiceAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) }));
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime.run(test_input(5), tx).await.unwrap();

    assert!(matches!(result.finish_reason, FinishReason::Error(_)));
    assert!(result.text.contains("bad_tool"));

    // UnknownToolRequested events should fire.
    let mut unknown_count = 0;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::UnknownToolRequested { .. }) {
            unknown_count += 1;
        }
    }
    assert!(unknown_count >= 1);
}

#[tokio::test]
async fn step_limit_with_successful_tools_returns_partial() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    // max_steps=1: one tool call loop then hit limit.
    let runtime = AgentRuntime::new(tools, Arc::new(LoopingAdapter));
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(test_input(1), tx).await.unwrap();

    // Synthesis fails (LoopingAdapter returns ToolCalls even with no tools),
    // so falls back to partial answer with StepLimitReached.
    assert!(matches!(result.finish_reason, FinishReason::StepLimitReached));
    assert!(result.text.contains("Reached step limit"));
}

#[tokio::test]
async fn valid_tool_list_appears_in_system_prompt() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct CheckToolListAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for CheckToolListAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                let sys = request.messages.first().unwrap();
                assert_eq!(sys.role, ChatRole::System);
                assert!(sys.content.contains("list_files"));
                assert!(sys.content.contains("read_file"));
                assert!(sys.content.contains("search_files"));
                assert!(sys.content.contains("apply_patch"));
                assert!(sys.content.contains("write_file"));
                assert!(sys.content.contains("Never invent tool names"));
                assert!(sys.content.contains("Tool budget"));
                assert!(sys.content.contains("1")); // max_steps
            }
            Ok(ModelResponse::Text("OK".to_string()))
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(CheckToolListAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) }));
    let mut input = test_input(1);
    input.prompt = "Hello, can you help?".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn read_file_cache_avoids_duplicate_reads() {
    let call_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    struct CachedReadTool { counter: Arc<std::sync::atomic::AtomicUsize> }
    #[async_trait::async_trait]
    impl AgentTool for CachedReadTool {
        fn name(&self) -> &'static str { "read_file" }
        fn description(&self) -> &'static str { "read" }
        fn risk(&self) -> ToolRisk { ToolRisk::Safe }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})
        }
        async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
            self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ToolResult { tool_call_id: String::new(), output: "file content".to_string(), is_error: false, cached: false })
        }
    }

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(CachedReadTool { counter: call_counter.clone() }));

    struct ReadTwiceAdapter { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for ReadTwiceAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match c {
                0 => Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "test.txt"}),
                }])),
                1 => Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "test.txt"}),
                }])),
                _ => Ok(ModelResponse::Text("Done.".to_string())),
            }
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(ReadTwiceAdapter { call_count: std::sync::atomic::AtomicUsize::new(0) }));
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime.run(test_input(5), tx).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
    assert_eq!(result.text, "Done.");

    // The underlying tool should have been called only once (second read was cached).
    assert_eq!(call_counter.load(std::sync::atomic::Ordering::SeqCst), 1);

    // At least one ToolCallFinished should have cached: true.
    let mut cached_count = 0;
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::ToolCallFinished { result, .. } = event {
            if result.cached {
                cached_count += 1;
            }
        }
    }
    assert_eq!(cached_count, 1, "Expected one cached read_file result");
}

#[tokio::test]
async fn unknown_tool_triggers_corrective_retry() {
    let tools = ToolRegistry::new();

    struct UnknownThenText { call_count: std::sync::atomic::AtomicUsize }
    #[async_trait::async_trait]
    impl ModelAdapter for UnknownThenText {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "nonexistent_tool".to_string(),
                    arguments: serde_json::json!({}),
                }]))
            } else {
                Ok(ModelResponse::Text("I see the tool was not found".to_string()))
            }
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(UnknownThenText { call_count: std::sync::atomic::AtomicUsize::new(0) }));
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime.run(test_input(5), tx).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
    assert!(result.text.contains("tool was not found"));

    // Verify corrective message was sent (UnknownToolRequested event).
    let mut got_unknown = false;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::UnknownToolRequested { .. }) {
            got_unknown = true;
        }
    }
    assert!(got_unknown);
}
