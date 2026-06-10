use std::path::PathBuf;
use std::sync::Arc;

use poly_agent_core::*;
use crate::agent_config::builtin_agent;
use poly_agent_providers::{ChatRequest, ModelAdapter, ModelResponse, ProviderError};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    AgentRuntime, AgentRuntimeConfig, AgentTool, AutoReviewer, ReviewContext, ReviewVerdict,
    ToolContext, ToolRegistry,
};

// --- Shared test helpers ---

struct LoopingAdapter;

#[async_trait::async_trait]
impl ModelAdapter for LoopingAdapter {
    async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
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
    async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        Ok(ModelResponse::Text(self.0.clone()))
    }
}

struct CapturingTextAdapter {
    seen_tools: Arc<tokio::sync::Mutex<Vec<usize>>>,
}

#[async_trait::async_trait]
impl ModelAdapter for CapturingTextAdapter {
    async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        self.seen_tools.lock().await.push(request.tools.len());
        Ok(ModelResponse::Text("Hello there.".to_string()))
    }
}

struct TestTool;

#[async_trait::async_trait]
impl AgentTool for TestTool {
    fn name(&self) -> &'static str {
        "test_tool"
    }
    fn description(&self) -> &'static str {
        "A test tool"
    }
    fn risk(&self) -> ToolRisk {
        ToolRisk::Safe
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: String::new(),
            output: "test output".to_string(),
            is_error: false,
            cached: false,
        })
    }
}

struct DangerousTool;

#[async_trait::async_trait]
impl AgentTool for DangerousTool {
    fn name(&self) -> &'static str {
        "dangerous_tool"
    }
    fn description(&self) -> &'static str {
        "A dangerous tool"
    }
    fn risk(&self) -> ToolRisk {
        ToolRisk::RequiresApproval
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: String::new(),
            output: "executed dangerous tool".to_string(),
            is_error: false,
            cached: false,
        })
    }
}

struct FakeApplyPatch;

#[async_trait::async_trait]
impl AgentTool for FakeApplyPatch {
    fn name(&self) -> &'static str {
        "apply_patch"
    }
    fn description(&self) -> &'static str {
        "fake"
    }
    fn risk(&self) -> ToolRisk {
        ToolRisk::RequiresApproval
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type":"object"})
    }
    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: String::new(),
            output: "patched".to_string(),
            is_error: false,
            cached: false,
        })
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
        limits: RuntimeLimits {
            max_steps,
            ..RuntimeLimits::default()
        },
        permission_preset: PermissionPreset::Default,
        resolved_context: None,
    }
}

fn cancellation() -> CancellationToken {
    CancellationToken::new()
}

// --- Tests ---

#[tokio::test]
async fn simple_greeting_uses_fast_path_without_tools() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));
    let seen_tools = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let runtime = AgentRuntime::new(
        tools,
        Arc::new(CapturingTextAdapter {
            seen_tools: seen_tools.clone(),
        }),
    );
    let (tx, mut rx) = mpsc::channel(64);
    let mut input = test_input(3);
    input.prompt = "Hello".to_string();

    let output = runtime.run(input, tx, cancellation()).await.unwrap();
    assert_eq!(output.text, "Hello there.");
    assert_eq!(*seen_tools.lock().await, vec![0]);

    let mut saw_tool_event = false;
    while let Ok(event) = rx.try_recv() {
        if matches!(
            event,
            AgentEvent::ToolCallRequested { .. }
                | AgentEvent::ToolCallStarted { .. }
                | AgentEvent::ToolCallFinished { .. }
        ) {
            saw_tool_event = true;
        }
    }
    assert!(!saw_tool_event);
}

#[tokio::test]
async fn runtime_stops_at_max_steps() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    let runtime = AgentRuntime::new(tools, Arc::new(LoopingAdapter));
    let (tx, mut rx) = mpsc::channel(64);

    let result = runtime
        .run(test_input(3), tx, cancellation())
        .await
        .unwrap();
    assert!(matches!(
        result.finish_reason,
        FinishReason::StepLimitReached
    ));

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

    struct UnknownThenTextAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for UnknownThenTextAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "nonexistent_tool".to_string(),
                    arguments: serde_json::json!({}),
                }]))
            } else {
                Ok(ModelResponse::Text(
                    "I see the tool was not found".to_string(),
                ))
            }
        }
    }

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(UnknownThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn runtime_returns_text_immediately() {
    let tools = ToolRegistry::new();
    let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Hello!".to_string())));
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
    assert_eq!(result.text, "Hello!");
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn runtime_pauses_and_resumes_on_approval() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(DangerousTool));

    struct DangerousThenTextAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for DangerousThenTextAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "dangerous_tool".to_string(),
                    arguments: serde_json::json!({}),
                }]))
            } else {
                Ok(ModelResponse::Text("Done!".to_string()))
            }
        }
    }

    let runtime = Arc::new(AgentRuntime::new(
        tools,
        Arc::new(DangerousThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    ));
    let (tx, mut rx) = mpsc::channel(64);
    let runtime_clone = runtime.clone();
    let handle =
        tokio::spawn(async move { runtime_clone.run(test_input(5), tx, cancellation()).await });

    let mut run_id = None;
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ApprovalRequired { run_id: id, call } = event {
            assert_eq!(call.name, "dangerous_tool");
            run_id = Some(id);
            break;
        }
    }
    runtime
        .approve_tool(run_id.unwrap(), "call_1")
        .await
        .unwrap();
    let result = handle.await.unwrap().unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn runtime_pauses_and_handles_rejection() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(DangerousTool));

    struct DangerousThenTextAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for DangerousThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "dangerous_tool".to_string(),
                    arguments: serde_json::json!({}),
                }]))
            } else {
                let last = request.messages.last().unwrap();
                assert_eq!(last.role, ChatRole::Tool);
                assert_eq!(last.content, "Tool execution denied by user.");
                Ok(ModelResponse::Text("Cancelled".to_string()))
            }
        }
    }

    let runtime = Arc::new(AgentRuntime::new(
        tools,
        Arc::new(DangerousThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    ));
    let (tx, mut rx) = mpsc::channel(64);
    let runtime_clone = runtime.clone();
    let handle =
        tokio::spawn(async move { runtime_clone.run(test_input(5), tx, cancellation()).await });

    let mut run_id = None;
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ApprovalRequired { run_id: id, .. } = event {
            run_id = Some(id);
            break;
        }
    }
    runtime
        .reject_tool(run_id.unwrap(), "call_1")
        .await
        .unwrap();
    let result = handle.await.unwrap().unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn runtime_handles_unknown_approval_id() {
    let runtime = AgentRuntime::new(
        ToolRegistry::new(),
        Arc::new(TextAdapter("Hello".to_string())),
    );
    let err = runtime.approve_tool(Uuid::new_v4(), "unknown_id").await;
    assert!(err.is_err());
    assert!(err
        .unwrap_err()
        .to_string()
        .contains("No pending approval found"));
}

#[test]
fn truncate_output_within_limit() {
    assert_eq!(
        AgentRuntime::truncate_output("hello world", 100),
        "hello world"
    );
}

#[test]
fn truncate_output_over_limit() {
    let r = AgentRuntime::truncate_output("hello world, this is a long string", 11);
    assert!(r.starts_with("hello world"));
    assert!(r.contains("[output truncated]"));
}

#[tokio::test]
async fn readme_title_change_auto_approves_apply_patch_under_default_preset() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeApplyPatch));

    struct ApplyPatchThenTextAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for ApplyPatchThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                let sys = request.messages.first().unwrap();
                assert_eq!(sys.role, ChatRole::System);
                assert!(sys.content.contains("apply_patch"));
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "apply_patch".to_string(),
                    arguments: serde_json::json!({"path":"README.md","expected_old_text":"# poly-agent","replacement_text":"# Poly Agent","reason":"rename project"}),
                }]))
            } else {
                Ok(ModelResponse::Text("I have updated the title.".to_string()))
            }
        }
    }

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(ApplyPatchThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let mut input = test_input(5);
    input.prompt = "Change README.md title from poly-agent to Poly Agent".to_string();
    let (tx, mut rx) = mpsc::channel(64);
    let handle = tokio::spawn(async move { runtime.run(input, tx, cancellation()).await });

    let mut auto_approved = false;
    let mut approval_seen = false;
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::ToolAutoApproved {
                tool_name, reason, ..
            } => {
                assert_eq!(tool_name, "apply_patch");
                assert_eq!(reason, AutoApproveReason::PresetDefault);
                auto_approved = true;
            }
            AgentEvent::ApprovalRequired { .. } => approval_seen = true,
            _ => {}
        }
        if auto_approved {
            break;
        }
    }
    assert!(auto_approved, "expected ToolAutoApproved for apply_patch");
    assert!(
        !approval_seen,
        "Default preset must not emit ApprovalRequired for file writes"
    );
    handle.abort();
}

#[tokio::test]
async fn read_only_summary_still_works() {
    let runtime = AgentRuntime::new(
        ToolRegistry::new(),
        Arc::new(TextAdapter("It's a Rust agent runtime.".to_string())),
    );
    let mut input = test_input(5);
    input.prompt = "Summarise what this project does".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx, cancellation()).await.unwrap();
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
            Ok(ModelResponse::Text(
                "I have updated the file. Done.".to_string(),
            ))
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(ReadThenFalseClaimAdapter));
    let mut input = test_input(5);
    input.prompt = "Change README.md title from poly-agent to Poly Agent".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx, cancellation()).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
    assert!(!result.text.contains("I have updated"));
    assert!(result.text.contains("no edit was applied"));
}

#[tokio::test]
async fn codebase_intent_triggers_tool_use() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct TestToolAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for TestToolAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                assert!(request
                    .messages
                    .iter()
                    .any(|m| m.content.contains("MUST inspect the workspace")));
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "test_tool".to_string(),
                    arguments: serde_json::json!({}),
                }]))
            } else {
                Ok(ModelResponse::Text("I found the answer.".to_string()))
            }
        }
    }

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(TestToolAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let mut input = test_input(5);
    input.prompt = "What is this codebase?".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx, cancellation()).await.unwrap();
    assert_eq!(result.text, "I found the answer.");
}

#[tokio::test]
async fn codebase_intent_with_zero_tools_returns_guard_message() {
    let tools = ToolRegistry::new();
    struct NoToolAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for NoToolAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            Ok(ModelResponse::Text(
                "Some answer without tools.".to_string(),
            ))
        }
    }
    let runtime = AgentRuntime::new(tools, Arc::new(NoToolAdapter));
    let mut input = test_input(5);
    input.prompt = "What is this codebase?".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx, cancellation()).await.unwrap();
    assert!(result.text.contains("I need to inspect the workspace"));
}

#[tokio::test]
async fn malformed_output_triggers_retry() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct MalformedThenTextAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for MalformedThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                Err(ProviderError::MalformedModelOutput)
            } else {
                assert!(request
                    .messages
                    .iter()
                    .any(|m| m.content.contains("invalid internal control tokens")));
                Ok(ModelResponse::Text("Corrected response.".to_string()))
            }
        }
    }

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(MalformedThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
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
            if c == 0 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "test_tool".to_string(),
                    arguments: serde_json::json!({}),
                }]))
            } else {
                Err(ProviderError::Api {
                    status: 500,
                    body: "Internal error".to_string(),
                })
            }
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(ToolThenErrorAdapter));
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();

    match result.finish_reason {
        FinishReason::PartialFailure {
            tool_name,
            last_tool_result,
        } => {
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
    async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        let _ = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(SynthesisAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let (tx, mut rx) = mpsc::channel(64);

    let result = runtime
        .run(test_input(3), tx, cancellation())
        .await
        .unwrap();
    assert!(matches!(
        result.finish_reason,
        FinishReason::StepLimitSynthesized
    ));
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

    struct UnknownTwiceAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for UnknownTwiceAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ModelResponse::ToolCalls(vec![ToolCall {
                id: format!("call_{}", c),
                name: "bad_tool".to_string(),
                arguments: serde_json::json!({}),
            }]))
        }
    }

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(UnknownTwiceAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();

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
    let result = runtime
        .run(test_input(1), tx, cancellation())
        .await
        .unwrap();

    // Synthesis fails (LoopingAdapter returns ToolCalls even with no tools),
    // so falls back to partial answer with StepLimitReached.
    assert!(matches!(
        result.finish_reason,
        FinishReason::StepLimitReached
    ));
    assert!(result.text.contains("Reached step limit"));
}

#[tokio::test]
async fn valid_tool_list_appears_in_system_prompt() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct CheckToolListAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for CheckToolListAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(CheckToolListAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let mut input = test_input(1);
    input.prompt = "Hello, can you help?".to_string();
    let (tx, _rx) = mpsc::channel(64);
    let result = runtime.run(input, tx, cancellation()).await.unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));
}

#[tokio::test]
async fn resolved_follow_up_context_is_sent_to_model() {
    struct ContextAdapter;

    #[async_trait::async_trait]
    impl ModelAdapter for ContextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let joined = request
                .messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(joined.contains("active file: test.txt"));
            assert!(joined.contains("Preserve existing content; do not remove anything."));
            assert!(joined.contains("Do not ask for the file again"));
            Ok(ModelResponse::Text("Done".to_string()))
        }
    }

    let runtime = AgentRuntime::new(ToolRegistry::new(), Arc::new(ContextAdapter));
    let mut input = test_input(2);
    input.prompt = "add another sentence".to_string();
    input.resolved_context = Some(AgentResolvedContext {
        active_workspace: Some("/tmp/workspace".to_string()),
        active_file: Some("test.txt".to_string()),
        recently_viewed_files: vec!["test.txt".to_string()],
        recently_edited_files: Vec::new(),
        recent_constraints: vec![
            "Preserve existing content; do not remove anything.".to_string(),
        ],
        last_tool_call: Some(AgentLastToolCall {
            tool_name: "read_file".to_string(),
            target_path: Some("test.txt".to_string()),
        }),
    });

    let (tx, _rx) = mpsc::channel(64);
    let output = runtime.run(input, tx, cancellation()).await.unwrap();
    assert_eq!(output.text, "Done");
}

#[tokio::test]
async fn follow_up_edit_does_not_mutate_without_tool_success() {
    let workspace = temp_workspace("follow-up-guard");
    let path = workspace.join("test.txt");
    tokio::fs::write(&path, "original\n").await.unwrap();

    let runtime = AgentRuntime::new(
        ToolRegistry::new(),
        Arc::new(TextAdapter("I will write to test.txt. Please confirm.".to_string())),
    );
    let mut input = test_input(2);
    input.workspace = workspace.clone();
    input.prompt = "Write to test.txt adding a few more sentences".to_string();
    input.resolved_context = Some(AgentResolvedContext {
        active_file: Some("test.txt".to_string()),
        recent_constraints: vec!["Preserve existing content; do not remove anything.".to_string()],
        ..AgentResolvedContext::default()
    });

    let (tx, _rx) = mpsc::channel(64);
    let output = runtime.run(input, tx, cancellation()).await.unwrap();

    assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), "original\n");
    assert!(output.text.contains("requires a file mutation"));
    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn edit_intent_empty_text_fails_without_mutating_tool() {
    let runtime = AgentRuntime::new(ToolRegistry::new(), Arc::new(TextAdapter(String::new())));
    let mut input = test_input(2);
    input.prompt = "Create me a file and add some nice text inside it.".to_string();

    let (tx, mut rx) = mpsc::channel(64);
    let output = runtime.run(input, tx, cancellation()).await.unwrap();

    assert!(matches!(output.finish_reason, FinishReason::Error(_)));
    assert!(output.text.contains("no file changes were produced"));

    let mut saw_error = false;
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::Error { error, .. } = event {
            saw_error = true;
            assert!(error.contains("no file changes were produced"));
        }
    }
    assert!(saw_error, "expected Error event for edit intent without mutation");
}

#[tokio::test]
async fn edit_intent_plain_noop_text_retries_then_fails_without_mutating_tool() {
    struct PlainNoopAdapter {
        count: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ModelAdapter for PlainNoopAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            self.count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ModelResponse::Text("No changes were made.".to_string()))
        }
    }

    let adapter = Arc::new(PlainNoopAdapter {
        count: std::sync::atomic::AtomicUsize::new(0),
    });
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeWriteFile));
    let runtime = AgentRuntime::new(tools, adapter.clone());
    let mut input = test_input(3);
    input.prompt = "Create me a file and add some nice text inside it.".to_string();

    let (tx, _rx) = mpsc::channel(64);
    let output = runtime.run(input, tx, cancellation()).await.unwrap();

    assert_eq!(adapter.count.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert!(matches!(output.finish_reason, FinishReason::Error(_)));
    assert!(output.text.contains("no file changes were produced"));
}

fn temp_workspace(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("poly-agent-{name}-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[tokio::test]
async fn read_file_cache_avoids_duplicate_reads() {
    let call_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    struct CachedReadTool {
        counter: Arc<std::sync::atomic::AtomicUsize>,
    }
    #[async_trait::async_trait]
    impl AgentTool for CachedReadTool {
        fn name(&self) -> &'static str {
            "read_file"
        }
        fn description(&self) -> &'static str {
            "read"
        }
        fn risk(&self) -> ToolRisk {
            ToolRisk::Safe
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})
        }
        async fn run(
            &self,
            _args: serde_json::Value,
            _ctx: ToolContext,
        ) -> anyhow::Result<ToolResult> {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ToolResult {
                tool_call_id: String::new(),
                output: "file content".to_string(),
                is_error: false,
                cached: false,
            })
        }
    }

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(CachedReadTool {
        counter: call_counter.clone(),
    }));

    struct ReadTwiceAdapter {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for ReadTwiceAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(ReadTwiceAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
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

    struct UnknownThenText {
        call_count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for UnknownThenText {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "nonexistent_tool".to_string(),
                    arguments: serde_json::json!({}),
                }]))
            } else {
                Ok(ModelResponse::Text(
                    "I see the tool was not found".to_string(),
                ))
            }
        }
    }

    let runtime = AgentRuntime::new(
        tools,
        Arc::new(UnknownThenText {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
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

// --- Permission preset tests ---

struct FakeRunCommand;

#[async_trait::async_trait]
impl AgentTool for FakeRunCommand {
    fn name(&self) -> &'static str {
        "run_command"
    }
    fn description(&self) -> &'static str {
        "fake run_command"
    }
    fn risk(&self) -> ToolRisk {
        ToolRisk::Dangerous
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]})
    }
    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: String::new(),
            output: "executed".to_string(),
            is_error: false,
            cached: false,
        })
    }
}

struct FakeWriteFile;

#[async_trait::async_trait]
impl AgentTool for FakeWriteFile {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "fake write"
    }
    fn risk(&self) -> ToolRisk {
        ToolRisk::RequiresApproval
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]})
    }
    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: String::new(),
            output: "wrote file".to_string(),
            is_error: false,
            cached: false,
        })
    }
}

struct ScriptedReviewer {
    verdict: ReviewVerdict,
}

#[async_trait::async_trait]
impl AutoReviewer for ScriptedReviewer {
    async fn review(&self, _call: &ToolCall, _ctx: &ReviewContext) -> ReviewVerdict {
        self.verdict.clone()
    }
}

fn runtime_with_preset(
    tools: ToolRegistry,
    adapter: Arc<dyn ModelAdapter>,
    preset: PermissionPreset,
    reviewer: Option<Arc<dyn AutoReviewer>>,
) -> AgentRuntime {
    AgentRuntime::with_config(
        tools,
        adapter,
        AgentRuntimeConfig {
            permission_preset: preset,
            reviewer,
            agent_config: builtin_agent("build"),
        },
    )
}

#[tokio::test]
async fn default_preset_prompts_for_run_command() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeRunCommand));

    struct CmdThenTextAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for CmdThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            if request.messages.iter().any(|m| m.tool_call_id.is_some()) {
                Ok(ModelResponse::Text("done".to_string()))
            } else {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "cmd_1".to_string(),
                    name: "run_command".to_string(),
                    arguments: serde_json::json!({"command": "echo hi"}),
                }]))
            }
        }
    }

    let runtime = Arc::new(runtime_with_preset(
        tools,
        Arc::new(CmdThenTextAdapter),
        PermissionPreset::Default,
        None,
    ));
    let (tx, mut rx) = mpsc::channel(64);
    let runtime_clone = runtime.clone();
    let handle =
        tokio::spawn(async move { runtime_clone.run(test_input(5), tx, cancellation()).await });

    let mut approval_seen = false;
    let mut auto_seen = false;
    let mut run_id: Option<Uuid> = None;
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::ApprovalRequired {
                run_id: id,
                ref call,
            } => {
                assert_eq!(call.name, "run_command");
                approval_seen = true;
                run_id = Some(id);
                break;
            }
            AgentEvent::ToolAutoApproved { ref tool_name, .. } => {
                if tool_name == "run_command" {
                    auto_seen = true;
                }
            }
            _ => {}
        }
    }
    if let Some(id) = run_id {
        let _ = runtime.approve_tool(id, "cmd_1").await;
    }
    assert!(approval_seen, "Default preset must prompt for run_command");
    assert!(
        !auto_seen,
        "Default preset must NOT auto-approve run_command"
    );
    let _ = handle.await;
}

#[tokio::test]
async fn default_preset_auto_approves_write_file() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeWriteFile));

    struct WriteThenTextAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for WriteThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            if request.messages.iter().any(|m| m.tool_call_id.is_some()) {
                Ok(ModelResponse::Text("wrote".to_string()))
            } else {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "wf_1".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({"path": "x.txt", "content": "hi"}),
                }]))
            }
        }
    }

    let runtime = runtime_with_preset(
        tools,
        Arc::new(WriteThenTextAdapter),
        PermissionPreset::Default,
        None,
    );
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));

    let mut saw_auto = false;
    let mut saw_approval = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ToolAutoApproved {
                tool_name, reason, ..
            } => {
                if tool_name == "write_file" {
                    assert_eq!(reason, AutoApproveReason::PresetDefault);
                    saw_auto = true;
                }
            }
            AgentEvent::ApprovalRequired { .. } => saw_approval = true,
            _ => {}
        }
    }
    assert!(saw_auto, "expected ToolAutoApproved for write_file");
    assert!(
        !saw_approval,
        "Default preset must not prompt for write_file"
    );
}

#[tokio::test]
async fn full_access_approves_everything() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeRunCommand));
    tools.register(Arc::new(FakeApplyPatch));

    struct FullAccessAdapter {
        count: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for FullAccessAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let n = self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "cmd_x".to_string(),
                    name: "run_command".to_string(),
                    arguments: serde_json::json!({"command": "ls"}),
                }]))
            } else if n == 1 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "patch_x".to_string(),
                    name: "apply_patch".to_string(),
                    arguments: serde_json::json!({"path": "x.txt", "expected_old_text": "a", "replacement_text": "b", "reason": "t"}),
                }]))
            } else {
                Ok(ModelResponse::Text("all done".to_string()))
            }
        }
    }

    let runtime = runtime_with_preset(
        tools,
        Arc::new(FullAccessAdapter {
            count: std::sync::atomic::AtomicUsize::new(0),
        }),
        PermissionPreset::FullAccess,
        None,
    );
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));

    let mut saw_cmd_auto = false;
    let mut saw_patch_auto = false;
    let mut saw_approval = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ToolAutoApproved {
                tool_name, reason, ..
            } => {
                assert_eq!(reason, AutoApproveReason::FullAccess);
                if tool_name == "run_command" {
                    saw_cmd_auto = true;
                }
                if tool_name == "apply_patch" {
                    saw_patch_auto = true;
                }
            }
            AgentEvent::ApprovalRequired { .. } => saw_approval = true,
            _ => {}
        }
    }
    assert!(saw_cmd_auto, "FullAccess must auto-approve run_command");
    assert!(saw_patch_auto, "FullAccess must auto-approve apply_patch");
    assert!(!saw_approval, "FullAccess must never emit ApprovalRequired");
}

#[tokio::test]
async fn auto_review_low_risk_auto_approves() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeWriteFile));

    struct WriteThenTextAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for WriteThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            if request.messages.iter().any(|m| m.tool_call_id.is_some()) {
                Ok(ModelResponse::Text("ok".to_string()))
            } else {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "wf_l".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({"path": "x.txt", "content": "hi"}),
                }]))
            }
        }
    }

    let reviewer: Arc<dyn AutoReviewer> = Arc::new(ScriptedReviewer {
        verdict: ReviewVerdict {
            risk: AutoReviewRisk::Low,
            decision: ReviewDecision::Approve,
            reason: "Looks safe.".to_string(),
        },
    });
    let runtime = runtime_with_preset(
        tools,
        Arc::new(WriteThenTextAdapter),
        PermissionPreset::AutoReview,
        Some(reviewer),
    );
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));

    let mut saw_decision = false;
    let mut saw_auto = false;
    let mut saw_approval = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::AutoReviewDecision {
                risk,
                decision,
                reason,
                ..
            } => {
                assert_eq!(risk, AutoReviewRisk::Low);
                assert_eq!(decision, ReviewDecision::Approve);
                assert_eq!(reason, "Looks safe.");
                saw_decision = true;
            }
            AgentEvent::ToolAutoApproved { reason, .. } => {
                assert_eq!(reason, AutoApproveReason::AutoReviewLow);
                saw_auto = true;
            }
            AgentEvent::ApprovalRequired { .. } => saw_approval = true,
            _ => {}
        }
    }
    assert!(saw_decision, "AutoReviewDecision event must fire");
    assert!(saw_auto, "Low-risk decision must auto-approve");
    assert!(!saw_approval, "Low-risk decision must NOT prompt user");
}

#[tokio::test]
async fn auto_review_medium_risk_asks_user() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeWriteFile));

    struct WriteThenTextAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for WriteThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            if request.messages.iter().any(|m| m.tool_call_id.is_some()) {
                Ok(ModelResponse::Text("done".to_string()))
            } else {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "wf_m".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({"path": "x.txt", "content": "hi"}),
                }]))
            }
        }
    }

    let reviewer: Arc<dyn AutoReviewer> = Arc::new(ScriptedReviewer {
        verdict: ReviewVerdict {
            risk: AutoReviewRisk::Medium,
            decision: ReviewDecision::Ask,
            reason: "Editable but ambiguous.".to_string(),
        },
    });
    let runtime = Arc::new(runtime_with_preset(
        tools,
        Arc::new(WriteThenTextAdapter),
        PermissionPreset::AutoReview,
        Some(reviewer),
    ));
    let (tx, mut rx) = mpsc::channel(64);
    let runtime_clone = runtime.clone();
    let handle =
        tokio::spawn(async move { runtime_clone.run(test_input(5), tx, cancellation()).await });

    let mut saw_decision = false;
    let mut saw_approval = false;
    let mut run_id: Option<Uuid> = None;
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::AutoReviewDecision { risk, decision, .. } => {
                assert_eq!(risk, AutoReviewRisk::Medium);
                assert_eq!(decision, ReviewDecision::Ask);
                saw_decision = true;
            }
            AgentEvent::ApprovalRequired {
                run_id: id,
                ref call,
            } => {
                assert_eq!(call.name, "write_file");
                saw_approval = true;
                run_id = Some(id);
                break;
            }
            _ => {}
        }
    }
    if let Some(id) = run_id {
        let _ = runtime.approve_tool(id, "wf_m").await;
    }
    assert!(saw_decision, "AutoReviewDecision event must fire");
    assert!(saw_approval, "Medium-risk decision must ask the user");
    let _ = handle.await;
}

#[tokio::test]
async fn auto_review_high_risk_auto_denies() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(FakeWriteFile));

    struct WriteThenTextAdapter;
    #[async_trait::async_trait]
    impl ModelAdapter for WriteThenTextAdapter {
        async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            if request.messages.iter().any(|m| m.tool_call_id.is_some()) {
                Ok(ModelResponse::Text("ok".to_string()))
            } else {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "wf_h".to_string(),
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({"path": "x.txt", "content": "hi"}),
                }]))
            }
        }
    }

    let reviewer: Arc<dyn AutoReviewer> = Arc::new(ScriptedReviewer {
        verdict: ReviewVerdict {
            risk: AutoReviewRisk::High,
            decision: ReviewDecision::Deny,
            reason: "Deleting critical file.".to_string(),
        },
    });
    let runtime = runtime_with_preset(
        tools,
        Arc::new(WriteThenTextAdapter),
        PermissionPreset::AutoReview,
        Some(reviewer),
    );
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime
        .run(test_input(5), tx, cancellation())
        .await
        .unwrap();
    assert!(matches!(result.finish_reason, FinishReason::Complete));

    let mut saw_decision = false;
    let mut saw_denied = false;
    let mut saw_approval = false;
    let mut saw_auto = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::AutoReviewDecision {
                risk,
                decision,
                reason,
                ..
            } => {
                assert_eq!(risk, AutoReviewRisk::High);
                assert_eq!(decision, ReviewDecision::Deny);
                assert_eq!(reason, "Deleting critical file.");
                saw_decision = true;
            }
            AgentEvent::ToolAutoDenied { reason, .. } => {
                assert!(reason.contains("Deleting critical file"));
                saw_denied = true;
            }
            AgentEvent::ApprovalRequired { .. } => saw_approval = true,
            AgentEvent::ToolAutoApproved { .. } => saw_auto = true,
            _ => {}
        }
    }
    assert!(
        saw_decision,
        "AutoReviewDecision event must fire even for Deny"
    );
    assert!(saw_denied, "High-risk decision must auto-deny");
    assert!(!saw_approval, "High-risk decision must NOT ask user");
    assert!(!saw_auto, "High-risk decision must NOT auto-approve");
}

// --- Streaming event tests ---

#[tokio::test]
async fn simple_prompt_emits_reasoning_activities() {
    let tools = ToolRegistry::new();
    let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Hello!".to_string())));
    for prompt in ["Hello", "How are you?", "Thanks"] {
        let (tx, mut rx) = mpsc::channel(64);
        let mut input = test_input(2);
        input.prompt = prompt.to_string();
        let _ = runtime.run(input, tx, cancellation()).await;

        let mut started = false;
        let mut thinking = false;
        let mut finished = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                AgentEvent::Started { .. } => started = true,
                AgentEvent::Activity { ref phase, .. } if phase == "thinking" => thinking = true,
                AgentEvent::Finished { .. } => finished = true,
                _ => {}
            }
        }
        assert!(started, "simple prompt {prompt:?}: must emit Started");
        assert!(thinking, "simple prompt {prompt:?}: must emit thinking Activity");
        assert!(finished, "simple prompt {prompt:?}: must emit Finished");
    }
}

#[tokio::test]
async fn final_response_delta_emitted_for_text_response() {
    let tools = ToolRegistry::new();
    let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Hello world!".to_string())));
    let (tx, mut rx) = mpsc::channel(64);
    let _ = runtime.run(test_input(2), tx, cancellation()).await;

    let mut deltas = 0;
    let mut final_text = String::new();
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::FinalResponseDelta { ref text, .. } = event {
            deltas += 1;
            final_text.push_str(text);
        }
    }
    assert!(deltas > 0, "must emit at least one FinalResponseDelta");
    assert_eq!(final_text, "Hello world!");
}

#[tokio::test]
async fn simple_chat_does_not_emit_file_read_activity() {
    let tools = ToolRegistry::new();
    let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Hello!".to_string())));
    let (tx, mut rx) = mpsc::channel(64);
    let mut input = test_input(2);
    input.prompt = "Hello".to_string();
    let _ = runtime.run(input, tx, cancellation()).await;

    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::Activity { ref phase, .. } = event {
            if phase == "context_loading" || phase == "file_read" {
                panic!("simple chat must not emit file-read activities, got phase: {phase}");
            }
        }
    }
}

#[tokio::test]
async fn project_prompt_emits_inspection_activities() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    struct InspectThenText;
    #[async_trait::async_trait]
    impl ModelAdapter for InspectThenText {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            Ok(ModelResponse::ToolCalls(vec![ToolCall {
                id: "call_1".to_string(),
                name: "test_tool".to_string(),
                arguments: serde_json::json!({}),
            }]))
        }
    }

    let runtime = AgentRuntime::new(tools, Arc::new(InspectThenText));
    let (tx, mut rx) = mpsc::channel(64);
    let mut input = test_input(3);
    input.prompt = "What is this project?".to_string();
    let _ = runtime.run(input, tx, cancellation()).await;

    let mut saw_workspace = false;
    let mut saw_reasoning = false;
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::Activity { ref phase, .. } = event {
            match phase.as_str() {
                "workspace_inspection" => saw_workspace = true,
                "thinking" => saw_reasoning = true,
                _ => {}
            }
        }
    }
    assert!(saw_workspace, "project prompt must emit workspace activity");
    assert!(saw_reasoning, "project prompt must emit thinking activity");
}

#[tokio::test]
async fn cancellation_emits_cancelled_event() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(TestTool));

    // Adapter that returns one tool call immediately on first invocation,
    // then waits for cancellation on the second invocation.
    // This lets the runtime reach the tool-execution step, where we can
    // cancel concurrently. The cancellation check at the top of the run
    // loop catches it before the second model call.
    struct CancelTestAdapter {
        call_count: std::sync::atomic::AtomicUsize,
        cancel: CancellationToken,
    }
    #[async_trait::async_trait]
    impl ModelAdapter for CancelTestAdapter {
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
            let c = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                Ok(ModelResponse::ToolCalls(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "test_tool".to_string(),
                    arguments: serde_json::json!({}),
                }]))
            } else {
                // Wait until cancelled, then return error to avoid hanging
                self.cancel.cancelled().await;
                Err(ProviderError::Parse("cancelled".to_string()))
            }
        }
    }

    let cancel = CancellationToken::new();
    let cancel_adapter = cancel.clone();
    let runtime = Arc::new(AgentRuntime::new(tools, Arc::new(CancelTestAdapter {
        call_count: std::sync::atomic::AtomicUsize::new(0),
        cancel: cancel_adapter,
    })));
    let cancel_for_task = cancel.clone();
    let (tx, mut rx) = mpsc::channel(64);
    let runtime_clone = runtime.clone();
    let handle = tokio::spawn(async move {
        runtime_clone.run(test_input(5), tx, cancel_for_task).await
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    cancel.cancel();

    let result = handle.await.unwrap();
    assert!(result.is_err(), "cancelled run must return error");

    let mut saw_cancelled = false;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::Cancelled { .. }) {
            saw_cancelled = true;
        }
    }
    assert!(saw_cancelled, "Cancelled event must be emitted when cancellation token fires");
}

#[tokio::test]
async fn run_finished_includes_final_text() {
    let tools = ToolRegistry::new();
    let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Final answer.".to_string())));
    let (tx, mut rx) = mpsc::channel(64);
    let result = runtime.run(test_input(2), tx, cancellation()).await.unwrap();

    let mut final_text = String::new();
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::Finished { ref text, .. } = event {
            final_text = text.clone();
        }
    }
    assert_eq!(result.text, "Final answer.");
    assert_eq!(final_text, "Final answer.");
}

#[tokio::test]
async fn activity_summaries_never_contain_raw_json() {
    let tools = ToolRegistry::new();
    let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Concise summary.".to_string())));
    let (tx, mut rx) = mpsc::channel(64);
    let _ = runtime.run(test_input(2), tx, cancellation()).await;

    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::Activity { ref summary, ref details, .. } = event {
            assert!(!summary.contains('{') && !summary.contains('['),
                "Activity summary must not contain raw JSON: {summary}");
            for detail in details {
                assert!(detail.len() <= 200,
                    "Activity detail too long (>200 chars): {detail}");
                assert!(!(detail.contains("arguments") && detail.contains('"')),
                    "Activity detail must not contain raw tool arguments JSON");
            }
        }
    }
}
