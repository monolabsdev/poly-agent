use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use poly_agent_core::{
    ChatRole, ModelConfig, ModelProvider, PermissionPreset, RuntimeLimits, ToolCall, ToolResult,
    ToolRisk,
};
use poly_agent_providers::{ChatRequest, ModelAdapter, ModelResponse, ProviderError};
use poly_agent_runtime::{AgentRuntime, AgentRuntimeConfig, AgentTool, ToolContext, ToolRegistry};
use tokio::time::sleep;
use tokio::time::timeout;

use crate::events::AgentUiEventPayload;
use crate::manager::AgentRunManager;
use crate::types::{AgentRunInput, RunStatus, MAX_STORED_EVENTS};
use crate::workspace::PreparedRun;

struct ScriptedAdapter {
    calls: std::sync::atomic::AtomicUsize,
    tool_name: &'static str,
    tool_args: serde_json::Value,
}

impl ScriptedAdapter {
    fn new(tool_name: &'static str, tool_args: serde_json::Value) -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
            tool_name,
            tool_args,
        }
    }
}

#[async_trait::async_trait]
impl ModelAdapter for ScriptedAdapter {
    async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            Ok(ModelResponse::ToolCalls(vec![ToolCall {
                id: "approval_1".to_string(),
                name: self.tool_name.to_string(),
                arguments: self.tool_args.clone(),
            }]))
        } else {
            assert_eq!(request.messages.last().unwrap().role, ChatRole::Tool);
            Ok(ModelResponse::Text("finished".to_string()))
        }
    }
}

struct SlowAdapter;

#[async_trait::async_trait]
impl ModelAdapter for SlowAdapter {
    async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        sleep(Duration::from_secs(30)).await;
        Ok(ModelResponse::Text("too late".to_string()))
    }
}

struct TextAdapter;

#[async_trait::async_trait]
impl ModelAdapter for TextAdapter {
    async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        Ok(ModelResponse::Text("ok".to_string()))
    }
}

struct FakeApplyPatch;

#[async_trait::async_trait]
impl AgentTool for FakeApplyPatch {
    fn name(&self) -> &'static str {
        "apply_patch"
    }
    fn description(&self) -> &'static str {
        "fake patch"
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
        serde_json::json!({"type":"object"})
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

fn input(workspace_path: Option<PathBuf>) -> AgentRunInput {
    AgentRunInput {
        prompt: "test".to_string(),
        model: ModelConfig {
            provider: ModelProvider::Ollama,
            model: "test".to_string(),
            base_url: None,
            api_key: None,
        },
        workspace_path,
        workspace_selection: None,
        limits: RuntimeLimits {
            max_steps: 3,
            ..RuntimeLimits::default()
        },
        permission_preset: Default::default(),
        resolved_context: None,
        debug: true,
    }
}

fn manager_with_adapter(adapter: Arc<dyn ModelAdapter>) -> AgentRunManager {
    AgentRunManager::with_runtime_factory(
        None,
        Arc::new(move |prepared: PreparedRun| {
            let mut tools = ToolRegistry::new();
            tools.register(Arc::new(FakeApplyPatch));
            tools.register(Arc::new(FakeRunCommand));
            Ok(AgentRuntime::with_config(
                tools,
                adapter.clone(),
                AgentRuntimeConfig {
                    permission_preset: prepared.input.permission_preset,
                    reviewer: None,
                    agent_config: None,
                },
            ))
        }),
    )
}

async fn wait_for_status(manager: &AgentRunManager, run_id: uuid::Uuid, status: RunStatus) {
    for _ in 0..100 {
        if manager.get_run_state(run_id).await.unwrap().status == status {
            return;
        }
        sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {status:?}");
}

#[tokio::test]
async fn run_state_transitions_running_waiting_finished() {
    let manager = manager_with_adapter(Arc::new(ScriptedAdapter::new(
        "run_command",
        serde_json::json!({"command": "echo hi", "reason": "test"}),
    )));
    let run_id = manager
        .start_run(input(Some(std::env::current_dir().unwrap())))
        .await
        .unwrap();

    assert_eq!(
        manager.get_run_state(run_id).await.unwrap().status,
        RunStatus::Running
    );
    wait_for_status(&manager, run_id, RunStatus::WaitingForApproval).await;

    manager
        .approve_tool_call(run_id, "approval_1")
        .await
        .unwrap();
    wait_for_status(&manager, run_id, RunStatus::Finished).await;
    assert_eq!(
        manager
            .get_run_state(run_id)
            .await
            .unwrap()
            .final_output
            .as_deref(),
        Some("finished")
    );
}

#[tokio::test]
async fn approval_resolution_is_idempotent_after_first_click() {
    let manager = manager_with_adapter(Arc::new(ScriptedAdapter::new(
        "run_command",
        serde_json::json!({"command": "echo hi", "reason": "test"}),
    )));
    let run_id = manager
        .start_run(input(Some(std::env::current_dir().unwrap())))
        .await
        .unwrap();

    wait_for_status(&manager, run_id, RunStatus::WaitingForApproval).await;
    manager
        .approve_tool_call(run_id, "approval_1")
        .await
        .unwrap();
    manager
        .approve_tool_call(run_id, "approval_1")
        .await
        .unwrap();
}

#[tokio::test]
async fn cancel_run_changes_status_and_emits_cancelled() {
    let manager = manager_with_adapter(Arc::new(SlowAdapter));
    let run_id = manager
        .start_run(input(Some(std::env::current_dir().unwrap())))
        .await
        .unwrap();

    manager.cancel_run(run_id).await.unwrap();
    let state = manager.get_run_state(run_id).await.unwrap();

    assert_eq!(state.status, RunStatus::Cancelled);
    assert!(state
        .events
        .iter()
        .any(|event| event.event_type == "cancelled"));
}

#[tokio::test]
async fn get_run_state_returns_events_and_stats() {
    let manager = manager_with_adapter(Arc::new(TextAdapter));
    let run_id = manager
        .start_run(input(Some(std::env::current_dir().unwrap())))
        .await
        .unwrap();

    wait_for_status(&manager, run_id, RunStatus::Finished).await;
    let state = manager.get_run_state(run_id).await.unwrap();

    assert!(!state.events.is_empty());
    assert!(state.stats.events_seen >= state.events.len());
    assert!(state.stats.model_calls >= 1);
}

#[tokio::test]
async fn event_buffer_is_bounded() {
    let mut state = crate::types::MutableRunState::new(uuid::Uuid::new_v4(), None, false);
    let run_id = state.run_id;
    for index in 0..(MAX_STORED_EVENTS + 10) {
        crate::events::push_event(
            &mut state,
            crate::events::AgentUiEvent {
                run_id,
                event_type: format!("event_{index}"),
                timestamp: std::time::SystemTime::now(),
                data: AgentUiEventPayload::Thinking,
            },
        );
    }

    assert_eq!(state.events.len(), MAX_STORED_EVENTS);
    assert_eq!(state.events.front().unwrap().event_type, "event_10");
}

#[tokio::test]
async fn approval_payload_contains_run_command_ui_data() {
    let manager = manager_with_adapter(Arc::new(ScriptedAdapter::new(
        "run_command",
        serde_json::json!({
            "command": "echo hello",
            "reason": "smoke test"
        }),
    )));
    let run_id = manager
        .start_run(input(Some(std::env::current_dir().unwrap())))
        .await
        .unwrap();

    wait_for_status(&manager, run_id, RunStatus::WaitingForApproval).await;
    let approval = manager
        .get_run_state(run_id)
        .await
        .unwrap()
        .pending_approval
        .unwrap();

    assert_eq!(approval.tool_name, "run_command");
    assert_eq!(approval.reason.as_deref(), Some("smoke test"));
    assert_eq!(approval.command_preview.as_deref(), Some("echo hello"));
    assert!(approval.raw_arguments.is_some());
}

#[tokio::test]
async fn default_preset_auto_approves_apply_patch_without_prompt() {
    let manager = manager_with_adapter(Arc::new(ScriptedAdapter::new(
        "apply_patch",
        serde_json::json!({
            "path": "README.md",
            "expected_old_text": "old",
            "replacement_text": "new",
            "reason": "test edit"
        }),
    )));
    let run_id = manager
        .start_run(input(Some(std::env::current_dir().unwrap())))
        .await
        .unwrap();

    wait_for_status(&manager, run_id, RunStatus::Finished).await;
    let state = manager.get_run_state(run_id).await.unwrap();

    assert!(state
        .events
        .iter()
        .any(|event| event.event_type == "tool_auto_approved"));
    assert!(state.pending_approval.is_none());
}

#[tokio::test]
async fn full_access_preset_auto_approves_run_command_without_prompt() {
    let manager = manager_with_adapter(Arc::new(ScriptedAdapter::new(
        "run_command",
        serde_json::json!({"command": "ls", "reason": "list"}),
    )));
    let mut run_input = input(Some(std::env::current_dir().unwrap()));
    run_input.permission_preset = PermissionPreset::FullAccess;
    let run_id = manager.start_run(run_input).await.unwrap();

    wait_for_status(&manager, run_id, RunStatus::Finished).await;
    let state = manager.get_run_state(run_id).await.unwrap();

    assert!(state
        .events
        .iter()
        .any(|event| event.event_type == "tool_auto_approved"));
    assert!(state.pending_approval.is_none());
}

#[tokio::test]
async fn workspace_path_is_canonicalized() {
    let manager = manager_with_adapter(Arc::new(TextAdapter));
    let run_id = manager
        .start_run(input(Some(PathBuf::from("."))))
        .await
        .unwrap();
    let state = manager.get_run_state(run_id).await.unwrap();

    assert_eq!(
        state.workspace_root,
        Some(std::env::current_dir().unwrap().canonicalize().unwrap())
    );
    assert!(state.local_tools_enabled);
}

#[tokio::test]
async fn missing_workspace_disables_local_tools() {
    let manager = manager_with_adapter(Arc::new(TextAdapter));
    let run_id = manager.start_run(input(None)).await.unwrap();
    let state = manager.get_run_state(run_id).await.unwrap();

    assert_eq!(state.workspace_root, None);
    assert!(!state.local_tools_enabled);
}

#[tokio::test]
async fn invalid_workspace_returns_clear_error() {
    let manager = manager_with_adapter(Arc::new(TextAdapter));
    let err = manager
        .start_run(input(Some(PathBuf::from("definitely-missing-workspace"))))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("workspace path does not exist"));
}

#[tokio::test]
async fn manager_state_lock_is_not_held_across_model_await() {
    let manager = manager_with_adapter(Arc::new(SlowAdapter));
    let run_id = manager
        .start_run(input(Some(std::env::current_dir().unwrap())))
        .await
        .unwrap();

    let state = timeout(Duration::from_millis(100), manager.get_run_state(run_id))
        .await
        .expect("state lock should stay available while model awaits")
        .unwrap();

    assert_eq!(state.status, RunStatus::Running);
    manager.cancel_run(run_id).await.unwrap();
}
