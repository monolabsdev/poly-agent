use std::collections::HashMap;
use std::sync::Arc;

use poly_agent_core::{
    AgentError, AgentEvent, AgentInput, AgentOutput, ChatMessage, FinishReason, RunId, ToolRisk,
};
use poly_agent_providers::{ChatRequest, ModelAdapter, ModelResponse, ProviderError};
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;

use crate::tool::{ToolContext, ToolRegistry};

const TOOL_USE_REMINDER: &str = "CRITICAL: When the user asks about the codebase, project, directory, program, app, repo, files, architecture, or how it works, you MUST inspect the workspace using list_files or read_file before answering. Do not provide a final answer without at least one tool call.";

/// Phrases that indicate the user is asking about the codebase/project.
const CODEBASE_INTENT_PHRASES: &[&str] = &[
    "what is this codebase",
    "what does this app do",
    "what does this program do",
    "how does this work",
    "explain this directory",
    "explain this repo",
    "summarise this project",
    "summarize this project",
    "what is this project",
    "what does this project do",
    "what is this repository",
    "what does this repository do",
];

/// Tools that actually mutate files on disk. Anything outside this set is
/// considered read-only and does not satisfy an edit-intent request.
const MUTATING_TOOLS: &[&str] = &["apply_patch", "write_file"];

/// Verbs that strongly indicate the user wants the agent to change a file.
const EDIT_VERBS: &[&str] = &[
    "change", "edit", "update", "modify", "replace", "fix", "create", "rename", "delete", "remove",
    "add", "set", "patch", "rewrite", "refactor", "remove", "insert", "append",
];

/// Default system prompt used by `AgentRuntime::run`.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful, careful coding agent.\n\
\n\
Tool selection rules:\n\
- For questions about the project, summaries, explanations, or other READ-ONLY tasks, use read_file, list_files, or search_files. Do not invent file contents.\n\
- When the user asks to change, edit, update, modify, replace, fix, create, rename, delete, or otherwise alter a file, you MUST use a file-mutation tool. Prefer `apply_patch` for small exact text replacements. Use `propose_edit` first when the change is broad or needs review. Never describe a change as done unless a mutating tool (apply_patch or write_file) actually succeeded.\n\
- If the user names a file path explicitly (e.g., README.md, package.json, src/main.rs), you may call read_file directly on it instead of calling list_files first.\n\
- If a tool requires approval, surface that to the user instead of pasting the whole rewritten file in your reply.\n";

fn contains_codebase_intent_phrase(text: &str) -> bool {
    let lower = text.to_lowercase();
    for phrase in CODEBASE_INTENT_PHRASES {
        if lower.contains(phrase) {
            return true;
        }
    }
    false
}

/// The agent runtime. Owns the tool registry and drives the agent loop.
pub struct AgentRuntime {
    tools: ToolRegistry,
    adapter: Arc<dyn ModelAdapter>,
    pending_approvals: Arc<Mutex<HashMap<(RunId, String), oneshot::Sender<bool>>>>,
}

/// Result of inspecting a user prompt for edit intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditIntent {
    /// True if the prompt contains an edit verb.
    pub is_edit: bool,
}

impl EditIntent {
    /// Inspect a user prompt for edit verbs. Match is case-insensitive and
    /// whole-word (alphabetic boundaries) so words like "updated" inside
    /// "updated_at" don't trigger.
    pub fn detect(prompt: &str) -> Self {
        let lower = prompt.to_lowercase();
        for verb in EDIT_VERBS {
            if contains_word(&lower, verb) {
                return Self { is_edit: true };
            }
        }
        Self { is_edit: false }
    }
}

/// True if `needle` appears in `haystack` surrounded by non-alphanumeric
/// boundaries (or string edges).
fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = haystack.as_bytes();
    let n = needle.as_bytes();
    let mut i = 0;
    while i + n.len() <= bytes.len() {
        if &bytes[i..i + n.len()] == n {
            let left_ok = i == 0 || !is_word_byte(bytes[i - 1]);
            let right_ok = i + n.len() == bytes.len() || !is_word_byte(bytes[i + n.len()]);
            if left_ok && right_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// True if `name` is in the curated list of tools that actually mutate files.
fn is_mutating_tool(name: &str) -> bool {
    MUTATING_TOOLS.contains(&name)
}

/// Phrases that imply the model is claiming it has already applied an edit.
const SUCCESS_CLAIMS: &[&str] = &[
    "i have updated",
    "i've updated",
    "i have changed",
    "i've changed",
    "i have edited",
    "i've edited",
    "i have modified",
    "i've modified",
    "i have replaced",
    "i've replaced",
    "i have fixed",
    "i've fixed",
    "i have created",
    "i've created",
    "i applied",
    "file has been updated",
    "file has been changed",
    "file has been edited",
    "file has been modified",
    "the file is now",
    "here is the updated",
    "here's the updated",
    "i rewrote",
    "i have rewritten",
    "i wrote it",
    "i have written",
    "done.",
    "done!",
];

/// Sanitise a final text response against the edit-intent guard.
///
/// - If the user asked for an edit and no mutating tool succeeded, replace
///   the response with a fixed warning so the model can't claim success.
/// - Otherwise return the text unchanged.
fn sanitise_final_text(
    text: String,
    intent: EditIntent,
    mutating_succeeded: bool,
    mutating_requested: bool,
) -> String {
    if !intent.is_edit || mutating_succeeded {
        return text;
    }

    if mutating_requested {
        // Model attempted an edit but the mutating call never succeeded
        // (e.g., pending approval or rejected). Warn explicitly.
        return format!(
            "{}\n\n[guard] An edit was requested but no mutating tool (apply_patch or write_file) succeeded. \
Approval may still be pending, or the tool was rejected. I have not modified the file.",
            text.trim()
        );
    }

    if claims_success(&text) {
        return "I inspected the file, but no edit was applied.".to_string();
    }

    text
}

fn claims_success(text: &str) -> bool {
    let lower = text.to_lowercase();
    SUCCESS_CLAIMS.iter().any(|p| lower.contains(p))
}

impl AgentRuntime {
    pub fn new(tools: ToolRegistry, adapter: Arc<dyn ModelAdapter>) -> Self {
        Self {
            tools,
            adapter,
            pending_approvals: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Approve a pending tool call.
    pub async fn approve_tool(&self, run_id: RunId, tool_call_id: &str) -> Result<(), AgentError> {
        let mut pending = self.pending_approvals.lock().await;
        if let Some(tx) = pending.remove(&(run_id, tool_call_id.to_string())) {
            let _ = tx.send(true);
            Ok(())
        } else {
            Err(AgentError::Other(format!(
                "No pending approval found for run_id {} and tool_call_id {}",
                run_id, tool_call_id
            )))
        }
    }

    /// Reject a pending tool call.
    pub async fn reject_tool(&self, run_id: RunId, tool_call_id: &str) -> Result<(), AgentError> {
        let mut pending = self.pending_approvals.lock().await;
        if let Some(tx) = pending.remove(&(run_id, tool_call_id.to_string())) {
            let _ = tx.send(false);
            Ok(())
        } else {
            Err(AgentError::Other(format!(
                "No pending approval found for run_id {} and tool_call_id {}",
                run_id, tool_call_id
            )))
        }
    }

    /// Run the agent loop. Returns `AgentOutput` and streams `AgentEvent`s
    /// through the provided sender.
    pub async fn run(
        &self,
        input: AgentInput,
        event_tx: mpsc::Sender<AgentEvent>,
    ) -> Result<AgentOutput, AgentError> {
        let run_id: RunId = Uuid::new_v4();
        let _ = event_tx.send(AgentEvent::Started { run_id }).await;

        let tool_ctx = ToolContext {
            workspace: input.workspace.clone(),
            limits: input.limits.clone(),
        };

        // Conversation history
        let edit_intent = EditIntent::detect(&input.prompt);
        let has_codebase_intent = contains_codebase_intent_phrase(&input.prompt);
        let mut messages: Vec<ChatMessage> = vec![
            ChatMessage {
                role: poly_agent_core::ChatRole::System,
                content: DEFAULT_SYSTEM_PROMPT.to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            ChatMessage::user(&input.prompt),
        ];

        // Inject tool-use reminder for codebase-intent prompts before first model call
        if has_codebase_intent {
            messages.push(ChatMessage::user(TOOL_USE_REMINDER));
        }

        let tool_specs = self.tools.tool_specs();

        // Track state for graceful handling
        let mut any_mutating_succeeded = false;
        let mut any_mutating_requested = false;
        let mut tool_call_count: usize = 0;
        let mut last_tool_result: Option<(String, String)> = None; // (tool_name, output)

        for step in 0..input.limits.max_steps {
            // Trim to max_context_messages to prevent unbounded growth.
            let window = Self::context_window(&messages, input.limits.max_context_messages);

            let request = ChatRequest {
                messages: window,
                tools: tool_specs.clone(),
            };

            let _ = event_tx
                .send(AgentEvent::ModelCallStarted { run_id, step })
                .await;

            let response = match self.adapter.chat(request).await {
                Ok(r) => r,
                Err(ProviderError::MalformedModelOutput) => {
                    // Retry once with corrective message
                    let _ = event_tx
                        .send(AgentEvent::ModelCallFinished { run_id, step })
                        .await;

                    messages.push(ChatMessage::user(
                        "Your previous response used invalid internal control tokens. Reply only with normal user-facing text or valid tool calls.",
                    ));

                    let retry_window = Self::context_window(&messages, input.limits.max_context_messages);
                    let retry_request = ChatRequest {
                        messages: retry_window,
                        tools: tool_specs.clone(),
                    };

                    let _ = event_tx
                        .send(AgentEvent::ModelCallStarted { run_id, step: step + 1000 })
                        .await;

                    match self.adapter.chat(retry_request).await {
                        Ok(r) => {
                            let _ = event_tx
                                .send(AgentEvent::ModelCallFinished { run_id, step: step + 1000 })
                                .await;
                            r
                        }
                        Err(e) => {
                            let _ = event_tx.send(AgentEvent::Error {
                                run_id,
                                error: format!("Model error: {}", e),
                            }).await;

                            // Graceful fallback if tools succeeded
                            if let Some((tool_name, output)) = last_tool_result {
                                let preview: String = output.chars().take(200).collect();
                                return Ok(AgentOutput {
                                    run_id,
                                    text: format!(
                                        "Model failed after workspace inspection.\n\nLast tool '{}' returned:\n{}",
                                        tool_name, preview
                                    ),
                                    finish_reason: FinishReason::PartialFailure {
                                        last_tool_result: output,
                                        tool_name,
                                    },
                                });
                            }
                            return Err(AgentError::Provider(e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    let _ = event_tx.send(AgentEvent::Error {
                        run_id,
                        error: format!("Model error: {}", e),
                    }).await;

                    // Graceful fallback if tools succeeded
                    if let Some((tool_name, output)) = last_tool_result {
                        let preview: String = output.chars().take(200).collect();
                        return Ok(AgentOutput {
                            run_id,
                            text: format!(
                                "Model failed after workspace inspection.\n\nLast tool '{}' returned:\n{}",
                                tool_name, preview
                            ),
                            finish_reason: FinishReason::PartialFailure {
                                last_tool_result: output,
                                tool_name,
                            },
                        });
                    }
                    return Err(AgentError::Provider(e.to_string()));
                }
            };

            let _ = event_tx
                .send(AgentEvent::ModelCallFinished { run_id, step })
                .await;

            match response {
                ModelResponse::Text(text) => {
                    // Guard for codebase-intent with zero tool calls
                    if has_codebase_intent && tool_call_count == 0 {
                        let _ = event_tx
                            .send(AgentEvent::Finished {
                                run_id,
                                text: "I need to inspect the workspace before answering that.".to_string(),
                            })
                            .await;
                        return Ok(AgentOutput {
                            run_id,
                            text: "I need to inspect the workspace before answering that.".to_string(),
                            finish_reason: FinishReason::Error(
                                "I need to inspect the workspace before answering that.".to_string(),
                            ),
                        });
                    }

                    let final_text = sanitise_final_text(
                        text,
                        edit_intent,
                        any_mutating_succeeded,
                        any_mutating_requested,
                    );

                    let _ = event_tx
                        .send(AgentEvent::Finished {
                            run_id,
                            text: final_text.clone(),
                        })
                        .await;

                    return Ok(AgentOutput {
                        run_id,
                        text: final_text,
                        finish_reason: FinishReason::Complete,
                    });
                }

                ModelResponse::ToolCalls(calls) => {
                    // Record assistant message with tool calls.
                    messages.push(ChatMessage::assistant_with_tool_calls(calls.clone()));

                    for call in calls {
                        let _ = event_tx
                            .send(AgentEvent::ToolCallRequested {
                                run_id,
                                call: call.clone(),
                            })
                            .await;

                        if is_mutating_tool(&call.name) {
                            any_mutating_requested = true;
                        }

                        // Look up the tool.
                        let tool = match self.tools.get(&call.name) {
                            Some(t) => t,
                            None => {
                                let err_result = poly_agent_core::ToolResult {
                                    tool_call_id: call.id.clone(),
                                    output: format!("Tool '{}' not found", call.name),
                                    is_error: true,
                                };
                                messages
                                    .push(ChatMessage::tool_result(&call.id, &err_result.output));
                                let _ = event_tx
                                    .send(AgentEvent::ToolCallFinished {
                                        run_id,
                                        result: err_result,
                                    })
                                    .await;
                                continue;
                            }
                        };

                        let risk = tool.risk();
                        let mut approved = true;

                        // For risky tools, pause and wait for approval.
                        if risk != ToolRisk::Safe {
                            let (tx, rx) = oneshot::channel();

                            {
                                let mut pending = self.pending_approvals.lock().await;
                                pending.insert((run_id, call.id.clone()), tx);
                            }

                            let _ = event_tx
                                .send(AgentEvent::ApprovalRequired {
                                    run_id,
                                    call: call.clone(),
                                })
                                .await;

                            // Pause execution until approved or rejected
                            approved = rx.await.unwrap_or(false);
                        }

                        if !approved {
                            let result = poly_agent_core::ToolResult {
                                tool_call_id: call.id.clone(),
                                output: "Tool execution denied by user.".to_string(),
                                is_error: true,
                            };
                            messages.push(ChatMessage::tool_result(&call.id, &result.output));
                            let _ = event_tx
                                .send(AgentEvent::ToolCallFinished { run_id, result })
                                .await;
                            continue; // Skip execution and continue with the loop (or next tool)
                        }

                        let _ = event_tx
                            .send(AgentEvent::ToolCallStarted {
                                run_id,
                                tool_call_id: call.id.clone(),
                                tool_name: call.name.clone(),
                            })
                            .await;

                        // Execute the tool.
                        let tool_output =
                            match tool.run(call.arguments.clone(), tool_ctx.clone()).await {
                                Ok(result) => result,
                                Err(e) => poly_agent_core::ToolResult {
                                    tool_call_id: call.id.clone(),
                                    output: format!("Tool error: {e}"),
                                    is_error: true,
                                },
                            };

                        if !tool_output.is_error && is_mutating_tool(&call.name) {
                            any_mutating_succeeded = true;
                        }

                        // Track tool usage for graceful fallback
                        tool_call_count += 1;
                        last_tool_result = Some((call.name.clone(), tool_output.output.clone()));

                        // Truncate output to max_tool_output_bytes.
                        let truncated = Self::truncate_output(
                            &tool_output.output,
                            input.limits.max_tool_output_bytes,
                        );

                        let final_result = poly_agent_core::ToolResult {
                            tool_call_id: call.id.clone(),
                            output: truncated,
                            is_error: tool_output.is_error,
                        };

                        // Append tool result as a message.
                        messages.push(ChatMessage::tool_result(&call.id, &final_result.output));

                        let _ = event_tx
                            .send(AgentEvent::ToolCallFinished {
                                run_id,
                                result: final_result,
                            })
                            .await;
                    }
                }
            }
        }

        // Hit step limit.
        let _ = event_tx
            .send(AgentEvent::StepLimitReached {
                run_id,
                max_steps: input.limits.max_steps,
            })
            .await;

        Ok(AgentOutput {
            run_id,
            text: String::new(),
            finish_reason: FinishReason::StepLimitReached,
        })
    }

    /// Returns the most recent `max` messages from the conversation.
    fn context_window(messages: &[ChatMessage], max: usize) -> Vec<ChatMessage> {
        if messages.len() <= max {
            messages.to_vec()
        } else {
            messages[messages.len() - max..].to_vec()
        }
    }

    /// Truncate tool output to fit within byte limit.
    fn truncate_output(output: &str, max_bytes: usize) -> String {
        if output.len() <= max_bytes {
            return output.to_string();
        }
        // Find a char boundary at or before max_bytes.
        let mut end = max_bytes;
        while end > 0 && !output.is_char_boundary(end) {
            end -= 1;
        }
        let mut truncated = output[..end].to_string();
        truncated.push_str("\n... [output truncated]");
        truncated
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::AgentTool;
    use poly_agent_core::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// A mock adapter that always returns tool calls, used to test max_steps.
    struct LoopingAdapter;

    #[async_trait::async_trait]
    impl ModelAdapter for LoopingAdapter {
        async fn chat(
            &self,
            _request: ChatRequest,
        ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
            Ok(ModelResponse::ToolCalls(vec![ToolCall {
                id: "call_1".to_string(),
                name: "test_tool".to_string(),
                arguments: serde_json::json!({}),
            }]))
        }
    }

    /// A mock adapter that returns text.
    struct TextAdapter(String);

    #[async_trait::async_trait]
    impl ModelAdapter for TextAdapter {
        async fn chat(
            &self,
            _request: ChatRequest,
        ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
            Ok(ModelResponse::Text(self.0.clone()))
        }
    }

    /// A simple safe test tool.
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
        async fn run(
            &self,
            _args: serde_json::Value,
            _ctx: crate::tool::ToolContext,
        ) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                tool_call_id: String::new(),
                output: "test output".to_string(),
                is_error: false,
            })
        }
    }

    /// A tool that requires approval.
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
        async fn run(
            &self,
            _args: serde_json::Value,
            _ctx: crate::tool::ToolContext,
        ) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                tool_call_id: String::new(),
                output: "executed dangerous tool".to_string(),
                is_error: false,
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
        }
    }

    #[tokio::test]
    async fn runtime_stops_at_max_steps() {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(TestTool));

        let runtime = AgentRuntime::new(tools, Arc::new(LoopingAdapter));
        let (tx, mut rx) = mpsc::channel(64);

        let result = runtime.run(test_input(3), tx).await.unwrap();

        assert!(matches!(
            result.finish_reason,
            FinishReason::StepLimitReached
        ));

        // Drain events and check we got StepLimitReached.
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
        // No tools registered.
        let tools = ToolRegistry::new();

        // Adapter returns a call to "nonexistent_tool", then on second call returns text.
        struct UnknownThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ModelAdapter for UnknownThenTextAdapter {
            async fn chat(
                &self,
                _request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 0 {
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

        let adapter = Arc::new(UnknownThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });

        let runtime = AgentRuntime::new(tools, adapter);
        let (tx, _rx) = mpsc::channel(64);

        let result = runtime.run(test_input(5), tx).await.unwrap();
        // Should complete — unknown tool produces an error result that gets sent back to the model.
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

        struct DangerousThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize,
        }
        #[async_trait::async_trait]
        impl ModelAdapter for DangerousThenTextAdapter {
            async fn chat(
                &self,
                _request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 0 {
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

        let adapter = Arc::new(DangerousThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });

        let runtime = Arc::new(AgentRuntime::new(tools, adapter));
        let (tx, mut rx) = mpsc::channel(64);

        let runtime_clone = runtime.clone();
        let handle = tokio::spawn(async move { runtime_clone.run(test_input(5), tx).await });

        // Wait for ApprovalRequired event
        let mut run_id = None;
        while let Some(event) = rx.recv().await {
            if let AgentEvent::ApprovalRequired { run_id: id, call } = event {
                assert_eq!(call.name, "dangerous_tool");
                run_id = Some(id);
                break;
            }
        }

        let run_id = run_id.unwrap();

        // Approve it
        runtime.approve_tool(run_id, "call_1").await.unwrap();

        // Wait for completion
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
            async fn chat(
                &self,
                request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 0 {
                    Ok(ModelResponse::ToolCalls(vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "dangerous_tool".to_string(),
                        arguments: serde_json::json!({}),
                    }]))
                } else {
                    // Make sure the last message from the tool result is "denied"
                    let last_msg = request.messages.last().unwrap();
                    assert_eq!(last_msg.role, ChatRole::Tool);
                    assert_eq!(last_msg.content, "Tool execution denied by user.");
                    Ok(ModelResponse::Text("Cancelled".to_string()))
                }
            }
        }

        let adapter = Arc::new(DangerousThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });

        let runtime = Arc::new(AgentRuntime::new(tools, adapter));
        let (tx, mut rx) = mpsc::channel(64);

        let runtime_clone = runtime.clone();
        let handle = tokio::spawn(async move { runtime_clone.run(test_input(5), tx).await });

        let mut run_id = None;
        while let Some(event) = rx.recv().await {
            if let AgentEvent::ApprovalRequired {
                run_id: id,
                call: _,
            } = event
            {
                run_id = Some(id);
                break;
            }
        }

        // Reject it
        runtime
            .reject_tool(run_id.unwrap(), "call_1")
            .await
            .unwrap();

        let result = handle.await.unwrap().unwrap();
        assert!(matches!(result.finish_reason, FinishReason::Complete));
    }

    #[tokio::test]
    async fn runtime_handles_unknown_approval_id() {
        let tools = ToolRegistry::new();
        let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Hello".to_string())));

        let err = runtime.approve_tool(Uuid::new_v4(), "unknown_id").await;
        assert!(err.is_err());
        assert!(err
            .unwrap_err()
            .to_string()
            .contains("No pending approval found"));
    }

    #[test]
    fn truncate_output_within_limit() {
        let output = "hello world";
        let result = AgentRuntime::truncate_output(output, 100);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn truncate_output_over_limit() {
        let output = "hello world, this is a long string";
        let result = AgentRuntime::truncate_output(output, 11);
        assert!(result.starts_with("hello world"));
        assert!(result.contains("[output truncated]"));
    }

    // --- Edit-intent guard tests ---

    /// A fake `apply_patch`-shaped tool that requires approval.
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
        async fn run(
            &self,
            _args: serde_json::Value,
            _ctx: crate::tool::ToolContext,
        ) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                tool_call_id: String::new(),
                output: "patched".to_string(),
                is_error: false,
            })
        }
    }

    #[test]
    fn edit_intent_detects_verbs() {
        assert!(EditIntent::detect("Change README.md title from poly-agent to Poly Agent").is_edit);
        assert!(EditIntent::detect("Please update the version field").is_edit);
        assert!(EditIntent::detect("Fix the typo in foo.rs").is_edit);
        assert!(EditIntent::detect("Add a new endpoint").is_edit);
        assert!(EditIntent::detect("Remove the unused import").is_edit);
        assert!(EditIntent::detect("CREATE a new file").is_edit);
    }

    #[test]
    fn edit_intent_negative_for_read_only() {
        assert!(!EditIntent::detect("What does this project do?").is_edit);
        assert!(!EditIntent::detect("Summarise README.md").is_edit);
        assert!(!EditIntent::detect("List the files in this project").is_edit);
        assert!(!EditIntent::detect("Explain how the runtime works").is_edit);
    }

    #[test]
    fn edit_intent_word_boundary() {
        // "remove" inside "removeit" should not trigger (word boundary check).
        assert!(!EditIntent::detect("describe the removeit helper").is_edit);
        // but a real edit request does
        assert!(EditIntent::detect("rename the updated_at field").is_edit);
    }

    #[test]
    fn sanitise_keeps_text_when_no_edit_intent() {
        let s = sanitise_final_text(
            "Here is the updated file".to_string(),
            EditIntent::detect("summarise readme"),
            false,
            false,
        );
        assert_eq!(s, "Here is the updated file");
    }

    #[test]
    fn sanitise_keeps_text_when_mutating_succeeded() {
        let s = sanitise_final_text(
            "I have updated the file.".to_string(),
            EditIntent::detect("change the title"),
            true,
            true,
        );
        assert_eq!(s, "I have updated the file.");
    }

    #[test]
    fn sanitise_replaces_claim_when_no_mutating_tool_ran() {
        let s = sanitise_final_text(
            "I have updated README.md to use Poly Agent as the title.".to_string(),
            EditIntent::detect("Change README.md title from poly-agent to Poly Agent"),
            false,
            false,
        );
        assert_eq!(s, "I inspected the file, but no edit was applied.");
    }

    #[test]
    fn sanitise_warns_when_mutating_requested_but_failed() {
        let s = sanitise_final_text(
            "I've updated the title.".to_string(),
            EditIntent::detect("update the title"),
            false,
            true,
        );
        assert!(s.contains("[guard]"));
        assert!(s.contains("no mutating tool"));
    }

    /// Simulates the exact prompt from the bug report: model requests
    /// `apply_patch`, which must produce an `ApprovalRequired` event.
    #[tokio::test]
    async fn readme_title_change_prompts_approval_for_apply_patch() {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(FakeApplyPatch));

        struct ApplyPatchThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize,
        }
        #[async_trait::async_trait]
        impl ModelAdapter for ApplyPatchThenTextAdapter {
            async fn chat(
                &self,
                request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 0 {
                    // Confirm the system prompt mentions apply_patch.
                    let sys = request.messages.first().unwrap();
                    assert_eq!(sys.role, ChatRole::System);
                    assert!(sys.content.contains("apply_patch"));
                    Ok(ModelResponse::ToolCalls(vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "apply_patch".to_string(),
                        arguments: serde_json::json!({
                            "path": "README.md",
                            "expected_old_text": "# poly-agent",
                            "replacement_text": "# Poly Agent",
                            "reason": "rename project",
                        }),
                    }]))
                } else {
                    Ok(ModelResponse::Text("I have updated the title.".to_string()))
                }
            }
        }

        let adapter = Arc::new(ApplyPatchThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });
        let runtime = AgentRuntime::new(tools, adapter);
        let mut input = test_input(5);
        input.prompt = "Change README.md title from poly-agent to Poly Agent".to_string();
        let (tx, mut rx) = mpsc::channel(64);

        let handle = tokio::spawn(async move { runtime.run(input, tx).await });

        // We expect an ApprovalRequired for apply_patch.
        let mut approval_seen = false;
        while let Some(event) = rx.recv().await {
            if let AgentEvent::ApprovalRequired { call, .. } = event {
                assert_eq!(call.name, "apply_patch");
                approval_seen = true;
                break;
            }
        }
        assert!(approval_seen, "expected ApprovalRequired for apply_patch");

        // Abort: we don't approve, the run is still in flight on approval.
        handle.abort();
    }

    /// If the model never invokes a mutating tool on an edit-intent prompt,
    /// the final response must be sanitised.
    #[tokio::test]
    async fn read_only_summary_still_works() {
        let tools = ToolRegistry::new();
        let runtime = AgentRuntime::new(
            tools,
            Arc::new(TextAdapter("It's a Rust agent runtime.".to_string())),
        );
        let mut input = test_input(5);
        input.prompt = "Summarise what this project does".to_string();
        let (tx, _rx) = mpsc::channel(64);
        let result = runtime.run(input, tx).await.unwrap();
        assert_eq!(result.text, "It's a Rust agent runtime.");
    }

    /// When the user asks for an edit and the model only reads the file then
    /// emits a "done"-style final message, the guard must overwrite it.
    #[tokio::test]
    async fn guard_overwrites_false_success_claim() {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(TestTool));

        struct ReadThenFalseClaimAdapter;
        #[async_trait::async_trait]
        impl ModelAdapter for ReadThenFalseClaimAdapter {
            async fn chat(
                &self,
                _request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                Ok(ModelResponse::Text(
                    "I have updated the file. Done.".to_string(),
                ))
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

    // --- Codebase-intent tests ---

    #[test]
    fn codebase_intent_detection() {
        assert!(contains_codebase_intent_phrase("What is this codebase?"));
        assert!(contains_codebase_intent_phrase("What does this app do?"));
        assert!(contains_codebase_intent_phrase("What does this program do?"));
        assert!(contains_codebase_intent_phrase("How does this work?"));
        assert!(contains_codebase_intent_phrase("Explain this directory"));
        assert!(contains_codebase_intent_phrase("Explain this repo"));
        assert!(contains_codebase_intent_phrase("Summarise this project"));
        assert!(!contains_codebase_intent_phrase("What is the weather?"));
        assert!(!contains_codebase_intent_phrase("Tell me a joke"));
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
            async fn chat(
                &self,
                request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 0 {
                    // First call should have the tool-use reminder in messages
                    let has_reminder = request
                        .messages
                        .iter()
                        .any(|m| m.content.contains("MUST inspect the workspace"));
                    assert!(has_reminder, "Expected tool-use reminder for codebase-intent");
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

        let adapter = Arc::new(TestToolAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });

        let runtime = AgentRuntime::new(tools, adapter);
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
            async fn chat(
                &self,
                _request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
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

    // --- Malformed output retry tests ---

    #[tokio::test]
    async fn malformed_output_triggers_retry() {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(TestTool));

        struct MalformedThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ModelAdapter for MalformedThenTextAdapter {
            async fn chat(
                &self,
                request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 0 {
                    // First call returns malformed output
                    Ok(ModelResponse::Text(
                        "<|start|>assistant<|channel|>analysis\n".to_string(),
                    ))
                } else {
                    // Check retry message is present
                    let has_retry_msg = request
                        .messages
                        .iter()
                        .any(|m| m.content.contains("invalid internal control tokens"));
                    assert!(has_retry_msg, "Expected retry corrective message");
                    Ok(ModelResponse::Text("Corrected response.".to_string()))
                }
            }
        }

        let adapter = Arc::new(MalformedThenTextAdapter {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });

        let runtime = AgentRuntime::new(tools, adapter);
        let (tx, _rx) = mpsc::channel(64);
        let result = runtime.run(test_input(5), tx).await.unwrap();
        assert_eq!(result.text, "Corrected response.");
    }

    // --- Graceful fallback after tool success tests ---

    #[tokio::test]
    async fn graceful_fallback_after_tool_success() {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(TestTool));

        struct ToolThenErrorAdapter;
        #[async_trait::async_trait]
        impl ModelAdapter for ToolThenErrorAdapter {
            async fn chat(
                &self,
                _request: ChatRequest,
            ) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                static CALLED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let count = CALLED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if count == 0 {
                    Ok(ModelResponse::ToolCalls(vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "test_tool".to_string(),
                        arguments: serde_json::json!({}),
                    }]))
                } else {
                    Err(poly_agent_providers::ProviderError::Api {
                        status: 500,
                        body: "Internal error".to_string(),
                    })
                }
            }
        }

        let adapter = Arc::new(ToolThenErrorAdapter);
        let runtime = AgentRuntime::new(tools, adapter);
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
}
