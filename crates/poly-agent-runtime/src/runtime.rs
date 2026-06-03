use std::collections::HashMap;
use std::sync::Arc;

use poly_agent_core::{
    AgentError, AgentEvent, AgentInput, AgentOutput, ChatMessage, FinishReason, RunId, ToolRisk,
};
use poly_agent_providers::{ChatRequest, ModelAdapter, ModelResponse};
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;

use crate::tool::{ToolContext, ToolRegistry};

/// The agent runtime. Owns the tool registry and drives the agent loop.
pub struct AgentRuntime {
    tools: ToolRegistry,
    adapter: Arc<dyn ModelAdapter>,
    pending_approvals: Arc<Mutex<HashMap<(RunId, String), oneshot::Sender<bool>>>>,
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
        let system_prompt = "You are a helpful agent. If the user explicitly names a file path (e.g., README.md, package.json, src/main.rs), prefer calling read_file directly on it instead of calling list_files first.";
        let mut messages: Vec<ChatMessage> = vec![
            ChatMessage {
                role: poly_agent_core::ChatRole::System,
                content: system_prompt.to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            ChatMessage::user(&input.prompt),
        ];

        let tool_specs = self.tools.tool_specs();

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

            let response = self
                .adapter
                .chat(request)
                .await
                .map_err(|e| AgentError::Provider(e.to_string()))?;

            let _ = event_tx
                .send(AgentEvent::ModelCallFinished { run_id, step })
                .await;

            match response {
                ModelResponse::Text(text) => {
                    let _ = event_tx
                        .send(AgentEvent::Finished {
                            run_id,
                            text: text.clone(),
                        })
                        .await;

                    return Ok(AgentOutput {
                        run_id,
                        text,
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

                        // Look up the tool.
                        let tool = match self.tools.get(&call.name) {
                            Some(t) => t,
                            None => {
                                let err_result = poly_agent_core::ToolResult {
                                    tool_call_id: call.id.clone(),
                                    output: format!("Tool '{}' not found", call.name),
                                    is_error: true,
                                };
                                messages.push(ChatMessage::tool_result(
                                    &call.id,
                                    &err_result.output,
                                ));
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
                            messages.push(ChatMessage::tool_result(
                                &call.id,
                                &result.output,
                            ));
                            let _ = event_tx
                                .send(AgentEvent::ToolCallFinished {
                                    run_id,
                                    result,
                                })
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
                        let tool_output = match tool.run(call.arguments.clone(), tool_ctx.clone()).await {
                            Ok(result) => result,
                            Err(e) => poly_agent_core::ToolResult {
                                tool_call_id: call.id.clone(),
                                output: format!("Tool error: {e}"),
                                is_error: true,
                            },
                        };

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
                        messages.push(ChatMessage::tool_result(
                            &call.id,
                            &final_result.output,
                        ));

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
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
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
        async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
            Ok(ModelResponse::Text(self.0.clone()))
        }
    }

    /// A simple safe test tool.
    struct TestTool;

    #[async_trait::async_trait]
    impl AgentTool for TestTool {
        fn name(&self) -> &'static str { "test_tool" }
        fn description(&self) -> &'static str { "A test tool" }
        fn risk(&self) -> ToolRisk { ToolRisk::Safe }
        fn parameters_schema(&self) -> serde_json::Value { serde_json::json!({}) }
        async fn run(&self, _args: serde_json::Value, _ctx: crate::tool::ToolContext) -> anyhow::Result<ToolResult> {
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
        fn name(&self) -> &'static str { "dangerous_tool" }
        fn description(&self) -> &'static str { "A dangerous tool" }
        fn risk(&self) -> ToolRisk { ToolRisk::RequiresApproval }
        fn parameters_schema(&self) -> serde_json::Value { serde_json::json!({}) }
        async fn run(&self, _args: serde_json::Value, _ctx: crate::tool::ToolContext) -> anyhow::Result<ToolResult> {
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

        assert!(matches!(result.finish_reason, FinishReason::StepLimitReached));

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
            async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 0 {
                    Ok(ModelResponse::ToolCalls(vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "nonexistent_tool".to_string(),
                        arguments: serde_json::json!({}),
                    }]))
                } else {
                    Ok(ModelResponse::Text("I see the tool was not found".to_string()))
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
            async fn chat(&self, _request: ChatRequest) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
        let handle = tokio::spawn(async move {
            runtime_clone.run(test_input(5), tx).await
        });

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
            async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, poly_agent_providers::ProviderError> {
                let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
        let handle = tokio::spawn(async move {
            runtime_clone.run(test_input(5), tx).await
        });

        let mut run_id = None;
        while let Some(event) = rx.recv().await {
            if let AgentEvent::ApprovalRequired { run_id: id, call: _ } = event {
                run_id = Some(id);
                break;
            }
        }

        // Reject it
        runtime.reject_tool(run_id.unwrap(), "call_1").await.unwrap();

        let result = handle.await.unwrap().unwrap();
        assert!(matches!(result.finish_reason, FinishReason::Complete));
    }

    #[tokio::test]
    async fn runtime_handles_unknown_approval_id() {
        let tools = ToolRegistry::new();
        let runtime = AgentRuntime::new(tools, Arc::new(TextAdapter("Hello".to_string())));

        let err = runtime.approve_tool(Uuid::new_v4(), "unknown_id").await;
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("No pending approval found"));
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
}
