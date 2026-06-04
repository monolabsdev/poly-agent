use poly_agent_core::{
    AgentError, AgentEvent, AgentInput, AgentOutput, ChatMessage, ChatRole, FinishReason, ToolResult,
    ToolRisk,
};
use poly_agent_providers::{ChatRequest, ModelResponse, ProviderError};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::engine::AgentRuntime;
use crate::intent::{
    build_grounding_prefix, contains_codebase_intent_phrase, is_mutating_tool,
    sanitise_final_text, tool_is_inspection, EditIntent,
};
use crate::tool::ToolContext;

const TOOL_USE_REMINDER: &str = "\
CRITICAL: When the user asks about the codebase, project, directory, program, app, repo, files, architecture, or how it works, you MUST inspect the workspace before answering.

Strategy for codebase summaries:
1. Call inspect_project first to get the high-level project structure.
2. Call read_important_files to read README, metadata, key entry points.
3. After 2-3 calls total, summarize using gathered evidence.
Do not inspect every module unless specifically asked. Gather evidence then answer.";

const VALID_TOOL_NAMES: &[&str] = &[
    "list_files", "read_file", "search_files", "propose_edit", "apply_patch", "write_file", "run_command",
    "inspect_project", "read_important_files",
];

fn build_system_prompt(max_steps: usize) -> String {
    format!(
        "You are a helpful, careful coding agent.\n\
\n\
Available tools: {}\n\
Never invent tool names. Only use tools from this list.\n\
\n\
Tool selection rules:\n\
- For questions about the project, summaries, explanations, or other READ-ONLY tasks, use inspect_project, read_important_files, read_file, list_files, or search_files. Do not invent file contents.\n\
- When the user asks to change, edit, update, modify, replace, fix, create, rename, delete, or otherwise alter a file, you MUST use a file-mutation tool. Prefer `apply_patch` for small exact text replacements. Use `propose_edit` first when the change is broad or needs review. Never describe a change as done unless a mutating tool (apply_patch or write_file) actually succeeded.\n\
- If the user names a file path explicitly (e.g., README.md, package.json, src/main.rs), you may call read_file directly on it instead of calling list_files first.\n\
- If a tool requires approval, surface that to the user instead of pasting the whole rewritten file in your reply.\n\
\n\
Tool budget: You have up to {} model calls. Use tools efficiently. \
Stop once you have enough information to produce the final answer.",
        VALID_TOOL_NAMES.join(", "),
        max_steps,
    )
}

struct RunContext {
    run_id: Uuid,
    event_tx: mpsc::Sender<AgentEvent>,
    tool_ctx: ToolContext,
    tool_specs: Vec<poly_agent_providers::ToolSpec>,
    edit_intent: EditIntent,
    has_codebase_intent: bool,
    max_steps: usize,
    max_context_messages: usize,
    max_tool_output_bytes: usize,
    tool_cache: Arc<Mutex<HashMap<(String, String), String>>>,
    empty_dirs: Arc<Mutex<HashSet<String>>>,
}

enum StepResult {
    Finished(AgentOutput),
    Continue,
}

impl AgentRuntime {
    pub async fn run(
        &self,
        input: AgentInput,
        event_tx: mpsc::Sender<AgentEvent>,
    ) -> Result<AgentOutput, AgentError> {
        let run_id = Uuid::new_v4();
        let _ = event_tx.send(AgentEvent::Started { run_id }).await;

        let ctx = RunContext {
            run_id,
            event_tx,
            tool_ctx: ToolContext {
                workspace: input.workspace,
                limits: input.limits.clone(),
            },
            tool_specs: self.tools.tool_specs(),
            edit_intent: EditIntent::detect(&input.prompt),
            has_codebase_intent: contains_codebase_intent_phrase(&input.prompt),
            max_steps: input.limits.max_steps,
            max_context_messages: input.limits.max_context_messages,
            max_tool_output_bytes: input.limits.max_tool_output_bytes,
            tool_cache: Arc::new(Mutex::new(HashMap::new())),
            empty_dirs: Arc::new(Mutex::new(HashSet::new())),
        };

        let mut state = RunState {
            messages: Self::build_initial_messages(&input.prompt, &ctx),
            any_mutating_succeeded: false,
            any_mutating_requested: false,
            tool_call_count: 0,
            last_tool_result: None,
            inspection_tools_used: false,
            unknown_tool_retried: None,
            repeated_unknown_tool: false,
        };

        for step in 0..ctx.max_steps {
            let result = self.run_step(step, &ctx, &mut state).await?;
            match result {
                StepResult::Finished(output) => return Ok(output),
                StepResult::Continue => {}
            }
        }

        // Step limit reached — attempt synthesis before giving up.
        let _ = ctx
            .event_tx
            .send(AgentEvent::StepLimitReached {
                run_id: ctx.run_id,
                max_steps: ctx.max_steps,
            })
            .await;

        if let Some(output) = self.attempt_synthesis(&ctx, &state).await {
            let _ = ctx
                .event_tx
                .send(AgentEvent::Finished {
                    run_id: ctx.run_id,
                    text: output.text.clone(),
                })
                .await;
            return Ok(output);
        }

        // Fallback: partial answer with last tool preview.
        if let Some((tool_name, output)) = state.last_tool_result {
            let preview: String = output.chars().take(200).collect();
            let cached_count = ctx.tool_cache.lock().await.len();
            let partial = AgentOutput {
                run_id: ctx.run_id,
                text: format!(
                    "Reached step limit ({}).\n\nInspected {} items via tool caches.\nLast tool '{}' returned:\n{}",
                    ctx.max_steps, cached_count, tool_name, preview
                ),
                finish_reason: FinishReason::StepLimitReached,
            };
            let _ = ctx
                .event_tx
                .send(AgentEvent::Finished {
                    run_id: ctx.run_id,
                    text: partial.text.clone(),
                })
                .await;
            return Ok(partial);
        }

        Ok(AgentOutput {
            run_id: ctx.run_id,
            text: String::new(),
            finish_reason: FinishReason::StepLimitReached,
        })
    }

    fn build_initial_messages(prompt: &str, ctx: &RunContext) -> Vec<ChatMessage> {
        let mut messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: build_system_prompt(ctx.max_steps),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            ChatMessage::user(prompt),
        ];
        if ctx.has_codebase_intent {
            messages.push(ChatMessage::user(TOOL_USE_REMINDER));
        }
        messages
    }

    async fn run_step(
        &self,
        step: usize,
        ctx: &RunContext,
        state: &mut RunState,
    ) -> Result<StepResult, AgentError> {
        let window = AgentRuntime::context_window(&state.messages, ctx.max_context_messages);
        let request = ChatRequest {
            messages: window,
            tools: ctx.tool_specs.clone(),
        };

        let _ = ctx
            .event_tx
            .send(AgentEvent::ModelCallStarted {
                run_id: ctx.run_id,
                step,
            })
            .await;

        let response = self.call_model(request).await;

        let response = match response {
            Ok(r) => r,
            Err(ProviderError::MalformedModelOutput) => {
                return self
                    .handle_malformed(step, ctx, state)
                    .await;
            }
            Err(e) => {
                return self
                    .handle_prompt_error(e, ctx, &mut state.last_tool_result)
                    .await;
            }
        };

        let _ = ctx
            .event_tx
            .send(AgentEvent::ModelCallFinished {
                run_id: ctx.run_id,
                step,
            })
            .await;

        self.process_response(response, ctx, state).await
    }

    async fn call_model(
        &self,
        request: ChatRequest,
    ) -> Result<ModelResponse, ProviderError> {
        self.adapter.chat(request).await
    }

    async fn handle_malformed(
        &self,
        step: usize,
        ctx: &RunContext,
        state: &mut RunState,
    ) -> Result<StepResult, AgentError> {
        let _ = ctx
            .event_tx
            .send(AgentEvent::ModelCallFinished {
                run_id: ctx.run_id,
                step,
            })
            .await;

        state.messages.push(ChatMessage::user(
            "Your previous response used invalid internal control tokens. Reply only with normal user-facing text or valid tool calls.",
        ));

        let retry_window =
            AgentRuntime::context_window(&state.messages, ctx.max_context_messages);
        let retry_request = ChatRequest {
            messages: retry_window,
            tools: ctx.tool_specs.clone(),
        };

        let _ = ctx
            .event_tx
            .send(AgentEvent::ModelCallStarted {
                run_id: ctx.run_id,
                step: step + 1000,
            })
            .await;

        match self.adapter.chat(retry_request).await {
            Ok(r) => {
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::ModelCallFinished {
                        run_id: ctx.run_id,
                        step: step + 1000,
                    })
                    .await;
                self.process_response(r, ctx, state).await
            }
            Err(e) => {
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::Error {
                        run_id: ctx.run_id,
                        error: format!("Model error: {}", e),
                    })
                    .await;
                self.fallback_or_error(e, ctx, &mut state.last_tool_result)
                    .await
            }
        }
    }

    async fn handle_prompt_error(
        &self,
        error: ProviderError,
        ctx: &RunContext,
        last_tool_result: &mut Option<(String, String)>,
    ) -> Result<StepResult, AgentError> {
        let _ = ctx
            .event_tx
            .send(AgentEvent::Error {
                run_id: ctx.run_id,
                error: format!("Model error: {}", error),
            })
            .await;
        self.fallback_or_error(error, ctx, last_tool_result).await
    }

    async fn fallback_or_error(
        &self,
        error: ProviderError,
        ctx: &RunContext,
        last_tool_result: &mut Option<(String, String)>,
    ) -> Result<StepResult, AgentError> {
        if let Some((tool_name, output)) = last_tool_result.take() {
            let preview: String = output.chars().take(200).collect();
            return Ok(StepResult::Finished(AgentOutput {
                run_id: ctx.run_id,
                text: format!(
                    "Model failed after workspace inspection.\n\nLast tool '{}' returned:\n{}",
                    tool_name, preview
                ),
                finish_reason: FinishReason::PartialFailure {
                    last_tool_result: output,
                    tool_name,
                },
            }));
        }
        Err(AgentError::Provider(error.to_string()))
    }

    /// Called after max_steps is reached. Injects a synthesis prompt with no tools,
    /// asks the model to produce its best final answer from gathered context.
    async fn attempt_synthesis(
        &self,
        ctx: &RunContext,
        state: &RunState,
    ) -> Option<AgentOutput> {
        let mut messages = state.messages.clone();
        let prefix = if state.inspection_tools_used {
            build_grounding_prefix()
        } else {
            String::new()
        };
        let synthesis_prompt = format!(
            "{}\n\n{}You have reached the tool-use budget. Stop calling tools and produce the best final answer using the information already gathered.",
            if !prefix.is_empty() { &prefix } else { "" },
            if !prefix.is_empty() { "\n" } else { "" },
        );
        messages.push(ChatMessage::user(&synthesis_prompt));
        let window = AgentRuntime::context_window(&messages, ctx.max_context_messages);
        let request = ChatRequest {
            messages: window,
            tools: Vec::new(),
        };

        let _ = ctx
            .event_tx
            .send(AgentEvent::ModelCallStarted {
                run_id: ctx.run_id,
                step: ctx.max_steps,
            })
            .await;

        match self.adapter.chat(request).await {
            Ok(ModelResponse::Text(text)) => {
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::ModelCallFinished {
                        run_id: ctx.run_id,
                        step: ctx.max_steps,
                    })
                    .await;
                Some(AgentOutput {
                    run_id: ctx.run_id,
                    text,
                    finish_reason: FinishReason::StepLimitSynthesized,
                })
            }
            Ok(ModelResponse::ToolCalls(_)) => {
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::ModelCallFinished {
                        run_id: ctx.run_id,
                        step: ctx.max_steps,
                    })
                    .await;
                None
            }
            Err(_) => None,
        }
    }

    async fn process_response(
        &self,
        response: ModelResponse,
        ctx: &RunContext,
        state: &mut RunState,
    ) -> Result<StepResult, AgentError> {
        match response {
            ModelResponse::Text(text) => {
                if ctx.has_codebase_intent && state.tool_call_count == 0 {
                    let msg = "I need to inspect the workspace before answering that.";
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::Finished {
                            run_id: ctx.run_id,
                            text: msg.to_string(),
                        })
                        .await;
                    return Ok(StepResult::Finished(AgentOutput {
                        run_id: ctx.run_id,
                        text: msg.to_string(),
                        finish_reason: FinishReason::Error(msg.to_string()),
                    }));
                }

                let final_text = sanitise_final_text(
                    text,
                    ctx.edit_intent,
                    state.any_mutating_succeeded,
                    state.any_mutating_requested,
                );
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::Finished {
                        run_id: ctx.run_id,
                        text: final_text.clone(),
                    })
                    .await;
                Ok(StepResult::Finished(AgentOutput {
                    run_id: ctx.run_id,
                    text: final_text,
                    finish_reason: FinishReason::Complete,
                }))
            }
            ModelResponse::ToolCalls(calls) => {
                let result = self.execute_tool_calls(calls, ctx, state).await;
                if state.repeated_unknown_tool {
                    let name = state.unknown_tool_retried.clone().unwrap_or_default();
                    let msg = format!(
                        "Model repeatedly requested unknown tool '{name}'. Available: {}",
                        VALID_TOOL_NAMES.join(", ")
                    );
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::Finished {
                            run_id: ctx.run_id,
                            text: msg.clone(),
                        })
                        .await;
                    return Ok(StepResult::Finished(AgentOutput {
                        run_id: ctx.run_id,
                        text: msg,
                        finish_reason: FinishReason::Error("Repeated unknown tool".to_string()),
                    }));
                }
                result
            }
        }
    }

    async fn execute_tool_calls(
        &self,
        calls: Vec<poly_agent_core::ToolCall>,
        ctx: &RunContext,
        state: &mut RunState,
    ) -> Result<StepResult, AgentError> {
        state
            .messages
            .push(ChatMessage::assistant_with_tool_calls(calls.clone()));

        for call in calls {
            if is_mutating_tool(&call.name) {
                state.any_mutating_requested = true;
            }

            if !self.lookup_and_check_approval(&call, ctx, state).await {
                continue;
            }

            self.run_tool(&call, ctx, state).await;
        }

        Ok(StepResult::Continue)
    }

    async fn lookup_and_check_approval(
        &self,
        call: &poly_agent_core::ToolCall,
        ctx: &RunContext,
        state: &mut RunState,
    ) -> bool {
        let _ = ctx
            .event_tx
            .send(AgentEvent::ToolCallRequested {
                run_id: ctx.run_id,
                call: call.clone(),
            })
            .await;

        let tool = match self.tools.get(&call.name) {
            Some(t) => t,
            None => {
                // Emit unknown tool event.
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::UnknownToolRequested {
                        run_id: ctx.run_id,
                        tool_name: call.name.clone(),
                    })
                    .await;

                if state.unknown_tool_retried.as_deref() == Some(&call.name) {
                    // Second time: fail cleanly.
                    state.repeated_unknown_tool = true;
                    let msg = format!(
                        "Tool '{}' does not exist. Available tools are: {}. Stopping.",
                        call.name,
                        VALID_TOOL_NAMES.join(", ")
                    );
                    state
                        .messages
                        .push(ChatMessage::tool_result(&call.id, &msg));
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::ToolCallFinished {
                            run_id: ctx.run_id,
                            result: ToolResult {
                                tool_call_id: call.id.clone(),
                                output: msg,
                                is_error: true,
                                cached: false,
                            },
                        })
                        .await;
                    return false;
                }

                // First time: corrective message, retry allowed.
                state.unknown_tool_retried = Some(call.name.clone());
                let corrective = format!(
                    "Tool '{}' does not exist. Available tools are: {}. Never invent tool names.",
                    call.name,
                    VALID_TOOL_NAMES.join(", ")
                );
                state
                    .messages
                    .push(ChatMessage::tool_result(&call.id, &corrective));
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::ToolCallFinished {
                        run_id: ctx.run_id,
                        result: ToolResult {
                            tool_call_id: call.id.clone(),
                            output: corrective,
                            is_error: true,
                            cached: false,
                        },
                    })
                    .await;
                return false;
            }
        };

        if tool.risk() == ToolRisk::Safe {
            return true;
        }

        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut pending = self.pending_approvals.lock().await;
            pending.insert((ctx.run_id, call.id.clone()), tx);
        }

        let _ = ctx
            .event_tx
            .send(AgentEvent::ApprovalRequired {
                run_id: ctx.run_id,
                call: call.clone(),
            })
            .await;

        let approved = rx.await.unwrap_or(false);

        if !approved {
            let result = ToolResult {
                tool_call_id: call.id.clone(),
                output: "Tool execution denied by user.".to_string(),
                is_error: true,
                cached: false,
            };
            state
                .messages
                .push(ChatMessage::tool_result(&call.id, &result.output));
            let _ = ctx
                .event_tx
                .send(AgentEvent::ToolCallFinished {
                    run_id: ctx.run_id,
                    result,
                })
                .await;
        }

        approved
    }

    async fn run_tool(
        &self,
        call: &poly_agent_core::ToolCall,
        ctx: &RunContext,
        state: &mut RunState,
    ) {
        let is_inspection_tool = tool_is_inspection(&call.name);

        let cache_key = (
            call.name.clone(),
            serde_json::to_string(&call.arguments).unwrap_or_default(),
        );

        // --- Check anti-loop guard: skip cached safe tool calls ---
        {
            let cache = ctx.tool_cache.lock().await;
            if cache.contains_key(&cache_key) {
                if let Some(cached_output) = cache.get(&cache_key).cloned() {
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::ToolCallStarted {
                            run_id: ctx.run_id,
                            tool_call_id: call.id.clone(),
                            tool_name: call.name.clone(),
                        })
                        .await;

                    state.tool_call_count += 1;
                    state.last_tool_result =
                        Some((call.name.clone(), cached_output.clone()));
                    state.inspection_tools_used =
                        state.inspection_tools_used || is_inspection_tool;

                    let truncated = AgentRuntime::truncate_output(
                        &cached_output,
                        ctx.max_tool_output_bytes,
                    );
                    let result = ToolResult {
                        tool_call_id: call.id.clone(),
                        output: truncated,
                        is_error: false,
                        cached: true,
                    };

                    state
                        .messages
                        .push(ChatMessage::tool_result(&call.id, &result.output));
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::ToolCallFinished {
                            run_id: ctx.run_id,
                            result,
                        })
                        .await;
                    return;
                }
            }
        }

        let _ = ctx
            .event_tx
            .send(AgentEvent::ToolCallStarted {
                run_id: ctx.run_id,
                tool_call_id: call.id.clone(),
                tool_name: call.name.clone(),
            })
            .await;

        let output = match self
            .tools
            .get(&call.name)
            .unwrap()
            .run(call.arguments.clone(), ctx.tool_ctx.clone())
            .await
        {
            Ok(r) => r,
            Err(e) => ToolResult {
                tool_call_id: call.id.clone(),
                output: format!("Tool error: {e}"),
                is_error: true,
                cached: false,
            },
        };

        if !output.is_error && is_mutating_tool(&call.name) {
            state.any_mutating_succeeded = true;
        }

        state.tool_call_count += 1;
        state.last_tool_result = Some((call.name.clone(), output.output.clone()));
        state.inspection_tools_used = state.inspection_tools_used || is_inspection_tool;

        // Cache successful safe tool outputs (anti-loop guard).
        if !output.is_error {
            let mut cache = ctx.tool_cache.lock().await;
            cache.insert(cache_key, output.output.clone());
        }

        // Track empty directories seen by list_files.
        if call.name == "list_files" && !output.is_error {
            if let Some(path) = call.arguments.get("path").and_then(|v| v.as_str()) {
                if output.output.trim().is_empty()
                    || output.output.trim() == "[]"
                    || output.output.trim() == "(empty)"
                {
                    let mut empty = ctx.empty_dirs.lock().await;
                    empty.insert(path.to_string());
                }
            }
        }

        let truncated = AgentRuntime::truncate_output(&output.output, ctx.max_tool_output_bytes);
        let result = ToolResult {
            tool_call_id: call.id.clone(),
            output: truncated,
            is_error: output.is_error,
            cached: false,
        };

        state
            .messages
            .push(ChatMessage::tool_result(&call.id, &result.output));

        let _ = ctx
            .event_tx
            .send(AgentEvent::ToolCallFinished {
                run_id: ctx.run_id,
                result,
            })
            .await;
    }
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod run_tests;

struct RunState {
    messages: Vec<ChatMessage>,
    any_mutating_succeeded: bool,
    any_mutating_requested: bool,
    tool_call_count: usize,
    last_tool_result: Option<(String, String)>,
    inspection_tools_used: bool,
    unknown_tool_retried: Option<String>,
    repeated_unknown_tool: bool,
}
