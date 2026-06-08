use poly_agent_core::{
    ActivityStatus, AgentError, AgentEvent, AgentInput, AgentOutput, AgentResolvedContext, AutoApproveReason,
    ChatMessage, ChatRole, FinishReason, PermissionPreset, ToolResult, ToolRisk,
};
use poly_agent_providers::{ChatRequest, ModelResponse, ProviderError};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::engine::AgentRuntime;
use crate::intent::{
    build_grounding_prefix, contains_codebase_intent_phrase, is_command_tool, is_file_write_tool,
    is_clarification_request, is_mutating_tool, is_prose_confirmation, mutation_guard_message,
    mutation_missing_message, mutation_retry_instruction, sanitise_final_text, tool_is_inspection,
    EditIntent,
};
use crate::review::ReviewContext;
use crate::tool::ToolContext;

/// Internal verdict for how the runtime should handle a single tool call.
pub(crate) enum ToolPermissionDecision {
    /// Run the tool without further user interaction.
    Approved(AutoApproveReason),
    /// Pause and wait for explicit user approval via the existing flow.
    RequiresApproval,
    /// Do not run the tool. Emit a `ToolAutoDenied` event and a denial result.
    Denied(String),
}

const TOOL_USE_REMINDER: &str = "\
CRITICAL: When the user asks about the codebase, project, directory, program, app, repo, files, architecture, or how it works, you MUST inspect the workspace before answering.

Strategy for codebase summaries:
1. Call inspect_project first to get the high-level project structure.
2. Call read_important_files to read README, metadata, key entry points.
3. After 2-3 calls total, summarize using gathered evidence.
Do not inspect every module unless specifically asked. Gather evidence then answer.";

const VALID_TOOL_NAMES: &[&str] = &[
    "list_files",
    "read_file",
    "search_files",
    "propose_edit",
    "apply_patch",
    "write_file",
    "run_command",
    "suggest_command",
    "inspect_project",
    "read_important_files",
];

const MODEL_CALL_TIMEOUT_SECS: u64 = 120;

async fn emit_activity(
    ctx: &RunContext,
    phase: &str,
    title: &str,
    summary: &str,
    status: ActivityStatus,
) {
    emit_activity_with_details(ctx, phase, title, summary, Vec::new(), status).await;
}

async fn emit_activity_with_details(
    ctx: &RunContext,
    phase: &str,
    title: &str,
    summary: &str,
    details: Vec<String>,
    status: ActivityStatus,
) {
    if summary.trim().is_empty() {
        return;
    }
    let _ = ctx
        .event_tx
        .send(AgentEvent::Activity {
            run_id: ctx.run_id,
            phase: phase.to_string(),
            title: title.to_string(),
            summary: summary.to_string(),
            details,
            status,
        })
        .await;
}

fn build_system_prompt(max_steps: usize) -> String {
    format!(
        "You are a helpful, careful coding agent.\n\
\n\
Available tools: {}\n\
Never invent tool names. Only use tools from this list.\n\
\n\
Tool selection rules:\n\
- For questions about the project, summaries, explanations, or other READ-ONLY tasks, use inspect_project, read_important_files, read_file, list_files, or search_files. Do not invent file contents.\n\
- When the user asks to create, write, edit, change, modify, append, delete, rename, move, replace, fix, or otherwise alter a file, you MUST call the appropriate tool. Do not ask the user for confirmation in plain text. The runtime approval system handles confirmation.\n\
- For creating or replacing a file, call `write_file`. For small edits to existing files, prefer `apply_patch`. For preview-only changes, use `propose_edit`.\n\
- If a mutating action is needed, call the mutating tool and let the runtime emit approval. Never say \"I will create/edit/write the file\" unless the tool has successfully completed. If approval is required, do not claim completion.\n\
- If the user names a file path explicitly (e.g., README.md, package.json, src/main.rs), you may call read_file directly on it instead of calling list_files first.\n\
- If a tool requires approval, surface that to the user instead of pasting the whole rewritten file in your reply.\n\
- For running shell commands, use `run_command`. It requires approval. Commands run in the workspace root.\n\
- For suggesting commands without executing, use `suggest_command`. It returns a structured suggestion.\n\
- Prefer `suggest_command` when you are unsure whether a command is safe.\n\
\n\
Follow-up context rules:\n\
- Use recent context to resolve pronouns and short follow-ups such as `it`, `that file`, `same file`, `add another`, `do it again`, `show it again`, and `review it`.\n\
- Do not ask for a file name again when the active file or recent target is clear from structured context.\n\
- Preserve recent user constraints such as `do not remove anything`, `append only`, `keep existing content`, `only edit this file`, and `do not run commands`.\n\
- For append-style requests like `add another sentence` on an active file, append to the end by default unless the user names another location.\n\
- Prefer acting when target and action are clear. Ask a clarification only when no active target exists, multiple targets are equally likely, the action is destructive, or constraints conflict.\n\
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
    cancellation: CancellationToken,
    tool_ctx: ToolContext,
    tool_specs: Vec<poly_agent_providers::ToolSpec>,
    edit_intent: EditIntent,
    has_codebase_intent: bool,
    simple_chat: bool,
    max_steps: usize,
    max_context_messages: usize,
    max_tool_output_bytes: usize,
    tool_cache: Arc<Mutex<HashMap<(String, String), String>>>,
    empty_dirs: Arc<Mutex<HashSet<String>>>,
    user_prompt: String,
    resolved_context: Option<AgentResolvedContext>,
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
        cancellation: CancellationToken,
    ) -> Result<AgentOutput, AgentError> {
        let run_id = Uuid::new_v4();
        let _ = event_tx.send(AgentEvent::Started { run_id }).await;

        let edit_intent = EditIntent::detect(&input.prompt);
        let has_codebase_intent = contains_codebase_intent_phrase(&input.prompt);
        let simple_chat = is_simple_chat_prompt(&input.prompt, edit_intent, has_codebase_intent);

        let ctx = RunContext {
            run_id,
            event_tx,
            cancellation: cancellation.clone(),
            tool_ctx: ToolContext {
                workspace: input.workspace,
                limits: input.limits.clone(),
                cancellation,
            },
            tool_specs: if simple_chat { Vec::new() } else { self.tools.tool_specs() },
            edit_intent,
            has_codebase_intent,
            simple_chat,
            max_steps: input.limits.max_steps,
            max_context_messages: input.limits.max_context_messages,
            max_tool_output_bytes: input.limits.max_tool_output_bytes,
            tool_cache: Arc::new(Mutex::new(HashMap::new())),
            empty_dirs: Arc::new(Mutex::new(HashSet::new())),
            user_prompt: input.prompt.clone(),
            resolved_context: input.resolved_context.clone(),
        };

        emit_activity(
            &ctx,
            "workspace_inspection",
            "Inspecting workspace",
            "Checking the selected project structure and metadata.",
            ActivityStatus::Completed,
        )
        .await;

        let mut state = RunState {
            messages: Self::build_initial_messages(&input.prompt, &ctx),
            any_mutating_succeeded: false,
            any_mutating_requested: false,
            tool_call_count: 0,
            last_tool_result: None,
            inspection_tools_used: false,
            unknown_tool_retried: None,
            repeated_unknown_tool: false,
            mutation_retry_attempted: false,
        };

        for step in 0..ctx.max_steps {
            if ctx.cancellation.is_cancelled() {
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::Cancelled {
                        run_id: ctx.run_id,
                    })
                    .await;
                return Err(AgentError::Other("Run cancelled".to_string()));
            }
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
        if !ctx.simple_chat {
            if let Some(context_message) = build_resolved_context_message(&ctx.resolved_context) {
                messages.push(ChatMessage::user(context_message));
            }
        }
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
        if state.inspection_tools_used && !state.any_mutating_requested {
            emit_activity(
                ctx,
                "summarizing",
                "Summarizing",
                "Preparing the final answer from the inspected project files.",
                ActivityStatus::Running,
            )
            .await;
        } else {
            emit_activity(
                ctx,
                "thinking",
                "Thinking",
                if ctx.has_codebase_intent {
                    "Understanding the request and deciding what needs to be inspected."
                } else {
                    "Waiting for the model response."
                },
                ActivityStatus::Running,
            )
            .await;
        }

        let response = match tokio::time::timeout(
            std::time::Duration::from_secs(MODEL_CALL_TIMEOUT_SECS),
            self.call_model(request, ctx),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(ProviderError::Parse("Model call timed out".to_string())),
        };

        if ctx.cancellation.is_cancelled() {
            let _ = ctx
                .event_tx
                .send(AgentEvent::Cancelled {
                    run_id: ctx.run_id,
                })
                .await;
            return Err(AgentError::Other("Run cancelled".to_string()));
        }

        let response = match response {
            Ok(r) => r,
            Err(ProviderError::MalformedModelOutput) => {
                return self.handle_malformed(step, ctx, state).await;
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
        emit_activity(
            ctx,
            if state.inspection_tools_used && !state.any_mutating_requested { "summarizing" } else { "thinking" },
            if state.inspection_tools_used && !state.any_mutating_requested { "Summarizing" } else { "Thinking" },
            "Model response received.",
            ActivityStatus::Completed,
        )
        .await;

        self.process_response(response, ctx, state).await
    }

    async fn call_model(
        &self,
        request: ChatRequest,
        ctx: &RunContext,
    ) -> Result<ModelResponse, ProviderError> {
        let mut stream = self.adapter.chat_stream(request).await?;
        let mut text = String::new();
        let mut first_token = true;
        while let Some(chunk) = stream.next().await {
            if ctx.cancellation.is_cancelled() {
                break;
            }
            match chunk? {
                ModelResponse::Text(delta) => {
                    if delta.is_empty() {
                        continue;
                    }
                    if first_token {
                        first_token = false;
                        emit_activity(
                            ctx,
                            "responding",
                            "Responding",
                            if ctx.simple_chat {
                                "Receiving the model response."
                            } else {
                                "Preparing the final response from gathered context."
                            },
                            ActivityStatus::Running,
                        )
                        .await;
                    }
                    text.push_str(&delta);
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::TextDelta {
                            run_id: ctx.run_id,
                            text: delta.clone(),
                        })
                        .await;
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::FinalResponseDelta {
                            run_id: ctx.run_id,
                            text: delta,
                        })
                        .await;
                }
                ModelResponse::ToolCalls(calls) => {
                    return Ok(ModelResponse::ToolCalls(calls));
                }
            }
        }
        Ok(ModelResponse::Text(text))
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

        let retry_window = AgentRuntime::context_window(&state.messages, ctx.max_context_messages);
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

        match self.call_model(retry_request, ctx).await {
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
    async fn attempt_synthesis(&self, ctx: &RunContext, state: &RunState) -> Option<AgentOutput> {
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

        match self.call_model(request, ctx).await {
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
                if self.should_retry_mutation_text(&text, ctx, state) {
                    state.mutation_retry_attempted = true;
                    state.messages.push(ChatMessage::assistant(text));
                    state
                        .messages
                        .push(ChatMessage::user(mutation_retry_instruction()));
                    return Ok(StepResult::Continue);
                }

                if ctx.edit_intent.is_edit
                    && !state.any_mutating_requested
                    && state.mutation_retry_attempted
                    && !state.any_mutating_succeeded
                    && !is_clarification_request(&text)
                {
                    let final_text = if is_prose_confirmation(&text) {
                        mutation_guard_message()
                    } else {
                        mutation_missing_message()
                    };
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::Error {
                            run_id: ctx.run_id,
                            error: final_text.clone(),
                        })
                        .await;
                    return Ok(StepResult::Finished(AgentOutput {
                        run_id: ctx.run_id,
                        text: final_text,
                        finish_reason: FinishReason::Error("Mutation tool required".to_string()),
                    }));
                }

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

                if ctx.edit_intent.is_edit
                    && !state.any_mutating_requested
                    && !state.any_mutating_succeeded
                    && text.trim().is_empty()
                {
                    let final_text = mutation_missing_message();
                    let _ = ctx
                        .event_tx
                        .send(AgentEvent::Error {
                            run_id: ctx.run_id,
                            error: final_text.clone(),
                        })
                        .await;
                    return Ok(StepResult::Finished(AgentOutput {
                        run_id: ctx.run_id,
                        text: final_text,
                        finish_reason: FinishReason::Error("Mutation tool required".to_string()),
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

            if self.reject_bad_follow_up_tool(&call, ctx, state).await {
                continue;
            }

            if !self.lookup_and_check_approval(&call, ctx, state).await {
                continue;
            }

            self.run_tool(&call, ctx, state).await;
        }

        Ok(StepResult::Continue)
    }

    async fn reject_bad_follow_up_tool(
        &self,
        call: &poly_agent_core::ToolCall,
        ctx: &RunContext,
        state: &mut RunState,
    ) -> bool {
        let lower = ctx.user_prompt.to_lowercase();
        let simple_file_edit = lower.contains("remove the last line")
            || lower.contains("delete the last line")
            || ((lower.contains("add") || lower.contains("append") || lower.contains("write"))
                && (lower.contains("sentence") || lower.contains("sentences")));
        if !simple_file_edit || call.name != "run_command" {
            return false;
        }

        let msg = "Do not use shell commands for this simple file edit. Use the active file and call apply_patch with the smallest exact text change only.";
        state
            .messages
            .push(ChatMessage::tool_result(&call.id, msg));
        let _ = ctx
            .event_tx
            .send(AgentEvent::ToolCallFinished {
                run_id: ctx.run_id,
                result: ToolResult {
                    tool_call_id: call.id.clone(),
                    output: msg.to_string(),
                    is_error: true,
                    cached: false,
                },
            })
            .await;
        true
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
        emit_activity(
            ctx,
            tool_phase(&call.name),
            tool_title(&call.name),
            &tool_summary(&call.name, &call.arguments, "planned"),
            ActivityStatus::Running,
        )
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

        let decision = self
            .evaluate_tool_permission(call, &call.name, ctx, state)
            .await;

        match decision {
            ToolPermissionDecision::Approved(reason) => {
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::ToolAutoApproved {
                        run_id: ctx.run_id,
                        tool_call_id: call.id.clone(),
                        tool_name: call.name.clone(),
                        reason,
                    })
                    .await;
                true
            }
            ToolPermissionDecision::RequiresApproval => {
                self.await_user_approval(call, ctx, state).await
            }
            ToolPermissionDecision::Denied(reason) => {
                let result = ToolResult {
                    tool_call_id: call.id.clone(),
                    output: format!("Tool execution denied: {reason}"),
                    is_error: true,
                    cached: false,
                };
                state
                    .messages
                    .push(ChatMessage::tool_result(&call.id, &result.output));
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::ToolAutoDenied {
                        run_id: ctx.run_id,
                        tool_call_id: call.id.clone(),
                        tool_name: call.name.clone(),
                        reason,
                    })
                    .await;
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::ToolCallFinished {
                        run_id: ctx.run_id,
                        result,
                    })
                    .await;
                false
            }
        }
    }

    /// Decide whether a non-Safe tool call should run, be auto-approved,
    /// require user approval, or be denied. Honours `self.permission_preset`.
    async fn evaluate_tool_permission(
        &self,
        call: &poly_agent_core::ToolCall,
        tool_name: &str,
        ctx: &RunContext,
        state: &RunState,
    ) -> ToolPermissionDecision {
        match self.permission_preset {
            PermissionPreset::FullAccess => {
                ToolPermissionDecision::Approved(AutoApproveReason::FullAccess)
            }
            PermissionPreset::Default => {
                if is_file_write_tool(tool_name) {
                    ToolPermissionDecision::Approved(AutoApproveReason::PresetDefault)
                } else if is_command_tool(tool_name) {
                    ToolPermissionDecision::RequiresApproval
                } else {
                    // Unknown non-Safe tool under Default preset: fall back to
                    // user approval to be safe. (In practice this branch is
                    // only reachable for tools that opt into RequiresApproval
                    // or Dangerous without being classified by the helpers.)
                    ToolPermissionDecision::RequiresApproval
                }
            }
            PermissionPreset::AutoReview => {
                let review_ctx = ReviewContext {
                    user_prompt: ctx.user_prompt.clone(),
                    recent_messages: AgentRuntime::context_window(
                        &state.messages,
                        ctx.max_context_messages,
                    ),
                };
                let verdict = self.reviewer.review(call, &review_ctx).await;
                let _ = ctx
                    .event_tx
                    .send(AgentEvent::AutoReviewDecision {
                        run_id: ctx.run_id,
                        tool_call_id: call.id.clone(),
                        risk: verdict.risk,
                        decision: verdict.decision,
                        reason: verdict.reason.clone(),
                    })
                    .await;
                match verdict.decision {
                    poly_agent_core::ReviewDecision::Approve => {
                        ToolPermissionDecision::Approved(AutoApproveReason::AutoReviewLow)
                    }
                    poly_agent_core::ReviewDecision::Ask => {
                        ToolPermissionDecision::RequiresApproval
                    }
                    poly_agent_core::ReviewDecision::Deny => {
                        ToolPermissionDecision::Denied(verdict.reason)
                    }
                }
            }
        }
    }

    /// Set up a oneshot, emit `ApprovalRequired`, and wait for the user.
    async fn await_user_approval(
        &self,
        call: &poly_agent_core::ToolCall,
        ctx: &RunContext,
        state: &mut RunState,
    ) -> bool {
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
                    state.last_tool_result = Some((call.name.clone(), cached_output.clone()));
                    state.inspection_tools_used = state.inspection_tools_used || is_inspection_tool;

                    let truncated =
                        AgentRuntime::truncate_output(&cached_output, ctx.max_tool_output_bytes);
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
        emit_activity(
            ctx,
            tool_phase(&call.name),
            tool_title(&call.name),
            &tool_summary(&call.name, &call.arguments, "started"),
            ActivityStatus::Running,
        )
        .await;

        let Some(tool) = self.tools.get(&call.name) else {
            let result = ToolResult {
                tool_call_id: call.id.clone(),
                output: format!("Unknown tool: {}", call.name),
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
            return;
        };

        let output = match tool.run(call.arguments.clone(), ctx.tool_ctx.clone()).await {
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
        emit_activity_with_details(
            ctx,
            tool_phase(&call.name),
            tool_title(&call.name),
            &tool_summary(&call.name, &call.arguments, if output.is_error { "failed" } else { "finished" }),
            tool_details(&call.name, &call.arguments, &output.output),
            if output.is_error { ActivityStatus::Failed } else { ActivityStatus::Completed },
        )
        .await;
    }

    fn should_retry_mutation_text(&self, text: &str, ctx: &RunContext, state: &RunState) -> bool {
        ctx.edit_intent.is_edit
            && !state.any_mutating_requested
            && !state.mutation_retry_attempted
            && !ctx.tool_specs.is_empty()
            && ctx
                .tool_specs
                .iter()
                .any(|tool| is_mutating_tool(&tool.name))
            && !is_clarification_request(text)
    }

}

fn build_resolved_context_message(context: &Option<AgentResolvedContext>) -> Option<String> {
    let context = context.as_ref()?;
    let has_context = context.active_file.is_some()
        || !context.recently_viewed_files.is_empty()
        || !context.recently_edited_files.is_empty()
        || !context.recent_constraints.is_empty()
        || context.last_tool_call.is_some();
    if !has_context {
        return None;
    }

    let mut lines = vec![
        "Structured recent agent context for resolving this follow-up:".to_string(),
    ];
    if let Some(workspace) = context.active_workspace.as_deref() {
        lines.push(format!("- active workspace: {workspace}"));
    }
    if let Some(file) = context.active_file.as_deref() {
        lines.push(format!("- active file: {file}"));
        lines.push(format!(
            "- if the user says `it`, `that file`, `same file`, or `add another`, use `{file}` unless the current prompt explicitly names another file."
        ));
    }
    if !context.recently_viewed_files.is_empty() {
        lines.push(format!(
            "- recently viewed files: {}",
            context.recently_viewed_files.join(", ")
        ));
    }
    if !context.recently_edited_files.is_empty() {
        lines.push(format!(
            "- recently edited files: {}",
            context.recently_edited_files.join(", ")
        ));
    }
    if !context.recent_constraints.is_empty() {
        lines.push(format!(
            "- constraints to preserve: {}",
            context.recent_constraints.join(" ")
        ));
    }
    if let Some(last) = &context.last_tool_call {
        lines.push(format!(
            "- last tool call: {}{}",
            last.tool_name,
            last.target_path
                .as_deref()
                .map(|path| format!(" on {path}"))
                .unwrap_or_default()
        ));
    }
    lines.push("Use this context silently. Do not mention it unless useful. Do not ask for the file again if the target is clear.".to_string());
    Some(lines.join("\n"))
}

fn tool_phase(tool_name: &str) -> &'static str {
    match tool_name {
        "read_file" | "read_important_files" => "file_read",
        "search_files" => "file_search",
        "list_files" | "inspect_project" => "workspace_inspection",
        "apply_patch" | "write_file" | "propose_edit" => "editing",
        "run_command" | "suggest_command" => "verifying",
        _ => "verifying",
    }
}

fn tool_title(tool_name: &str) -> &'static str {
    match tool_name {
        "read_file" | "read_important_files" => "Reading files",
        "list_files" | "inspect_project" => "Inspecting workspace",
        "search_files" => "Searching files",
        "apply_patch" | "write_file" | "propose_edit" => "Editing file",
        "run_command" => "Running command",
        "suggest_command" => "Preparing command",
        _ => "Using tool",
    }
}

fn tool_summary(tool_name: &str, args: &serde_json::Value, stage: &str) -> String {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("workspace");
    match (tool_name, stage) {
        ("read_file", "planned") => format!("Preparing to read {path}."),
        ("read_file", "started") => format!("Loading {path} before answering."),
        ("read_file", "finished") => format!("Loaded {path}."),
        ("read_important_files", "finished") => "Loaded key project files.".to_string(),
        ("list_files", "finished") | ("inspect_project", "finished") => {
            "Workspace structure inspected.".to_string()
        }
        ("search_files", "finished") => "File search completed.".to_string(),
        ("apply_patch", "planned") | ("write_file", "planned") => {
            format!("Preparing targeted edit for {path}.")
        }
        ("apply_patch", "started") | ("write_file", "started") => {
            format!("Applying file change to {path}.")
        }
        ("apply_patch", "finished") | ("write_file", "finished") => {
            format!("Updated {path}.")
        }
        (_, "failed") => format!("{} failed.", tool_title(tool_name)),
        (_, "started") => format!("{} started.", tool_title(tool_name)),
        (_, "finished") => format!("{} completed.", tool_title(tool_name)),
        _ => format!("{} planned.", tool_title(tool_name)),
    }
}

fn tool_details(tool_name: &str, args: &serde_json::Value, output: &str) -> Vec<String> {
    let mut details = Vec::new();
    if let Some(path) = args
        .get("path")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        details.push(format!("Target: {path}"));
    }
    if let Some(query) = args
        .get("query")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        details.push(format!("Search: {query}"));
    }
    match tool_name {
        "inspect_project" | "list_files" => {
            let count = output.lines().filter(|line| !line.trim().is_empty()).take(50).count();
            if count > 0 {
                details.push(format!("Found {count} workspace entries."));
            }
        }
        "search_files" => {
            let count = output.lines().filter(|line| !line.trim().is_empty()).take(50).count();
            if count > 0 {
                details.push(format!("Matched {count} file results."));
            }
        }
        "read_important_files" => {
            details.push("Loaded key project files.".to_string());
        }
        "read_file" => {
            if !output.trim().is_empty() {
                details.push("Loaded file contents.".to_string());
            }
        }
        "apply_patch" | "write_file" => {
            details.push("File change completed.".to_string());
        }
        _ => {}
    }
    details.truncate(4);
    details
}

fn is_simple_chat_prompt(prompt: &str, edit_intent: EditIntent, has_codebase_intent: bool) -> bool {
    if edit_intent.is_edit || has_codebase_intent {
        return false;
    }
    let lower = prompt.trim().to_lowercase();
    if lower.is_empty() {
        return true;
    }
    let project_terms = [
        "file",
        "folder",
        "directory",
        "repo",
        "project",
        "code",
        "function",
        "component",
        "test",
        "run",
        "build",
        "error",
        "bug",
        "fix",
        "readme",
        "package.json",
        "src/",
        ".rs",
        ".ts",
        ".tsx",
        ".js",
        ".json",
        ".md",
    ];
    if project_terms.iter().any(|term| lower.contains(term)) {
        return false;
    }
    let simple_exact = [
        "hello",
        "hi",
        "hey",
        "how are you?",
        "thanks",
        "thank you",
        "ok",
        "okay",
    ];
    simple_exact.contains(&lower.as_str()) || lower.split_whitespace().count() <= 12
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
    mutation_retry_attempted: bool,
}
