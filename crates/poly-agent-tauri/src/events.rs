use std::sync::Arc;
use std::time::SystemTime;

use poly_agent_core::{AgentEvent, RunId, ToolRisk};
use serde::{Deserialize, Serialize};

use crate::approval::ApprovalPayload;
use crate::types::{MutableRunState, RunStatus, MAX_STORED_EVENTS};

pub const POLY_AGENT_EVENT: &str = "poly-agent:event";
pub type AgentEventSink = Arc<dyn Fn(AgentUiEvent) + Send + Sync>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentUiEvent {
    pub run_id: RunId,
    pub event_type: String,
    pub timestamp: SystemTime,
    pub data: AgentUiEventPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum AgentUiEventPayload {
    Started,
    Activity {
        phase: String,
        title: String,
        summary: String,
        details: Vec<String>,
        status: String,
    },
    Thinking,
    ModelCallStarted {
        step: usize,
    },
    ModelCallFinished {
        step: usize,
    },
    ToolCallRequested {
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    ToolCallStarted {
        tool_call_id: String,
        tool_name: String,
    },
    ToolCallFinished {
        tool_call_id: String,
        output: String,
        is_error: bool,
        cached: bool,
    },
    ToolCallDelta {
        tool_call_id: String,
        delta: String,
    },
    TextDelta {
        text: String,
    },
    FinalResponseDelta {
        text: String,
    },
    ApprovalRequired(ApprovalPayload),
    ToolAutoApproved {
        tool_call_id: String,
        tool_name: String,
        reason: String,
    },
    ToolAutoDenied {
        tool_call_id: String,
        tool_name: String,
        reason: String,
    },
    AutoReviewDecision {
        tool_call_id: String,
        tool_name: String,
        risk: String,
        decision: String,
        reason: String,
    },
    Finished {
        text: String,
    },
    Failed {
        error: String,
    },
    Cancelled,
    StepLimitReached {
        max_steps: usize,
    },
    UnknownToolRequested {
        tool_name: String,
    },
}

pub(crate) fn cancelled_event(run_id: RunId) -> AgentUiEvent {
    AgentUiEvent {
        run_id,
        event_type: "cancelled".to_string(),
        timestamp: SystemTime::now(),
        data: AgentUiEventPayload::Cancelled,
    }
}

pub(crate) fn push_event(state: &mut MutableRunState, event: AgentUiEvent) {
    state.stats.events_seen += 1;
    if state.events.len() == MAX_STORED_EVENTS {
        state.events.pop_front();
    }
    state.events.push_back(event);
}

pub(crate) fn map_runtime_event(
    external_run_id: RunId,
    event: AgentEvent,
    debug: bool,
) -> AgentUiEvent {
    let timestamp = SystemTime::now();
    let (event_type, data) = match event {
        AgentEvent::Started { .. } => ("run_started", AgentUiEventPayload::Started),
        AgentEvent::Activity {
            phase,
            title,
            summary,
            details,
            status,
            ..
        } => (
            "activity",
            AgentUiEventPayload::Activity {
                phase,
                title,
                summary,
                details,
                status: serde_json::to_string(&status)
                    .unwrap_or_else(|_| "\"running\"".to_string())
                    .trim_matches('"')
                    .to_string(),
            },
        ),
        AgentEvent::ModelCallStarted { step, .. } => (
            "model_call_started",
            AgentUiEventPayload::ModelCallStarted { step },
        ),
        AgentEvent::ModelCallFinished { step, .. } => {
            ("thinking", AgentUiEventPayload::ModelCallFinished { step })
        }
        AgentEvent::ToolCallRequested { call, .. } => (
            "tool_call_planned",
            AgentUiEventPayload::ToolCallRequested {
                tool_call_id: call.id,
                tool_name: call.name,
                arguments: call.arguments,
            },
        ),
        AgentEvent::ToolCallStarted {
            tool_call_id,
            tool_name,
            ..
        } => (
            "tool_call_started",
            AgentUiEventPayload::ToolCallStarted {
                tool_call_id,
                tool_name,
            },
        ),
        AgentEvent::ToolCallFinished { result, .. } => (
            "tool_call_finished",
            AgentUiEventPayload::ToolCallFinished {
                tool_call_id: result.tool_call_id,
                output: result.output,
                is_error: result.is_error,
                cached: result.cached,
            },
        ),
        AgentEvent::ApprovalRequired { call, .. } => (
            "approval_required",
            AgentUiEventPayload::ApprovalRequired(ApprovalPayload::from_call(
                &call,
                ToolRisk::RequiresApproval,
                debug,
            )),
        ),
        AgentEvent::ToolAutoApproved {
            tool_call_id,
            tool_name,
            reason,
            ..
        } => {
            let reason_str =
                serde_json::to_string(&reason).unwrap_or_else(|_| "\"unknown\"".to_string());
            (
                "tool_auto_approved",
                AgentUiEventPayload::ToolAutoApproved {
                    tool_call_id,
                    tool_name,
                    reason: reason_str,
                },
            )
        }
        AgentEvent::ToolAutoDenied {
            tool_call_id,
            tool_name,
            reason,
            ..
        } => (
            "tool_auto_denied",
            AgentUiEventPayload::ToolAutoDenied {
                tool_call_id,
                tool_name,
                reason,
            },
        ),
        AgentEvent::AutoReviewDecision {
            tool_call_id,
            risk,
            decision,
            reason,
            ..
        } => {
            let risk_str =
                serde_json::to_string(&risk).unwrap_or_else(|_| "\"unknown\"".to_string());
            let decision_str =
                serde_json::to_string(&decision).unwrap_or_else(|_| "\"unknown\"".to_string());
            (
                "auto_review_decision",
                AgentUiEventPayload::AutoReviewDecision {
                    tool_call_id,
                    tool_name: String::new(),
                    risk: risk_str,
                    decision: decision_str,
                    reason,
                },
            )
        }
        AgentEvent::StepLimitReached { max_steps, .. } => (
            "thinking",
            AgentUiEventPayload::StepLimitReached { max_steps },
        ),
        AgentEvent::UnknownToolRequested { tool_name, .. } => (
            "thinking",
            AgentUiEventPayload::UnknownToolRequested { tool_name },
        ),
        AgentEvent::TextDelta { text, .. } => {
            ("model_token_delta", AgentUiEventPayload::TextDelta { text })
        }
        AgentEvent::FinalResponseDelta { text, .. } => {
            ("final_response_delta", AgentUiEventPayload::FinalResponseDelta { text })
        }
        AgentEvent::ToolCallDelta {
            tool_call_id,
            delta,
            ..
        } => (
            "tool_call_delta",
            AgentUiEventPayload::ToolCallDelta {
                tool_call_id,
                delta,
            },
        ),
        AgentEvent::Finished { text, .. } => ("run_finished", AgentUiEventPayload::Finished { text }),
        AgentEvent::Error { error, .. } => ("run_failed", AgentUiEventPayload::Failed { error }),
        AgentEvent::Cancelled { .. } => ("run_cancelled", AgentUiEventPayload::Cancelled),
        _ => ("thinking", AgentUiEventPayload::Thinking),
    };

    AgentUiEvent {
        run_id: external_run_id,
        event_type: event_type.to_string(),
        timestamp,
        data,
    }
}

pub(crate) fn apply_event_to_state(
    state: &mut MutableRunState,
    runtime_event: &AgentEvent,
    ui_event: &AgentUiEvent,
) {
    match runtime_event {
        AgentEvent::Started { run_id } => state.runtime_run_id = Some(*run_id),
        AgentEvent::ModelCallStarted { .. } => state.stats.model_calls += 1,
        AgentEvent::ToolCallRequested { .. } => state.stats.tool_calls_requested += 1,
        AgentEvent::ToolCallFinished { .. } => {
            state.stats.tool_calls_finished += 1;
            if state.status == RunStatus::WaitingForApproval {
                state.status = RunStatus::Running;
                state.pending_approval = None;
            }
        }
        AgentEvent::ApprovalRequired { .. } => {
            state.stats.approvals_requested += 1;
            state.status = RunStatus::WaitingForApproval;
            if let AgentUiEventPayload::ApprovalRequired(payload) = &ui_event.data {
                state.pending_approval = Some(payload.clone());
            }
        }
        AgentEvent::Finished { text, .. } => {
            if state.status != RunStatus::Cancelled {
                state.status = RunStatus::Finished;
                state.final_output = Some(text.clone());
                state.finished_at = Some(SystemTime::now());
            }
        }
        AgentEvent::Error { error, .. } => {
            if state.status != RunStatus::Cancelled {
                state.status = RunStatus::Failed;
                state.last_error = Some(error.clone());
                state.finished_at = Some(SystemTime::now());
            }
        }
        AgentEvent::Cancelled { .. } => {
            state.status = RunStatus::Cancelled;
            state.finished_at = Some(SystemTime::now());
        }
        _ => {}
    }
}
