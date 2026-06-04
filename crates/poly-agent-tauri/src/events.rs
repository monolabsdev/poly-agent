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
    Thinking,
    ModelCallStarted { step: usize },
    ModelCallFinished { step: usize },
    ToolCallRequested { tool_call_id: String, tool_name: String },
    ToolCallStarted { tool_call_id: String, tool_name: String },
    ToolCallFinished {
        tool_call_id: String,
        output: String,
        is_error: bool,
        cached: bool,
    },
    ApprovalRequired(ApprovalPayload),
    Finished { text: String },
    Failed { error: String },
    Cancelled,
    StepLimitReached { max_steps: usize },
    UnknownToolRequested { tool_name: String },
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
        AgentEvent::Started { .. } => ("started", AgentUiEventPayload::Started),
        AgentEvent::ModelCallStarted { step, .. } => (
            "model_call_started",
            AgentUiEventPayload::ModelCallStarted { step },
        ),
        AgentEvent::ModelCallFinished { step, .. } => (
            "thinking",
            AgentUiEventPayload::ModelCallFinished { step },
        ),
        AgentEvent::ToolCallRequested { call, .. } => (
            "tool_call_requested",
            AgentUiEventPayload::ToolCallRequested {
                tool_call_id: call.id,
                tool_name: call.name,
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
        AgentEvent::StepLimitReached { max_steps, .. } => (
            "thinking",
            AgentUiEventPayload::StepLimitReached { max_steps },
        ),
        AgentEvent::UnknownToolRequested { tool_name, .. } => (
            "thinking",
            AgentUiEventPayload::UnknownToolRequested { tool_name },
        ),
        AgentEvent::Finished { text, .. } => {
            ("finished", AgentUiEventPayload::Finished { text })
        }
        AgentEvent::Error { error, .. } => ("failed", AgentUiEventPayload::Failed { error }),
        _ => (
            "thinking",
            AgentUiEventPayload::Thinking,
        ),
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
        _ => {}
    }
}
