//! Rust-side integration layer for embedding poly-agent in Poly UI.

mod approval;
mod events;
mod manager;
#[cfg(feature = "tauri")]
mod tauri_commands;
mod types;
mod workspace;

pub use approval::ApprovalPayload;
pub use events::{AgentEventSink, AgentUiEvent, AgentUiEventPayload, POLY_AGENT_EVENT};
pub use manager::AgentRunManager;
#[cfg(feature = "tauri")]
pub use tauri_commands::{
    agent_approve_tool_call, agent_cancel, agent_delete_chat_sandbox, agent_get_run_state,
    agent_reject_tool_call, agent_run, tauri_event_sink,
};
pub use types::{
    AgentRunError, AgentRunInput, AgentRunStats, RunStateSnapshot, RunStatus, MAX_STORED_EVENTS,
};

#[cfg(test)]
#[path = "manager_tests.rs"]
mod manager_tests;
