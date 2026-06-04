use poly_agent_core::RunId;
use std::sync::Arc;
use tauri::Emitter;

use crate::events::{AgentEventSink, AgentUiEvent, POLY_AGENT_EVENT};
use crate::manager::AgentRunManager;
use crate::types::{AgentRunInput, RunStateSnapshot};

pub fn tauri_event_sink(app: tauri::AppHandle) -> AgentEventSink {
    Arc::new(move |event: AgentUiEvent| {
        let _ = app.emit(POLY_AGENT_EVENT, event);
    })
}

#[tauri::command]
pub async fn agent_run(
    manager: tauri::State<'_, AgentRunManager>,
    input: AgentRunInput,
) -> Result<RunId, String> {
    manager.start_run(input).await.map_err(|err| err.to_string())
}

#[tauri::command]
pub async fn agent_cancel(
    manager: tauri::State<'_, AgentRunManager>,
    run_id: RunId,
) -> Result<(), String> {
    manager.cancel_run(run_id).await.map_err(|err| err.to_string())
}

#[tauri::command]
pub async fn agent_approve_tool_call(
    manager: tauri::State<'_, AgentRunManager>,
    run_id: RunId,
    approval_id: String,
) -> Result<(), String> {
    manager
        .approve_tool_call(run_id, &approval_id)
        .await
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub async fn agent_reject_tool_call(
    manager: tauri::State<'_, AgentRunManager>,
    run_id: RunId,
    approval_id: String,
) -> Result<(), String> {
    manager
        .reject_tool_call(run_id, &approval_id)
        .await
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub async fn agent_get_run_state(
    manager: tauri::State<'_, AgentRunManager>,
    run_id: RunId,
) -> Result<RunStateSnapshot, String> {
    manager
        .get_run_state(run_id)
        .await
        .map_err(|err| err.to_string())
}
