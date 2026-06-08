use poly_agent_core::RunId;
use std::sync::Arc;
use tauri::{Emitter, Manager};

use crate::events::{AgentEventSink, AgentUiEvent, POLY_AGENT_EVENT};
use crate::manager::AgentRunManager;
use crate::types::{AgentRunInput, RunStateSnapshot};
use crate::workspace::{resolve_workspace_selection, sandbox_root};

pub fn tauri_event_sink(app: tauri::AppHandle) -> AgentEventSink {
    Arc::new(move |event: AgentUiEvent| {
        let _ = app.emit(POLY_AGENT_EVENT, event);
    })
}

#[tauri::command]
pub async fn agent_run(
    app_handle: tauri::AppHandle,
    manager: tauri::State<'_, AgentRunManager>,
    mut input: AgentRunInput,
) -> Result<RunId, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|err| err.to_string())?;
    resolve_workspace_selection(&mut input, &app_data_dir).map_err(|err| err.to_string())?;
    manager
        .start_run(input)
        .await
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub async fn agent_delete_chat_sandbox(
    app_handle: tauri::AppHandle,
    chat_id: String,
) -> Result<(), String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|err| err.to_string())?;
    let root = sandbox_root(&app_data_dir, &chat_id).map_err(|err| err.to_string())?;
    if root.exists() {
        tokio::fs::remove_dir_all(root)
            .await
            .map_err(|err| err.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub async fn agent_cancel(
    manager: tauri::State<'_, AgentRunManager>,
    run_id: RunId,
) -> Result<(), String> {
    manager
        .cancel_run(run_id)
        .await
        .map_err(|err| err.to_string())
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
