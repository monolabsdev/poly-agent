use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::SystemTime;

use poly_agent_core::{ModelConfig, RunId, RuntimeLimits};
use serde::{Deserialize, Serialize};

use crate::approval::ApprovalPayload;
use crate::events::AgentUiEvent;

pub const MAX_STORED_EVENTS: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    WaitingForApproval,
    Finished,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunInput {
    pub prompt: String,
    pub model: ModelConfig,
    #[serde(default)]
    pub workspace_path: Option<PathBuf>,
    #[serde(default)]
    pub limits: RuntimeLimits,
    #[serde(default)]
    pub debug: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentRunStats {
    pub events_seen: usize,
    pub model_calls: usize,
    pub tool_calls_requested: usize,
    pub tool_calls_finished: usize,
    pub approvals_requested: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStateSnapshot {
    pub run_id: RunId,
    pub status: RunStatus,
    pub workspace_root: Option<PathBuf>,
    pub local_tools_enabled: bool,
    pub events: Vec<AgentUiEvent>,
    pub pending_approval: Option<ApprovalPayload>,
    pub stats: AgentRunStats,
    pub final_output: Option<String>,
    pub last_error: Option<String>,
    pub started_at: SystemTime,
    pub finished_at: Option<SystemTime>,
}

#[derive(Debug)]
pub(crate) struct MutableRunState {
    pub run_id: RunId,
    pub runtime_run_id: Option<RunId>,
    pub status: RunStatus,
    pub workspace_root: Option<PathBuf>,
    pub local_tools_enabled: bool,
    pub events: VecDeque<AgentUiEvent>,
    pub pending_approval: Option<ApprovalPayload>,
    pub stats: AgentRunStats,
    pub final_output: Option<String>,
    pub last_error: Option<String>,
    pub started_at: SystemTime,
    pub finished_at: Option<SystemTime>,
}

impl MutableRunState {
    pub(crate) fn new(
        run_id: RunId,
        workspace_root: Option<PathBuf>,
        local_tools_enabled: bool,
    ) -> Self {
        Self {
            run_id,
            runtime_run_id: None,
            status: RunStatus::Running,
            workspace_root,
            local_tools_enabled,
            events: VecDeque::with_capacity(MAX_STORED_EVENTS),
            pending_approval: None,
            stats: AgentRunStats::default(),
            final_output: None,
            last_error: None,
            started_at: SystemTime::now(),
            finished_at: None,
        }
    }

    pub(crate) fn snapshot(&self) -> RunStateSnapshot {
        RunStateSnapshot {
            run_id: self.run_id,
            status: self.status,
            workspace_root: self.workspace_root.clone(),
            local_tools_enabled: self.local_tools_enabled,
            events: self.events.iter().cloned().collect(),
            pending_approval: self.pending_approval.clone(),
            stats: self.stats.clone(),
            final_output: self.final_output.clone(),
            last_error: self.last_error.clone(),
            started_at: self.started_at,
            finished_at: self.finished_at,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentRunError {
    #[error("workspace path does not exist or is not a directory: {0}")]
    InvalidWorkspace(String),

    #[error("run not found: {0}")]
    RunNotFound(RunId),

    #[error("run has no pending approval: {0}")]
    ApprovalNotFound(RunId),

    #[error("{0}")]
    Other(String),
}
