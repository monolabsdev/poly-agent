use poly_agent_core::{AgentError, RunId};

use crate::engine::AgentRuntime;

impl AgentRuntime {
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
}
