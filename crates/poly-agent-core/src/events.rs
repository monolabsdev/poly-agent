use serde::{Deserialize, Serialize};

use crate::{RunId, ToolCall, ToolResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AgentEvent {
    Started {
        run_id: RunId,
    },
    ModelCallStarted {
        run_id: RunId,
        step: usize,
    },
    ModelCallFinished {
        run_id: RunId,
        step: usize,
    },
    ToolCallRequested {
        run_id: RunId,
        call: ToolCall,
    },
    ToolCallStarted {
        run_id: RunId,
        tool_call_id: String,
        tool_name: String,
    },
    ToolCallFinished {
        run_id: RunId,
        result: ToolResult,
    },
    ApprovalRequired {
        run_id: RunId,
        call: ToolCall,
    },
    StepLimitReached {
        run_id: RunId,
        max_steps: usize,
    },
    UnknownToolRequested {
        run_id: RunId,
        tool_name: String,
    },
    Finished {
        run_id: RunId,
        text: String,
    },
    Error {
        run_id: RunId,
        error: String,
    },
}
