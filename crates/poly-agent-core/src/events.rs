use serde::{Deserialize, Serialize};

use crate::{AutoApproveReason, AutoReviewRisk, ReviewDecision, RunId, ToolCall, ToolResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AgentEvent {
    Started {
        run_id: RunId,
    },
    Activity {
        run_id: RunId,
        phase: String,
        title: String,
        summary: String,
        details: Vec<String>,
        status: ActivityStatus,
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
    TextDelta {
        run_id: RunId,
        text: String,
    },
    FinalResponseDelta {
        run_id: RunId,
        text: String,
    },
    ToolCallDelta {
        run_id: RunId,
        tool_call_id: String,
        delta: String,
    },
    /// The runtime auto-approved a tool call without asking the user.
    ToolAutoApproved {
        run_id: RunId,
        tool_call_id: String,
        tool_name: String,
        reason: AutoApproveReason,
    },
    /// The runtime denied a tool call (currently only from AutoReview high risk).
    ToolAutoDenied {
        run_id: RunId,
        tool_call_id: String,
        tool_name: String,
        reason: String,
    },
    /// The auto-reviewer classified a tool call. Always emitted in AutoReview
    /// mode before the corresponding Approve/Ask/Deny action is taken.
    AutoReviewDecision {
        run_id: RunId,
        tool_call_id: String,
        risk: AutoReviewRisk,
        decision: ReviewDecision,
        reason: String,
    },
    Finished {
        run_id: RunId,
        text: String,
    },
    Error {
        run_id: RunId,
        error: String,
    },
    Cancelled {
        run_id: RunId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityStatus {
    Running,
    Completed,
    Failed,
    Waiting,
}
