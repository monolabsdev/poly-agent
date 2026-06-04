use poly_agent_core::{ToolCall, ToolRisk};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApprovalPayload {
    pub approval_id: String,
    pub tool_name: String,
    pub risk: ToolRisk,
    pub reason: Option<String>,
    pub path: Option<String>,
    pub command_preview: Option<String>,
    pub diff_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<serde_json::Value>,
}

impl ApprovalPayload {
    pub(crate) fn from_call(call: &ToolCall, risk: ToolRisk, debug: bool) -> Self {
        let reason = string_arg(call, "reason");
        let path = string_arg(call, "path");
        let command_preview = string_arg(call, "command");
        let diff_preview = if call.name == "apply_patch" {
            build_apply_patch_preview(call)
        } else {
            None
        };

        Self {
            approval_id: call.id.clone(),
            tool_name: call.name.clone(),
            risk,
            reason,
            path,
            command_preview,
            diff_preview,
            raw_arguments: debug.then(|| call.arguments.clone()),
        }
    }
}

fn string_arg(call: &ToolCall, key: &str) -> Option<String> {
    call.arguments
        .get(key)
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

fn build_apply_patch_preview(call: &ToolCall) -> Option<String> {
    let old = string_arg(call, "expected_old_text")?;
    let new = string_arg(call, "replacement_text")?;
    Some(format!("--- expected\n+++ replacement\n-{}\n+{}", old, new))
}
