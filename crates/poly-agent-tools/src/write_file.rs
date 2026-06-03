use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};

/// Placeholder — not auto-executed. Requires user approval.
pub struct WriteFileTool;

#[async_trait::async_trait]
impl AgentTool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Write content to a file in the workspace. Requires user approval."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::RequiresApproval
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to workspace root."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write."
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        // This will not be called without prior approval.
        Ok(ToolResult {
            tool_call_id: String::new(),
            output: "write_file is not yet implemented".to_string(),
            is_error: true,
        })
    }
}
