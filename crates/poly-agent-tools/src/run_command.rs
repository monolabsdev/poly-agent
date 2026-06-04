use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};

/// Placeholder — blocked by default. Dangerous.
pub struct RunCommandTool;

#[async_trait::async_trait]
impl AgentTool for RunCommandTool {
    fn name(&self) -> &'static str {
        "run_command"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in the workspace. Blocked by default — dangerous."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::Dangerous
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute."
                }
            },
            "required": ["command"]
        })
    }

    async fn run(&self, _args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        // This will not be called without prior approval.
        Ok(ToolResult {
            tool_call_id: String::new(),
            output: "run_command is not yet implemented".to_string(),
            is_error: true,
            cached: false,
        })
    }
}
