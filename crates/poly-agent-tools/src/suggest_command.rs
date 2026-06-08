use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};

/// Suggests a command for the user to run manually or approve.
/// Does not execute anything.
pub struct SuggestCommandTool;

#[async_trait::async_trait]
impl AgentTool for SuggestCommandTool {
    fn name(&self) -> &'static str {
        "suggest_command"
    }

    fn description(&self) -> &'static str {
        "Suggest a shell command for the user to run or approve. Does not execute anything."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::Safe
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to suggest."
                },
                "explanation": {
                    "type": "string",
                    "description": "Why this command is needed and what it does."
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Estimated risk level of the command."
                },
                "expected_outcome": {
                    "type": "string",
                    "description": "What the command should produce or accomplish."
                }
            },
            "required": ["command", "explanation"]
        })
    }

    async fn run(&self, args: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        let explanation = args
            .get("explanation")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let risk_level = args
            .get("risk_level")
            .and_then(|v| v.as_str())
            .unwrap_or("medium");
        let expected_outcome = args
            .get("expected_outcome")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let output = serde_json::json!({
            "command": command,
            "explanation": explanation,
            "risk_level": risk_level,
            "expected_outcome": expected_outcome,
            "type": "command_suggestion"
        });

        Ok(ToolResult {
            tool_call_id: String::new(),
            output: output.to_string(),
            is_error: false,
            cached: false,
        })
    }
}

#[cfg(test)]
#[path = "suggest_command_tests.rs"]
mod suggest_command_tests;
