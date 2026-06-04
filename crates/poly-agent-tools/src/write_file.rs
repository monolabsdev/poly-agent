use crate::common::{is_ignored_dir, resolve_and_validate};
use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;

/// Placeholder — not auto-executed. Requires user approval.
pub struct WriteFileTool;

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

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

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: WriteFileArgs = serde_json::from_value(args)?;
        if is_ignored_dir(args.path.split(['/', '\\']).next().unwrap_or("")) {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Ignored directory".to_string(),
                is_error: true,
                cached: false,
            });
        }

        let target = resolve_and_validate(&ctx.workspace, &args.path)?;
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let temp_path = target.with_extension("tmp");
        tokio::fs::write(&temp_path, args.content).await?;
        tokio::fs::rename(&temp_path, &target).await?;

        Ok(ToolResult {
            tool_call_id: String::new(),
            output: format!("Wrote {}", args.path),
            is_error: false,
            cached: false,
        })
    }
}

#[cfg(test)]
#[path = "write_file_tests.rs"]
mod write_file_tests;
