use crate::common::{is_ignored_dir, resolve_and_validate};
use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;

pub struct ApplyPatchTool;

#[derive(Deserialize)]
#[allow(dead_code)]
struct ApplyPatchArgs {
    path: String,
    expected_old_text: String,
    replacement_text: String,
    reason: String,
}

#[async_trait::async_trait]
impl AgentTool for ApplyPatchTool {
    fn name(&self) -> &'static str {
        "apply_patch"
    }
    fn description(&self) -> &'static str {
        "Use this to actually modify a file. Requires approval. Use for exact text replacement edits."
    }
    fn risk(&self) -> ToolRisk {
        ToolRisk::RequiresApproval
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path within workspace"},
                "expected_old_text": {"type": "string", "description": "Exact text expected in file"},
                "replacement_text": {"type": "string", "description": "Text to replace with"},
                "reason": {"type": "string", "description": "Why this change is needed"}
            },
            "required": ["path", "expected_old_text", "replacement_text", "reason"]
        })
    }
    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: ApplyPatchArgs = serde_json::from_value(args)?;
        // Validate path
        if is_ignored_dir(args.path.split('/').next().unwrap_or("")) {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Ignored directory".into(),
                is_error: true,
            });
        }
        let target = resolve_and_validate(&ctx.workspace, &args.path)?;
        // Read current content
        let current_content = match tokio::fs::read_to_string(&target).await {
            Ok(c) => c,
            Err(_) => {
                return Ok(ToolResult {
                    tool_call_id: String::new(),
                    output: "File not found".into(),
                    is_error: true,
                })
            }
        };
        // Validate exact match count
        let count = current_content.matches(&args.expected_old_text).count();
        if count == 0 {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Text not found".into(),
                is_error: true,
            });
        }
        if count > 1 {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Text appears multiple times".into(),
                is_error: true,
            });
        }
        // Atomic write approach
        let temp_path = target.with_extension(".tmp");
        let new_content = current_content.replace(&args.expected_old_text, &args.replacement_text);
        tokio::fs::write(&temp_path, &new_content).await?;
        tokio::fs::rename(&temp_path, &target).await?;
        Ok(ToolResult {
            tool_call_id: String::new(),
            output: "Applied".to_string(),
            is_error: false,
        })
    }
}
