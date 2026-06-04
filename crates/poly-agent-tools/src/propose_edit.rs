use crate::common::{is_ignored_dir, resolve_and_validate};
use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;
use similar::{ChangeTag, TextDiff};

pub struct ProposeEditTool;

#[derive(Deserialize)]
struct ProposeEditArgs {
    path: String,
    new_content: String,
    reason: String,
}

#[async_trait::async_trait]
impl AgentTool for ProposeEditTool {
    fn name(&self) -> &'static str {
        "propose_edit"
    }
    fn description(&self) -> &'static str {
        "Use this when you want to preview a file change without writing it."
    }
    fn risk(&self) -> ToolRisk {
        ToolRisk::Safe
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path within workspace"},
                "new_content": {"type": "string", "description": "Full new file content"},
                "reason": {"type": "string", "description": "Why this change is needed"}
            },
            "required": ["path", "new_content", "reason"]
        })
    }
    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: ProposeEditArgs = serde_json::from_value(args)?;
        // Validate path
        if is_ignored_dir(args.path.split('/').next().unwrap_or("")) {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Ignored directory".into(),
                is_error: true,
            });
        }
        let target = resolve_and_validate(&ctx.workspace, &args.path)?;
        // Read old content if exists
        let old_content = match tokio::fs::read_to_string(&target).await {
            Ok(c) => c,
            Err(_) => String::new(),
        };
        // Generate unified diff (compact)
        let diff = TextDiff::configure()
            .algorithm(similar::Algorithm::Patience)
            .diff_lines(&old_content, &args.new_content);
        let mut diff_str = String::new();
        for change in diff.iter_all_changes() {
            let prefix = match change.tag() {
                ChangeTag::Delete => "-",
                ChangeTag::Insert => "+",
                ChangeTag::Equal => " ",
            };
            diff_str.push_str(&format!("{}{}\n", prefix, change.value()));
        }
        // Truncate if too large
        let max_bytes = ctx.limits.max_tool_output_bytes;
        let truncated = diff_str.len() > max_bytes;
        if truncated {
            diff_str.truncate(max_bytes);
        }
        let output = serde_json::json!({
            "path": args.path,
            "reason": args.reason,
            "exists": !old_content.is_empty(),
            "diff_preview": diff_str,
            "old_bytes": old_content.len(),
            "new_bytes": args.new_content.len(),
            "truncated": truncated
        })
        .to_string();
        Ok(ToolResult {
            tool_call_id: String::new(),
            output,
            is_error: false,
        })
    }
}
