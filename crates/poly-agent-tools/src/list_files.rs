use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;
use walkdir::WalkDir;

use crate::common::{is_ignored_dir, resolve_and_validate};

pub struct ListFilesTool;

#[derive(Deserialize)]
struct ListFilesArgs {
    /// Path relative to workspace root. Defaults to ".".
    #[serde(default = "default_path")]
    path: String,
}

fn default_path() -> String {
    ".".to_string()
}

#[async_trait::async_trait]
impl AgentTool for ListFilesTool {
    fn name(&self) -> &'static str {
        "list_files"
    }

    fn description(&self) -> &'static str {
        "List files and directories at the given path within the workspace. Returns a compact listing."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::Safe
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to workspace root. Defaults to '.'."
                }
            }
        })
    }

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: ListFilesArgs = serde_json::from_value(args)?;
        let target = resolve_and_validate(&ctx.workspace, &args.path)?;

        let mut entries = Vec::new();
        let walker = WalkDir::new(&target)
            .max_depth(3)
            .into_iter()
            .filter_entry(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| !is_ignored_dir(s))
                    .unwrap_or(true)
            });

        for entry in walker.flatten() {
            if let Ok(rel) = entry.path().strip_prefix(&target) {
                let rel_str = rel.to_string_lossy();
                if rel_str.is_empty() {
                    continue;
                }
                let suffix = if entry.file_type().is_dir() { "/" } else { "" };
                entries.push(format!("{rel_str}{suffix}"));
            }
        }

        let output = if entries.is_empty() {
            "(empty directory)".to_string()
        } else {
            entries.join("\n")
        };

        Ok(ToolResult {
            tool_call_id: String::new(),
            output,
            is_error: false,
            cached: false,
        })
    }
}
