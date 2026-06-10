use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;

pub struct GlobFilesTool;

#[derive(Deserialize)]
struct GlobFilesArgs {
    /// Glob pattern to match files against.
    pattern: String,
}

#[async_trait::async_trait]
impl AgentTool for GlobFilesTool {
    fn name(&self) -> &'static str {
        "glob_files"
    }

    fn description(&self) -> &'static str {
        "List files matching a glob pattern within the workspace. Returns relative paths of matching files."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::Safe
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files against. Paths are relative to workspace root."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: GlobFilesArgs = serde_json::from_value(args)?;

        // Resolve the pattern relative to workspace
        let pattern = if std::path::Path::new(&args.pattern).is_absolute() {
            args.pattern.clone()
        } else {
            ctx.workspace.join(&args.pattern).to_string_lossy().to_string()
        };

        let mut results = Vec::new();

        // Use glob to find matching files
        if let Ok(paths) = glob::glob(&pattern) {
            for entry in paths {
                match entry {
                    Ok(path) => {
                        if path.is_file() {
                            if let Ok(rel_path) = path.strip_prefix(&ctx.workspace) {
                                results.push(rel_path.to_string_lossy().to_string());
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        let output = if results.is_empty() {
            format!("No files found matching pattern '{}'", args.pattern)
        } else {
            results.join("\n")
        };

        Ok(ToolResult {
            tool_call_id: String::new(),
            output,
            is_error: false,
            cached: false,
        })
    }
}
