use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;
use regex::Regex;

use crate::common::{is_ignored_dir, resolve_and_validate};

pub struct GrepFilesTool;

#[derive(Deserialize)]
struct GrepFilesArgs {
    /// Regular expression pattern to search for.
    pattern: String,
    /// Path relative to workspace root. Defaults to ".".
    #[serde(default = "default_path")]
    path: String,
    /// Maximum number of results to return. Defaults to 50.
    #[serde(default = "default_max_results")]
    max_results: u32,
}

fn default_path() -> String {
    ".".to_string()
}

fn default_max_results() -> u32 {
    50
}

#[async_trait::async_trait]
impl AgentTool for GrepFilesTool {
    fn name(&self) -> &'static str {
        "grep_files"
    }

    fn description(&self) -> &'static str {
        "Search for a regular expression pattern across files in the workspace. Returns matching lines with file paths and line numbers."
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
                    "description": "Regular expression pattern to search for."
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in, relative to workspace root. Defaults to '.'."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 50."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: GrepFilesArgs = serde_json::from_value(args)?;
        let target = resolve_and_validate(&ctx.workspace, &args.path)?;

        let max_results = args.max_results as usize;
        let max_read = ctx.limits.max_file_read_bytes;
        let mut results = Vec::new();

        // Compile the regex pattern
        let pattern = match Regex::new(&args.pattern) {
            Ok(re) => re,
            Err(e) => {
                return Ok(ToolResult {
                    tool_call_id: String::new(),
                    output: format!("Invalid regex pattern: {}", e),
                    is_error: true,
                    cached: false,
                });
            }
        };

        let walker = walkdir::WalkDir::new(&target).into_iter().filter_entry(|e| {
            e.file_name()
                .to_str()
                .map(|s| !is_ignored_dir(s))
                .unwrap_or(true)
        });

        'outer: for entry in walker.flatten() {
            if !entry.file_type().is_file() {
                continue;
            }

            // Skip files that are too large or likely binary.
            if let Ok(meta) = entry.metadata() {
                if meta.len() as usize > max_read {
                    continue;
                }
            }

            let content = match std::fs::read_to_string(entry.path()) {
                Ok(c) => c,
                Err(_) => continue, // Skip binary / unreadable files.
            };

            let rel_path = entry
                .path()
                .strip_prefix(&ctx.workspace)
                .unwrap_or(entry.path());

            for (line_num, line) in content.lines().enumerate() {
                if pattern.is_match(line) {
                    results.push(serde_json::json!({
                        "file": rel_path.to_string_lossy().to_string(),
                        "line_number": line_num + 1,
                        "line_content": line.chars().take(500).collect::<String>()
                    }));

                    if results.len() >= max_results {
                        results.push(serde_json::json!({
                            "file": "",
                            "line_number": 0,
                            "line_content": format!("... [stopped at {} results]", max_results)
                        }));
                        break 'outer;
                    }
                }
            }
        }

        let output = if results.is_empty() {
            format!("No matches found for pattern '{}'", args.pattern)
        } else {
            serde_json::to_string(&results)?
        };

        Ok(ToolResult {
            tool_call_id: String::new(),
            output,
            is_error: false,
            cached: false,
        })
    }
}
