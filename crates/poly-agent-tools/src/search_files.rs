use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;
use walkdir::WalkDir;

use crate::common::{is_ignored_dir, resolve_and_validate};

pub struct SearchFilesTool;

#[derive(Deserialize)]
struct SearchFilesArgs {
    /// Text pattern to search for.
    query: String,
    /// Path relative to workspace root. Defaults to ".".
    #[serde(default = "default_path")]
    path: String,
}

fn default_path() -> String {
    ".".to_string()
}

#[async_trait::async_trait]
impl AgentTool for SearchFilesTool {
    fn name(&self) -> &'static str {
        "search_files"
    }

    fn description(&self) -> &'static str {
        "Search for a text pattern across files in the workspace. Returns matching lines with file paths and line numbers."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::Safe
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text pattern to search for (case-sensitive substring match)."
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in, relative to workspace root. Defaults to '.'."
                }
            },
            "required": ["query"]
        })
    }

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: SearchFilesArgs = serde_json::from_value(args)?;
        let target = resolve_and_validate(&ctx.workspace, &args.path)?;

        let max_results = ctx.limits.max_search_results;
        let max_read = ctx.limits.max_file_read_bytes;
        let mut results = Vec::new();

        let walker = WalkDir::new(&target).into_iter().filter_entry(|e| {
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
                if line.contains(&args.query) {
                    results.push(format!(
                        "{}:{}:{}",
                        rel_path.display(),
                        line_num + 1,
                        line.chars().take(200).collect::<String>()
                    ));

                    if results.len() >= max_results {
                        results.push(format!("... [stopped at {max_results} results]"));
                        break 'outer;
                    }
                }
            }
        }

        let output = if results.is_empty() {
            format!("No matches found for '{}'", args.query)
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

#[cfg(test)]
mod tests {
    use super::*;
    use poly_agent_core::RuntimeLimits;

    #[tokio::test]
    async fn search_respects_max_results() {
        let dir = std::env::temp_dir().join("poly_agent_test_search");
        let _ = std::fs::create_dir_all(&dir);

        // Create a file with many matching lines.
        let content: String = (0..100).map(|i| format!("match line {i}\n")).collect();
        std::fs::write(dir.join("many_matches.txt"), &content).unwrap();

        let tool = SearchFilesTool;
        let ctx = ToolContext {
            workspace: dir.clone(),
            limits: RuntimeLimits {
                max_search_results: 5,
                ..RuntimeLimits::default()
            },
            cancellation: tokio_util::sync::CancellationToken::new(),
        };

        let result = tool
            .run(
                serde_json::json!({ "query": "match line", "path": "." }),
                ctx,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        let lines: Vec<&str> = result.output.lines().collect();
        // 5 results + 1 truncation message.
        assert_eq!(lines.len(), 6);
        assert!(lines.last().unwrap().contains("stopped at 5 results"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
