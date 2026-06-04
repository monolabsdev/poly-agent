use std::collections::BTreeMap;

use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;

use crate::common::resolve_and_validate;

pub struct ReadImportantFilesTool;

#[derive(Deserialize)]
struct ReadImportantFilesArgs {
    /// Specific paths to read. If omitted, read auto-detected important files.
    #[serde(default)]
    paths: Vec<String>,
    /// Maximum total bytes across all files read.
    #[serde(default = "default_max_total_bytes")]
    max_total_bytes: usize,
}

fn default_max_total_bytes() -> usize {
    256 * 1024
}

/// Default important file patterns to look for.
const IMPORTANT_FILE_PATTERNS: &[&str] = &[
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "Cargo.toml",
];

#[async_trait::async_trait]
impl AgentTool for ReadImportantFilesTool {
    fn name(&self) -> &'static str {
        "read_important_files"
    }

    fn description(&self) -> &'static str {
        "Read important project files (README, Cargo.toml, etc.) with size limits. If paths omitted, auto-detects files from workspace structure. Returns compact JSON with content previews."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::Safe
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific file paths to read (relative to workspace). If empty, auto-detects important files."
                },
                "max_total_bytes": {
                    "type": "integer",
                    "description": "Maximum total bytes across all files read. Default 256KB."
                }
            }
        })
    }

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: ReadImportantFilesArgs = serde_json::from_value(args)?;
        let workspace = &ctx.workspace;

        let files_to_read: Vec<String> = if args.paths.is_empty() {
            discover_important_files(workspace)
        } else {
            args.paths
        };

        let max_total = args.max_total_bytes;
        let mut total_bytes = 0usize;
        let mut truncated = false;
        let mut file_results: BTreeMap<String, serde_json::Value> = BTreeMap::new();

        for path_str in &files_to_read {
            if total_bytes >= max_total {
                truncated = true;
                continue;
            }

            let target = match resolve_and_validate(workspace, path_str) {
                Ok(p) => p,
                Err(_) => {
                    file_results.insert(
                        path_str.clone(),
                        serde_json::json!({"error": "path invalid or outside workspace"}),
                    );
                    continue;
                }
            };

            if !target.exists() {
                file_results.insert(
                    path_str.clone(),
                    serde_json::json!({"error": "file not found"}),
                );
                continue;
            }

            let metadata = match tokio::fs::metadata(&target).await {
                Ok(m) => m,
                Err(e) => {
                    file_results.insert(
                        path_str.clone(),
                        serde_json::json!({"error": format!("cannot read: {e}")}),
                    );
                    continue;
                }
            };

            if !metadata.is_file() {
                file_results.insert(
                    path_str.clone(),
                    serde_json::json!({"error": "not a file"}),
                );
                continue;
            }

            let file_size = metadata.len() as usize;

            // Per-file cap: at most half of max_total.
            let per_file_max = (max_total / 2).max(4096);
            let read_limit = per_file_max.min(max_total.saturating_sub(total_bytes));

            let content = if file_size <= read_limit {
                match tokio::fs::read_to_string(&target).await {
                    Ok(c) => {
                        total_bytes += c.len();
                        c
                    }
                    Err(e) => {
                        file_results.insert(
                            path_str.clone(),
                            serde_json::json!({"error": format!("read error: {e}")}),
                        );
                        continue;
                    }
                }
            } else {
                let bytes = tokio::fs::read(&target).await?;
                let truncated_bytes = &bytes[..read_limit.min(bytes.len())];
                total_bytes += truncated_bytes.len();
                let text = String::from_utf8_lossy(truncated_bytes).into_owned();
                format!(
                    "{}\n\n... [truncated — file is {} bytes, limit is {} bytes]",
                    text, file_size, read_limit
                )
            };

            let content_preview: String = content.chars().take(2000).collect();
            let is_truncated = file_size > read_limit || content.len() > 2000;

            file_results.insert(
                path_str.clone(),
                serde_json::json!({
                    "content_preview": content_preview,
                    "bytes": file_size,
                    "truncated": is_truncated,
                }),
            );

            if total_bytes >= max_total {
                truncated = true;
                break;
            }
        }

        let output = serde_json::json!({
            "files": file_results,
            "total_bytes": total_bytes,
            "truncated": truncated,
        });

        Ok(ToolResult {
            tool_call_id: String::new(),
            output: serde_json::to_string_pretty(&output)?,
            is_error: false,
            cached: false,
        })
    }
}

/// Discover important files at the workspace root by checking common patterns.
fn discover_important_files(workspace: &std::path::Path) -> Vec<String> {
    let mut files = Vec::new();

    for pattern in IMPORTANT_FILE_PATTERNS {
        let candidate = workspace.join(*pattern);
        if candidate.exists() {
            files.push(pattern.to_string());
        }
    }

    // Cargo.toml at workspace root.
    let root_cargo = workspace.join("Cargo.toml");
    if root_cargo.exists() && !files.contains(&"Cargo.toml".to_string()) {
        files.push("Cargo.toml".to_string());
    }

    // package.json at workspace root.
    let pkg_json = workspace.join("package.json");
    if pkg_json.exists() {
        files.push("package.json".to_string());
    }

    // src/main.rs, src/lib.rs at workspace root.
    for sub in &["src/main.rs", "src/lib.rs"] {
        let candidate = workspace.join(sub);
        if candidate.exists() {
            files.push(sub.to_string());
        }
    }

    // crates/*/Cargo.toml, crates/*/src/lib.rs, crates/*/src/main.rs
    let crates_dir = workspace.join("crates");
    if crates_dir.is_dir() {
        if let Ok(rd) = std::fs::read_dir(&crates_dir) {
            let mut crate_dirs: Vec<_> = rd
                .flatten()
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .collect();
            crate_dirs.sort_by_key(|e| e.file_name());
            for entry in crate_dirs {
                let name = entry.file_name().to_string_lossy().to_string();
                let cargo = entry.path().join("Cargo.toml");
                if cargo.exists() {
                    files.push(format!("crates/{name}/Cargo.toml"));
                }
                for sub in &["src/lib.rs", "src/main.rs"] {
                    let candidate = entry.path().join(sub);
                    if candidate.exists() {
                        files.push(format!("crates/{name}/{sub}"));
                    }
                }
            }
        }
    }

    // examples/*/Cargo.toml, examples/*/src/main.rs
    let examples_dir = workspace.join("examples");
    if examples_dir.is_dir() {
        if let Ok(rd) = std::fs::read_dir(&examples_dir) {
            let mut ex_dirs: Vec<_> = rd
                .flatten()
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .collect();
            ex_dirs.sort_by_key(|e| e.file_name());
            for entry in ex_dirs {
                let name = entry.file_name().to_string_lossy().to_string();
                let cargo = entry.path().join("Cargo.toml");
                if cargo.exists() {
                    files.push(format!("examples/{name}/Cargo.toml"));
                }
                let main = entry.path().join("src/main.rs");
                if main.exists() {
                    files.push(format!("examples/{name}/src/main.rs"));
                }
            }
        }
    }

    // AGENTS.md / agents.md (checked but not already in patterns)
    let agents = workspace.join("AGENTS.md");
    if agents.exists() && !files.contains(&"AGENTS.md".to_string()) {
        files.push("AGENTS.md".to_string());
    }

    files
}
