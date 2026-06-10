use crate::common::{is_ignored_dir, resolve_and_validate};
use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;

pub struct ApplyPatchTool;

#[derive(Deserialize)]
#[allow(dead_code)]
struct ApplyPatchArgs {
    path: String,
    #[serde(default)]
    expected_old_text: String,
    #[serde(default)]
    replacement_text: String,
    reason: String,
    /// Unified diff patch to apply. If provided, expected_old_text and replacement_text are ignored.
    #[serde(default)]
    patch: Option<String>,
}

#[async_trait::async_trait]
impl AgentTool for ApplyPatchTool {
    fn name(&self) -> &'static str {
        "apply_patch"
    }
    fn description(&self) -> &'static str {
        "Use this to actually modify a file. Requires approval. Supports two modes: (1) exact text replacement via expected_old_text/replacement_text, or (2) unified diff patch via the `patch` parameter. For unified diff, provide a standard diff/patch format with target file path. Never pass the whole file unless explicitly asked to replace the whole file."
    }
    fn risk(&self) -> ToolRisk {
        ToolRisk::RequiresApproval
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path within workspace"},
                "expected_old_text": {"type": "string", "description": "Exact text expected in file. Use an empty string only for append-only edits."},
                "replacement_text": {"type": "string", "description": "Text to replace with, or text to append when expected_old_text is empty."},
                "reason": {"type": "string", "description": "Why this change is needed"},
                "patch": {"type": "string", "description": "Unified diff patch to apply. Overrides expected_old_text/replacement_text when provided."}
            },
            "required": ["path", "reason"]
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
                cached: false,
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
                    cached: false,
                })
            }
        };

        // If patch parameter is provided, apply unified diff
        if let Some(patch) = &args.patch {
            return apply_unified_diff(&target, &current_content, patch).await;
        }

        // Fall back to exact text replacement
        if looks_like_whole_file_rewrite(&current_content, &args.expected_old_text) {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Rejected whole-file replacement. Use a minimal exact text replacement, append-only patch, or targeted line removal.".into(),
                is_error: true,
                cached: false,
            });
        }
        if args.expected_old_text.is_empty() {
            let temp_path = target.with_extension(".tmp");
            let separator = if current_content.is_empty()
                || current_content.ends_with('\n')
                || args.replacement_text.starts_with('\n')
            {
                ""
            } else {
                "\n"
            };
            let new_content = format!("{current_content}{separator}{}", args.replacement_text);
            tokio::fs::write(&temp_path, &new_content).await?;
            tokio::fs::rename(&temp_path, &target).await?;
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Appended".to_string(),
                is_error: false,
                cached: false,
            });
        }
        // Validate exact match count
        let count = current_content.matches(&args.expected_old_text).count();
        if count == 0 {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Text not found".into(),
                is_error: true,
                cached: false,
            });
        }
        if count > 1 {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Text appears multiple times".into(),
                is_error: true,
                cached: false,
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
            cached: false,
        })
    }
}

/// Apply a unified diff patch to the target file.
async fn apply_unified_diff(
    target: &std::path::Path,
    current_content: &str,
    patch: &str,
) -> anyhow::Result<ToolResult> {
    let mut result = current_content.to_string();

    // Split the patch into lines and find all hunks
    let lines: Vec<&str> = patch.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        // Skip header lines (--- a/..., +++ b/..., diff --git, index, etc.)
        if lines[i].starts_with("@@") {
            // Parse hunk header
            let header = lines[i];
            let parts: Vec<&str> = header.split(' ').collect();
            if parts.len() < 3 {
                i += 1;
                continue;
            }
            // Parse the old file range: -start,count
            let old_range = parts[1].trim_start_matches('-');
            let old_start: usize = old_range.split(',').next().unwrap_or("1").parse().unwrap_or(1);

            // Collect the hunk body
            i += 1;
            let mut hunk_context: Vec<String> = Vec::new();
            let mut hunk_old: Vec<String> = Vec::new();
            let mut hunk_new: Vec<String> = Vec::new();
            let mut in_hunk = false;

            while i < lines.len() && !lines[i].starts_with("@@") {
                let line = lines[i];
                if let Some(content) = line.strip_prefix('-') {
                    hunk_old.push(content.to_string());
                    in_hunk = true;
                } else if let Some(content) = line.strip_prefix('+') {
                    hunk_new.push(content.to_string());
                    in_hunk = true;
                } else if let Some(content) = line.strip_prefix(' ') {
                    // Context line - flush any pending change
                    if in_hunk {
                        // Apply current hunk
                        match apply_hunk(&result, &hunk_context, &hunk_old, &hunk_new, old_start) {
                            Ok(new_result) => result = new_result,
                            Err(e) => {
                                return Ok(ToolResult {
                                    tool_call_id: String::new(),
                                    output: format!("Failed to apply hunk: {}", e),
                                    is_error: true,
                                    cached: false,
                                });
                            }
                        }
                        hunk_old.clear();
                        hunk_new.clear();
                        hunk_context.clear();
                        in_hunk = false;
                    }
                    hunk_context.push(content.to_string());
                }
                // Skip diff headers that might appear mid-patch (single-file patch typically has just one)
                i += 1;
            }

            // Apply last hunk if any
            if in_hunk || !hunk_old.is_empty() || !hunk_new.is_empty() {
                match apply_hunk(&result, &hunk_context, &hunk_old, &hunk_new, old_start) {
                    Ok(new_result) => result = new_result,
                    Err(e) => {
                        return Ok(ToolResult {
                            tool_call_id: String::new(),
                            output: format!("Failed to apply hunk: {}", e),
                            is_error: true,
                            cached: false,
                        });
                    }
                }
            }
        } else {
            i += 1;
        }
    }

    // Write result atomically
    let temp_path = target.with_extension(".tmp");
    tokio::fs::write(&temp_path, &result).await?;
    tokio::fs::rename(&temp_path, target).await?;

    Ok(ToolResult {
        tool_call_id: String::new(),
        output: "Applied unified diff patch".to_string(),
        is_error: false,
        cached: false,
    })
}

/// Apply a single hunk: find context + old text in `content` and replace with context + new text.
fn apply_hunk(
    content: &str,
    context: &[String],
    old: &[String],
    new: &[String],
    _old_start: usize,
) -> Result<String, String> {
    // Build the search string: context lines + old lines (removals)
    let mut search = String::new();
    for line in context {
        search.push_str(line);
        search.push('\n');
    }
    for line in old {
        search.push_str(line);
        search.push('\n');
    }
    // Trim trailing newline for matching, but we always appended one
    if !search.is_empty() {
        // Remove trailing newline for comparison
        let search_trimmed = search.trim_end_matches('\n');

        // Build replacement: context lines + new lines (additions)
        let mut replacement = String::new();
        for line in context {
            replacement.push_str(line);
            replacement.push('\n');
        }
        for line in new {
            replacement.push_str(line);
            replacement.push('\n');
        }
        let replacement_trimmed = replacement.trim_end_matches('\n');

        // Find and replace in content
        if content.contains(search_trimmed) {
            let count = content.matches(search_trimmed).count();
            if count > 1 {
                return Err(format!(
                    "Text matches {} times (ambiguous hunk)",
                    count
                ));
            }
            if content.trim() == search_trimmed && content.lines().count() > 3 {
                return Err("Rejected whole-file replacement via diff".to_string());
            }
            return Ok(content.replace(search_trimmed, replacement_trimmed));
        }
    }

    // If no context, just try to match old lines directly
    if !old.is_empty() && context.is_empty() {
        let old_text = old.join("\n");
        if content.contains(&old_text) {
            let new_text = new.join("\n");
            let count = content.matches(&old_text).count();
            if count > 1 {
                return Err(format!("Old text matches {} times (ambiguous hunk)", count));
            }
            return Ok(content.replace(&old_text, &new_text));
        }
    }

    Err("Hunk context does not match file content".to_string())
}

fn looks_like_whole_file_rewrite(current: &str, expected_old: &str) -> bool {
    let current_trimmed = current.trim();
    let expected_trimmed = expected_old.trim();
    !current_trimmed.is_empty()
        && current_trimmed == expected_trimmed
        && current_trimmed.lines().count() > 3
}

#[cfg(test)]
mod tests {
    use super::{looks_like_whole_file_rewrite, ApplyPatchTool};
    use poly_agent_runtime::{AgentTool, ToolContext};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn detects_whole_file_rewrite() {
        assert!(looks_like_whole_file_rewrite("a\nb\nc\nd\n", "a\nb\nc\nd\n"));
    }

    #[test]
    fn allows_small_targeted_patch() {
        assert!(!looks_like_whole_file_rewrite("a\nb\nc\nd\n", "d\n"));
    }

    #[tokio::test]
    async fn empty_expected_old_text_appends() {
        let workspace = temp_workspace();
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("test.txt"), "one\n").await.unwrap();

        let result = ApplyPatchTool
            .run(
                serde_json::json!({
                    "path": "test.txt",
                    "expected_old_text": "",
                    "replacement_text": "two\n",
                    "reason": "append"
                }),
                ctx(workspace.clone()),
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(
            tokio::fs::read_to_string(workspace.join("test.txt"))
                .await
                .unwrap(),
            "one\ntwo\n"
        );
        let _ = tokio::fs::remove_dir_all(workspace).await;
    }

    #[tokio::test]
    async fn rejects_whole_file_replacement() {
        let workspace = temp_workspace();
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("test.txt"), "one\ntwo\nthree\nfour\n")
            .await
            .unwrap();

        let result = ApplyPatchTool
            .run(
                serde_json::json!({
                    "path": "test.txt",
                    "expected_old_text": "one\ntwo\nthree\nfour\n",
                    "replacement_text": "one\ntwo\nthree\n",
                    "reason": "remove last line"
                }),
                ctx(workspace.clone()),
            )
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.output.contains("Rejected whole-file replacement"));
        assert_eq!(
            tokio::fs::read_to_string(workspace.join("test.txt"))
                .await
                .unwrap(),
            "one\ntwo\nthree\nfour\n"
        );
        let _ = tokio::fs::remove_dir_all(workspace).await;
    }

    #[tokio::test]
    async fn applies_unified_diff_patch() {
        let workspace = temp_workspace();
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("test.txt"), "line one\nline two\nline three\n")
            .await
            .unwrap();

        let patch = "\
@@ -1,3 +1,3 @@
 line one
-line two
+line modified
 line three
";

        let result = ApplyPatchTool
            .run(
                serde_json::json!({
                    "path": "test.txt",
                    "patch": patch,
                    "reason": "modify line two"
                }),
                ctx(workspace.clone()),
            )
            .await
            .unwrap();

        assert!(!result.is_error, "Error: {}", result.output);
        assert_eq!(
            tokio::fs::read_to_string(workspace.join("test.txt"))
                .await
                .unwrap(),
            "line one\nline modified\nline three\n"
        );
        let _ = tokio::fs::remove_dir_all(workspace).await;
    }

    #[tokio::test]
    async fn rejects_patch_outside_workspace() {
        let workspace = temp_workspace();
        tokio::fs::create_dir_all(&workspace).await.unwrap();

        let result = ApplyPatchTool
            .run(
                serde_json::json!({
                    "path": "../outside.txt",
                    "expected_old_text": "old",
                    "replacement_text": "new",
                    "reason": "test"
                }),
                ctx(workspace.clone()),
            )
            .await;

        assert!(result.is_err() || result.unwrap().is_error);
        let _ = tokio::fs::remove_dir_all(workspace).await;
    }

    fn temp_workspace() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("poly-agent-apply-patch-test-{nonce}"))
    }

    fn ctx(workspace: std::path::PathBuf) -> ToolContext {
        ToolContext {
            workspace,
            limits: Default::default(),
            cancellation: tokio_util::sync::CancellationToken::new(),
        }
    }
}
