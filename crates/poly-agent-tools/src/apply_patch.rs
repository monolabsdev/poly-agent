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
        "Use this to actually modify a file. Requires approval. Use for exact minimal text replacement edits. For append-only edits, set expected_old_text to an empty string and replacement_text to only the text to append. Never pass the whole file unless the user explicitly asked to replace the whole file."
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
