use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;

use crate::common::resolve_and_validate;

pub struct ReadFileTool;

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
}

#[async_trait::async_trait]
impl AgentTool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read the contents of a file within the workspace. This is read-only and does not modify any files. Large files are truncated."
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
                    "description": "Path to the file, relative to workspace root."
                }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: ReadFileArgs = serde_json::from_value(args)?;
        let target = resolve_and_validate(&ctx.workspace, &args.path)?;

        let metadata = tokio::fs::metadata(&target).await?;
        if !metadata.is_file() {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: format!("'{}' is not a file", args.path),
                is_error: true,
                cached: false,
            });
        }

        let max_bytes = ctx.limits.max_file_read_bytes;
        let file_size = metadata.len() as usize;

        let content = if file_size <= max_bytes {
            tokio::fs::read_to_string(&target).await?
        } else {
            // Read only up to the limit.
            let bytes = tokio::fs::read(&target).await?;
            let truncated = &bytes[..max_bytes.min(bytes.len())];
            let text = String::from_utf8_lossy(truncated).into_owned();
            format!(
                "{}\n\n... [truncated — file is {} bytes, limit is {} bytes]",
                text, file_size, max_bytes
            )
        };

        Ok(ToolResult {
            tool_call_id: String::new(),
            output: content,
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
    async fn read_file_respects_size_limit() {
        // Create a temp dir with a file larger than the limit.
        let dir = std::env::temp_dir().join("poly_agent_test_read");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("big.txt");
        let content = "A".repeat(1024);
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool;
        let ctx = ToolContext {
            workspace: dir.clone(),
            limits: RuntimeLimits {
                max_file_read_bytes: 100,
                ..RuntimeLimits::default()
            },
            cancellation: tokio_util::sync::CancellationToken::new(),
        };

        let result = tool
            .run(serde_json::json!({ "path": "big.txt" }), ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("[truncated"));
        // Output should be bounded.
        assert!(result.output.len() < 300);

        // Clean up.
        let _ = std::fs::remove_dir_all(&dir);
    }
}
