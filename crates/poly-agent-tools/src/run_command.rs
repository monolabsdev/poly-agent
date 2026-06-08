use std::path::PathBuf;
use std::time::Instant;

use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};

/// Executes a shell command inside the workspace.
/// Requires approval (ToolRisk::Dangerous).
pub struct RunCommandTool;

#[async_trait::async_trait]
impl AgentTool for RunCommandTool {
    fn name(&self) -> &'static str {
        "run_command"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in the workspace. Requires approval."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::Dangerous
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory override. Must be within the workspace."
                }
            },
            "required": ["command"]
        })
    }

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required argument: command"))?;

        if command.trim().is_empty() {
            return Ok(ToolResult {
                tool_call_id: String::new(),
                output: "Error: empty command".to_string(),
                is_error: true,
                cached: false,
            });
        }

        let cwd = resolve_cwd(&ctx, args.get("cwd").and_then(|v| v.as_str()))?;

        let timeout = std::time::Duration::from_secs(ctx.limits.command_timeout_secs);
        let start = Instant::now();

        #[cfg(windows)]
        let mut cmd = {
            let mut c = std::process::Command::new("cmd");
            c.arg("/C").arg(command);
            c
        };

        #[cfg(not(windows))]
        let mut cmd = {
            let mut c = std::process::Command::new("sh");
            c.arg("-c").arg(command);
            c
        };

        cmd.current_dir(&cwd);

        let child = tokio::process::Command::from(cmd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn command '{}': {}", command, e))?;

        let output = tokio::time::timeout(timeout, child.wait_with_output()).await;
        let duration_ms = start.elapsed().as_millis() as u64;

        match output {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);
                let success = output.status.success();

                let result_output = format_command_output(
                    command,
                    &cwd,
                    exit_code,
                    &stdout,
                    &stderr,
                    duration_ms,
                    success,
                );

                Ok(ToolResult {
                    tool_call_id: String::new(),
                    output: result_output,
                    is_error: !success,
                    cached: false,
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                tool_call_id: String::new(),
                output: format!("Command '{}' failed to complete: {}", command, e),
                is_error: true,
                cached: false,
            }),
            Err(_) => {
                // Timeout expired — the kill_on_drop(true) will terminate the process.
                Ok(ToolResult {
                    tool_call_id: String::new(),
                    output: format!(
                        "Command '{}' timed out after {} seconds",
                        command, ctx.limits.command_timeout_secs
                    ),
                    is_error: true,
                    cached: false,
                })
            }
        }
    }
}

fn resolve_cwd(ctx: &ToolContext, cwd_override: Option<&str>) -> anyhow::Result<PathBuf> {
    match cwd_override {
        Some(override_path) => {
            let requested = PathBuf::from(override_path);
            let resolved = if requested.is_absolute() {
                requested
            } else {
                ctx.workspace.join(&requested)
            };

            // Normalize the path by resolving . and .. components without
            // touching the filesystem (avoids Windows canonicalize issues).
            let normalised = normalize_path(&resolved);

            // Verify it's within the workspace.
            if !normalised.starts_with(&ctx.workspace) {
                return Err(anyhow::anyhow!(
                    "Working directory '{}' resolves outside the workspace (workspace: {})",
                    override_path,
                    ctx.workspace.display()
                ));
            }

            Ok(normalised)
        }
        None => Ok(ctx.workspace.clone()),
    }
}

/// Resolve `.` and `..` components in a path without filesystem access.
fn normalize_path(path: &std::path::Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                components.pop();
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}

fn format_command_output(
    command: &str,
    cwd: &std::path::Path,
    exit_code: i32,
    stdout: &str,
    stderr: &str,
    duration_ms: u64,
    success: bool,
) -> String {
    let status = if success { "success" } else { "failed" };
    let mut output = format!(
        "Command: {}\nWorking directory: {}\nExit code: {}\nStatus: {}\nDuration: {}ms",
        command,
        cwd.display(),
        exit_code,
        status,
        duration_ms,
    );

    if !stdout.is_empty() {
        output.push_str("\n\nStdout:\n");
        output.push_str(stdout);
    }

    if !stderr.is_empty() {
        output.push_str("\n\nStderr:\n");
        output.push_str(stderr);
    }

    output
}

#[cfg(test)]
#[path = "run_command_tests.rs"]
mod run_command_tests;
