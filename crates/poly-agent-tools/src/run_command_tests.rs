use std::path::PathBuf;

use poly_agent_core::RuntimeLimits;

use super::*;

fn test_ctx() -> ToolContext {
    ToolContext {
        workspace: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        limits: Default::default(),
        cancellation: tokio_util::sync::CancellationToken::new(),
    }
}

#[tokio::test]
async fn run_command_captures_stdout() {
    let tool = RunCommandTool;
    let ctx = test_ctx();
    let args = serde_json::json!({"command": "echo hello"});

    let result = tool.run(args, ctx).await.unwrap();
    assert!(!result.is_error);
    assert!(result.output.contains("hello"));
    assert!(result.output.contains("Exit code: 0"));
}

#[tokio::test]
async fn run_command_captures_stderr() {
    let tool = RunCommandTool;
    let ctx = test_ctx();
    let args = serde_json::json!({"command": "echo error >&2"});

    let result = tool.run(args, ctx).await.unwrap();
    assert!(result.output.contains("error"));
}

#[tokio::test]
async fn run_command_returns_nonzero_exit_code() {
    let tool = RunCommandTool;
    let ctx = test_ctx();
    let args = serde_json::json!({"command": "exit 1"});

    let result = tool.run(args, ctx).await.unwrap();
    assert!(result.is_error);
    assert!(result.output.contains("Exit code: 1"));
    assert!(result.output.contains("failed"));
}

#[tokio::test]
async fn run_command_empty_command() {
    let tool = RunCommandTool;
    let ctx = test_ctx();
    let args = serde_json::json!({"command": ""});

    let result = tool.run(args, ctx).await.unwrap();
    assert!(result.is_error);
    assert!(result.output.contains("empty command"));
}

#[tokio::test]
async fn run_command_cwd_within_workspace() {
    let tool = RunCommandTool;
    let ctx = test_ctx();
    #[cfg(windows)]
    let args = serde_json::json!({"command": "cd", "cwd": "."});
    #[cfg(not(windows))]
    let args = serde_json::json!({"command": "pwd", "cwd": "."});

    let result = tool.run(args, ctx).await.unwrap();
    assert!(!result.is_error);
    assert!(result.output.contains("Exit code: 0"));
}

#[tokio::test]
async fn run_command_cwd_outside_workspace_rejected() {
    let tool = RunCommandTool;
    let ctx = test_ctx();
    let args = serde_json::json!({"command": "pwd", "cwd": "../../../.."});

    let result = tool.run(args, ctx).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("outside the workspace"));
}

#[tokio::test]
async fn run_command_is_dangerous() {
    let tool = RunCommandTool;
    assert_eq!(tool.risk(), ToolRisk::Dangerous);
}

#[tokio::test]
async fn run_command_cancellation() {
    let tool = RunCommandTool;
    let ctx = ToolContext {
        workspace: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        limits: RuntimeLimits {
            command_timeout_secs: 1,
            ..Default::default()
        },
        cancellation: tokio_util::sync::CancellationToken::new(),
    };
    #[cfg(windows)]
    let args = serde_json::json!({"command": "ping -n 30 127.0.0.1"});
    #[cfg(not(windows))]
    let args = serde_json::json!({"command": "sleep 30"});

    let result = tool.run(args, ctx).await.unwrap();
    assert!(result.is_error);
    assert!(result.output.contains("timed out"));
}
