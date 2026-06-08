use serde_json::json;

use super::*;

#[tokio::test]
async fn suggest_command_returns_structured_output() {
    let tool = SuggestCommandTool;
    let ctx = ToolContext {
        workspace: std::path::PathBuf::from("/tmp/test"),
        limits: Default::default(),
        cancellation: tokio_util::sync::CancellationToken::new(),
    };
    let args = json!({
        "command": "bun install",
        "explanation": "Install workspace dependencies",
        "risk_level": "medium",
        "expected_outcome": "node_modules populated, lockfile updated"
    });

    let result = tool.run(args, ctx).await.unwrap();
    assert!(!result.is_error);

    let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
    assert_eq!(parsed["command"], "bun install");
    assert_eq!(parsed["explanation"], "Install workspace dependencies");
    assert_eq!(parsed["risk_level"], "medium");
    assert_eq!(
        parsed["expected_outcome"],
        "node_modules populated, lockfile updated"
    );
    assert_eq!(parsed["type"], "command_suggestion");
}

#[tokio::test]
async fn suggest_command_defaults_optional_fields() {
    let tool = SuggestCommandTool;
    let ctx = ToolContext {
        workspace: std::path::PathBuf::from("/tmp/test"),
        limits: Default::default(),
        cancellation: tokio_util::sync::CancellationToken::new(),
    };
    let args = json!({
        "command": "ls -la",
        "explanation": "List files"
    });

    let result = tool.run(args, ctx).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
    assert_eq!(parsed["risk_level"], "medium");
    assert_eq!(parsed["expected_outcome"], "");
}

#[test]
fn suggest_command_is_safe() {
    let tool = SuggestCommandTool;
    assert_eq!(tool.risk(), ToolRisk::Safe);
}
