use std::time::{SystemTime, UNIX_EPOCH};

use poly_agent_runtime::{AgentTool, ToolContext};

use super::WriteFileTool;

fn temp_workspace() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("poly-agent-write-file-test-{nonce}"))
}

fn ctx(workspace: std::path::PathBuf) -> ToolContext {
    ToolContext {
        workspace,
        limits: Default::default(),
    }
}

#[tokio::test]
async fn writes_new_file_and_parent_dirs() {
    let workspace = temp_workspace();
    tokio::fs::create_dir_all(&workspace).await.unwrap();

    let result = WriteFileTool
        .run(
            serde_json::json!({
                "path": "nested/test.txt",
                "content": "hello\n"
            }),
            ctx(workspace.clone()),
        )
        .await
        .unwrap();

    assert!(!result.is_error);
    assert_eq!(result.output, "Wrote nested/test.txt");
    assert_eq!(
        tokio::fs::read_to_string(workspace.join("nested/test.txt")).await.unwrap(),
        "hello\n"
    );
    let _ = tokio::fs::remove_dir_all(workspace).await;
}

#[tokio::test]
async fn overwrites_existing_file() {
    let workspace = temp_workspace();
    tokio::fs::create_dir_all(&workspace).await.unwrap();
    tokio::fs::write(workspace.join("test.txt"), "old").await.unwrap();

    let result = WriteFileTool
        .run(
            serde_json::json!({
                "path": "test.txt",
                "content": "new"
            }),
            ctx(workspace.clone()),
        )
        .await
        .unwrap();

    assert!(!result.is_error);
    assert_eq!(tokio::fs::read_to_string(workspace.join("test.txt")).await.unwrap(), "new");
    let _ = tokio::fs::remove_dir_all(workspace).await;
}

#[tokio::test]
async fn blocks_path_traversal() {
    let workspace = temp_workspace();
    tokio::fs::create_dir_all(&workspace).await.unwrap();

    let result = WriteFileTool
        .run(
            serde_json::json!({
                "path": "../escape.txt",
                "content": "bad"
            }),
            ctx(workspace.clone()),
        )
        .await;

    assert!(result.is_err());
    assert!(!workspace.with_file_name("escape.txt").exists());
    let _ = tokio::fs::remove_dir_all(workspace).await;
}

#[tokio::test]
async fn blocks_ignored_directory() {
    let workspace = temp_workspace();
    tokio::fs::create_dir_all(&workspace).await.unwrap();

    let result = WriteFileTool
        .run(
            serde_json::json!({
                "path": "target/out.txt",
                "content": "bad"
            }),
            ctx(workspace.clone()),
        )
        .await
        .unwrap();

    assert!(result.is_error);
    assert_eq!(result.output, "Ignored directory");
    assert!(!workspace.join("target/out.txt").exists());
    let _ = tokio::fs::remove_dir_all(workspace).await;
}
