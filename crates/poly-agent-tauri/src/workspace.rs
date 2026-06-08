use std::path::{Path, PathBuf};

use poly_agent_core::AgentInput;
use poly_agent_runtime::{AgentRuntime, AgentRuntimeConfig, ToolRegistry};
use poly_agent_tools::{register_all_tools, register_safe_tools};

use crate::types::{AgentRunError, AgentRunInput, AgentWorkspaceSelection};

pub(crate) struct PreparedRun {
    pub input: AgentInput,
    pub tools: ToolRegistry,
    pub workspace_root: Option<PathBuf>,
    pub local_tools_enabled: bool,
}

pub(crate) fn prepare_run(input: AgentRunInput) -> Result<PreparedRun, AgentRunError> {
    let workspace_root = match input.workspace_path {
        Some(path) => {
            if !path.exists() || !path.is_dir() {
                return Err(AgentRunError::InvalidWorkspace(path.display().to_string()));
            }
            Some(path.canonicalize().map_err(|err| {
                AgentRunError::Other(format!("failed to canonicalize workspace: {err}"))
            })?)
        }
        None => None,
    };

    let mut tools = ToolRegistry::new();
    let local_tools_enabled = workspace_root.is_some();
    if local_tools_enabled {
        register_all_tools(&mut tools);
    } else {
        register_safe_tools(&mut tools);
    }

    if !local_tools_enabled {
        tools = ToolRegistry::new();
    }

    Ok(PreparedRun {
        input: AgentInput {
            prompt: input.prompt,
            workspace: workspace_root.clone().unwrap_or_default(),
            model: input.model,
            limits: input.limits,
            permission_preset: input.permission_preset,
            resolved_context: input.resolved_context,
        },
        tools,
        workspace_root,
        local_tools_enabled,
    })
}

pub(crate) fn resolve_workspace_selection(
    input: &mut AgentRunInput,
    app_data_dir: &Path,
) -> Result<(), AgentRunError> {
    let Some(selection) = input.workspace_selection.clone() else {
        return Ok(());
    };

    input.workspace_path = Some(match selection {
        AgentWorkspaceSelection::Project { project_id, .. } => {
            let path = PathBuf::from(project_id);
            if !path.exists() || !path.is_dir() {
                return Err(AgentRunError::InvalidWorkspace(path.display().to_string()));
            }
            path.canonicalize().map_err(|err| {
                AgentRunError::Other(format!("failed to canonicalize project workspace: {err}"))
            })?
        }
        AgentWorkspaceSelection::Sandbox { chat_id } => {
            let safe_chat_id = sanitize_chat_id(&chat_id)?;
            let root = app_data_dir.join("agent-sandboxes").join(safe_chat_id);
            std::fs::create_dir_all(&root).map_err(|err| {
                AgentRunError::Other(format!("failed to create chat sandbox: {err}"))
            })?;
            root.canonicalize().map_err(|err| {
                AgentRunError::Other(format!("failed to canonicalize chat sandbox: {err}"))
            })?
        }
    });

    Ok(())
}

pub(crate) fn sandbox_root(app_data_dir: &Path, chat_id: &str) -> Result<PathBuf, AgentRunError> {
    Ok(app_data_dir
        .join("agent-sandboxes")
        .join(sanitize_chat_id(chat_id)?))
}

fn sanitize_chat_id(chat_id: &str) -> Result<String, AgentRunError> {
    let trimmed = chat_id.trim();
    if trimmed.is_empty()
        || trimmed.contains("..")
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains(':')
        || !trimmed
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err(AgentRunError::Other("invalid chat id for sandbox".to_string()));
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use poly_agent_core::{ModelConfig, ModelProvider, RuntimeLimits};

    fn input(selection: AgentWorkspaceSelection) -> AgentRunInput {
        AgentRunInput {
            prompt: "test".to_string(),
            model: ModelConfig {
                provider: ModelProvider::Ollama,
                model: "test".to_string(),
                base_url: None,
                api_key: None,
            },
            workspace_path: None,
            workspace_selection: Some(selection),
            limits: RuntimeLimits::default(),
            permission_preset: Default::default(),
            resolved_context: None,
            debug: false,
        }
    }

    #[test]
    fn sandbox_selection_resolves_chat_lifetime_path() {
        let root = std::env::temp_dir().join(format!("poly-agent-sandbox-test-{}", uuid::Uuid::new_v4()));
        let mut first = input(AgentWorkspaceSelection::Sandbox {
            chat_id: "chat-1".to_string(),
        });
        let mut second = input(AgentWorkspaceSelection::Sandbox {
            chat_id: "chat-1".to_string(),
        });

        resolve_workspace_selection(&mut first, &root).unwrap();
        std::fs::write(first.workspace_path.as_ref().unwrap().join("test.txt"), "hello").unwrap();
        resolve_workspace_selection(&mut second, &root).unwrap();

        assert_eq!(first.workspace_path, second.workspace_path);
        assert!(second.workspace_path.as_ref().unwrap().join("test.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sandbox_selection_rejects_path_traversal_chat_id() {
        let root = std::env::temp_dir();
        let mut run = input(AgentWorkspaceSelection::Sandbox {
            chat_id: "../escape".to_string(),
        });

        assert!(resolve_workspace_selection(&mut run, &root).is_err());
    }
}

pub(crate) fn default_runtime(prepared: PreparedRun) -> Result<AgentRuntime, AgentRunError> {
    use poly_agent_core::ModelProvider;
    use poly_agent_providers::{OllamaAdapter, OpenAICompatibleAdapter};
    use std::sync::Arc;

    let model = prepared.input.model.clone();
    let adapter: Arc<dyn poly_agent_providers::ModelAdapter> = match model.provider {
        ModelProvider::Ollama => Arc::new(OllamaAdapter::new(model.model, model.base_url)),
        ModelProvider::OpenAICompatible => Arc::new(OpenAICompatibleAdapter::new(
            model
                .base_url
                .unwrap_or_else(|| "http://localhost:8080/v1".to_string()),
            model.model,
            model.api_key,
        )),
        _ => {
            return Err(AgentRunError::Other(
                "unsupported model provider for Poly UI run".to_string(),
            ));
        }
    };

    let preset = prepared.input.permission_preset;
    Ok(AgentRuntime::with_config(
        prepared.tools,
        adapter,
        AgentRuntimeConfig {
            permission_preset: preset,
            reviewer: None,
        },
    ))
}
