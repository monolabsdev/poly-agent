use std::path::PathBuf;

use poly_agent_core::AgentInput;
use poly_agent_runtime::{AgentRuntime, ToolRegistry};
use poly_agent_tools::{register_all_tools, register_safe_tools};

use crate::types::{AgentRunError, AgentRunInput};

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
        },
        tools,
        workspace_root,
        local_tools_enabled,
    })
}

pub(crate) fn default_runtime(
    prepared: PreparedRun,
) -> Result<AgentRuntime, AgentRunError> {
    use poly_agent_core::ModelProvider;
    use poly_agent_providers::{OllamaAdapter, OpenAICompatibleAdapter};
    use std::sync::Arc;

    let model = prepared.input.model.clone();
    let adapter: Arc<dyn poly_agent_providers::ModelAdapter> = match model.provider {
        ModelProvider::Ollama => Arc::new(OllamaAdapter::new(
            model.model,
            model.base_url,
        )),
        ModelProvider::OpenAICompatible => Arc::new(OpenAICompatibleAdapter::new(
            model.base_url.unwrap_or_else(|| "http://localhost:8080/v1".to_string()),
            model.model,
            model.api_key,
        )),
        _ => {
            return Err(AgentRunError::Other(
                "unsupported model provider for Poly UI run".to_string(),
            ));
        }
    };

    Ok(AgentRuntime::new(prepared.tools, adapter))
}
