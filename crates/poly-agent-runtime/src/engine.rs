use std::sync::Arc;

use poly_agent_core::{ChatMessage, PermissionPreset};
use poly_agent_providers::ModelAdapter;

use crate::review::{AutoReviewer, ModelAdapterReviewer};
use crate::tool::ToolRegistry;

pub struct AgentRuntime {
    pub tools: ToolRegistry,
    pub adapter: Arc<dyn ModelAdapter>,
    pub permission_preset: PermissionPreset,
    pub reviewer: Arc<dyn AutoReviewer>,
    pub pending_approvals: std::sync::Arc<
        tokio::sync::Mutex<
            std::collections::HashMap<(uuid::Uuid, String), tokio::sync::oneshot::Sender<bool>>,
        >,
    >,
}

/// Optional configuration for `AgentRuntime::with_config`.
///
/// `permission_preset` always defaults to `Default`. When `reviewer` is
/// `None`, a `ModelAdapterReviewer` is built from the supplied model adapter
/// so the default reviewer uses the same model.
pub struct AgentRuntimeConfig {
    pub permission_preset: PermissionPreset,
    pub reviewer: Option<Arc<dyn AutoReviewer>>,
}

impl Default for AgentRuntimeConfig {
    fn default() -> Self {
        Self {
            permission_preset: PermissionPreset::Default,
            reviewer: None,
        }
    }
}

impl AgentRuntime {
    /// Backward-compatible constructor. Uses `PermissionPreset::Default` and
    /// builds a default `ModelAdapterReviewer` from `adapter`.
    pub fn new(tools: ToolRegistry, adapter: Arc<dyn ModelAdapter>) -> Self {
        Self::with_config(
            tools,
            adapter,
            AgentRuntimeConfig {
                permission_preset: PermissionPreset::Default,
                reviewer: None,
            },
        )
    }

    /// Construct a runtime with a custom configuration.
    pub fn with_config(
        tools: ToolRegistry,
        adapter: Arc<dyn ModelAdapter>,
        config: AgentRuntimeConfig,
    ) -> Self {
        let reviewer = config
            .reviewer
            .unwrap_or_else(|| Arc::new(ModelAdapterReviewer::new(adapter.clone())));
        Self {
            tools,
            adapter,
            permission_preset: config.permission_preset,
            reviewer,
            pending_approvals: std::sync::Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    pub fn context_window(messages: &[ChatMessage], max: usize) -> Vec<ChatMessage> {
        if messages.len() <= max {
            messages.to_vec()
        } else {
            messages[messages.len() - max..].to_vec()
        }
    }

    pub fn truncate_output(output: &str, max_bytes: usize) -> String {
        if output.len() <= max_bytes {
            return output.to_string();
        }
        let mut end = max_bytes;
        while end > 0 && !output.is_char_boundary(end) {
            end -= 1;
        }
        let mut truncated = output[..end].to_string();
        truncated.push_str("\n... [output truncated]");
        truncated
    }
}
