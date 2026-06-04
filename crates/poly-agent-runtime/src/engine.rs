use std::sync::Arc;

use poly_agent_core::ChatMessage;
use poly_agent_providers::ModelAdapter;

use crate::tool::ToolRegistry;

pub struct AgentRuntime {
    pub tools: ToolRegistry,
    pub adapter: Arc<dyn ModelAdapter>,
    pub pending_approvals:
        std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<(uuid::Uuid, String), tokio::sync::oneshot::Sender<bool>>>>,
}

impl AgentRuntime {
    pub fn new(tools: ToolRegistry, adapter: Arc<dyn ModelAdapter>) -> Self {
        Self {
            tools,
            adapter,
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
