use poly_agent_core::{ChatMessage, ToolCall};
use serde::{Deserialize, Serialize};

#[async_trait::async_trait]
pub trait ModelAdapter: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, super::error::ProviderError>;
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum ModelResponse {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}
