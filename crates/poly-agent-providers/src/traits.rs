use poly_agent_core::{ChatMessage, ToolCall};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::Stream;

pub type ModelStream =
    Pin<Box<dyn Stream<Item = Result<ModelResponse, super::error::ProviderError>> + Send>>;

#[async_trait::async_trait]
pub trait ModelAdapter: Send + Sync {
    async fn chat(
        &self,
        request: ChatRequest,
    ) -> Result<ModelResponse, super::error::ProviderError>;

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<ModelStream, super::error::ProviderError> {
        let response = self.chat(request).await?;
        Ok(Box::pin(async_stream::stream! {
            yield Ok(response);
        }))
    }
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
