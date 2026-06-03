//! LLM provider adapters for poly-agent.
//!
//! Provides a common `ModelAdapter` trait and implementations for
//! Ollama and OpenAI-compatible APIs.

mod ollama;
mod openai;

pub use ollama::OllamaAdapter;
pub use openai::OpenAICompatibleAdapter;

use poly_agent_core::{ChatMessage, ToolCall};
use serde::{Deserialize, Serialize};

/// Trait for LLM provider adapters.
#[async_trait::async_trait]
pub trait ModelAdapter: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError>;
}

/// A chat request to send to a provider.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSpec>,
}

/// Specification of a tool to advertise to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Response from a model.
#[derive(Debug, Clone)]
pub enum ModelResponse {
    /// The model produced a text response.
    Text(String),
    /// The model wants to call one or more tools.
    ToolCalls(Vec<ToolCall>),
}

/// Errors from provider operations.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    Parse(String),

    #[error("API error: {status} — {body}")]
    Api { status: u16, body: String },
}
