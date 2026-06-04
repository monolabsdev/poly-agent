use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Unique identifier for an agent run.
pub type RunId = Uuid;

/// Which LLM provider to use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelProvider {
    Ollama,
    OpenAICompatible,
}

/// Configuration for the model to call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider: ModelProvider,
    pub model: String,
    /// Base URL for the provider API. Defaults are applied per-provider if None.
    pub base_url: Option<String>,
    /// Optional API key (used by OpenAI-compatible providers).
    pub api_key: Option<String>,
}

/// Role in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Tool calls requested by the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// When role is Tool, which tool call this result corresponds to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: String::new(),
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: output.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// A tool call from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub output: String,
    pub is_error: bool,
}

/// Risk level of a tool. Determines whether auto-execution is allowed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolRisk {
    /// Safe to auto-execute without user approval.
    Safe,
    /// Requires explicit user approval before execution.
    RequiresApproval,
    /// Dangerous operation — blocked by default.
    Dangerous,
}

/// Input to start an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInput {
    pub prompt: String,
    pub workspace: PathBuf,
    pub model: ModelConfig,
    pub limits: RuntimeLimits,
}

/// Why the agent run finished.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    /// Model produced a final text response.
    Complete,
    /// Hit the maximum step count.
    StepLimitReached,
    /// An error occurred.
    Error(String),
    /// Model failed after successful tool calls.
    PartialFailure {
        last_tool_result: String,
        tool_name: String,
    },
}

/// Output of a completed agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    pub run_id: RunId,
    pub text: String,
    pub finish_reason: FinishReason,
}

/// Hard limits to prevent unbounded resource usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeLimits {
    pub max_steps: usize,
    pub max_file_read_bytes: usize,
    pub max_tool_output_bytes: usize,
    pub max_search_results: usize,
    pub max_context_messages: usize,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_file_read_bytes: 256 * 1024,
            max_tool_output_bytes: 64 * 1024,
            max_search_results: 50,
            max_context_messages: 32,
        }
    }
}
