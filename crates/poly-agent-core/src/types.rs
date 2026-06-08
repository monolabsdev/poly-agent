use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Unique identifier for an agent run.
pub type RunId = Uuid;

/// Which LLM provider to use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// True when the result was served from a read-file cache (same path read twice).
    #[serde(default)]
    pub cached: bool,
}

/// Risk level of a tool. Determines whether auto-execution is allowed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolRisk {
    Safe,
    RequiresApproval,
    Dangerous,
}

/// Policy preset that decides how the runtime handles tool approvals.
///
/// Presets are policy defaults, not hardcoded states. The existing
/// `approve_tool` / `reject_tool` API continues to work as a per-call override.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionPreset {
    /// Auto-approve file writes (apply_patch, write_file). Prompt for run_command.
    Default,
    /// Use an auto-reviewer to classify tool calls.
    /// Low -> approve, Medium -> ask user, High -> deny.
    AutoReview,
    /// Auto-approve every tool call. Runtime safety guards still apply.
    FullAccess,
}

impl Default for PermissionPreset {
    fn default() -> Self {
        Self::Default
    }
}

/// Why a tool call was auto-approved. Emitted to the UI for transparency.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AutoApproveReason {
    /// Default preset auto-approves file writes.
    PresetDefault,
    /// FullAccess preset auto-approves every tool.
    FullAccess,
    /// AutoReviewer classified this call as low risk.
    AutoReviewLow,
}

/// Risk classification returned by the auto-reviewer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AutoReviewRisk {
    Low,
    Medium,
    High,
}

/// Final decision the auto-reviewer recommends for a tool call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ReviewDecision {
    Approve,
    Ask,
    Deny,
}

/// Input to start an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInput {
    pub prompt: String,
    pub workspace: PathBuf,
    pub model: ModelConfig,
    pub limits: RuntimeLimits,
    #[serde(default)]
    pub permission_preset: PermissionPreset,
    #[serde(default)]
    pub resolved_context: Option<AgentResolvedContext>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentResolvedContext {
    #[serde(default)]
    pub active_workspace: Option<String>,
    #[serde(default)]
    pub active_file: Option<String>,
    #[serde(default)]
    pub recently_viewed_files: Vec<String>,
    #[serde(default)]
    pub recently_edited_files: Vec<String>,
    #[serde(default)]
    pub recent_constraints: Vec<String>,
    #[serde(default)]
    pub last_tool_call: Option<AgentLastToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentLastToolCall {
    pub tool_name: String,
    #[serde(default)]
    pub target_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FinishReason {
    Complete,
    StepLimitReached,
    Error(String),
    PartialFailure {
        last_tool_result: String,
        tool_name: String,
    },
    /// The model hit max_steps but the runtime synthesized a final answer from gathered context.
    StepLimitSynthesized,
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
    /// Timeout in seconds for `run_command` tool execution. Default: 60.
    pub command_timeout_secs: u64,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_file_read_bytes: 256 * 1024,
            max_tool_output_bytes: 64 * 1024,
            max_search_results: 50,
            max_context_messages: 32,
            command_timeout_secs: 60,
        }
    }
}
