use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use poly_agent_core::{AgentConfig, RuntimeLimits, ToolResult, ToolRisk};
use tokio_util::sync::CancellationToken;

/// Context passed to each tool invocation.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Workspace root — tools must not access files outside this path.
    pub workspace: PathBuf,
    /// Runtime limits for bounding output.
    pub limits: RuntimeLimits,
    /// Cancellation token — tools should check this and abort early if cancelled.
    pub cancellation: CancellationToken,
}

/// Trait that all agent tools must implement.
#[async_trait::async_trait]
pub trait AgentTool: Send + Sync {
    /// Unique tool name (e.g. "list_files").
    fn name(&self) -> &'static str;

    /// Human-readable description for the model.
    fn description(&self) -> &'static str;

    /// Risk level — determines auto-execution policy.
    fn risk(&self) -> ToolRisk;

    /// JSON Schema for the tool's parameters.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given arguments.
    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult>;
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<&'static str, Arc<dyn AgentTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. Overwrites any existing tool with the same name.
    pub fn register(&mut self, tool: Arc<dyn AgentTool>) {
        self.tools.insert(tool.name(), tool);
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn AgentTool>> {
        self.tools.get(name)
    }

    /// Return specs for all registered tools (for sending to the model).
    pub fn tool_specs(&self) -> Vec<poly_agent_providers::ToolSpec> {
        self.tools
            .values()
            .map(|t| poly_agent_providers::ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }

    /// Return the names of all registered tools.
    pub fn names(&self) -> Vec<&'static str> {
        self.tools.keys().copied().collect()
    }

    /// Return tool specs filtered by agent configuration.
    /// When `allowed_tools` is empty, all tools except dangerous ones (unless allow_dangerous) are included.
    /// When `allowed_tools` is non-empty, only those tools are included.
    pub fn tool_specs_for_agent(&self, config: &AgentConfig) -> Vec<poly_agent_providers::ToolSpec> {
        self.tools
            .values()
            .filter(|t| self.is_tool_allowed(t, config))
            .map(|t| poly_agent_providers::ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }

    /// Check whether a given tool is allowed per the agent configuration.
    fn is_tool_allowed(&self, tool: &Arc<dyn AgentTool>, config: &AgentConfig) -> bool {
        // If allowed_tools is non-empty, only include explicitly listed tools.
        if !config.allowed_tools.is_empty() {
            return config.allowed_tools.contains(&tool.name().to_string());
        }
        // Empty allowed_tools means all safe tools by default.
        // Dangerous tools require allow_dangerous.
        if tool.risk() == poly_agent_core::ToolRisk::Dangerous {
            return config.allow_dangerous;
        }
        true
    }

    /// Check whether a named tool is available for the given agent config.
    pub fn is_available(&self, name: &str, config: &AgentConfig) -> bool {
        self.tools.get(name).is_some_and(|t| self.is_tool_allowed(t, config))
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
