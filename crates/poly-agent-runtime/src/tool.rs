use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use poly_agent_core::{RuntimeLimits, ToolResult, ToolRisk};
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
