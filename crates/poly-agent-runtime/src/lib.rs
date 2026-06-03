//! Agent runtime — the core execution loop for poly-agent.
//!
//! Provides `AgentRuntime`, the `AgentTool` trait, and `ToolRegistry`.

mod runtime;
mod tool;

pub use runtime::AgentRuntime;
pub use tool::{AgentTool, ToolContext, ToolRegistry};
