use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AgentError {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Tool error: {tool_name}: {message}")]
    Tool { tool_name: String, message: String },

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Step limit reached: {0}")]
    StepLimitReached(usize),

    #[error("Path traversal blocked: {0}")]
    PathTraversal(String),

    #[error("{0}")]
    Other(String),
}
