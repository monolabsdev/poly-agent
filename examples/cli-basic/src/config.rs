use std::collections::HashSet;
use std::path::PathBuf;

use poly_agent_core::PermissionPreset;

#[derive(Debug)]
pub struct SessionConfig {
    pub provider: String,
    pub model: String,
    pub workspace: PathBuf,
    pub preset: PermissionPreset,
}

pub struct RunStats {
    pub model_calls: usize,
    pub tool_calls: usize,
    pub tools_used: HashSet<String>,
}

pub struct RunSummary {
    pub output_text: String,
    pub elapsed_seconds: f64,
    pub stats: RunStats,
}
