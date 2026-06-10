//! Built-in tools for the poly-agent runtime.
//!
//! Provides safe file-system tools that operate within a workspace sandbox.

mod apply_patch;
mod common;
mod grep_files;
mod glob_files;
mod inspect_project;
mod list_files;
mod propose_edit;
mod read_file;
mod read_important_files;
mod run_command;
mod search_files;
mod suggest_command;
mod write_file;

pub use apply_patch::ApplyPatchTool;
pub use grep_files::GrepFilesTool;
pub use glob_files::GlobFilesTool;
pub use inspect_project::InspectProjectTool;
pub use list_files::ListFilesTool;
pub use propose_edit::ProposeEditTool;
pub use read_file::ReadFileTool;
pub use read_important_files::ReadImportantFilesTool;
pub use run_command::RunCommandTool;
pub use search_files::SearchFilesTool;
pub use suggest_command::SuggestCommandTool;
pub use write_file::WriteFileTool;

use poly_agent_runtime::ToolRegistry;
use std::sync::Arc;

/// Register all safe (auto-executable) tools into the registry.
pub fn register_safe_tools(registry: &mut ToolRegistry) {
    registry.register(Arc::new(ListFilesTool));
    registry.register(Arc::new(ReadFileTool));
    registry.register(Arc::new(SearchFilesTool));
    registry.register(Arc::new(GrepFilesTool));
    registry.register(Arc::new(GlobFilesTool));
    registry.register(Arc::new(ProposeEditTool));
    registry.register(Arc::new(InspectProjectTool));
    registry.register(Arc::new(ReadImportantFilesTool));
    registry.register(Arc::new(SuggestCommandTool));
}

/// Register all tools that mutate files. These require user approval.
pub fn register_mutation_tools(registry: &mut ToolRegistry) {
    registry.register(Arc::new(ApplyPatchTool));
    registry.register(Arc::new(WriteFileTool));
}

/// Register tools that execute commands. These require user approval.
pub fn register_command_tools(registry: &mut ToolRegistry) {
    registry.register(Arc::new(RunCommandTool));
}

/// Register every built-in tool (safe + mutation + dangerous).
pub fn register_all_tools(registry: &mut ToolRegistry) {
    register_safe_tools(registry);
    register_mutation_tools(registry);
    register_command_tools(registry);
}
