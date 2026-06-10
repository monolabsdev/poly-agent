use poly_agent_core::AgentConfig;

/// Name for the general-purpose read-only agent.
pub const GENERAL_AGENT: &str = "general";
/// Name for the build agent with full tool access.
pub const BUILD_AGENT: &str = "build";
/// Name for the exploration agent (read-only, file-discovery tools only).
pub const EXPLORE_AGENT: &str = "explore";

/// Return the built-in agent configuration for the given name, or None if unknown.
pub fn builtin_agent(name: &str) -> Option<AgentConfig> {
    match name {
        GENERAL_AGENT => Some(AgentConfig {
            name: GENERAL_AGENT.to_string(),
            description: "Read-only codebase exploration and Q&A".to_string(),
            system_prompt: include_str!("agents/general.md").to_string(),
            allowed_tools: vec![
                "list_files".to_string(),
                "read_file".to_string(),
                "search_files".to_string(),
                "grep_files".to_string(),
                "glob_files".to_string(),
                "propose_edit".to_string(),
                "inspect_project".to_string(),
                "read_important_files".to_string(),
                "suggest_command".to_string(),
            ],
            allow_dangerous: false,
            model_override: None,
        }),
        BUILD_AGENT => Some(AgentConfig {
            name: BUILD_AGENT.to_string(),
            description: "Full file mutation and command execution".to_string(),
            system_prompt: include_str!("agents/build.md").to_string(),
            allowed_tools: vec![], // empty = all tools
            allow_dangerous: true,
            model_override: None,
        }),
        EXPLORE_AGENT => Some(AgentConfig {
            name: EXPLORE_AGENT.to_string(),
            description: "Fast file and content discovery".to_string(),
            system_prompt: include_str!("agents/explore.md").to_string(),
            allowed_tools: vec![
                "list_files".to_string(),
                "read_file".to_string(),
                "search_files".to_string(),
                "grep_files".to_string(),
                "glob_files".to_string(),
            ],
            allow_dangerous: false,
            model_override: None,
        }),
        _ => None,
    }
}

/// Return all built-in agent names and their descriptions.
pub fn builtin_agents() -> Vec<(String, String)> {
    vec![
        (
            GENERAL_AGENT.to_string(),
            "Read-only codebase exploration and Q&A".to_string(),
        ),
        (
            BUILD_AGENT.to_string(),
            "Full file mutation and command execution".to_string(),
        ),
        (
            EXPLORE_AGENT.to_string(),
            "Fast file and content discovery".to_string(),
        ),
    ]
}
