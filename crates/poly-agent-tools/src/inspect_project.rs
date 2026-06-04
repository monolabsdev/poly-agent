use std::collections::BTreeMap;
use std::path::Path;

use poly_agent_core::{ToolResult, ToolRisk};
use poly_agent_runtime::{AgentTool, ToolContext};
use serde::Deserialize;

use crate::common::is_ignored_dir;

pub struct InspectProjectTool;

#[derive(Deserialize)]
struct InspectProjectArgs {
    #[serde(default = "default_max_files")]
    max_files: usize,
}

fn default_max_files() -> usize {
    100
}

#[async_trait::async_trait]
impl AgentTool for InspectProjectTool {
    fn name(&self) -> &'static str {
        "inspect_project"
    }

    fn description(&self) -> &'static str {
        "Inspect project structure: detect project types, find important files (README, Cargo.toml, package.json, etc.), list top-level directories. Lightweight and fast."
    }

    fn risk(&self) -> ToolRisk {
        ToolRisk::Safe
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "max_files": {
                    "type": "integer",
                    "description": "Maximum files to inspect. Default 100."
                }
            }
        })
    }

    async fn run(&self, args: serde_json::Value, ctx: ToolContext) -> anyhow::Result<ToolResult> {
        let args: InspectProjectArgs = serde_json::from_value(args)?;
        let workspace = &ctx.workspace;

        let mut result = serde_json::json!({
            "workspace_root": workspace.to_string_lossy(),
            "project_types": [],
            "important_files": {},
            "top_level_dirs": [],
            "ignored_dir_count": 0,
        });

        // Detect project types from workspace root contents.
        let mut project_types = Vec::new();
        let mut important_files: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut top_level_dirs = Vec::new();
        let mut ignored_count = 0usize;

        // Track found Cargo.toml for crate names.
        let mut has_cargo_toml = false;
        let mut has_package_json = false;

        // Quick scan of workspace root (depth 1).
        let root_entries = match std::fs::read_dir(workspace) {
            Ok(rd) => rd,
            Err(e) => {
                return Ok(ToolResult {
                    tool_call_id: String::new(),
                    output: format!("Cannot read workspace: {e}"),
                    is_error: true,
                    cached: false,
                });
            }
        };

        for entry in root_entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false);
            let is_file = entry.file_type().ok().map(|ft| ft.is_file()).unwrap_or(false);

            if is_ignored_dir(&name) || name.starts_with('.') {
                if is_ignored_dir(&name) {
                    ignored_count += 1;
                }
                continue;
            }

            if is_dir {
                top_level_dirs.push(name.clone());
                // Check for crate directories (has Cargo.toml inside).
                let sub_cargo = entry.path().join("Cargo.toml");
                if sub_cargo.exists() {
                    has_cargo_toml = true;
                    if let Some(cname) = read_crate_name(&sub_cargo) {
                        important_files.insert(
                            format!("crates/{}/Cargo.toml", &name),
                            serde_json::json!({"path": sub_cargo.to_string_lossy(), "crate_name": cname}),
                        );
                    }
                    // Check for src/lib.rs and src/main.rs inside crate dir.
                    for sub_path in &["src/lib.rs", "src/main.rs"] {
                        let candidate = entry.path().join(sub_path);
                        if candidate.exists() {
                            important_files.insert(
                                format!("crates/{}/{}", &name, sub_path),
                                serde_json::json!({"path": candidate.to_string_lossy()}),
                            );
                        }
                    }
                    // Check for examples in crate dir.
                    let examples_dir = entry.path().join("examples");
                    if examples_dir.is_dir() {
                        if let Ok(ex_rd) = std::fs::read_dir(&examples_dir) {
                            for ex_entry in ex_rd.flatten() {
                                let ex_name = ex_entry.file_name().to_string_lossy().to_string();
                                let ex_cargo = ex_entry.path().join("Cargo.toml");
                                if ex_cargo.exists() {
                                    if let Some(ecname) = read_crate_name(&ex_cargo) {
                                        important_files.insert(
                                            format!("crates/{}/examples/{}/Cargo.toml", &name, &ex_name),
                                            serde_json::json!({"path": ex_cargo.to_string_lossy(), "crate_name": ecname}),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            } else if is_file {
                match name.as_str() {
                    "Cargo.toml" => {
                        has_cargo_toml = true;
                        if let Some(cname) = read_crate_name(&entry.path()) {
                            important_files.insert("Cargo.toml".to_string(), serde_json::json!({"path": entry.path().to_string_lossy(), "crate_name": cname}));
                        } else {
                            important_files.insert("Cargo.toml".to_string(), serde_json::json!({"path": entry.path().to_string_lossy()}));
                        }
                    }
                    "package.json" => {
                        has_package_json = true;
                        if let Some(pname) = read_package_name(&entry.path()) {
                            important_files.insert("package.json".to_string(), serde_json::json!({"path": entry.path().to_string_lossy(), "package_name": pname}));
                        } else {
                            important_files.insert("package.json".to_string(), serde_json::json!({"path": entry.path().to_string_lossy()}));
                        }
                    }
                    "README.md" => {
                        important_files.insert("README.md".to_string(), serde_json::json!({"path": entry.path().to_string_lossy()}));
                    }
                    "AGENTS.md" | "CLAUDE.md" => {
                        important_files.insert(name, serde_json::json!({"path": entry.path().to_string_lossy()}));
                    }
                    _ => {}
                }
            }
        }

        // Check for src/main.rs, src/lib.rs at workspace root.
        for sub_path in &["src/main.rs", "src/lib.rs"] {
            let candidate = workspace.join(sub_path);
            if candidate.exists() {
                important_files.insert(
                    sub_path.to_string(),
                    serde_json::json!({"path": candidate.to_string_lossy()}),
                );
            }
        }

        // Check for examples at workspace root.
        let examples_dir = workspace.join("examples");
        if examples_dir.is_dir() {
            if let Ok(ex_rd) = std::fs::read_dir(&examples_dir) {
                for ex_entry in ex_rd.flatten() {
                    let ex_name = ex_entry.file_name().to_string_lossy().to_string();
                    let ex_cargo = ex_entry.path().join("Cargo.toml");
                    let ex_main = ex_entry.path().join("src/main.rs");
                    if ex_cargo.exists() {
                        if let Some(ecname) = read_crate_name(&ex_cargo) {
                            important_files.insert(
                                format!("examples/{}/Cargo.toml", &ex_name),
                                serde_json::json!({"path": ex_cargo.to_string_lossy(), "crate_name": ecname}),
                            );
                        } else {
                            important_files.insert(
                                format!("examples/{}/Cargo.toml", &ex_name),
                                serde_json::json!({"path": ex_cargo.to_string_lossy()}),
                            );
                        }
                    }
                    if ex_main.exists() {
                        important_files.insert(
                            format!("examples/{}/src/main.rs", &ex_name),
                            serde_json::json!({"path": ex_main.to_string_lossy()}),
                        );
                    }
                }
            }
        }

        // Detect project types.
        if has_cargo_toml {
            project_types.push("rust");
        }
        if has_package_json {
            project_types.push("node");
        }
        // Check for Tauri indicators.
        let tauri_cargo = workspace.join("src-tauri").join("Cargo.toml");
        if tauri_cargo.exists() {
            project_types.push("tauri");
        }
        let tauri_config = workspace.join("tauri.conf.json");
        if tauri_config.exists() && !project_types.contains(&"tauri") {
            project_types.push("tauri");
        }
        if project_types.is_empty() {
            // Generic detection.
            let has_readme = workspace.join("README.md").exists();
            if has_readme {
                project_types.push("generic");
            }
        }

        // Truncate results if too many.
        if important_files.len() > args.max_files {
            important_files = important_files.into_iter().take(args.max_files).collect();
        }

        top_level_dirs.sort();

        result["project_types"] = serde_json::json!(project_types);
        result["important_files"] = serde_json::json!(important_files);
        result["top_level_dirs"] = serde_json::json!(top_level_dirs);
        result["ignored_dir_count"] = serde_json::json!(ignored_count);

        Ok(ToolResult {
            tool_call_id: String::new(),
            output: serde_json::to_string_pretty(&result)?,
            is_error: false,
            cached: false,
        })
    }
}

/// Read crate name from a Cargo.toml if cheaply available.
fn read_crate_name(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_prefix("name = ") {
            return Some(name.trim_matches('"').to_string());
        }
        if trimmed.starts_with("[") {
            return None;
        }
    }
    None
}

/// Read package name from package.json if cheaply available.
fn read_package_name(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    // Simple JSON scan without full parse.
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("\"name\":") {
            let val = rest.trim().trim_matches(',').trim_matches('"');
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
    }
    None
}
