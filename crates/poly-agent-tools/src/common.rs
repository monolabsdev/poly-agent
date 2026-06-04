use std::path::{Path, PathBuf};

use poly_agent_core::AgentError;

/// Directories to skip when walking the file system.
pub const IGNORED_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    "target",
    "dist",
    "build",
    ".next",
    ".turbo",
];

/// Normalize path separators: replace backslashes with forward slashes.
/// On Windows this converts `\` to `/` for consistent key comparisons.
/// On Unix this is a no-op.
pub fn normalize_separators(path: &str) -> String {
    path.replace('\\', "/")
}

/// Resolve `path` relative to `workspace` and ensure it stays within bounds.
/// Returns the canonicalized path on success.
/// Handles both Windows (`\`) and Unix (`/`) path separators consistently.
pub fn resolve_and_validate(workspace: &Path, path: &str) -> Result<PathBuf, AgentError> {
    let normalized_path = normalize_separators(path);
    let requested = if Path::new(&normalized_path).is_absolute() {
        PathBuf::from(&normalized_path)
    } else {
        workspace.join(&normalized_path)
    };

    // Normalize by resolving components manually (canonicalize needs the path to exist).
    let normalized = normalize_path(&requested);
    let ws_normalized = normalize_path(workspace);

    if !normalized.starts_with(&ws_normalized) {
        return Err(AgentError::PathTraversal(format!(
            "Path '{}' escapes workspace '{}'",
            normalized_path,
            workspace.display()
        )));
    }

    Ok(normalized)
}

/// Normalize a path by resolving `.` and `..` components without touching the filesystem.
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

/// Check whether a directory name should be skipped.
pub fn is_ignored_dir(name: &str) -> bool {
    IGNORED_DIRS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_valid_relative_path() {
        let ws = Path::new("/workspace/project");
        let result = resolve_and_validate(ws, "src/main.rs");
        assert!(result.is_ok());
    }

    #[test]
    fn block_path_traversal_dotdot() {
        let ws = Path::new("/workspace/project");
        let result = resolve_and_validate(ws, "../../etc/passwd");
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::PathTraversal(_) => {}
            other => panic!("Expected PathTraversal, got: {other:?}"),
        }
    }

    #[test]
    fn block_absolute_path_outside_workspace() {
        let ws = Path::new("/workspace/project");
        let result = resolve_and_validate(ws, "/etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn allow_absolute_path_inside_workspace() {
        let ws = Path::new("/workspace/project");
        let result = resolve_and_validate(ws, "/workspace/project/src/lib.rs");
        assert!(result.is_ok());
    }

    #[test]
    fn block_sneaky_traversal() {
        let ws = Path::new("/workspace/project");
        let result = resolve_and_validate(ws, "src/../../..");
        assert!(result.is_err());
    }

    #[test]
    fn ignored_dirs_are_detected() {
        assert!(is_ignored_dir("node_modules"));
        assert!(is_ignored_dir(".git"));
        assert!(is_ignored_dir("target"));
        assert!(!is_ignored_dir("src"));
    }

    #[test]
    fn normalize_separators_handles_backslash() {
        assert_eq!(normalize_separators(r"crates\foo\src\lib.rs"), "crates/foo/src/lib.rs");
    }

    #[test]
    fn normalize_separators_preserves_forward_slash() {
        assert_eq!(normalize_separators("crates/foo/src/lib.rs"), "crates/foo/src/lib.rs");
    }

    #[test]
    fn resolve_and_validate_handles_backslash_paths() {
        let ws = Path::new("/workspace/project");
        let result = resolve_and_validate(ws, r"src\main.rs");
        assert!(result.is_ok());
        if cfg!(windows) {
            // On Windows, the result may have backslashes but should be valid.
            let p = result.unwrap();
            assert!(p.to_string_lossy().contains("src"));
            assert!(p.to_string_lossy().contains("main"));
        } else {
            assert_eq!(result.unwrap(), Path::new("/workspace/project/src/main.rs"));
        }
    }

    #[test]
    fn normalize_path_is_pub() {
        // Just verify the function compiles and runs.
        let p = normalize_path(Path::new("/a/b/../c/./d"));
        let s = p.to_string_lossy().replace('\\', "/");
        assert!(s.contains("a/c/d") || s == "/a/c/d", "got: {s}");
    }
}
