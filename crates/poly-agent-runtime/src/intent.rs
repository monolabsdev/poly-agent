/// Phrases that indicate the user is asking about the codebase/project.
const CODEBASE_INTENT_PHRASES: &[&str] = &[
    "what is this codebase",
    "what does this app do",
    "what does this program do",
    "how does this work",
    "explain this directory",
    "explain this repo",
    "summarise this project",
    "summarize this project",
    "what is this project",
    "what does this project do",
    "what is this repository",
    "what does this repository do",
];

/// Tools that actually mutate files on disk.
pub const MUTATING_TOOLS: &[&str] = &["apply_patch", "write_file"];

/// Tools used for codebase inspection (not cached by anti-loop guard after first call).
pub const INSPECTION_TOOLS: &[&str] = &["inspect_project", "read_important_files"];

pub fn tool_is_inspection(name: &str) -> bool {
    INSPECTION_TOOLS.contains(&name)
}

/// Prefix added to model output when inspection tools were used, to ground claims.
pub fn build_grounding_prefix() -> String {
    "Based on the files inspected:".to_string()
}

/// Verbs that strongly indicate the user wants the agent to change a file.
const EDIT_VERBS: &[&str] = &[
    "change", "edit", "update", "modify", "replace", "fix", "create", "rename", "delete", "remove",
    "add", "set", "patch", "rewrite", "refactor", "remove", "insert", "append",
];

/// Phrases that imply the model is claiming it has already applied an edit.
const SUCCESS_CLAIMS: &[&str] = &[
    "i have updated", "i've updated", "i have changed", "i've changed",
    "i have edited", "i've edited", "i have modified", "i've modified",
    "i have replaced", "i've replaced", "i have fixed", "i've fixed",
    "i have created", "i've created", "i applied",
    "file has been updated", "file has been changed", "file has been edited",
    "file has been modified", "the file is now", "here is the updated",
    "here's the updated", "i rewrote", "i have rewritten", "i wrote it",
    "i have written", "done.", "done!",
];

pub fn contains_codebase_intent_phrase(text: &str) -> bool {
    let lower = text.to_lowercase();
    CODEBASE_INTENT_PHRASES.iter().any(|p| lower.contains(p))
}

/// Whether `name` is a mutating tool.
pub fn is_mutating_tool(name: &str) -> bool {
    MUTATING_TOOLS.contains(&name)
}

/// Result of inspecting a user prompt for edit intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditIntent {
    pub is_edit: bool,
}

impl EditIntent {
    /// Inspect user prompt for edit verbs. Case-insensitive, whole-word.
    pub fn detect(prompt: &str) -> Self {
        let lower = prompt.to_lowercase();
        if EDIT_VERBS.iter().any(|v| contains_word(&lower, v)) {
            return Self { is_edit: true };
        }
        Self { is_edit: false }
    }
}

/// True if `needle` appears surrounded by non-word boundaries.
fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = haystack.as_bytes();
    let n = needle.as_bytes();
    let mut i = 0;
    while i + n.len() <= bytes.len() {
        if &bytes[i..i + n.len()] == n {
            let left_ok = i == 0 || !is_word_byte(bytes[i - 1]);
            let right_ok = i + n.len() == bytes.len() || !is_word_byte(bytes[i + n.len()]);
            if left_ok && right_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn claims_success(text: &str) -> bool {
    let lower = text.to_lowercase();
    SUCCESS_CLAIMS.iter().any(|p| lower.contains(p))
}

/// Sanitise final text against the edit-intent guard.
///
/// If user asked for edit and no mutating tool succeeded, replace with warning.
/// If model claimed success but no mutating tool was even requested, replace with flat denial.
pub fn sanitise_final_text(
    text: String,
    intent: EditIntent,
    mutating_succeeded: bool,
    mutating_requested: bool,
) -> String {
    if !intent.is_edit || mutating_succeeded {
        return text;
    }

    if mutating_requested {
        return format!(
            "{}\n\n[guard] An edit was requested but no mutating tool (apply_patch or write_file) succeeded. \
Approval may still be pending, or the tool was rejected. I have not modified the file.",
            text.trim()
        );
    }

    if claims_success(&text) {
        return "I inspected the file, but no edit was applied.".to_string();
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_intent_detects_verbs() {
        assert!(EditIntent::detect("Change README.md title from poly-agent to Poly Agent").is_edit);
        assert!(EditIntent::detect("Please update the version field").is_edit);
        assert!(EditIntent::detect("Fix the typo in foo.rs").is_edit);
        assert!(EditIntent::detect("Add a new endpoint").is_edit);
        assert!(EditIntent::detect("Remove the unused import").is_edit);
        assert!(EditIntent::detect("CREATE a new file").is_edit);
    }

    #[test]
    fn edit_intent_negative_for_read_only() {
        assert!(!EditIntent::detect("What does this project do?").is_edit);
        assert!(!EditIntent::detect("Summarise README.md").is_edit);
        assert!(!EditIntent::detect("List the files in this project").is_edit);
        assert!(!EditIntent::detect("Explain how the runtime works").is_edit);
    }

    #[test]
    fn edit_intent_word_boundary() {
        assert!(!EditIntent::detect("describe the removeit helper").is_edit);
        assert!(EditIntent::detect("rename the updated_at field").is_edit);
    }

    #[test]
    fn sanitise_keeps_text_when_no_edit_intent() {
        let s = sanitise_final_text(
            "Here is the updated file".to_string(),
            EditIntent::detect("summarise readme"),
            false,
            false,
        );
        assert_eq!(s, "Here is the updated file");
    }

    #[test]
    fn sanitise_keeps_text_when_mutating_succeeded() {
        let s = sanitise_final_text(
            "I have updated the file.".to_string(),
            EditIntent::detect("change the title"),
            true,
            true,
        );
        assert_eq!(s, "I have updated the file.");
    }

    #[test]
    fn sanitise_replaces_claim_when_no_mutating_tool_ran() {
        let s = sanitise_final_text(
            "I have updated README.md to use Poly Agent as the title.".to_string(),
            EditIntent::detect("Change README.md title from poly-agent to Poly Agent"),
            false,
            false,
        );
        assert_eq!(s, "I inspected the file, but no edit was applied.");
    }

    #[test]
    fn sanitise_warns_when_mutating_requested_but_failed() {
        let s = sanitise_final_text(
            "I've updated the title.".to_string(),
            EditIntent::detect("update the title"),
            false,
            true,
        );
        assert!(s.contains("[guard]"));
        assert!(s.contains("no mutating tool"));
    }

    #[test]
    fn codebase_intent_detection() {
        assert!(contains_codebase_intent_phrase("What is this codebase?"));
        assert!(contains_codebase_intent_phrase("What does this app do?"));
        assert!(contains_codebase_intent_phrase("What does this program do?"));
        assert!(contains_codebase_intent_phrase("How does this work?"));
        assert!(contains_codebase_intent_phrase("Explain this directory"));
        assert!(contains_codebase_intent_phrase("Explain this repo"));
        assert!(contains_codebase_intent_phrase("Summarise this project"));
        assert!(!contains_codebase_intent_phrase("What is the weather?"));
        assert!(!contains_codebase_intent_phrase("Tell me a joke"));
    }
}
