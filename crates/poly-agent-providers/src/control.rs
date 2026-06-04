const CONTROL_TOKEN_PATTERNS: &[&str] = &[
    "assistant<|channel|>analysis",
    "assistant<|channel|>commentary",
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
];

pub fn contains_control_tokens(text: &str) -> bool {
    for pattern in CONTROL_TOKEN_PATTERNS {
        if text.contains(pattern) {
            return true;
        }
    }
    false
}

pub fn strip_control_tokens(text: &str) -> String {
    let mut result = text.to_string();
    for pattern in CONTROL_TOKEN_PATTERNS {
        result = result.replace(pattern, "");
    }
    result = result.replace("assistant\n", "");
    result = result.trim().to_string();
    result
}
