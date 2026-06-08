use std::sync::Arc;

use poly_agent_core::{AutoReviewRisk, ChatMessage, ChatRole, ReviewDecision, ToolCall};
use poly_agent_providers::{ChatRequest, ModelAdapter};
use serde::Deserialize;

const REVIEW_SYSTEM_PROMPT: &str = "\
You are a safety reviewer for an AI agent. You will be shown a single tool call the agent wants to make, \
together with the user's most recent request. Classify the tool call as low, medium, or high risk.

Definitions:
- low: read-only or trivial, clearly aligned with the user's request, no side effects outside the workspace.
- medium: makes a reversible change to the workspace (e.g. editing a file) or has unclear intent.
- high: deletes, overwrites, or runs shell commands with destructive potential, or clearly diverges from the user's request.

Return STRICT JSON only, with no markdown, no commentary, no code fences. The JSON object must contain exactly these three fields:
- \"risk\": one of \"low\", \"medium\", \"high\"
- \"decision\": one of \"approve\", \"ask\", \"deny\"
- \"reason\": a short user-visible sentence explaining the decision

Do not include any other text.";

/// Context passed to an auto-reviewer for a single tool call.
#[derive(Debug, Clone)]
pub struct ReviewContext {
    pub user_prompt: String,
    pub recent_messages: Vec<ChatMessage>,
}

/// Verdict returned by an auto-reviewer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewVerdict {
    pub risk: AutoReviewRisk,
    pub decision: ReviewDecision,
    pub reason: String,
}

impl ReviewVerdict {
    /// Fallback verdict used when the reviewer fails or returns invalid output.
    /// Fails closed: medium risk, ask the user.
    pub fn fallback_unavailable() -> Self {
        Self {
            risk: AutoReviewRisk::Medium,
            decision: ReviewDecision::Ask,
            reason: "Auto-review failed to return a valid structured decision.".to_string(),
        }
    }
}

#[async_trait::async_trait]
pub trait AutoReviewer: Send + Sync {
    async fn review(&self, call: &ToolCall, ctx: &ReviewContext) -> ReviewVerdict;
}

/// Default reviewer that uses the model adapter to classify the call.
pub struct ModelAdapterReviewer {
    adapter: Arc<dyn ModelAdapter>,
}

impl ModelAdapterReviewer {
    pub fn new(adapter: Arc<dyn ModelAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait::async_trait]
impl AutoReviewer for ModelAdapterReviewer {
    async fn review(&self, call: &ToolCall, ctx: &ReviewContext) -> ReviewVerdict {
        let messages = build_review_messages(call, ctx);
        let request = ChatRequest {
            messages,
            tools: Vec::new(),
        };

        let response = match self.adapter.chat(request).await {
            Ok(poly_agent_providers::ModelResponse::Text(text)) => text,
            Ok(_) => return ReviewVerdict::fallback_unavailable(),
            Err(_) => return ReviewVerdict::fallback_unavailable(),
        };

        parse_review_response(&response)
    }
}

fn build_review_messages(call: &ToolCall, ctx: &ReviewContext) -> Vec<ChatMessage> {
    let tool_json =
        serde_json::to_string_pretty(&call.arguments).unwrap_or_else(|_| "{}".to_string());

    let user_payload = format!(
        "User request:\n{}\n\nTool the agent wants to call:\n  name: {}\n  arguments: {}\n\nReturn strict JSON only.",
        ctx.user_prompt, call.name, tool_json
    );

    vec![
        ChatMessage {
            role: ChatRole::System,
            content: REVIEW_SYSTEM_PROMPT.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        ChatMessage::user(user_payload),
    ]
}

#[derive(Debug, Deserialize)]
struct ReviewJson {
    risk: Option<String>,
    decision: Option<String>,
    reason: Option<String>,
}

pub(crate) fn parse_review_response(raw: &str) -> ReviewVerdict {
    let trimmed = raw.trim();
    let candidate = extract_json_object(trimmed);
    let parsed: ReviewJson = match serde_json::from_str(candidate) {
        Ok(v) => v,
        Err(_) => return ReviewVerdict::fallback_unavailable(),
    };

    let risk = match parsed.risk.as_deref() {
        Some("low") => AutoReviewRisk::Low,
        Some("medium") => AutoReviewRisk::Medium,
        Some("high") => AutoReviewRisk::High,
        _ => return ReviewVerdict::fallback_unavailable(),
    };

    let decision = match parsed.decision.as_deref() {
        Some("approve") => ReviewDecision::Approve,
        Some("ask") => ReviewDecision::Ask,
        Some("deny") => ReviewDecision::Deny,
        _ => return ReviewVerdict::fallback_unavailable(),
    };

    let reason = parsed
        .reason
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Auto-reviewer provided no reason.".to_string());

    ReviewVerdict {
        risk,
        decision,
        reason,
    }
}

/// Pull the first balanced `{...}` block from `text`. Returns the original
/// text if no balanced block is found.
fn extract_json_object(text: &str) -> &str {
    let bytes = text.as_bytes();
    let mut start: Option<usize> = None;
    let mut depth: i32 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'{' {
            if start.is_none() {
                start = Some(i);
            }
            depth += 1;
        } else if b == b'}' {
            if depth > 0 {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        return &text[s..=i];
                    }
                }
            }
        }
    }
    text
}

#[cfg(test)]
#[path = "review_tests.rs"]
mod review_tests;
