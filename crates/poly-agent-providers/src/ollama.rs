use crate::{ChatRequest, ModelAdapter, ModelResponse, ProviderError, ToolSpec};
use poly_agent_core::{ChatMessage, ChatRole, ToolCall};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Adapter for the Ollama chat API.
pub struct OllamaAdapter {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaAdapter {
    pub fn new(model: impl Into<String>, base_url: Option<String>) -> Self {
        Self {
            base_url: base_url.unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string()),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    fn build_messages(messages: &[ChatMessage]) -> Vec<OllamaMessage> {
        messages.iter().map(|m| {
            let role = match m.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
            };

            let tool_calls = if m.tool_calls.is_empty() {
                None
            } else {
                Some(m.tool_calls.iter().map(|tc| OllamaToolCall {
                    function: OllamaFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    },
                }).collect())
            };

            OllamaMessage {
                role: role.to_string(),
                content: m.content.clone(),
                tool_calls,
            }
        }).collect()
    }

    fn build_tools(tools: &[ToolSpec]) -> Option<Vec<OllamaTool>> {
        if tools.is_empty() {
            return None;
        }
        Some(tools.iter().map(|t| OllamaTool {
            r#type: "function".to_string(),
            function: OllamaToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        }).collect())
    }

    fn parse_response(body: OllamaResponse) -> Result<ModelResponse, ProviderError> {
        if let Some(tool_calls) = body.message.tool_calls {
            if !tool_calls.is_empty() {
                let calls = tool_calls.into_iter().map(|tc| {
                    ToolCall {
                        // Ollama doesn't always provide an ID, so generate one
                        id: Uuid::new_v4().to_string(),
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    }
                }).collect();
                return Ok(ModelResponse::ToolCalls(calls));
            }
        }

        let text = body.message.content;
        Ok(ModelResponse::Text(text))
    }
}

#[async_trait::async_trait]
impl ModelAdapter for OllamaAdapter {
    async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));

        let body = OllamaRequest {
            model: self.model.clone(),
            messages: Self::build_messages(&request.messages),
            tools: Self::build_tools(&request.tools),
            stream: false,
        };

        tracing::debug!(url = %url, model = %self.model, "Sending Ollama request");

        let resp = self.client.post(&url).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: body_text,
            });
        }

        let response_body: OllamaResponse = resp.json().await
            .map_err(|e| ProviderError::Parse(format!("Failed to deserialize Ollama response: {e}")))?;

        Self::parse_response(response_body)
    }
}

// --- Serde models for the Ollama API ---

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaToolCall {
    function: OllamaFunction,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct OllamaTool {
    r#type: String,
    function: OllamaToolFunction,
}

#[derive(Serialize)]
struct OllamaToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct OllamaResponse {
    message: OllamaMessage,
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ollama_text_response() {
        let json = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "Here is the list of files."
            }
        });
        let body: OllamaResponse = serde_json::from_value(json).unwrap();
        let result = OllamaAdapter::parse_response(body).unwrap();
        match result {
            ModelResponse::Text(t) => assert_eq!(t, "Here is the list of files."),
            _ => panic!("Expected text response"),
        }
    }

    #[test]
    fn parse_ollama_empty_response() {
        let json = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": ""
            }
        });
        let body: OllamaResponse = serde_json::from_value(json).unwrap();
        let result = OllamaAdapter::parse_response(body).unwrap();
        match result {
            ModelResponse::Text(t) => assert!(t.is_empty()),
            _ => panic!("Expected text response"),
        }
    }

    #[test]
    fn parse_ollama_tool_call_response() {
        let json = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "list_files",
                        "arguments": { "path": "." }
                    }
                }]
            }
        });
        let body: OllamaResponse = serde_json::from_value(json).unwrap();
        let result = OllamaAdapter::parse_response(body).unwrap();
        match result {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "list_files");
                assert_eq!(calls[0].arguments["path"], ".");
                assert!(!calls[0].id.is_empty());
            }
            _ => panic!("Expected tool calls"),
        }
    }
}
