use crate::control::{contains_control_tokens, strip_control_tokens};
use crate::error::ProviderError;
use crate::retry;
use crate::traits::{ChatRequest, ModelAdapter, ModelResponse, ToolSpec};
use poly_agent_core::{ChatMessage, ChatRole, ToolCall};
use serde::{Deserialize, Serialize};

pub struct OpenAICompatibleAdapter {
    base_url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl OpenAICompatibleAdapter {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key,
            client: reqwest::Client::new(),
        }
    }

    fn build_messages(messages: &[ChatMessage]) -> Vec<OpenAIMessage> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                    _ => "user",
                };

                let tool_calls = if m.tool_calls.is_empty() {
                    None
                } else {
                    Some(
                        m.tool_calls
                            .iter()
                            .map(|tc| OpenAIToolCall {
                                id: tc.id.clone(),
                                r#type: "function".to_string(),
                                function: OpenAIFunction {
                                    name: tc.name.clone(),
                                    arguments: serde_json::to_string(&tc.arguments)
                                        .unwrap_or_default(),
                                },
                            })
                            .collect(),
                    )
                };

                OpenAIMessage {
                    role: role.to_string(),
                    content: if m.content.is_empty() {
                        None
                    } else {
                        Some(m.content.clone())
                    },
                    tool_calls,
                    tool_call_id: m.tool_call_id.clone(),
                }
            })
            .collect()
    }

    fn build_tools(tools: &[ToolSpec]) -> Option<Vec<OpenAITool>> {
        if tools.is_empty() {
            return None;
        }
        Some(
            tools
                .iter()
                .map(|t| OpenAITool {
                    r#type: "function".to_string(),
                    function: OpenAIToolFunction {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                })
                .collect(),
        )
    }

    fn parse_response(body: OpenAIResponse) -> Result<ModelResponse, ProviderError> {
        let choice = body
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Parse("No choices in response".into()))?;

        if let Some(tool_calls) = choice.message.tool_calls {
            if !tool_calls.is_empty() {
                let calls = tool_calls
                    .into_iter()
                    .map(|tc| {
                        let args = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        ToolCall {
                            id: tc.id,
                            name: tc.function.name,
                            arguments: args,
                        }
                    })
                    .collect();
                return Ok(ModelResponse::ToolCalls(calls));
            }
        }

        let text = choice.message.content.unwrap_or_default();
        if contains_control_tokens(&text) {
            let stripped = strip_control_tokens(&text);
            if stripped.trim().is_empty() {
                return Err(ProviderError::MalformedModelOutput);
            }
            tracing::warn!("Provider returned control tokens, stripping them");
            Ok(ModelResponse::Text(stripped))
        } else {
            Ok(ModelResponse::Text(text))
        }
    }
}

#[async_trait::async_trait]
impl ModelAdapter for OpenAICompatibleAdapter {
    async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let body = OpenAIRequest {
            model: self.model.clone(),
            messages: Self::build_messages(&request.messages),
            tools: Self::build_tools(&request.tools),
            stream: false,
        };

        let response = retry::send_with_retry(|| async {
            let mut req = self.client.post(&url).json(&body);
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }
            req.send().await.map_err(ProviderError::Http)
        })
        .await?;

        let response_body: OpenAIResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("Failed to deserialize response: {e}")))?;

        Self::parse_response(response_body)
    }
}

// --- Serde models ---

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAIToolCall {
    id: String,
    r#type: String,
    function: OpenAIFunction,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OpenAITool {
    r#type: String,
    function: OpenAIToolFunction,
}

#[derive(Serialize)]
struct OpenAIToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[cfg(test)]
#[path = "openai_tests.rs"]
mod openai_tests;
