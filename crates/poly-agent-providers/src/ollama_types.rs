use crate::{
    error::ProviderError,
    traits::{ModelResponse, ToolSpec},
};
use ollama_rs::generation::{
    chat::{ChatMessage as OllamaChatMessage, ChatMessageResponse},
    tools::ToolCall as OllamaToolCall,
};
use poly_agent_core::{ChatMessage, ChatRole, ToolCall};
use serde::Serialize;
use uuid::Uuid;

pub(crate) fn build_messages(messages: &[ChatMessage]) -> Vec<OllamaChatMessage> {
    messages
        .iter()
        .map(|message| {
            let mut ollama_message = match message.role {
                ChatRole::System => OllamaChatMessage::system(message.content.clone()),
                ChatRole::User => OllamaChatMessage::user(message.content.clone()),
                ChatRole::Assistant => OllamaChatMessage::assistant(message.content.clone()),
                ChatRole::Tool => OllamaChatMessage::tool(message.content.clone()),
                _ => OllamaChatMessage::user(message.content.clone()),
            };

            if !message.tool_calls.is_empty() {
                ollama_message.tool_calls = message
                    .tool_calls
                    .iter()
                    .map(|call| OllamaToolCall {
                        function: ollama_rs::generation::tools::ToolCallFunction {
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        },
                    })
                    .collect();
            }

            ollama_message
        })
        .collect()
}

/// Our own tool serialization that outputs `"type": "function"` (lowercase).
/// `ollama_rs::ToolType::Function` serializes as `"Function"` (PascalCase)
/// which Ollama does not recognise, silently dropping all tools.
#[derive(Serialize)]
pub(crate) struct OllamaTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: OllamaToolFunction,
}

#[derive(Serialize)]
pub(crate) struct OllamaToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

pub(crate) fn build_tools(tools: &[ToolSpec]) -> Vec<OllamaTool> {
    tools
        .iter()
        .map(|tool| OllamaTool {
            tool_type: "function",
            function: OllamaToolFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            },
        })
        .collect()
}

/// The full Ollama request body with correctly-serialised tools.
#[derive(Serialize)]
pub(crate) struct OllamaRequestBody {
    pub(crate) model: String,
    pub(crate) messages: Vec<OllamaChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<OllamaTool>,
    pub(crate) stream: bool,
}

pub(crate) fn build_request_body(
    model: String,
    messages: Vec<OllamaChatMessage>,
    tools: Vec<OllamaTool>,
) -> OllamaRequestBody {
    OllamaRequestBody {
        model,
        messages,
        tools,
        stream: false,
    }
}

pub(crate) fn response_to_model_response(
    response: ChatMessageResponse,
) -> Result<ModelResponse, ProviderError> {
    if !response.message.tool_calls.is_empty() {
        let calls = response
            .message
            .tool_calls
            .into_iter()
            .map(|call| ToolCall {
                id: Uuid::new_v4().to_string(),
                name: call.function.name,
                arguments: call.function.arguments,
            })
            .collect();
        return Ok(ModelResponse::ToolCalls(calls));
    }

    let text = response.message.content;
    if crate::contains_control_tokens(&text) {
        let stripped = crate::strip_control_tokens(&text);
        if stripped.trim().is_empty() {
            return Err(ProviderError::MalformedModelOutput);
        }
        tracing::warn!("Provider returned control tokens, stripping them");
        Ok(ModelResponse::Text(stripped))
    } else {
        Ok(ModelResponse::Text(text))
    }
}

