use crate::{
    error::ProviderError,
    traits::{ModelResponse, ToolSpec},
};
use ollama_rs::generation::{
    chat::{request::ChatMessageRequest, ChatMessage as OllamaChatMessage, ChatMessageResponse},
    tools::{ToolCall as OllamaToolCall, ToolFunctionInfo, ToolInfo, ToolType},
};
use poly_agent_core::{ChatMessage, ChatRole, ToolCall};
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

pub(crate) fn build_tools(tools: &[ToolSpec]) -> Vec<ToolInfo> {
    tools
        .iter()
        .map(|tool| ToolInfo {
            tool_type: ToolType::Function,
                function: ToolFunctionInfo {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: serde_json::from_value(tool.parameters.clone())
                        .unwrap_or_else(|_| serde_json::from_value(serde_json::json!({})).expect("empty schema")),
                },
            })
        .collect()
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

pub(crate) fn build_request(
    model: String,
    messages: Vec<OllamaChatMessage>,
    tools: Vec<ToolInfo>,
) -> ChatMessageRequest {
    let mut request = ChatMessageRequest::new(model, messages);
    request.tools = tools;
    request
}
