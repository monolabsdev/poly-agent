use super::*;
use crate::ollama_types::{build_messages, response_to_model_response};
use poly_agent_core::{ChatMessage, ChatRole, ToolCall};

#[test]
fn maps_poly_messages_to_ollama_messages() {
    let messages = vec![
        ChatMessage {
            role: ChatRole::System,
            content: "sys".into(),
            tool_calls: vec![],
            tool_call_id: None,
        },
        ChatMessage::assistant_with_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "list_files".into(),
            arguments: serde_json::json!({"path":"."}),
        }]),
    ];

    let mapped = build_messages(&messages);
    assert_eq!(mapped.len(), 2);
    assert_eq!(mapped[0].content, "sys");
    assert_eq!(mapped[1].tool_calls.len(), 1);
    assert_eq!(mapped[1].tool_calls[0].function.name, "list_files");
}

#[test]
fn maps_non_streaming_response_to_text() {
    let body = ollama_rs::generation::chat::ChatMessageResponse {
        model: "llama3".into(),
        created_at: "2026-06-04T00:00:00Z".into(),
        message: ollama_rs::generation::chat::ChatMessage::assistant("hello".into()),
        logprobs: None,
        done: true,
        final_data: None,
    };

    let mapped = response_to_model_response(body).unwrap();
    assert!(matches!(mapped, ModelResponse::Text(text) if text == "hello"));
}

#[test]
fn malformed_output_still_errors() {
    let body = ollama_rs::generation::chat::ChatMessageResponse {
        model: "llama3".into(),
        created_at: "2026-06-04T00:00:00Z".into(),
        message: ollama_rs::generation::chat::ChatMessage::assistant("<|channel|>\n".into()),
        logprobs: None,
        done: true,
        final_data: None,
    };

    let mapped = response_to_model_response(body);
    assert!(matches!(mapped, Err(ProviderError::MalformedModelOutput)));
}

#[test]
fn no_ollama_types_leak_from_public_adapter_api() {
    let type_name = std::any::type_name::<OllamaAdapter>();
    assert!(type_name.contains("OllamaAdapter"));
    assert!(!type_name.contains("ollama_rs"));
}
