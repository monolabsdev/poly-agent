use super::*;

#[test]
fn parse_text_response() {
    let json = serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": "Hello! How can I help you?" }
        }]
    });
    let body: OpenAIResponse = serde_json::from_value(json).unwrap();
    let result = OpenAICompatibleAdapter::parse_response(body).unwrap();
    match result {
        ModelResponse::Text(t) => assert_eq!(t, "Hello! How can I help you?"),
        _ => panic!("Expected text response"),
    }
}

#[test]
fn parse_tool_call_response() {
    let json = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant", "content": null,
                "tool_calls": [{
                    "id": "call_123", "type": "function",
                    "function": { "name": "list_files", "arguments": "{\"path\": \".\"}" }
                }]
            }
        }]
    });
    let body: OpenAIResponse = serde_json::from_value(json).unwrap();
    let result = OpenAICompatibleAdapter::parse_response(body).unwrap();
    match result {
        ModelResponse::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "list_files");
            assert_eq!(calls[0].id, "call_123");
            assert_eq!(calls[0].arguments["path"], ".");
        }
        _ => panic!("Expected tool calls"),
    }
}

#[test]
fn parse_multiple_tool_calls() {
    let json = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant", "content": null,
                "tool_calls": [
                    { "id": "call_1", "type": "function", "function": { "name": "list_files", "arguments": "{\"path\": \".\"}" } },
                    { "id": "call_2", "type": "function", "function": { "name": "read_file", "arguments": "{\"path\": \"README.md\"}" } }
                ]
            }
        }]
    });
    let body: OpenAIResponse = serde_json::from_value(json).unwrap();
    let result = OpenAICompatibleAdapter::parse_response(body).unwrap();
    match result {
        ModelResponse::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].name, "list_files");
            assert_eq!(calls[1].name, "read_file");
        }
        _ => panic!("Expected tool calls"),
    }
}

#[test]
fn parse_empty_choices_fails() {
    let body: OpenAIResponse = serde_json::from_value(serde_json::json!({ "choices": [] })).unwrap();
    assert!(OpenAICompatibleAdapter::parse_response(body).is_err());
}

#[test]
fn control_tokens_detected_and_stripped() {
    let json = serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": "<|start|>assistant<|channel|>commentary\nHello world." }
        }]
    });
    let body: OpenAIResponse = serde_json::from_value(json).unwrap();
    let result = OpenAICompatibleAdapter::parse_response(body).unwrap();
    match result {
        ModelResponse::Text(t) => assert_eq!(t, "Hello world."),
        _ => panic!("Expected text response"),
    }
}

#[test]
fn only_control_tokens_returns_error() {
    let json = serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": "<|start|>assistant<|channel|>analysis" }
        }]
    });
    let body: OpenAIResponse = serde_json::from_value(json).unwrap();
    let result = OpenAICompatibleAdapter::parse_response(body);
    assert!(matches!(result, Err(ProviderError::MalformedModelOutput)));
}
