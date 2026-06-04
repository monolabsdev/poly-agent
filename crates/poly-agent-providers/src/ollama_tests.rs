use super::*;

#[test]
fn parse_ollama_text_response() {
    let json = serde_json::json!({
        "message": { "role": "assistant", "content": "Here is the list of files." }
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
        "message": { "role": "assistant", "content": "" }
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
            "role": "assistant", "content": "",
            "tool_calls": [{ "function": { "name": "list_files", "arguments": { "path": "." } } }]
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

#[test]
fn control_tokens_detected_and_stripped() {
    let json = serde_json::json!({
        "message": { "role": "assistant", "content": "<|start|>assistant<|channel|>analysis\nHere is the answer." }
    });
    let body: OllamaResponse = serde_json::from_value(json).unwrap();
    let result = OllamaAdapter::parse_response(body).unwrap();
    match result {
        ModelResponse::Text(t) => assert_eq!(t, "Here is the answer."),
        _ => panic!("Expected text response"),
    }
}

#[test]
fn only_control_tokens_returns_error() {
    let json = serde_json::json!({
        "message": { "role": "assistant", "content": "<|channel|>\n" }
    });
    let body: OllamaResponse = serde_json::from_value(json).unwrap();
    let result = OllamaAdapter::parse_response(body);
    assert!(matches!(result, Err(ProviderError::MalformedModelOutput)));
}
