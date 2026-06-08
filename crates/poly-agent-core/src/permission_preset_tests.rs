use crate::{AgentInput, ModelConfig, ModelProvider, PermissionPreset, RuntimeLimits};

#[test]
fn default_preset_is_default_variant() {
    assert_eq!(PermissionPreset::default(), PermissionPreset::Default);
}

#[test]
fn serde_uses_kebab_case_for_all_presets() {
    for (preset, expected) in [
        (PermissionPreset::Default, "default"),
        (PermissionPreset::AutoReview, "auto-review"),
        (PermissionPreset::FullAccess, "full-access"),
    ] {
        let json = serde_json::to_string(&preset).unwrap();
        assert_eq!(json, format!("\"{expected}\""), "preset {preset:?}");

        let parsed: PermissionPreset = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, preset, "roundtrip failed for {preset:?}");
    }
}

#[test]
fn agent_input_defaults_to_default_preset() {
    let input = AgentInput {
        prompt: "hi".to_string(),
        workspace: std::path::PathBuf::from("/tmp"),
        model: ModelConfig {
            provider: ModelProvider::Ollama,
            model: "test".to_string(),
            base_url: None,
            api_key: None,
        },
        limits: RuntimeLimits::default(),
        permission_preset: PermissionPreset::default(),
        resolved_context: None,
    };
    assert_eq!(input.permission_preset, PermissionPreset::Default);
}

#[test]
fn old_json_without_preset_field_still_deserializes() {
    let legacy = r#"{
        "prompt": "summarise this",
        "workspace": "/tmp",
        "model": {
            "provider": "Ollama",
            "model": "gpt-oss"
        },
        "limits": {
            "max_steps": 8,
            "max_file_read_bytes": 262144,
            "max_tool_output_bytes": 65536,
            "max_search_results": 50,
            "max_context_messages": 32,
            "command_timeout_secs": 60
        }
    }"#;
    let parsed: AgentInput = serde_json::from_str(legacy).unwrap();
    assert_eq!(parsed.prompt, "summarise this");
    assert_eq!(parsed.permission_preset, PermissionPreset::Default);
}

#[test]
fn new_json_with_preset_field_round_trips() {
    let json = r#"{
        "prompt": "run tests",
        "workspace": "/tmp",
        "model": {
            "provider": "Ollama",
            "model": "gpt-oss"
        },
        "limits": {
            "max_steps": 8,
            "max_file_read_bytes": 262144,
            "max_tool_output_bytes": 65536,
            "max_search_results": 50,
            "max_context_messages": 32,
            "command_timeout_secs": 60
        },
        "permission_preset": "auto-review"
    }"#;
    let parsed: AgentInput = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.permission_preset, PermissionPreset::AutoReview);
}
