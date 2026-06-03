# poly-agent

A minimal, low-RAM Rust agent runtime for [Poly UI](https://github.com/nicholasgriffintn/poly-ui).

## What this is

- An internal AI agent runtime written in Rust
- Designed for the Poly UI Tauri desktop app
- Supports Ollama and OpenAI-compatible APIs
- Provides safe, sandboxed local tool execution
- Prioritises low memory usage and speed

## What this is NOT

- Not a general-purpose agent framework
- Not a plugin marketplace or SDK
- Not a multi-agent orchestration system
- Not a vector database or memory system

## Architecture

```
poly-agent/
├── crates/
│   ├── poly-agent-core/       # Shared types, events, errors
│   ├── poly-agent-providers/   # LLM adapters (Ollama, OpenAI-compatible)
│   ├── poly-agent-runtime/     # Agent loop, tool registry, event emission
│   └── poly-agent-tools/       # Built-in tools (list_files, read_file, search_files)
├── examples/
│   └── cli-basic/              # CLI example for testing
└── docs/
    └── tauri-integration.md    # Future Tauri integration design
```

## Quick Start

### With Ollama

```bash
cargo run -p cli-basic -- \
  --provider ollama \
  --model qwen2.5-coder:7b \
  --workspace . \
  --prompt "List the files in this project"
```

### With OpenAI-compatible API

```bash
cargo run -p cli-basic -- \
  --provider openai-compatible \
  --base-url http://localhost:1234/v1 \
  --model some-model \
  --workspace . \
  --prompt "What does this project do?"
```

## Development

```bash
cargo check      # Type-check all crates
cargo test       # Run all tests
cargo clippy     # Lint
```

## License

Internal — not yet published.
