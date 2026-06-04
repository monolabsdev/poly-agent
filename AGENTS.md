# poly-agent — cheat sheet for agents

## Build & test
```bash
cargo build -p <crate>    # no default member, always -p
cargo test                # all 42 tests, 0 skips
cargo clippy              # no config file exists
```
No formatter config, no CI, no pre-commit hooks, no toolchain pin.

## Workspace layout
```
poly-agent-core  ← poly-agent-providers  ← poly-agent-runtime  ← poly-agent-tools  ← cli-basic
```
4 crates + 1 example (linear dep, no cycles). `cli-basic` workspace member (`examples/cli-basic/`), not `[[example]]`.

## Test pattern
Tests in separate files (`*_tests.rs`) via attribute at bottom of source file:
```rust
#[cfg(test)]
#[path = "openai_tests.rs"]
mod openai_tests;
```
Tests access `pub(crate)` + private items. No `tests/` dir.

Test files at:
- `crates/poly-agent-providers/src/openai_tests.rs` (7 tests)
- `crates/poly-agent-providers/src/ollama_tests.rs` (5 tests)
- `crates/poly-agent-runtime/src/run_tests.rs` (14 tests)
- `crates/poly-agent-runtime/src/intent.rs` has inline `#[cfg(test)] mod tests`

## Tool safety model
| Risk | Tools | Execution |
|---|---|---|
| `Safe` | `list_files`, `read_file`, `search_files`, `propose_edit` | Auto-executed |
| `RequiresApproval` | `apply_patch`, `write_file` | Pauses for user y/N |
| `Dangerous` | `run_command` | Blocked by default |

**`write_file` + `run_command` stubs** — return `"not yet implemented"` with `is_error: true`.

## Edit-intent guard (`intent.rs`)
- `is_mutating_tool("apply_patch")` / `is_mutating_tool("write_file")` — checks `MUTATING_TOOLS` array
- `EditIntent::detect(prompt)` — matches 18 edit verbs
- `sanitise_final_text()` — strips false success claims (e.g. "I have updated the file") when no mutating tool ran
- `contains_codebase_intent_phrase(text)` — matches 12 codebase-explanation phrases; triggers zero-tool guard message

## Control tokens (`control.rs`)
Tokens like `<|start|>assistant<|channel|>analysis` stripped from model output in provider layer before returning to runtime. `only_control_tokens_returns_error` test validates edge case.

## Approval flow
`AgentRuntime` stores `HashMap<(RunId, String), Sender<bool>>` pending approvals. `approve_tool()` / `reject_tool()` send bool via oneshot. `cli-basic` UI calls these from event handler on `AgentEvent::ApprovalRequired`.

## Quirks
- `pub const MUTATING_TOOLS` in **both** `runtime/src/intent.rs` (semantic guard) and `tools/src/lib.rs` (registration grouping, `register_mutation_tools`). Intentional duplication — different layers.
- `RunCommandTool` registered by `register_all_tools()` but `Dangerous` stub.
- `RuntimeLimits` has sensible `Default` — no need to set manually for basic use.

## Tauri integration
Future design in `docs/tauri-integration.md`. Plan: Tauri commands `agent_run`, `agent_cancel`, `agent_approve_tool_call`; events emitted via `app_handle.emit("poly-agent:event", ...)`.
