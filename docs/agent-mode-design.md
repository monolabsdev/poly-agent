# Agent Mode: poly-agent Runtime Enhancements

## Scope

All changes are in the `poly-agent` Rust crates. No Poly UI frontend code is modified. After implementation, this document explains how to wire the enhanced runtime into the existing Poly UI agent infrastructure.

## Current State

The poly-agent runtime already has:
- Multi-step agent loop with tool calling (`run.rs`)
- 8 working tools (list_files, read_file, search_files, propose_edit, apply_patch, write_file, inspect_project, read_important_files)
- Approval flow for mutating tools via oneshot channels
- Event system (`AgentEvent` enum with 11 variants)
- Tauri bridge (`poly-agent-tauri`) with `AgentRunManager`, event mapping, cancellation
- Provider adapters (OpenAI-compatible, Ollama) with streaming support

**What's missing:**
1. `run_command` is a stub (returns "not yet implemented")
2. No `suggest_command` tool
3. Agent loop falls back to non-streaming `chat()` when tools are present
4. No cancellation token propagation into the runtime (only manager-level abort)
5. No workspace boundary enforcement for shell commands
6. No command safety/classification model

## Architecture Changes

### 1. `run_command` Tool Implementation

**File:** `crates/poly-agent-tools/src/run_command.rs`

Replace the stub with a real implementation that:

- Executes shell commands via `tokio::process::Command`
- Runs in the workspace root by default
- Captures stdout, stderr, exit code, and duration
- Streams output via a channel (for long-running commands)
- Respects workspace boundaries
- Requires approval based on command risk classification

**Parameters schema:**
```json
{
  "type": "object",
  "properties": {
    "command": { "type": "string", "description": "Shell command to execute" },
    "cwd": { "type": "string", "description": "Working directory override (must be within workspace)" }
  },
  "required": ["command"]
}
```

**Output format (JSON):**
```json
{
  "exit_code": 0,
  "stdout": "file contents here",
  "stderr": "",
  "duration_ms": 142,
  "command": "ls -la",
  "workspace": "/path/to/workspace",
  "success": true
}
```

**Risk classification:**
- `run_command` remains `ToolRisk::Dangerous` (requires approval before execution)
- The tool itself does the risk classification internally for informational purposes, but the approval gate is handled by the runtime's existing approval flow

**Safety model:**
- Commands run with `current_dir` set to workspace root (or validated `cwd` override)
- `cwd` override is validated: must be a subdirectory of workspace root, path traversal blocked
- Process group is created so cancellation can kill the entire tree
- Timeout: 60 seconds default, configurable via `RuntimeLimits`
- Output truncation: respects `max_tool_output_bytes`
- No shell injection: commands are run via `Command::new("sh").arg("-c").arg(command)` on Unix, `Command::new("cmd").arg("/C").arg(command)` on Windows

**Cancellation:** The tool receives a `CancellationToken` through `ToolContext` (new field). When cancelled, the child process is killed via `kill_on_drop(true)` and the process group is terminated.

### 2. `suggest_command` Tool

**File:** `crates/poly-agent-tools/src/suggest_command.rs`

A new safe tool that formats a command suggestion without executing anything.

**Parameters schema:**
```json
{
  "type": "object",
  "properties": {
    "command": { "type": "string", "description": "The command to suggest" },
    "explanation": { "type": "string", "description": "Why this command is needed" },
    "risk_level": { "type": "string", "enum": ["low", "medium", "high"], "description": "Estimated risk" },
    "expected_outcome": { "type": "string", "description": "What the command should produce" }
  },
  "required": ["command", "explanation"]
}
```

**Risk:** `ToolRisk::Safe` — no approval needed, no execution happens.

**Output:** JSON with the structured suggestion data, which the Tauri bridge maps to `AgentUiEventPayload::ToolCallFinished` with the JSON as output. The frontend renders it as a suggestion card.

**Registration:** Added to `register_safe_tools()` in `lib.rs`.

### 3. Streaming With Tools Present

**File:** `crates/poly-agent-runtime/src/run.rs`

Modify `call_model()` to use `chat_stream()` even when tools are present. The current code:

```rust
async fn call_model(&self, request: ChatRequest, ctx: &RunContext) -> Result<ModelResponse, ProviderError> {
    if !request.tools.is_empty() {
        return self.adapter.chat(request).await;  // <-- non-streaming
    }
    // ... streaming path
}
```

**New approach:**
- Always use `adapter.chat_stream(request)` regardless of whether tools are present
- Accumulate text deltas and emit `TextDelta` events as they arrive
- When a `ToolCalls` variant arrives in the stream, return `ModelResponse::ToolCalls`
- When the stream ends with only text, return `ModelResponse::Text`
- If the provider's `chat_stream()` default implementation wraps `chat()` in a single-element stream (the current fallback), this still works — the stream yields one `ModelResponse` chunk

**Key insight:** The `ModelStream` type is `Pin<Box<dyn Stream<Item = Result<ModelResponse, ProviderError>>>>`. Each item is already a full `ModelResponse` (either `Text` or `ToolCalls`). The OpenAI adapter already parses SSE chunks and accumulates tool calls. So streaming with tools "just works" if we always use `chat_stream()`.

**Benefit:** Users see assistant text streaming in real-time even when the model will eventually call tools. The agent feels responsive instead of showing nothing until the full response arrives.

### 4. Cancellation Token in ToolContext

**File:** `crates/poly-agent-runtime/src/tool.rs`

Add a `CancellationToken` to `ToolContext`:

```rust
pub struct ToolContext {
    pub workspace: PathBuf,
    pub limits: RuntimeLimits,
    pub cancellation: tokio_util::sync::CancellationToken,
}
```

**Propagation:** The runtime creates a `CancellationToken` per run and passes it through `ToolContext`. The `run_command` tool checks this token and kills the child process on cancellation.

**Backward compatibility:** Existing tools ignore the token (they don't need it). The new field has a default in construction.

### 5. Runtime-Level Cancellation

**File:** `crates/poly-agent-runtime/src/run.rs`

Add cancellation awareness to the agent loop:

- Accept a `CancellationToken` parameter in `AgentRuntime::run()`
- Check `cancellation.is_cancelled()` at the start of each step
- When cancelled, emit `AgentEvent::Error` with a cancellation message and return early
- The existing manager-level abort (`handle.abort()`) still works as a hard kill

**Signature change:**
```rust
pub async fn run(
    &self,
    input: AgentInput,
    event_tx: mpsc::Sender<AgentEvent>,
    cancellation: CancellationToken,
) -> Result<AgentOutput, AgentError>
```

The manager (`poly-agent-tauri/src/manager.rs`) already has a `CancellationToken` per run slot — just pass it through to `runtime.run()`.

### 6. Command Safety Classification

**File:** `crates/poly-agent-tools/src/run_command.rs` (inline)

A conservative pattern-matching classifier that determines if a command is safe, needs approval, or is outright dangerous.

**Classification rules:**

| Pattern | Risk | Action |
|---|---|---|
| `ls`, `pwd`, `echo`, `cat`, `head`, `tail`, `wc`, `file`, `stat`, `which`, `whoami` | Safe-ish | Still requires approval (all run_command is Dangerous) |
| `rm -rf /`, `sudo`, `chmod 777`, `mkfs`, `dd` | Dangerous | Blocked even with approval |
| `rm`, `mv`, `cp`, `rmdir` | Dangerous | Requires approval |
| `git commit`, `git push`, `git add`, `git checkout` | Dangerous | Requires approval |
| `npm install`, `bun install`, `cargo install`, `pip install` | Dangerous | Requires approval |
| `curl`, `wget`, `ssh`, `scp` | Dangerous | Requires approval |
| Commands with `|`, `&&`, `||`, `>`, `>>`, backticks | Dangerous | Requires approval |
| Anything else | Dangerous | Requires approval |

**Note:** Since `run_command` is registered as `ToolRisk::Dangerous`, ALL commands require approval through the runtime's existing approval flow. The classification is informational (included in the approval payload) but doesn't bypass the approval gate.

### 7. Event Type Updates

**File:** `crates/poly-agent-core/src/events.rs`

Add two new event variants for command output streaming:

```rust
pub enum AgentEvent {
    // ... existing variants ...
    ToolCallDelta {
        run_id: RunId,
        tool_call_id: String,
        delta: String,
    },
}
```

This allows `run_command` to stream partial stdout/stderr as it runs. The Tauri bridge maps this to `AgentUiEventPayload::ToolCallDelta`.

**File:** `crates poly-agent-tauri/src/events.rs`

Add mapping for the new event:
```rust
AgentEvent::ToolCallDelta { tool_call_id, delta, .. } => (
    "tool_call_delta",
    AgentUiEventPayload::ToolCallDelta { tool_call_id, delta },
),
```

### 8. System Prompt Updates

**File:** `crates/poly-agent-runtime/src/run.rs`

Update `build_system_prompt()` and `VALID_TOOL_NAMES` to include the new tools:

```rust
const VALID_TOOL_NAMES: &[&str] = &[
    "list_files",
    "read_file",
    "search_files",
    "propose_edit",
    "apply_patch",
    "write_file",
    "run_command",
    "suggest_command",
    "inspect_project",
    "read_important_files",
];
```

Add tool guidance to the system prompt:
```
- For running shell commands, use `run_command`. It requires approval. Commands run in the workspace root.
- For suggesting commands without executing, use `suggest_command`. It returns a structured suggestion.
- Prefer `suggest_command` when you are unsure whether a command is safe.
```

## File Changes Summary

| File | Change |
|---|---|
| `crates/poly-agent-core/src/events.rs` | Add `ToolCallDelta` event variant |
| `crates/poly-agent-core/src/types.rs` | Add `command_timeout_secs` to `RuntimeLimits` |
| `crates/poly-agent-runtime/src/tool.rs` | Add `CancellationToken` to `ToolContext` |
| `crates/poly-agent-runtime/src/run.rs` | Always use `chat_stream()`, add cancellation param, update system prompt + VALID_TOOL_NAMES |
| `crates/poly-agent-runtime/src/engine.rs` | Update `run()` signature to accept cancellation token |
| `crates/poly-agent-tools/Cargo.toml` | Add `tokio-util` dependency |
| `crates/poly-agent-tools/src/run_command.rs` | Full implementation |
| `crates/poly-agent-tools/src/suggest_command.rs` | New file |
| `crates/poly-agent-tools/src/lib.rs` | Register `SuggestCommandTool`, add `register_agent_tools()` |
| `crates/poly-agent-tauri/src/events.rs` | Map `ToolCallDelta` event |
| `crates/poly-agent-tauri/src/manager.rs` | Pass cancellation token to `runtime.run()` |
| `crates/poly-agent-tauri/src/workspace.rs` | Register agent tools for workspaces |

## Testing Strategy

### Unit Tests

Add to existing test files:

**`crates/poly-agent-tools/src/run_command.rs`:**
- `run_command_captures_stdout` — run `echo hello`, verify stdout
- `run_command_captures_stderr` — run command that writes to stderr
- `run_command_returns_exit_code` — run `exit 1`, verify exit_code = 1
- `run_command_respects_timeout` — run `sleep 30` with short timeout, verify timeout error
- `run_command_cwd_validation` — verify cwd outside workspace is rejected
- `run_command_path_traversal` — verify `../../etc/passwd` is blocked

**`crates/poly-agent-tools/src/suggest_command.rs`:**
- `suggest_command_returns_structured_output` — verify JSON output shape
- `suggest_command_does_not_execute` — verify no process is spawned

**`crates/poly-agent-runtime/src/run_tests.rs`:**
- `streaming_text_before_tool_call` — verify TextDelta events arrive before ToolCallRequested
- `cancellation_stops_agent_loop` — cancel mid-run, verify cancelled state
- `run_command_requires_approval` — verify approval flow for run_command

### Manual Testing Checklist

```
1. Start a run with a workspace attached
2. Ask "list files in the current directory"
   - Verify: TextDelta events stream before tool call
   - Verify: ToolCallRequested -> ToolCallStarted -> ToolCallFinished
3. Ask "run `echo hello`"
   - Verify: ApprovalRequired event fires
   - Approve: verify stdout contains "hello"
4. Ask "suggest running `bun install`"
   - Verify: ToolCallFinished with structured JSON output
   - Verify: No process was spawned
5. Ask "delete the entire project"
   - Verify: ApprovalRequired with high-risk classification
   - Reject: verify "Tool execution denied" message
6. During a long command, click Cancel
   - Verify: Run status becomes "cancelled"
   - Verify: No zombie processes remain
7. Ask a question that triggers multiple tool calls
   - Verify: Assistant text streams between tool calls
   - Verify: Activity timeline shows all events
8. Test with no workspace attached (chat-only mode)
   - Verify: run_command and suggest_command are not available
   - Verify: read-only tools still work
```

## Poly UI Integration Guide

The existing Poly UI agent infrastructure (`src/features/agent/`) already handles:
- Tauri command invocation (`agentClient.ts`)
- Event subscription (`listenToAgentEvents`)
- Activity timeline rendering (`AgentActivityDisclosure`)
- Approval UI (`AgentApprovalBar`)
- Workspace selection (`AgentWorkspaceSelector`)

### What's Already Wired

| Feature | Status |
|---|---|
| `agent_run` command | ✅ Wired in `agentClient.ts` |
| `agent_cancel` command | ✅ Wired in `agentClient.ts` |
| `agent_approve_tool_call` | ✅ Wired in `agentClient.ts` |
| `agent_reject_tool_call` | ✅ Wired in `agentClient.ts` |
| `poly-agent:event` listener | ✅ Wired in `agentClient.ts` |
| Agent activity timeline | ✅ `AgentActivityDisclosure.tsx` |
| Approval bar | ✅ `AgentApprovalBar.tsx` |
| Workspace selector | ✅ `AgentWorkspaceSelector.tsx` |
| Agent mode toggle | ✅ `agentStore.enabled` |
| Run state management | ✅ `useAgentRun.ts` |

### New Events to Handle in Frontend

After these poly-agent changes, the frontend will receive new event types:

1. **`tool_call_delta`** — partial output from `run_command` as it streams
2. **`ToolCallFinished` with `suggest_command` output** — structured JSON for command suggestions

### Frontend Changes Needed (Future)

These are NOT part of the current poly-agent scope, but document what Poly UI should add:

#### 1. Render `tool_call_delta` for command output streaming

In the agent activity timeline, when a `tool_call_started` event has `tool_name: "run_command"`, subsequent `tool_call_delta` events should be appended to a live output area.

```typescript
// In appendAgentEvent() in activity.ts
case "tool_call_delta":
  // Append delta.text to the current tool call's output buffer
  // Render as a live-updating <pre> block in the timeline
```

#### 2. Render `suggest_command` output as a card

When `tool_call_finished` arrives with `tool_name: "suggest_command"`, parse the JSON output and render a suggestion card:

```typescript
// In AgentActivityDisclosure.tsx
const suggestion = JSON.parse(output);
// Render: command, explanation, risk_level badge, expected_outcome
// Actions: Copy command, Run (triggers run_command with same command), Approve
```

#### 3. Command output display in timeline

For `run_command` tool calls, show:
- The command that was run
- Exit code (with color: green for 0, red for non-zero)
- Duration
- Stdout/stderr in a collapsible `<pre>` block
- Whether it was cancelled

#### 4. Agent mode wiring

Ensure the agent mode toggle in `ChatInput.tsx` or `Header.tsx` is connected to `agentStore.enabled`. When enabled:
- User messages route through `sendAgentMessage()` instead of `sendMessage()`
- The agent activity timeline appears below assistant messages
- Approval cards appear for tool calls requiring approval
- Cancel button appears during active runs

### Build & Test

```bash
# In poly-agent repo
cargo test                    # All 42+ tests pass
cargo clippy                  # No warnings

# In poly-ui repo
cargo build --features tauri  # Build with poly-agent-tauri
bun run tauri dev             # Start the app
```

## Limitations & Follow-up Work

1. **No provider-level cancellation** — Model API calls are not aborted when cancelled. The runtime checks cancellation between steps but an in-flight HTTP request to the LLM will complete. Full provider cancellation requires abort handles in the `ModelAdapter` trait.

2. **Command timeout is hardcoded** — `RuntimeLimits` should gain a `command_timeout_secs` field. For now, 60 seconds is a reasonable default.

3. **No command history/audit log** — Commands executed by `run_command` are not persisted beyond the event buffer. A future enhancement could log them to SQLite.

4. **Shell injection is limited, not eliminated** — Running via `sh -c` is standard but means the model could craft malicious arguments. The approval gate is the primary defense.

5. **No streaming of tool output to model** — The model receives the full tool output after completion, not partial streams. This is fine for most tools but means long-running commands produce a single large tool result.

6. **`delete_file` and `rename_file` are still not implemented** — Referenced in `MUTATING_TOOLS` but have no tool implementation. These are separate from the agent mode scope.
