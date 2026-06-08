# Poly UI Integration — Agent Mode

This document explains how to wire the enhanced `poly-agent` runtime into the existing Poly UI frontend for Agent Mode.

## What's Already Wired

Poly UI (`src/features/agent/`) already has complete agent infrastructure:

| Feature | File | Status |
|---|---|---|
| Tauri command wrappers | `agentClient.ts` | `runAgent`, `cancelAgent`, `approveAgentToolCall`, `rejectAgentToolCall` |
| Event subscription | `agentClient.ts` | `listenToAgentEvents()` → `listen("poly-agent:event", ...)` |
| Agent activity timeline | `AgentActivityDisclosure.tsx` | Renders `AgentActivityItem[]` |
| Approval bar | `AgentApprovalBar.tsx` | Approve/reject with file review |
| Workspace selector | `AgentWorkspaceSelector.tsx` | Project/sandbox selection |
| Agent mode toggle | `agentStore.ts` | `enabled` state per chat |
| Run lifecycle | `useAgentRun.ts` | Start, stream, cancel, finish |

## New Events to Handle

After the poly-agent changes, two new event types arrive via `poly-agent:event`:

### 1. `tool_call_delta` — Streaming command output

Emitted by `run_command` as stdout/stderr streams in. Payload:

```json
{
  "run_id": "...",
  "event_type": "tool_call_delta",
  "data": {
    "kind": "tool_call_delta",
    "value": {
      "tool_call_id": "call_1",
      "delta": "partial output text"
    }
  }
}
```

**Frontend handling:** In `appendAgentEvent()` (`activity.ts`), when a `tool_call_started` event has `tool_name: "run_command"`, subsequent `tool_call_delta` events append to a live output buffer for that tool call. Render as a live-updating `<pre>` block.

### 2. `suggest_command` output — Structured command suggestion

When `tool_call_finished` arrives with `tool_name: "suggest_command"`, the output is structured JSON:

```json
{
  "command": "bun install",
  "explanation": "Install workspace dependencies",
  "risk_level": "medium",
  "expected_outcome": "node_modules populated, lockfile updated",
  "type": "command_suggestion"
}
```

**Frontend handling:** Parse the JSON output and render as a suggestion card with:
- Command text
- Explanation
- Risk level badge (low/medium/high)
- Expected outcome
- Actions: Copy, Run (triggers `run_command`), Approve

### 3. Updated `tool_call_finished` for `run_command`

Output format for `run_command`:

```
Command: echo hello
Working directory: /path/to/workspace
Exit code: 0
Status: success
Duration: 42ms

Stdout:
hello
```

**Frontend handling:** Parse this structured output and render:
- Command string
- Exit code with color (green for 0, red for non-zero)
- Duration
- Collapsible stdout/stderr sections
- Whether it was cancelled (check for "timed out" in output)

## What Needs Frontend Changes

### In `activity.ts` — `appendAgentEvent()`

Add handling for the new `tool_call_delta` event kind:

```typescript
case "tool_call_delta":
  // Find the current tool call in the activity timeline
  // Append delta.value.delta to its output buffer
  // Trigger re-render of the live output area
  break;
```

### In `AgentActivityDisclosure.tsx`

Add rendering for:
- Live command output (for `run_command` tool calls with streaming deltas)
- Command suggestion cards (for `suggest_command` tool calls)
- Exit code badges, duration display
- Collapsible stdout/stderr sections

### In `useAgentRun.ts`

The existing hook already handles:
- Starting runs via `runAgent()`
- Listening to events via `listenToAgentEvents()`
- Building activity timeline via `appendAgentEvent()`
- Handling finish/fail/cancel states

No changes needed in the hook — the new event types flow through the existing pipeline.

## Build & Test

```bash
# In poly-agent repo
cargo test                    # All 83 tests pass
cargo clippy                  # No new warnings

# In poly-ui repo
bun run tauri dev             # Start the app
```

## Manual Testing Checklist

```
1. Enable Agent Mode via the toggle
2. Select a workspace (folder/project)
3. Ask "list files in the current directory"
   - Verify: TextDelta streams before tool call
   - Verify: ToolCallRequested → ToolCallStarted → ToolCallFinished
4. Ask "run `echo hello`"
   - Verify: ApprovalRequired fires
   - Approve: verify stdout shows "hello"
   - Verify: Exit code 0, duration displayed
5. Ask "suggest running `bun install`"
   - Verify: ToolCallFinished with JSON output
   - Verify: No process was spawned
6. Ask "delete the entire project"
   - Verify: ApprovalRequired with high-risk
   - Reject: verify "Tool execution denied" message
7. During a long command, click Cancel
   - Verify: Run status becomes "cancelled"
   - Verify: No zombie processes
8. Ask a multi-step question
   - Verify: Assistant text streams between tool calls
   - Verify: Activity timeline shows all events
9. Test with no workspace (chat-only mode)
   - Verify: run_command/suggest_command unavailable
   - Verify: read-only tools still work
```

## Limitations

1. **No provider-level cancellation** — In-flight LLM HTTP requests complete even after cancel. Cancellation only takes effect between steps.
2. **Command timeout is 60s default** — Configurable via `RuntimeLimits.command_timeout_secs` but not yet exposed in the frontend.
3. **No command audit log** — Commands are not persisted beyond the event buffer (max 500 events per run).
4. **Shell injection is limited, not eliminated** — Commands run via `sh -c` / `cmd /C`. The approval gate is the primary defense.
5. **`delete_file`/`rename_file` still not implemented** — Referenced in `MUTATING_TOOLS` but not yet built.
