# Poly UI Integration

`poly-agent-tauri` is the Rust bridge for embedding poly-agent in Poly UI. It keeps Poly UI talking to stable run/event APIs while providers, tools, and runtime internals stay inside Rust crates.

## Workspace Attachment

Poly UI should pass `workspace_path` when a chat is attached to a folder. The manager validates that path, canonicalizes it, stores it in `RunState`, and enables local file/project/edit tools.

When no workspace is attached, the run enters chat-only mode: `workspace_root: null`, `local_tools_enabled: false`, and local tools are not registered for the run.

## Tauri Commands

Feature-gated commands live behind the `tauri` feature:

```ts
agent_run(input): Promise<RunId>
agent_cancel(runId): Promise<void>
agent_approve_tool_call(runId, approvalId): Promise<void>
agent_reject_tool_call(runId, approvalId): Promise<void>
agent_get_run_state(runId): Promise<RunStateSnapshot>
```

All commands are designed to use Tauri managed state containing `AgentRunManager`. Build the manager with `tauri_event_sink(app_handle)` so each run event is emitted to the frontend.

## Event Stream

Rust emits one Tauri event name:

```text
poly-agent:event
```

Payload shape:

```json
{
  "run_id": "uuid",
  "event_type": "approval_required",
  "timestamp": { "secs_since_epoch": 0, "nanos_since_epoch": 0 },
  "data": {}
}
```

Common `event_type` values:

```text
started
thinking
model_call_started
text_delta
tool_call_requested
tool_call_started
tool_call_finished
approval_required
finished
failed
cancelled
```

Approval event payload:

```json
{
  "kind": "approval_required",
  "value": {
    "approval_id": "call_1",
    "tool_name": "apply_patch",
    "risk": "RequiresApproval",
    "reason": "Rename project title",
    "path": "README.md",
    "command_preview": null,
    "diff_preview": "--- expected\n+++ replacement\n-old\n+new"
  }
}
```

`raw_arguments` is included only when debug mode is enabled.

## Frontend Wrapper Draft

```ts
export async function runAgent(input: AgentRunInput) {
  return invoke<string>("agent_run", { input })
}

export async function cancelAgent(runId: string) {
  return invoke("agent_cancel", { runId })
}

export async function approveToolCall(runId: string, approvalId: string) {
  return invoke("agent_approve_tool_call", { runId, approvalId })
}

export async function rejectToolCall(runId: string, approvalId: string) {
  return invoke("agent_reject_tool_call", { runId, approvalId })
}

export function listenToAgentEvents(handler: (event: AgentEvent) => void) {
  return listen<AgentEvent>("poly-agent:event", (event) => handler(event.payload))
}
```

## UI Placement

Place Agent Mode toggle in chat input or header. Show Agent Activity as a disclosure under assistant messages, backed by stored run events. Render approval cards for diffs and commands using `approval_required` payloads.

Cancellation is cooperative at manager level today: it marks the run cancelled, emits `cancelled`, aborts the run task, and leaves provider-level cancellation as future work.
