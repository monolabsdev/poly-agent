# Tauri Integration Design

This document describes how `poly-agent` will integrate into the Poly UI Tauri app.

## Overview

The Rust runtime runs inside the Tauri backend. The TypeScript/React frontend communicates with it exclusively through Tauri's `invoke` (for commands) and `listen` (for events).

## Rust Commands

### `agent_run`

Starts an agent run. Returns the `RunId` immediately.

```rust
#[tauri::command]
async fn agent_run(
    state: tauri::State<'_, AppState>,
    prompt: String,
    provider: String,
    model: String,
    base_url: Option<String>,
    api_key: Option<String>,
    workspace: String,
) -> Result<String, String> {
    // Creates AgentInput, spawns the runtime on a background task,
    // and returns the RunId as a string.
}
```

### `agent_cancel`

Cancels a running agent by `RunId`.

```rust
#[tauri::command]
async fn agent_cancel(
    state: tauri::State<'_, AppState>,
    run_id: String,
) -> Result<(), String> {
    // Sends a cancellation signal via CancellationToken or similar.
}
```

### `agent_approve_tool_call`

Approves a pending tool call that requires user permission.

```rust
#[tauri::command]
async fn agent_approve_tool_call(
    state: tauri::State<'_, AppState>,
    run_id: String,
    tool_call_id: String,
    approved: bool,
) -> Result<(), String> {
    // Sends approval/denial through a channel to the runtime.
}
```

## Event Emission

The runtime emits `AgentEvent` values through a `tokio::sync::mpsc` channel. The Tauri plugin layer receives these and forwards them to the frontend:

```rust
app_handle.emit("poly-agent:event", &event)?;
```

## Frontend Usage (TypeScript)

### Starting a run

```typescript
import { invoke } from '@tauri-apps/api/core';

const runId = await invoke<string>('agent_run', {
  prompt: userMessage,
  provider: 'ollama',
  model: 'qwen2.5-coder:7b',
  workspace: '/path/to/project',
});
```

### Listening to events

```typescript
import { listen } from '@tauri-apps/api/event';

const unlisten = await listen<AgentEvent>('poly-agent:event', (event) => {
  const agentEvent = event.payload;
  switch (agentEvent.type) {
    case 'Started':
      // Show "Agent started" in timeline
      break;
    case 'ToolCallStarted':
      // Show tool execution spinner
      break;
    case 'ToolCallFinished':
      // Show tool result
      break;
    case 'ApprovalRequired':
      // Show approval dialog
      break;
    case 'Finished':
      // Show final response
      break;
    case 'Error':
      // Show error
      break;
  }
});
```

### Approving a tool call

```typescript
await invoke('agent_approve_tool_call', {
  runId,
  toolCallId: event.call.id,
  approved: true,
});
```

### Cancelling a run

```typescript
await invoke('agent_cancel', { runId });
```

## React Component Shape

The frontend renders an **agent activity timeline**:

```
┌─────────────────────────────────┐
│ 🟢 Agent Started                │
│ 🔄 Calling model...             │
│ 🔧 Tool: list_files             │
│    └─ Found 12 files            │
│ 🔄 Calling model...             │
│ 🔧 Tool: read_file              │
│    └─ Read src/main.rs (245b)   │
│ 🔄 Calling model...             │
│ ✅ Agent finished                │
│                                 │
│ "Here's what I found..."        │
└─────────────────────────────────┘
```

Each `AgentEvent` maps to a timeline entry. The `type` field on the event (via `serde(tag = "type")`) makes frontend dispatch simple.

## State Management

The Tauri plugin should hold:

- A `HashMap<RunId, JoinHandle>` for active runs
- A `HashMap<RunId, CancellationToken>` for cancellation
- A `HashMap<RunId, Sender<bool>>` for pending approvals

This stays in `AppState` managed by Tauri.
