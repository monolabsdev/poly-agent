use std::collections::HashMap;
use std::sync::Arc;

use poly_agent_core::AgentEvent;
use poly_agent_core::{AgentInput, RunId};
use poly_agent_runtime::AgentRuntime;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::events::{
    apply_event_to_state, cancelled_event, map_runtime_event, push_event, AgentEventSink,
};
use crate::types::{AgentRunError, AgentRunInput, MutableRunState, RunStateSnapshot, RunStatus};
use crate::workspace::{default_runtime, prepare_run, PreparedRun};

type RuntimeFactory = Arc<dyn Fn(PreparedRun) -> Result<AgentRuntime, AgentRunError> + Send + Sync>;

#[derive(Clone)]
pub struct AgentRunManager {
    inner: Arc<ManagerInner>,
}

struct ManagerInner {
    runs: RwLock<HashMap<RunId, Arc<RunSlot>>>,
    event_sink: Option<AgentEventSink>,
    runtime_factory: RuntimeFactory,
}

struct RunSlot {
    state: Mutex<MutableRunState>,
    runtime: Arc<AgentRuntime>,
    task: Mutex<Option<JoinHandle<()>>>,
    cancellation: CancellationToken,
}

impl AgentRunManager {
    pub fn new(event_sink: Option<AgentEventSink>) -> Self {
        Self::with_runtime_factory(event_sink, Arc::new(default_runtime))
    }

    pub(crate) fn with_runtime_factory(
        event_sink: Option<AgentEventSink>,
        runtime_factory: RuntimeFactory,
    ) -> Self {
        Self {
            inner: Arc::new(ManagerInner {
                runs: RwLock::new(HashMap::new()),
                event_sink,
                runtime_factory,
            }),
        }
    }

    pub async fn start_run(&self, input: AgentRunInput) -> Result<RunId, AgentRunError> {
        let debug = input.debug;
        let prepared = prepare_run(input)?;
        let agent_input = prepared.input.clone();
        let workspace_root = prepared.workspace_root.clone();
        let local_tools_enabled = prepared.local_tools_enabled;
        let runtime = Arc::new((self.inner.runtime_factory)(prepared)?);
        let run_id = RunId::new_v4();
        let slot = Arc::new(RunSlot {
            state: Mutex::new(MutableRunState::new(
                run_id,
                workspace_root,
                local_tools_enabled,
            )),
            runtime,
            task: Mutex::new(None),
            cancellation: CancellationToken::new(),
        });

        self.inner.runs.write().await.insert(run_id, slot.clone());
        self.spawn_run(run_id, slot, agent_input, debug).await;
        Ok(run_id)
    }

    pub async fn cancel_run(&self, run_id: RunId) -> Result<(), AgentRunError> {
        let slot = self.slot(run_id).await?;
        slot.cancellation.cancel();

        {
            let mut state = slot.state.lock().await;
            state.status = RunStatus::Cancelled;
            state.finished_at = Some(std::time::SystemTime::now());
            state.pending_approval = None;
            let event = cancelled_event(run_id);
            push_event(&mut state, event.clone());
            self.emit(event);
        }

        if let Some(handle) = slot.task.lock().await.take() {
            handle.abort();
        }
        Ok(())
    }

    pub async fn approve_tool_call(
        &self,
        run_id: RunId,
        approval_id: &str,
    ) -> Result<(), AgentRunError> {
        let slot = self.slot(run_id).await?;
        let runtime_run_id = { slot.state.lock().await.runtime_run_id };
        let runtime_run_id = runtime_run_id.ok_or_else(|| {
            AgentRunError::Other(format!("run {run_id} has not started in runtime"))
        })?;
        slot.runtime
            .approve_tool(runtime_run_id, approval_id)
            .await
            .map_err(|err| AgentRunError::Other(err.to_string()))?;
        {
            let mut state = slot.state.lock().await;
            if state
                .pending_approval
                .as_ref()
                .map(|a| a.approval_id.as_str())
                == Some(approval_id)
            {
                state.pending_approval = None;
            }
            if state.status == RunStatus::WaitingForApproval {
                state.status = RunStatus::Running;
            }
        }
        Ok(())
    }

    pub async fn reject_tool_call(
        &self,
        run_id: RunId,
        approval_id: &str,
    ) -> Result<(), AgentRunError> {
        let slot = self.slot(run_id).await?;
        let runtime_run_id = { slot.state.lock().await.runtime_run_id };
        let runtime_run_id = runtime_run_id.ok_or_else(|| {
            AgentRunError::Other(format!("run {run_id} has not started in runtime"))
        })?;
        slot.runtime
            .reject_tool(runtime_run_id, approval_id)
            .await
            .map_err(|err| AgentRunError::Other(err.to_string()))?;
        {
            let mut state = slot.state.lock().await;
            if state
                .pending_approval
                .as_ref()
                .map(|a| a.approval_id.as_str())
                == Some(approval_id)
            {
                state.pending_approval = None;
            }
            if state.status == RunStatus::WaitingForApproval {
                state.status = RunStatus::Running;
            }
        }
        Ok(())
    }

    pub async fn get_run_state(&self, run_id: RunId) -> Result<RunStateSnapshot, AgentRunError> {
        let slot = self.slot(run_id).await?;
        let snapshot = slot.state.lock().await.snapshot();
        Ok(snapshot)
    }

    pub async fn list_active_runs(&self) -> Vec<RunStateSnapshot> {
        let slots: Vec<_> = self.inner.runs.read().await.values().cloned().collect();
        let mut snapshots = Vec::with_capacity(slots.len());
        for slot in slots {
            let snapshot = slot.state.lock().await.snapshot();
            if matches!(
                snapshot.status,
                RunStatus::Running | RunStatus::WaitingForApproval
            ) {
                snapshots.push(snapshot);
            }
        }
        snapshots
    }

    async fn spawn_run(&self, run_id: RunId, slot: Arc<RunSlot>, input: AgentInput, debug: bool) {
        let manager = self.clone();
        let runtime = slot.runtime.clone();
        let cancellation = slot.cancellation.clone();
        let (tx, mut rx) = mpsc::channel::<AgentEvent>(128);

        let event_slot = slot.clone();
        let event_manager = manager.clone();
        let event_task = tokio::spawn(async move {
            while let Some(runtime_event) = rx.recv().await {
                let ui_event = map_runtime_event(run_id, runtime_event.clone(), debug);
                {
                    let mut state = event_slot.state.lock().await;
                    if state.status == RunStatus::Cancelled {
                        continue;
                    }
                    apply_event_to_state(&mut state, &runtime_event, &ui_event);
                    push_event(&mut state, ui_event.clone());
                }
                event_manager.emit(ui_event);
            }
        });

        let run_slot = slot.clone();
        let run_cancellation = cancellation.clone();
        let run_task = tokio::spawn(async move {
            let mut cancelled = false;
            tokio::select! {
                result = runtime.run(input, tx, run_cancellation) => {
                    if let Err(err) = result {
                        let mut state = run_slot.state.lock().await;
                        if state.status != RunStatus::Cancelled {
                            state.status = RunStatus::Failed;
                            state.last_error = Some(err.to_string());
                            state.finished_at = Some(std::time::SystemTime::now());
                        }
                    }
                }
                _ = cancellation.cancelled() => {
                    cancelled = true;
                }
            }
            if cancelled {
                event_task.abort();
            } else {
                let _ = event_task.await;
            }
        });

        *slot.task.lock().await = Some(run_task);
    }

    async fn slot(&self, run_id: RunId) -> Result<Arc<RunSlot>, AgentRunError> {
        self.inner
            .runs
            .read()
            .await
            .get(&run_id)
            .cloned()
            .ok_or(AgentRunError::RunNotFound(run_id))
    }

    fn emit(&self, event: crate::events::AgentUiEvent) {
        if let Some(sink) = &self.inner.event_sink {
            sink(event);
        }
    }
}
