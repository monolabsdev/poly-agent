use std::collections::HashSet;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use poly_agent_core::{AgentEvent, AgentInput, ModelConfig, ModelProvider, RuntimeLimits};
use poly_agent_runtime::AgentRuntime;
use tokio::sync::mpsc;

use crate::config::{RunStats, RunSummary, SessionConfig};

pub async fn run_prompt(
    runtime: Arc<AgentRuntime>,
    config: &SessionConfig,
    prompt: String,
) -> anyhow::Result<RunSummary> {
    let input = AgentInput {
        prompt,
        workspace: config.workspace.clone(),
        model: ModelConfig {
            provider: match config.provider.as_str() {
                "ollama" => ModelProvider::Ollama,
                _ => ModelProvider::OpenAICompatible,
            },
            model: config.model.clone(),
            base_url: None,
            api_key: None,
        },
        limits: RuntimeLimits::default(),
        permission_preset: config.preset,
        resolved_context: None,
    };

    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(32);
    let event_runtime = runtime.clone();

    let event_handle = tokio::spawn(async move {
        let mut stats = RunStats {
            model_calls: 0,
            tool_calls: 0,
            tools_used: HashSet::new(),
        };

        while let Some(event) = event_rx.recv().await {
            print_event(&event);
            match event {
                AgentEvent::ModelCallStarted { .. } => stats.model_calls += 1,
                AgentEvent::ToolCallStarted { tool_name, .. } => {
                    stats.tool_calls += 1;
                    stats.tools_used.insert(tool_name);
                }
                AgentEvent::ApprovalRequired { run_id, call } => {
                    if prompt_for_approval(&call.name) {
                        let _ = event_runtime.approve_tool(run_id, &call.id).await;
                    } else {
                        let _ = event_runtime.reject_tool(run_id, &call.id).await;
                    }
                }
                _ => {}
            }
        }

        stats
    });

    eprintln!("\nStarting agent run...\n");

    let start_time = Instant::now();
    let cancellation = tokio_util::sync::CancellationToken::new();
    let output = runtime.run(input, event_tx, cancellation).await?;
    let elapsed = start_time.elapsed();
    let stats = event_handle.await?;

    Ok(RunSummary {
        output_text: output.text,
        elapsed_seconds: elapsed.as_secs_f64(),
        stats,
    })
}

pub fn print_run_summary(summary: &RunSummary) {
    eprintln!("\n─────────────────────────────────");
    eprintln!("Stats:");
    eprintln!("  Model Calls: {}", summary.stats.model_calls);
    eprintln!("  Tool Calls:  {}", summary.stats.tool_calls);

    let mut tools_vec: Vec<_> = summary.stats.tools_used.iter().cloned().collect();
    tools_vec.sort();
    if !tools_vec.is_empty() {
        eprintln!("  Tools Used:  {}", tools_vec.join(", "));
    }
    eprintln!("  Time:        {:.2}s", summary.elapsed_seconds);
}

fn print_event(event: &AgentEvent) {
    match event {
        AgentEvent::Started { .. } => eprintln!("Started"),
        AgentEvent::ModelCallStarted { .. } => eprintln!("Thinking..."),
        AgentEvent::ModelCallFinished { .. } => {}
        AgentEvent::ToolCallRequested { call, .. } => {
            eprintln!("Tool requested: {}", call.name);
        }
        AgentEvent::ToolCallStarted { tool_name, .. } => {
            eprintln!("Running: {tool_name}");
        }
        AgentEvent::ToolCallFinished { result, .. } => {
            let tag = if result.cached { " [cached]" } else { "" };
            let single_line = result.output.replace('\n', " ");
            let preview: String = single_line.chars().take(60).collect();
            let suffix = if single_line.chars().count() > 60 {
                "..."
            } else {
                ""
            };
            eprintln!("Result: {preview}{suffix}{tag}");
        }
        AgentEvent::ApprovalRequired { call, .. } => {
            eprintln!("Approval required for: {}", call.name);
        }
        AgentEvent::StepLimitReached { max_steps, .. } => {
            eprintln!("Step limit reached ({max_steps})");
        }
        AgentEvent::Finished { .. } => eprintln!("Finished"),
        AgentEvent::Error { error, .. } => eprintln!("Error: {error}"),
        AgentEvent::UnknownToolRequested { tool_name, .. } => {
            eprintln!("Unknown tool: {tool_name}");
        }
        _ => {}
    }
}

fn prompt_for_approval(tool_name: &str) -> bool {
    loop {
        eprint!("Approve tool `{tool_name}`? [y/N] ");
        let _ = std::io::Write::write_all(&mut std::io::stderr(), b"").ok();
        let _ = std::io::stderr().flush();

        let mut answer = String::new();
        if std::io::stdin().read_line(&mut answer).is_err() {
            return false;
        }

        match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => return true,
            "" | "n" | "no" => return false,
            _ => eprintln!("Please answer y or n."),
        }
    }
}
