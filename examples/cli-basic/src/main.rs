use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use poly_agent_core::{AgentEvent, AgentInput, ModelConfig, ModelProvider, RuntimeLimits};
use poly_agent_providers::{ModelAdapter, OllamaAdapter, OpenAICompatibleAdapter};
use poly_agent_runtime::{AgentRuntime, ToolRegistry};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(name = "poly-agent-cli", about = "Minimal CLI for poly-agent")]
struct Cli {
    /// Provider: "ollama" or "openai-compatible".
    #[arg(long, default_value = "ollama")]
    provider: String,

    /// Model name to use.
    #[arg(long, default_value = "gpt-oss:20b-cloud")]
    model: String,

    /// Base URL for the provider API.
    #[arg(long)]
    base_url: Option<String>,

    /// API key (for OpenAI-compatible providers).
    #[arg(long)]
    api_key: Option<String>,

    /// Workspace directory to operate in.
    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    /// Prompt to send once. If omitted, starts an interactive chat.
    #[arg(long)]
    prompt: Option<String>,

    /// Enable debug mode for verbose tracing logs.
    #[arg(long)]
    debug: bool,
}

struct SessionConfig {
    provider: String,
    model: String,
    workspace: PathBuf,
}

struct RunStats {
    model_calls: usize,
    tool_calls: usize,
    tools_used: HashSet<String>,
}

struct RunSummary {
    output_text: String,
    elapsed_seconds: f64,
    stats: RunStats,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let env_filter = if cli.debug {
        tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("poly_agent=debug".parse().unwrap())
    } else {
        tracing_subscriber::EnvFilter::from_default_env()
    };

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .compact()
        .init();

    if cli.debug {
        eprintln!("DEBUG Configuration:");
        eprintln!("  Provider: {}", cli.provider);
        eprintln!("  Model: {}", cli.model);
        if let Some(ref url) = cli.base_url {
            eprintln!("  Base URL: {}", url);
        }
        eprintln!("  Workspace: {:?}", cli.workspace);
    }

    let adapter: Arc<dyn ModelAdapter> = match cli.provider.as_str() {
        "ollama" => Arc::new(OllamaAdapter::new(&cli.model, cli.base_url)),
        "openai-compatible" => {
            let base_url = cli
                .base_url
                .unwrap_or_else(|| "http://localhost:1234/v1".to_string());
            Arc::new(OpenAICompatibleAdapter::new(
                base_url,
                &cli.model,
                cli.api_key,
            ))
        }
        other => anyhow::bail!("Unknown provider: '{other}'. Use 'ollama' or 'openai-compatible'."),
    };

    let mut registry = ToolRegistry::new();
    poly_agent_tools::register_all_tools(&mut registry);
    eprintln!("Registered {} tools", registry.len());

    let runtime = Arc::new(AgentRuntime::new(registry, adapter));
    let config = SessionConfig {
        provider: cli.provider,
        model: cli.model,
        workspace: std::fs::canonicalize(&cli.workspace)?,
    };

    if let Some(prompt) = cli.prompt {
        if prompt.trim().is_empty() {
            anyhow::bail!("Prompt cannot be empty");
        }

        let summary = run_prompt(runtime, &config, prompt).await?;
        print_run_summary(&summary);
        if !summary.output_text.is_empty() {
            println!("\n{}", summary.output_text);
        }
        return Ok(());
    }

    run_chat(runtime, config).await
}

async fn run_chat(runtime: Arc<AgentRuntime>, config: SessionConfig) -> Result<()> {
    let mut editor = DefaultEditor::new()?;
    let mut history: Vec<(String, String)> = Vec::new();

    eprintln!("poly-agent chat");
    eprintln!("Provider: {} | Model: {}", config.provider, config.model);
    eprintln!("Workspace: {}", config.workspace.display());
    eprintln!("Type /exit or /quit, or press Ctrl+D to quit.\n");

    loop {
        match editor.readline("you> ") {
            Ok(line) => {
                let prompt = line.trim();
                if prompt.is_empty() {
                    continue;
                }
                if matches!(prompt, "/exit" | "/quit") {
                    break;
                }

                let _ = editor.add_history_entry(prompt);
                let contextual_prompt = build_contextual_prompt(&history, prompt);
                let summary = run_prompt(runtime.clone(), &config, contextual_prompt).await?;

                print_run_summary(&summary);
                if summary.output_text.is_empty() {
                    println!();
                } else {
                    println!("\nagent> {}\n", summary.output_text);
                }

                history.push((prompt.to_string(), summary.output_text));
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }

    Ok(())
}

fn build_contextual_prompt(history: &[(String, String)], prompt: &str) -> String {
    if history.is_empty() {
        return prompt.to_string();
    }

    let recent: Vec<_> = history.iter().rev().take(8).collect();
    let mut contextual = String::from(
        "Previous conversation in this CLI session follows. Use it as context, but treat the final user message as the current task.\n\n",
    );

    for (idx, (user, assistant)) in recent.into_iter().rev().enumerate() {
        contextual.push_str(&format!("Turn {} user:\n{}\n\n", idx + 1, user));
        contextual.push_str(&format!("Turn {} assistant:\n{}\n\n", idx + 1, assistant));
    }

    contextual.push_str("Current user message:\n");
    contextual.push_str(prompt);
    contextual
}

async fn run_prompt(
    runtime: Arc<AgentRuntime>,
    config: &SessionConfig,
    prompt: String,
) -> Result<RunSummary> {
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
    let output = runtime.run(input, event_tx).await?;
    let elapsed = start_time.elapsed();
    let stats = event_handle.await?;

    Ok(RunSummary {
        output_text: output.text,
        elapsed_seconds: elapsed.as_secs_f64(),
        stats,
    })
}

fn print_run_summary(summary: &RunSummary) {
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
        AgentEvent::Started { .. } => {
            eprintln!("Started");
        }
        AgentEvent::ModelCallStarted { .. } => {
            eprintln!("Thinking...");
        }
        AgentEvent::ModelCallFinished { .. } => {}
        AgentEvent::ToolCallRequested { call, .. } => {
            eprintln!("Tool requested: {}", call.name);
        }
        AgentEvent::ToolCallStarted { tool_name, .. } => {
            eprintln!("Running: {tool_name}");
        }
        AgentEvent::ToolCallFinished { result, .. } => {
            let single_line = result.output.replace('\n', " ");
            let preview: String = single_line.chars().take(60).collect();
            let suffix = if single_line.chars().count() > 60 {
                "..."
            } else {
                ""
            };
            eprintln!("Result: {preview}{suffix}");
        }
        AgentEvent::ApprovalRequired { call, .. } => {
            eprintln!("Approval required for: {}", call.name);
        }
        AgentEvent::StepLimitReached { max_steps, .. } => {
            eprintln!("Step limit reached ({max_steps})");
        }
        AgentEvent::Finished { .. } => {
            eprintln!("Finished");
        }
        AgentEvent::Error { error, .. } => {
            eprintln!("Error: {error}");
        }
    }
}

fn prompt_for_approval(tool_name: &str) -> bool {
    loop {
        eprint!("Approve tool `{tool_name}`? [y/N] ");
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
