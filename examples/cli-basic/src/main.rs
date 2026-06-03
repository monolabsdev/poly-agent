use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use poly_agent_core::{
    AgentEvent, AgentInput, ModelConfig, ModelProvider, RuntimeLimits,
};
use poly_agent_providers::{ModelAdapter, OllamaAdapter, OpenAICompatibleAdapter};
use poly_agent_runtime::{AgentRuntime, ToolRegistry};
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(name = "poly-agent-cli", about = "Minimal CLI for poly-agent")]
struct Cli {
    /// Provider: "ollama" or "openai-compatible".
    #[arg(long)]
    provider: String,

    /// Model name to use.
    #[arg(long)]
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

    /// Prompt to send. If omitted, reads from stdin.
    #[arg(long)]
    prompt: Option<String>,

    /// Enable debug mode for verbose tracing logs.
    #[arg(long)]
    debug: bool,
}

struct RunStats {
    model_calls: usize,
    tool_calls: usize,
    tools_used: HashSet<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing based on --debug or RUST_LOG.
    let env_filter = if cli.debug {
        tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("poly_agent=debug".parse().unwrap())
    } else {
        tracing_subscriber::EnvFilter::from_default_env()
    };

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr) // Ensure tracing always goes to stderr
        .compact()
        .init();

    // Read prompt from arg or stdin.
    let prompt = match cli.prompt {
        Some(p) => p,
        None => {
            eprintln!("Enter your prompt (then press Ctrl+D / Ctrl+Z):");
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            buf.trim().to_string()
        }
    };

    if prompt.is_empty() {
        anyhow::bail!("Prompt cannot be empty");
    }

    if cli.debug {
        eprintln!("DEBUG Configuration:");
        eprintln!("  Provider: {}", cli.provider);
        eprintln!("  Model: {}", cli.model);
        if let Some(ref url) = cli.base_url {
            eprintln!("  Base URL: {}", url);
        }
        eprintln!("  Workspace: {:?}", cli.workspace);
    }

    // Build the provider adapter.
    let adapter: Arc<dyn ModelAdapter> = match cli.provider.as_str() {
        "ollama" => Arc::new(OllamaAdapter::new(&cli.model, cli.base_url)),
        "openai-compatible" => {
            let base_url = cli
                .base_url
                .unwrap_or_else(|| "http://localhost:1234/v1".to_string());
            Arc::new(OpenAICompatibleAdapter::new(
                base_url, &cli.model, cli.api_key,
            ))
        }
        other => anyhow::bail!("Unknown provider: '{other}'. Use 'ollama' or 'openai-compatible'."),
    };

    // Build tool registry with safe tools.
    let mut registry = ToolRegistry::new();
    poly_agent_tools::register_safe_tools(&mut registry);

    eprintln!(
        "🔧 Registered {} tools",
        registry.len()
    );

    let runtime = AgentRuntime::new(registry, adapter);

    // Resolve workspace path.
    let workspace = std::fs::canonicalize(&cli.workspace)?;

    let input = AgentInput {
        prompt,
        workspace,
        model: ModelConfig {
            provider: match cli.provider.as_str() {
                "ollama" => ModelProvider::Ollama,
                _ => ModelProvider::OpenAICompatible,
            },
            model: cli.model,
            base_url: None,
            api_key: None,
        },
        limits: RuntimeLimits::default(),
    };

    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(32);

    // Spawn event printer and stats collector.
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
                _ => {}
            }
        }
        stats
    });

    eprintln!("\n🚀 Starting agent run...\n");

    let start_time = Instant::now();
    let output = runtime.run(input, event_tx).await?;
    let elapsed = start_time.elapsed();

    // Wait for all events to print and get stats.
    let stats = event_handle.await?;

    eprintln!("\n─────────────────────────────────");
    eprintln!("Run ID:  {}", output.run_id);
    eprintln!("Reason:  {:?}", output.finish_reason);
    eprintln!("Stats:");
    eprintln!("  Model Calls: {}", stats.model_calls);
    eprintln!("  Tool Calls:  {}", stats.tool_calls);
    
    let mut tools_vec: Vec<_> = stats.tools_used.into_iter().collect();
    tools_vec.sort();
    if !tools_vec.is_empty() {
        eprintln!("  Tools Used:  {}", tools_vec.join(", "));
    }
    eprintln!("  Time:        {:.2}s", elapsed.as_secs_f64());

    if !output.text.is_empty() {
        println!("\n{}", output.text); // Final text goes to stdout
    }

    Ok(())
}

fn print_event(event: &AgentEvent) {
    match event {
        AgentEvent::Started { run_id: _ } => {
            eprintln!("🟢 Started");
        }
        AgentEvent::ModelCallStarted { .. } => {
            eprintln!("🔄 Thinking...");
        }
        AgentEvent::ModelCallFinished { .. } => {}
        AgentEvent::ToolCallRequested { call, .. } => {
            eprintln!("🔧 Tool requested: {}", call.name);
        }
        AgentEvent::ToolCallStarted { tool_name, .. } => {
            eprintln!("⚙️  Running: {tool_name}");
        }
        AgentEvent::ToolCallFinished { result, .. } => {
            // Replace newlines with spaces for a single-line preview
            let single_line: String = result.output.replace('\n', " ");
            let preview: String = single_line.chars().take(60).collect();
            let suffix = if single_line.chars().count() > 60 { "..." } else { "" };
            eprintln!("📋 Result: {preview}{suffix}");
        }
        AgentEvent::ApprovalRequired { call, .. } => {
            eprintln!("⚠️  Approval required for: {}", call.name);
        }
        AgentEvent::StepLimitReached { max_steps, .. } => {
            eprintln!("🛑 Step limit reached ({max_steps})");
        }
        AgentEvent::Finished { .. } => {
            eprintln!("🏁 Finished");
        }
        AgentEvent::Error { error, .. } => {
            eprintln!("❌ Error: {error}");
        }
    }
}
