mod chat;
mod config;
mod events;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use poly_agent_providers::{ModelAdapter, OllamaAdapter, OpenAICompatibleAdapter};
use poly_agent_runtime::{AgentRuntime, ToolRegistry};

use crate::chat::run_chat;
use crate::config::SessionConfig;
use crate::events::{print_run_summary, run_prompt};

#[derive(Parser)]
#[command(name = "poly-agent-cli", about = "Minimal CLI for poly-agent")]
struct Cli {
    #[arg(long, default_value = "ollama")]
    provider: String,
    #[arg(long, default_value = "gpt-oss:20b-cloud")]
    model: String,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
            Arc::new(OpenAICompatibleAdapter::new(base_url, &cli.model, cli.api_key))
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
