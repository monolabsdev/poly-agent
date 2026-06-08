use std::sync::Arc;

use poly_agent_runtime::AgentRuntime;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::config::SessionConfig;
use crate::events::{print_run_summary, run_prompt};

pub async fn run_chat(runtime: Arc<AgentRuntime>, config: SessionConfig) -> anyhow::Result<()> {
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
    let mut contextual =
        "Previous conversation in this CLI session follows. Use it as context, but treat the final user message as the current task.\n\n".to_string();

    for (idx, (user, assistant)) in recent.into_iter().rev().enumerate() {
        contextual.push_str(&format!("Turn {} user:\n{}\n\n", idx + 1, user));
        contextual.push_str(&format!("Turn {} assistant:\n{}\n\n", idx + 1, assistant));
    }

    contextual.push_str("Current user message:\n");
    contextual.push_str(prompt);
    contextual
}
