#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    Parse(String),

    #[error("API error: {status} — {body}")]
    Api { status: u16, body: String },

    #[error("Malformed model output: control tokens detected in response")]
    MalformedModelOutput,
}
