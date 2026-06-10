use crate::{
    error::ProviderError,
    ollama_types::{
        build_messages, build_request_body, build_tools, response_to_model_response, OllamaRequestBody,
    },
    traits::{ChatRequest, ModelAdapter, ModelResponse, ModelStream},
};
use ollama_rs::generation::chat::ChatMessageResponse;
use ollama_rs::Ollama;
use url::Url;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

pub struct OllamaAdapter {
    /// Used only for streaming when no tools are present.
    _client: Ollama,
    model: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl OllamaAdapter {
    pub fn new(model: impl Into<String>, base_url: Option<String>) -> Self {
        let resolved = base_url.unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
        let ollama = Ollama::from_url(Url::parse(&resolved).expect("invalid Ollama base URL"));

        Self {
            _client: ollama,
            model: model.into(),
            base_url: resolved,
            http_client: reqwest::Client::new(),
        }
    }

    fn chat_api_url(&self) -> String {
        format!("{}/api/chat", self.base_url.trim_end_matches('/'))
    }

    async fn send_non_streaming(&self, body: &OllamaRequestBody) -> Result<ModelResponse, ProviderError> {
        let res = self
            .http_client
            .post(self.chat_api_url())
            .json(body)
            .send()
            .await
            .map_err(|e| ProviderError::Parse(format!("Ollama request failed: {e}")))?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            return Err(ProviderError::Parse(format!("Ollama error ({status}): {text}")));
        }

        let response: ChatMessageResponse = res
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("Failed to deserialize Ollama response: {e}")))?;
        response_to_model_response(response)
    }

    async fn stream_chat(&self, request: ChatRequest) -> Result<ModelStream, ProviderError> {
        if !request.tools.is_empty() {
            tracing::debug!("Ollama tool-call stream fallback: using non-streaming response path");
            let response = self.send_non_streaming(
                &build_request_body(
                    self.model.clone(),
                    build_messages(&request.messages),
                    build_tools(&request.tools),
                ),
            )
            .await?;
            return Ok(Box::pin(async_stream::stream! {
                yield Ok(response);
            }));
        }

        use ollama_rs::generation::chat::request::ChatMessageRequest;
        let request = ChatMessageRequest::new(
            self.model.clone(),
            build_messages(&request.messages),
        );
        let stream = self
            ._client
            .send_chat_messages_stream(request)
            .await
            .map_err(map_ollama_error)?;
        Ok(Box::pin(async_stream::try_stream! {
            use tokio_stream::StreamExt;
            tokio::pin!(stream);
            while let Some(item) = stream.next().await {
                let response = item.map_err(|_| ProviderError::Parse("Failed to read Ollama stream".into()))?;
                yield response_to_model_response(response)?;
            }
        }))
    }
}

#[async_trait::async_trait]
impl ModelAdapter for OllamaAdapter {
    async fn chat(&self, request: ChatRequest) -> Result<ModelResponse, ProviderError> {
        self.send_non_streaming(
            &build_request_body(
                self.model.clone(),
                build_messages(&request.messages),
                build_tools(&request.tools),
            ),
        )
        .await
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ModelStream, ProviderError> {
        self.stream_chat(request).await
    }
}

fn map_ollama_error(error: impl std::fmt::Display) -> ProviderError {
    ProviderError::Parse(format!("Ollama error: {error}"))
}

#[cfg(test)]
#[path = "ollama_tests.rs"]
mod ollama_tests;
