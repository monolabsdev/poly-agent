use crate::{
    error::ProviderError,
    ollama_types::{build_messages, build_request, build_tools, response_to_model_response},
    traits::{ChatRequest, ModelAdapter, ModelResponse},
};
use ollama_rs::Ollama;
use std::pin::Pin;
use tokio_stream::Stream;
use url::Url;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

pub struct OllamaAdapter {
    client: Ollama,
    model: String,
}

impl OllamaAdapter {
    pub fn new(model: impl Into<String>, base_url: Option<String>) -> Self {
        let client = match base_url {
            Some(url) => Ollama::from_url(Url::parse(&url).expect("invalid Ollama base URL")),
            None => Ollama::from_url(Url::parse(DEFAULT_OLLAMA_URL).expect("invalid default Ollama URL")),
        };

        Self {
            client,
            model: model.into(),
        }
    }

    pub(crate) fn build_chat_request(&self, request: &ChatRequest) -> ollama_rs::generation::chat::request::ChatMessageRequest {
        build_request(
            self.model.clone(),
            build_messages(&request.messages),
            build_tools(&request.tools),
        )
    }

    pub async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ModelResponse, ProviderError>> + Send>>, ProviderError>
    {
        if !request.tools.is_empty() {
            tracing::debug!("Ollama tool-call stream fallback: using non-streaming response path");
            let response = self.chat(request).await?;
            return Ok(Box::pin(async_stream::stream! {
                yield Ok(response);
            }));
        }

        let request = self.build_chat_request(&request);
        let stream = self.client.send_chat_messages_stream(request).await.map_err(map_ollama_error)?;
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
        let request = self.build_chat_request(&request);
        let response = self
            .client
            .send_chat_messages(request)
            .await
            .map_err(map_ollama_error)?;
        response_to_model_response(response)
    }
}

fn map_ollama_error(error: impl std::fmt::Display) -> ProviderError {
    ProviderError::Parse(format!("Ollama error: {error}"))
}

#[cfg(test)]
#[path = "ollama_tests.rs"]
mod ollama_tests;
