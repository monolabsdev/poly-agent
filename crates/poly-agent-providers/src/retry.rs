use crate::error::ProviderError;
use reqwest::{Response, StatusCode};

const RETRYABLE_STATUS_CODES: &[StatusCode] = &[
    StatusCode::INTERNAL_SERVER_ERROR,
    StatusCode::BAD_GATEWAY,
    StatusCode::SERVICE_UNAVAILABLE,
    StatusCode::GATEWAY_TIMEOUT,
];
const MAX_RETRIES: usize = 2;

pub fn backoff_duration(attempt: usize) -> tokio::time::Duration {
    tokio::time::Duration::from_millis(100 * (1 << attempt))
}

pub async fn send_with_retry<F, Fut>(request_fn: F) -> Result<Response, ProviderError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<Response, ProviderError>>,
{
    let mut retry_count = 0;
    loop {
        let resp = request_fn().await?;
        let status = resp.status();

        if status.is_success() {
            return Ok(resp);
        }

        if RETRYABLE_STATUS_CODES.contains(&status) && retry_count < MAX_RETRIES {
            retry_count += 1;
            tracing::debug!(attempt = retry_count, status = %status, "Retryable HTTP error");
            tokio::time::sleep(backoff_duration(retry_count)).await;
            continue;
        }

        let body_text = resp.text().await.unwrap_or_default();
        return Err(ProviderError::Api {
            status: status.as_u16(),
            body: body_text,
        });
    }
}
