use crate::parser::{image_entry_download_key, normalize_split, ImageEntry};
use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use reqwest::{header::RETRY_AFTER, Client, StatusCode};
use serde::Serialize;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::ipc::Channel;
use thiserror::Error;
use tokio::sync::Mutex;
use url::{Host, Url};

const MAX_DOWNLOAD_BYTES: usize = 50 * 1024 * 1024; // 50 MiB per image
const MAX_DOWNLOAD_ATTEMPTS: usize = 3;
const MAX_RETRY_AFTER: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
struct RetryPolicy {
    max_attempts: usize,
    base_delay: Duration,
    max_retry_after: Duration,
}

impl RetryPolicy {
    fn production() -> Self {
        Self {
            max_attempts: MAX_DOWNLOAD_ATTEMPTS,
            base_delay: Duration::from_secs(1),
            max_retry_after: MAX_RETRY_AFTER,
        }
    }

    fn delay_after(&self, attempt: usize, retry_after: Option<Duration>) -> Duration {
        if let Some(delay) = retry_after {
            return delay.min(self.max_retry_after);
        }

        let multiplier = 1_u32 << attempt.saturating_sub(1);
        self.base_delay.saturating_mul(multiplier)
    }
}

#[derive(Debug, Error)]
enum DownloadError {
    #[error("server returned HTTP {0}")]
    Http(StatusCode),
    #[error("request failed: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("failed to read response body: {0}")]
    Body(#[source] reqwest::Error),
    #[error("response too large ({actual} bytes, max {maximum})")]
    TooLarge { actual: u64, maximum: usize },
    #[error("response body size overflow")]
    SizeOverflow,
}

impl DownloadError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Http(status) => {
                *status == StatusCode::REQUEST_TIMEOUT
                    || *status == StatusCode::TOO_MANY_REQUESTS
                    || status.is_server_error()
            }
            Self::Transport(_) | Self::Body(_) => true,
            Self::TooLarge { .. } | Self::SizeOverflow => false,
        }
    }

    fn is_possible_expired_url(&self) -> bool {
        matches!(self, Self::Http(StatusCode::FORBIDDEN))
    }
}

#[derive(Clone, Serialize)]
pub struct ProgressEvent {
    pub phase: String,
    pub current: u32,
    pub total: u32,
    pub item: Option<String>,
}

pub struct Downloader {
    client: Client,
    concurrency: usize,
}

impl Downloader {
    pub fn new(concurrency: usize) -> Result<Self, String> {
        let client = Client::builder()
            .pool_max_idle_per_host(concurrency)
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        Ok(Self {
            client,
            concurrency,
        })
    }

    pub async fn download_all(
        &self,
        images: &[ImageEntry],
        channel: &Channel<ProgressEvent>,
    ) -> DownloadResult {
        let images_with_urls: Vec<_> = images
            .iter()
            .filter(|img| !img.url.is_empty())
            .map(|img| {
                let split = normalize_split(&img.split);
                let item_label = format!("{}/{}", split, img.effective_file_name());
                let download_key = image_entry_download_key(img);
                (item_label, download_key, img.url.clone())
            })
            .collect();

        let total = images_with_urls.len() as u32;

        if total == 0 {
            return DownloadResult {
                files: HashMap::new(),
                total: 0,
                failed: 0,
                expired_url_failures: 0,
            };
        }

        let _ = channel.send(ProgressEvent {
            phase: "downloading".to_string(),
            current: 0,
            total,
            item: None,
        });

        let downloaded = Arc::new(Mutex::new(HashMap::new()));
        let counter = Arc::new(AtomicU32::new(0));
        let failed = Arc::new(AtomicU32::new(0));
        let expired_url_failures = Arc::new(AtomicU32::new(0));
        let client = self.client.clone();

        stream::iter(images_with_urls)
            .map(|(item_label, download_key, url)| {
                let client = client.clone();
                let downloaded = Arc::clone(&downloaded);
                let counter = Arc::clone(&counter);
                let failed = Arc::clone(&failed);
                let expired_url_failures = Arc::clone(&expired_url_failures);
                let channel = channel.clone();

                async move {
                    if let Err(err) = validate_download_url(&url).await {
                        eprintln!("Skipping download for '{}': {}", item_label, err);
                        failed.fetch_add(1, Ordering::SeqCst);
                        let current = counter.fetch_add(1, Ordering::SeqCst) + 1;
                        let _ = channel.send(ProgressEvent {
                            phase: "downloading".to_string(),
                            current,
                            total,
                            item: Some(item_label.clone()),
                        });
                        return;
                    }

                    match download_prevalidated_url(
                        &client,
                        &url,
                        MAX_DOWNLOAD_BYTES,
                        RetryPolicy::production(),
                    )
                    .await
                    {
                        Ok(bytes) => {
                            let mut map = downloaded.lock().await;
                            map.insert(download_key, bytes);
                        }
                        Err(err) => {
                            eprintln!("Skipping download for '{}': {}", item_label, err);
                            failed.fetch_add(1, Ordering::SeqCst);
                            if err.is_possible_expired_url() {
                                expired_url_failures.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }

                    let current = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = channel.send(ProgressEvent {
                        phase: "downloading".to_string(),
                        current,
                        total,
                        item: Some(item_label),
                    });
                }
            })
            .buffer_unordered(self.concurrency)
            .collect::<Vec<()>>()
            .await;

        let files = match Arc::try_unwrap(downloaded) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.lock().await.clone(),
        };

        let failed_count = match Arc::try_unwrap(failed) {
            Ok(counter) => counter.into_inner(),
            Err(counter) => counter.load(Ordering::SeqCst),
        };

        let expired_url_failure_count = match Arc::try_unwrap(expired_url_failures) {
            Ok(counter) => counter.into_inner(),
            Err(counter) => counter.load(Ordering::SeqCst),
        };

        DownloadResult {
            files,
            total,
            failed: failed_count as usize,
            expired_url_failures: expired_url_failure_count as usize,
        }
    }
}

pub struct DownloadResult {
    pub files: HashMap<String, Vec<u8>>,
    pub total: u32,
    pub failed: usize,
    pub expired_url_failures: usize,
}

// Caller must run validate_download_url before invoking this helper.
async fn download_prevalidated_url(
    client: &Client,
    url: &str,
    max_bytes: usize,
    retry_policy: RetryPolicy,
) -> Result<Vec<u8>, DownloadError> {
    let max_attempts = retry_policy.max_attempts.max(1);

    for attempt in 1..=max_attempts {
        let (result, retry_after) = match client.get(url).send().await {
            Ok(response) if response.status().is_success() => {
                (read_response_with_limit(response, max_bytes).await, None)
            }
            Ok(response) => {
                let status = response.status();
                let retry_after = if matches!(
                    status,
                    StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
                ) {
                    response
                        .headers()
                        .get(RETRY_AFTER)
                        .and_then(|value| parse_retry_after(value.to_str().ok()?, Utc::now()))
                } else {
                    None
                };
                (Err(DownloadError::Http(status)), retry_after)
            }
            Err(error) => (Err(DownloadError::Transport(error)), None),
        };

        match result {
            Ok(bytes) => return Ok(bytes),
            Err(error) if error.is_retryable() && attempt < max_attempts => {
                tokio::time::sleep(retry_policy.delay_after(attempt, retry_after)).await;
            }
            Err(error) => return Err(error),
        }
    }

    unreachable!("retry loop always returns on final attempt")
}

fn parse_retry_after(value: &str, now: DateTime<Utc>) -> Option<Duration> {
    if let Ok(seconds) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = DateTime::parse_from_rfc2822(value)
        .ok()?
        .with_timezone(&Utc);
    (retry_at - now).to_std().ok().or(Some(Duration::ZERO))
}

async fn validate_download_url(url: &str) -> Result<(), String> {
    let parsed = Url::parse(url).map_err(|_| "Invalid URL".to_string())?;
    match parsed.scheme() {
        "http" | "https" => {}
        _ => return Err("Only HTTP/HTTPS URLs are allowed".to_string()),
    }

    let host = parsed
        .host()
        .ok_or_else(|| "URL must include a hostname".to_string())?;
    match host {
        Host::Ipv4(v4) => {
            if is_forbidden_ip(IpAddr::V4(v4)) {
                return Err("Private or local IPs are not allowed".to_string());
            }
        }
        Host::Ipv6(v6) => {
            if is_forbidden_ip(IpAddr::V6(v6)) {
                return Err("Private or local IPs are not allowed".to_string());
            }
        }
        Host::Domain(domain) => {
            let host_lower = domain.to_ascii_lowercase();
            if host_lower == "localhost"
                || host_lower.ends_with(".localhost")
                || host_lower.ends_with(".local")
            {
                return Err("Localhost addresses are not allowed".to_string());
            }

            let port = parsed.port_or_known_default().unwrap_or(80);
            let mut addrs = tokio::net::lookup_host((domain, port))
                .await
                .map_err(|_| "Failed to resolve download host".to_string())?;
            let mut resolved_any = false;

            for addr in addrs.by_ref() {
                resolved_any = true;
                if is_forbidden_ip(addr.ip()) {
                    return Err("Private or local IPs are not allowed".to_string());
                }
            }

            if !resolved_any {
                return Err("Failed to resolve download host".to_string());
            }
        }
    }

    Ok(())
}

fn is_forbidden_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_multicast()
                || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            if let Some(mapped_v4) = v6.to_ipv4_mapped() {
                return is_forbidden_ip(IpAddr::V4(mapped_v4));
            }

            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
        }
    }
}

async fn read_response_with_limit(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, DownloadError> {
    if let Some(content_length) = response.content_length() {
        if content_length > max_bytes as u64 {
            return Err(DownloadError::TooLarge {
                actual: content_length,
                maximum: max_bytes,
            });
        }
    }

    let mut downloaded = Vec::new();
    let mut stream = response.bytes_stream();
    let mut total_bytes = 0usize;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(DownloadError::Body)?;
        total_bytes = total_bytes
            .checked_add(chunk.len())
            .ok_or(DownloadError::SizeOverflow)?;

        if total_bytes > max_bytes {
            return Err(DownloadError::TooLarge {
                actual: total_bytes as u64,
                maximum: max_bytes,
            });
        }

        downloaded.extend_from_slice(&chunk);
    }

    Ok(downloaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;
    use tokio::time::{timeout, Duration};

    async fn spawn_response_server(
        responses: Vec<String>,
    ) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let server_request_count = Arc::clone(&request_count);
        let server = tokio::spawn(async move {
            for response in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 1024];
                let bytes_read = stream.read(&mut request).await.unwrap();
                assert!(bytes_read > 0);
                server_request_count.fetch_add(1, Ordering::SeqCst);
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        (format!("http://{address}/image.jpg"), request_count, server)
    }

    fn test_retry_policy() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
            max_retry_after: Duration::ZERO,
        }
    }

    #[test]
    fn production_retry_policy_uses_one_then_two_second_backoff() {
        let policy = RetryPolicy::production();

        assert_eq!(policy.delay_after(1, None), Duration::from_secs(1));
        assert_eq!(policy.delay_after(2, None), Duration::from_secs(2));
    }

    #[test]
    fn production_retry_policy_clamps_retry_after_to_five_seconds() {
        let policy = RetryPolicy::production();

        assert_eq!(
            policy.delay_after(1, Some(Duration::from_secs(30))),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn retries_only_transient_http_statuses() {
        for status in [
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
        ] {
            assert!(DownloadError::Http(status).is_retryable(), "{status}");
        }

        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
        ] {
            assert!(!DownloadError::Http(status).is_retryable(), "{status}");
        }
    }

    #[tokio::test]
    async fn retries_500_then_returns_successful_body() {
        let (url, requests, server) = spawn_response_server(vec![
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_string(),
            "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nimage".to_string(),
        ])
        .await;

        let bytes = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            test_retry_policy(),
        )
        .await
        .unwrap();

        server.await.unwrap();
        assert_eq!(bytes, b"image");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn stops_after_three_retryable_failures() {
        let responses = (0..3)
            .map(|_| {
                "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    .to_string()
            })
            .collect();
        let (url, requests, server) = spawn_response_server(responses).await;

        let error = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            test_retry_policy(),
        )
        .await
        .unwrap_err();

        server.await.unwrap();
        assert!(matches!(
            error,
            DownloadError::Http(StatusCode::INTERNAL_SERVER_ERROR)
        ));
        assert_eq!(requests.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retries_transport_failure_then_returns_successful_body() {
        let (url, requests, server) = spawn_response_server(vec![
            String::new(),
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".to_string(),
        ])
        .await;

        let bytes = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            test_retry_policy(),
        )
        .await
        .unwrap();

        server.await.unwrap();
        assert_eq!(bytes, b"ok");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retries_429_and_honors_zero_retry_after_in_tests() {
        let (url, requests, server) = spawn_response_server(vec![
            "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_string(),
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".to_string(),
        ])
        .await;

        let bytes = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            test_retry_policy(),
        )
        .await
        .unwrap();

        server.await.unwrap();
        assert_eq!(bytes, b"ok");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn does_not_retry_404() {
        let (url, requests, server) = spawn_response_server(vec![
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
        ])
        .await;

        let error = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            test_retry_policy(),
        )
        .await
        .unwrap_err();

        server.await.unwrap();
        assert!(
            matches!(error, DownloadError::Http(status) if status == reqwest::StatusCode::NOT_FOUND)
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retries_midstream_body_failure() {
        let (url, requests, server) = spawn_response_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nab".to_string(),
            "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nimage".to_string(),
        ])
        .await;

        let bytes = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            test_retry_policy(),
        )
        .await
        .unwrap();

        server.await.unwrap();
        assert_eq!(bytes, b"image");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn does_not_retry_oversized_response() {
        let (url, requests, server) = spawn_response_server(vec![format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            MAX_DOWNLOAD_BYTES + 1
        )])
        .await;

        let error = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            test_retry_policy(),
        )
        .await
        .unwrap_err();

        server.await.unwrap();
        assert!(matches!(error, DownloadError::TooLarge { .. }));
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn marks_403_as_possible_expired_url_without_retrying() {
        let (url, requests, server) = spawn_response_server(vec![
            "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
        ])
        .await;

        let error = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            test_retry_policy(),
        )
        .await
        .unwrap_err();

        server.await.unwrap();
        assert!(error.is_possible_expired_url());
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn download_all_rejects_loopback_before_request() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let content = format!(
            r#"{{"type":"dataset","class_names":{{}}}}
{{"type":"image","file":"blocked.jpg","width":1,"height":1,"split":"train","url":"http://{address}/blocked.jpg"}}"#
        );
        let data = crate::parser::parse_ndjson(&content).unwrap();
        let channel = Channel::new(|_| Ok(()));

        let result = Downloader::new(1)
            .unwrap()
            .download_all(&data.images, &channel)
            .await;

        assert_eq!(result.failed, 1);
        assert_eq!(result.expired_url_failures, 0);
        assert!(timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn downloader_client_does_not_follow_redirects() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let server_request_count = Arc::clone(&request_count);

        let server = tokio::spawn(async move {
            for request_number in 0..2 {
                let accepted = timeout(Duration::from_millis(250), listener.accept()).await;
                let Ok(Ok((mut stream, _))) = accepted else {
                    break;
                };

                let mut request = [0_u8; 1024];
                let bytes_read = stream.read(&mut request).await.unwrap();
                if bytes_read == 0 {
                    break;
                }
                server_request_count.fetch_add(1, Ordering::SeqCst);

                if request_number == 0 {
                    stream
                        .write_all(
                            format!(
                                "HTTP/1.1 302 Found\r\nLocation: http://{address}/internal\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            )
                            .as_bytes(),
                        )
                        .await
                        .unwrap();
                } else {
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                }
            }
        });

        let downloader = Downloader::new(1).unwrap();
        let response = downloader
            .client
            .get(format!("http://{address}/public"))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        server.await.unwrap();
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn validate_url_accepts_public_ipv4_https() {
        let result = validate_download_url("https://1.1.1.1/image.jpg").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn validate_url_accepts_public_ipv4_http() {
        let result = validate_download_url("http://8.8.8.8/image.jpg").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn validate_url_rejects_localhost() {
        let result = validate_download_url("http://127.0.0.1/image.jpg").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Private or local"));
    }

    #[tokio::test]
    async fn validate_url_rejects_private_ip_10() {
        let result = validate_download_url("http://10.0.0.1/image.jpg").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Private or local"));
    }

    #[tokio::test]
    async fn validate_url_rejects_private_ip_192() {
        let result = validate_download_url("http://192.168.1.1/image.jpg").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Private or local"));
    }

    #[tokio::test]
    async fn validate_url_rejects_ipv4_mapped_ipv6_loopback() {
        let result = validate_download_url("http://[::ffff:127.0.0.1]/image.jpg").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Private or local"));
    }

    #[tokio::test]
    async fn validate_url_rejects_localhost_hostname() {
        let result = validate_download_url("http://localhost/image.jpg").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Localhost"));
    }
}
