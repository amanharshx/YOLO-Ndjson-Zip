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
const CLOCK_SKEW_TOLERANCE_SECS: i64 = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    MissingUrl,
    ExpiredUrl,
    AccessDenied,
    NotFound,
    Timeout,
    Connect,
    Dns,
    BlockedAddress,
    MalformedUrl,
    UnsupportedScheme,
    ServerError,
    ResponseError,
    TooLarge,
    SizeOverflow,
    HttpError,
    DownloadError,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FailureGroup {
    pub kind: FailureKind,
    pub count: usize,
    pub examples: Vec<String>,
    pub http_statuses: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpirySummary {
    pub urls_with_expiry: usize,
    pub expired_urls: usize,
    pub all_expired: bool,
    pub latest_expired_at: Option<i64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct FailureSummary {
    pub groups: Vec<FailureGroup>,
    pub expiry: Option<ExpirySummary>,
}

impl FailureSummary {
    fn record(&mut self, kind: FailureKind, file_name: &str, http_status: Option<u16>) {
        let index = match self.groups.binary_search_by_key(&kind, |group| group.kind) {
            Ok(index) => index,
            Err(index) => {
                self.groups.insert(
                    index,
                    FailureGroup {
                        kind,
                        count: 0,
                        examples: Vec::new(),
                        http_statuses: Vec::new(),
                    },
                );
                index
            }
        };
        let group = &mut self.groups[index];
        group.count += 1;

        let example = safe_file_name(file_name);
        if !group.examples.contains(&example) {
            group.examples.push(example);
            group.examples.sort();
            group.examples.truncate(3);
        }

        if let Some(status) = http_status {
            match group.http_statuses.binary_search(&status) {
                Ok(_) => {}
                Err(index) => group.http_statuses.insert(index, status),
            }
        }
    }

    pub fn count(&self, kind: FailureKind) -> usize {
        self.groups
            .iter()
            .find(|group| group.kind == kind)
            .map_or(0, |group| group.count)
    }

    fn expired_url_failures(&self) -> usize {
        self.count(FailureKind::ExpiredUrl) + self.count(FailureKind::AccessDenied)
    }
}

fn safe_file_name(value: &str) -> String {
    let normalized = value.replace('\\', "/");
    let basename = normalized
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("unknown file");
    basename
        .split(['?', '#'])
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("unknown file")
        .to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UrlExpiry {
    Missing,
    Malformed,
    Active(DateTime<Utc>),
    Expired(DateTime<Utc>),
}

fn parse_url_expiry(url: &str, now: DateTime<Utc>) -> UrlExpiry {
    let Ok(parsed) = Url::parse(url) else {
        return UrlExpiry::Missing;
    };
    let Some(value) = parsed
        .query_pairs()
        .find_map(|(key, value)| (key == "Expires").then_some(value))
    else {
        return UrlExpiry::Missing;
    };
    let Ok(timestamp) = value.parse::<i64>() else {
        return UrlExpiry::Malformed;
    };
    let Some(expires_at) = DateTime::from_timestamp(timestamp, 0) else {
        return UrlExpiry::Malformed;
    };
    let confidently_expired = expires_at
        .checked_add_signed(chrono::Duration::seconds(CLOCK_SKEW_TOLERANCE_SECS))
        .is_some_and(|deadline| deadline < now);

    if confidently_expired {
        UrlExpiry::Expired(expires_at)
    } else {
        UrlExpiry::Active(expires_at)
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
enum UrlRejection {
    #[error("download URL is malformed")]
    MalformedUrl,
    #[error("download URL uses an unsupported scheme")]
    UnsupportedScheme,
    #[error("download host resolved to a blocked address")]
    BlockedAddress,
    #[error("download host could not be resolved")]
    DnsFailure,
}

impl UrlRejection {
    fn failure_kind(self) -> FailureKind {
        match self {
            Self::MalformedUrl => FailureKind::MalformedUrl,
            Self::UnsupportedScheme => FailureKind::UnsupportedScheme,
            Self::BlockedAddress => FailureKind::BlockedAddress,
            Self::DnsFailure => FailureKind::Dns,
        }
    }
}

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

    fn failure_kind(&self) -> FailureKind {
        match self {
            Self::Http(StatusCode::FORBIDDEN) => FailureKind::AccessDenied,
            Self::Http(StatusCode::NOT_FOUND) => FailureKind::NotFound,
            Self::Http(status) if status.is_server_error() => FailureKind::ServerError,
            Self::Http(_) => FailureKind::HttpError,
            Self::Transport(error) if error.is_timeout() => FailureKind::Timeout,
            Self::Transport(error) if error.is_connect() => FailureKind::Connect,
            Self::Transport(_) => FailureKind::DownloadError,
            Self::Body(_) => FailureKind::ResponseError,
            Self::TooLarge { .. } => FailureKind::TooLarge,
            Self::SizeOverflow => FailureKind::SizeOverflow,
        }
    }

    fn http_status(&self) -> Option<u16> {
        match self {
            Self::Http(status) => Some(status.as_u16()),
            _ => None,
        }
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
        let mut failure_summary = FailureSummary::default();
        let mut images_with_urls = Vec::new();
        for img in images {
            if img.url.is_empty() {
                failure_summary.record(FailureKind::MissingUrl, img.effective_file_name(), None);
            } else {
                let split = normalize_split(&img.split);
                let item_label = format!("{}/{}", split, img.effective_file_name());
                let file_name = safe_file_name(img.effective_file_name());
                let download_key = image_entry_download_key(img);
                images_with_urls.push((item_label, file_name, download_key, img.url.clone()));
            }
        }

        let total = images_with_urls.len() as u32;

        if total == 0 {
            return DownloadResult {
                files: HashMap::new(),
                total: 0,
                failed: 0,
                expired_url_failures: 0,
                failure_summary,
            };
        }

        let now = Utc::now();
        let mut ready_downloads = Vec::with_capacity(images_with_urls.len());
        let mut urls_with_expiry = 0usize;
        let mut expired_urls = 0usize;
        let mut latest_expired_at = None;
        for (item_label, file_name, download_key, url) in images_with_urls {
            match parse_url_expiry(&url, now) {
                UrlExpiry::Expired(expires_at) => {
                    urls_with_expiry += 1;
                    expired_urls += 1;
                    latest_expired_at = Some(
                        latest_expired_at
                            .map_or(expires_at, |current: DateTime<Utc>| current.max(expires_at)),
                    );
                    failure_summary.record(FailureKind::ExpiredUrl, &file_name, None);
                }
                UrlExpiry::Active(_) => {
                    urls_with_expiry += 1;
                    ready_downloads.push((item_label, file_name, download_key, url));
                }
                UrlExpiry::Missing | UrlExpiry::Malformed => {
                    ready_downloads.push((item_label, file_name, download_key, url));
                }
            }
        }
        if urls_with_expiry > 0 {
            failure_summary.expiry = Some(ExpirySummary {
                urls_with_expiry,
                expired_urls,
                all_expired: expired_urls == total as usize,
                latest_expired_at: latest_expired_at.map(|value| value.timestamp()),
            });
        }

        let _ = channel.send(ProgressEvent {
            phase: "downloading".to_string(),
            current: expired_urls as u32,
            total,
            item: None,
        });

        if ready_downloads.is_empty() {
            return DownloadResult {
                files: HashMap::new(),
                total,
                failed: expired_urls,
                expired_url_failures: failure_summary.expired_url_failures(),
                failure_summary,
            };
        }

        let downloaded = Arc::new(Mutex::new(HashMap::new()));
        let counter = Arc::new(AtomicU32::new(expired_urls as u32));
        let failed = Arc::new(AtomicU32::new(expired_urls as u32));
        let failure_summary = Arc::new(Mutex::new(failure_summary));
        let client = self.client.clone();

        stream::iter(ready_downloads)
            .map(|(item_label, file_name, download_key, url)| {
                let client = client.clone();
                let downloaded = Arc::clone(&downloaded);
                let counter = Arc::clone(&counter);
                let failed = Arc::clone(&failed);
                let failure_summary = Arc::clone(&failure_summary);
                let channel = channel.clone();

                async move {
                    if let Err(err) = validate_download_url(&url).await {
                        let kind = err.failure_kind();
                        eprintln!("Skipping download for '{}': {:?}", file_name, kind);
                        failure_summary.lock().await.record(kind, &file_name, None);
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
                            let kind = err.failure_kind();
                            eprintln!("Skipping download for '{}': {:?}", file_name, kind);
                            failure_summary.lock().await.record(
                                kind,
                                &file_name,
                                err.http_status(),
                            );
                            failed.fetch_add(1, Ordering::SeqCst);
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

        let failure_summary = match Arc::try_unwrap(failure_summary) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.lock().await.clone(),
        };
        let expired_url_failure_count = failure_summary.expired_url_failures();

        DownloadResult {
            files,
            total,
            failed: failed_count as usize,
            expired_url_failures: expired_url_failure_count,
            failure_summary,
        }
    }
}

pub struct DownloadResult {
    pub files: HashMap<String, Vec<u8>>,
    pub total: u32,
    pub failed: usize,
    pub expired_url_failures: usize,
    pub failure_summary: FailureSummary,
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

async fn validate_download_url(url: &str) -> Result<(), UrlRejection> {
    let parsed = Url::parse(url).map_err(|_| UrlRejection::MalformedUrl)?;
    match parsed.scheme() {
        "http" | "https" => {}
        _ => return Err(UrlRejection::UnsupportedScheme),
    }

    let host = parsed.host().ok_or(UrlRejection::MalformedUrl)?;
    match host {
        Host::Ipv4(v4) => {
            if is_forbidden_ip(IpAddr::V4(v4)) {
                return Err(UrlRejection::BlockedAddress);
            }
        }
        Host::Ipv6(v6) => {
            if is_forbidden_ip(IpAddr::V6(v6)) {
                return Err(UrlRejection::BlockedAddress);
            }
        }
        Host::Domain(domain) => {
            let host_lower = domain.to_ascii_lowercase();
            if host_lower == "localhost"
                || host_lower.ends_with(".localhost")
                || host_lower.ends_with(".local")
            {
                return Err(UrlRejection::BlockedAddress);
            }

            let port = parsed.port_or_known_default().unwrap_or(80);
            let mut addrs = tokio::net::lookup_host((domain, port))
                .await
                .map_err(|_| UrlRejection::DnsFailure)?;
            let mut resolved_any = false;

            for addr in addrs.by_ref() {
                resolved_any = true;
                if is_forbidden_ip(addr.ip()) {
                    return Err(UrlRejection::BlockedAddress);
                }
            }

            if !resolved_any {
                return Err(UrlRejection::DnsFailure);
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
        assert_eq!(error.failure_kind(), FailureKind::AccessDenied);
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn distinguishes_timeout_from_connect_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let timeout_address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
        });
        let timeout_client = Client::builder()
            .timeout(Duration::from_millis(10))
            .build()
            .unwrap();
        let timeout_error = timeout_client
            .get(format!("http://{timeout_address}/image.jpg"))
            .send()
            .await
            .unwrap_err();

        assert!(timeout_error.is_timeout());
        assert_eq!(
            DownloadError::Transport(timeout_error).failure_kind(),
            FailureKind::Timeout
        );
        server.await.unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let connect_address = listener.local_addr().unwrap();
        drop(listener);
        let connect_error = Client::new()
            .get(format!("http://{connect_address}/image.jpg"))
            .send()
            .await
            .unwrap_err();

        assert!(connect_error.is_connect());
        assert_eq!(
            DownloadError::Transport(connect_error).failure_kind(),
            FailureKind::Connect
        );
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
        assert_eq!(result, Err(UrlRejection::BlockedAddress));
    }

    #[tokio::test]
    async fn validate_url_rejects_private_ip_10() {
        let result = validate_download_url("http://10.0.0.1/image.jpg").await;
        assert_eq!(result, Err(UrlRejection::BlockedAddress));
    }

    #[tokio::test]
    async fn validate_url_rejects_private_ip_192() {
        let result = validate_download_url("http://192.168.1.1/image.jpg").await;
        assert_eq!(result, Err(UrlRejection::BlockedAddress));
    }

    #[tokio::test]
    async fn validate_url_rejects_ipv4_mapped_ipv6_loopback() {
        let result = validate_download_url("http://[::ffff:127.0.0.1]/image.jpg").await;
        assert_eq!(result, Err(UrlRejection::BlockedAddress));
    }

    #[tokio::test]
    async fn validate_url_rejects_localhost_hostname() {
        let result = validate_download_url("http://localhost/image.jpg").await;
        assert_eq!(result, Err(UrlRejection::BlockedAddress));
    }

    #[tokio::test]
    async fn validate_url_returns_typed_rejections() {
        assert_eq!(
            validate_download_url("not a url").await,
            Err(UrlRejection::MalformedUrl)
        );
        assert_eq!(
            validate_download_url("file:///tmp/image.jpg").await,
            Err(UrlRejection::UnsupportedScheme)
        );
        assert_eq!(
            validate_download_url("http://missing.invalid/image.jpg").await,
            Err(UrlRejection::DnsFailure)
        );
    }

    #[test]
    fn parses_expiry_with_clock_skew_tolerance() {
        let now = DateTime::from_timestamp(2_000, 0).unwrap();

        assert_eq!(
            parse_url_expiry("https://example.com/a.jpg", now),
            UrlExpiry::Missing
        );
        assert_eq!(
            parse_url_expiry("https://example.com/a.jpg?Expires=bad", now),
            UrlExpiry::Malformed
        );
        assert_eq!(
            parse_url_expiry("https://example.com/a.jpg?Expires=2120", now),
            UrlExpiry::Active(DateTime::from_timestamp(2_120, 0).unwrap())
        );
        assert_eq!(
            parse_url_expiry("https://example.com/a.jpg?Expires=1879", now),
            UrlExpiry::Expired(DateTime::from_timestamp(1_879, 0).unwrap())
        );
        assert_eq!(
            parse_url_expiry("https://example.com/a.jpg?Expires=1880", now),
            UrlExpiry::Active(DateTime::from_timestamp(1_880, 0).unwrap())
        );
    }

    #[test]
    fn aggregates_failures_with_safe_bounded_examples() {
        let mut summary = FailureSummary::default();
        let names = [
            "https://secret.example/five.jpg?Signature=token",
            "folder/one.jpg",
            r"folder\two.jpg",
            "../three.jpg",
            "four.jpg",
        ];

        for name in names {
            summary.record(FailureKind::NotFound, name, Some(404));
        }
        summary.record(FailureKind::NotFound, "folder/one.jpg", Some(404));

        let mut reverse_summary = FailureSummary::default();
        for name in names.into_iter().rev() {
            reverse_summary.record(FailureKind::NotFound, name, Some(404));
        }

        assert_eq!(summary.groups.len(), 1);
        assert_eq!(summary.groups[0].count, 6);
        assert_eq!(
            summary.groups[0].examples,
            vec!["five.jpg", "four.jpg", "one.jpg"]
        );
        assert_eq!(
            summary.groups[0].examples,
            reverse_summary.groups[0].examples
        );
        assert_eq!(summary.groups[0].http_statuses, vec![404]);
        assert!(!format!("{summary:?}").contains("Signature"));
    }

    #[test]
    fn classifies_http_and_size_failures() {
        assert_eq!(
            DownloadError::Http(StatusCode::FORBIDDEN).failure_kind(),
            FailureKind::AccessDenied
        );
        assert_eq!(
            DownloadError::Http(StatusCode::NOT_FOUND).failure_kind(),
            FailureKind::NotFound
        );
        assert_eq!(
            DownloadError::Http(StatusCode::BAD_GATEWAY).failure_kind(),
            FailureKind::ServerError
        );
        assert_eq!(
            DownloadError::Http(StatusCode::UNAUTHORIZED).failure_kind(),
            FailureKind::HttpError
        );
        assert_eq!(
            DownloadError::TooLarge {
                actual: 2,
                maximum: 1
            }
            .failure_kind(),
            FailureKind::TooLarge
        );
        assert_eq!(
            DownloadError::SizeOverflow.failure_kind(),
            FailureKind::SizeOverflow
        );
    }

    #[tokio::test]
    async fn classifies_body_and_generic_transport_failures() {
        let (url, _, server) = spawn_response_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nab".to_string(),
        ])
        .await;
        let body_error = download_prevalidated_url(
            &Client::new(),
            &url,
            MAX_DOWNLOAD_BYTES,
            RetryPolicy {
                max_attempts: 1,
                base_delay: Duration::ZERO,
                max_retry_after: Duration::ZERO,
            },
        )
        .await
        .unwrap_err();
        server.await.unwrap();

        assert_eq!(body_error.failure_kind(), FailureKind::ResponseError);

        let generic_error = Client::new().get("://invalid").send().await.unwrap_err();
        assert!(!generic_error.is_timeout());
        assert!(!generic_error.is_connect());
        assert_eq!(
            DownloadError::Transport(generic_error).failure_kind(),
            FailureKind::DownloadError
        );
    }

    #[tokio::test]
    async fn all_expired_urls_make_no_request() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let expires = Utc::now().timestamp() - CLOCK_SKEW_TOLERANCE_SECS - 1;
        let content = format!(
            r#"{{"type":"dataset","class_names":{{}}}}
{{"type":"image","file":"expired.jpg","width":1,"height":1,"split":"train","url":"http://{address}/expired.jpg?Expires={expires}&Signature=secret"}}"#
        );
        let data = crate::parser::parse_ndjson(&content).unwrap();
        let channel = Channel::new(|_| Ok(()));

        let result = Downloader::new(1)
            .unwrap()
            .download_all(&data.images, &channel)
            .await;

        assert_eq!(result.failed, 1);
        assert_eq!(result.expired_url_failures, 1);
        assert_eq!(
            result.failure_summary.groups[0].kind,
            FailureKind::ExpiredUrl
        );
        assert!(result.failure_summary.expiry.as_ref().unwrap().all_expired);
        assert!(timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err());
    }
}
