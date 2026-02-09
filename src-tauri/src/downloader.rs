use crate::parser::ImageEntry;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tauri::ipc::Channel;
use tokio::sync::Mutex;
use url::{Host, Url};

const MAX_DOWNLOAD_BYTES: usize = 50 * 1024 * 1024; // 50 MiB per image

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
            .map(|img| (img.file.clone(), img.url.clone()))
            .collect();

        let total = images_with_urls.len() as u32;

        if total == 0 {
            return DownloadResult {
                files: HashMap::new(),
                total: 0,
                failed: 0,
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
        let client = self.client.clone();

        stream::iter(images_with_urls)
            .map(|(file, url)| {
                let client = client.clone();
                let downloaded = Arc::clone(&downloaded);
                let counter = Arc::clone(&counter);
                let failed = Arc::clone(&failed);
                let channel = channel.clone();

                async move {
                    if let Err(err) = validate_download_url(&url).await {
                        eprintln!("Skipping download for '{}': {}", file, err);
                        failed.fetch_add(1, Ordering::SeqCst);
                        let current = counter.fetch_add(1, Ordering::SeqCst) + 1;
                        let _ = channel.send(ProgressEvent {
                            phase: "downloading".to_string(),
                            current,
                            total,
                            item: Some(file),
                        });
                        return;
                    }

                    match client.get(&url).send().await {
                        Ok(response) => {
                            if response.status().is_success() {
                                match read_response_with_limit(response, MAX_DOWNLOAD_BYTES).await {
                                    Ok(bytes) => {
                                        let mut map = downloaded.lock().await;
                                        map.insert(file.clone(), bytes);
                                    }
                                    Err(err) => {
                                        eprintln!("Skipping download for '{}': {}", file, err);
                                        failed.fetch_add(1, Ordering::SeqCst);
                                    }
                                }
                            } else {
                                failed.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to download '{}': {}", file, e);
                            failed.fetch_add(1, Ordering::SeqCst);
                        }
                    }

                    let current = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = channel.send(ProgressEvent {
                        phase: "downloading".to_string(),
                        current,
                        total,
                        item: Some(file),
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

        DownloadResult {
            files,
            total,
            failed: failed_count as usize,
        }
    }
}

pub struct DownloadResult {
    pub files: HashMap<String, Vec<u8>>,
    pub total: u32,
    pub failed: usize,
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
) -> Result<Vec<u8>, String> {
    if let Some(content_length) = response.content_length() {
        if content_length > max_bytes as u64 {
            return Err(format!(
                "Response too large ({} bytes, max {})",
                content_length, max_bytes
            ));
        }
    }

    let mut downloaded = Vec::new();
    let mut stream = response.bytes_stream();
    let mut total_bytes = 0usize;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Failed to read response body: {}", e))?;
        total_bytes = total_bytes
            .checked_add(chunk.len())
            .ok_or_else(|| "Response body size overflow".to_string())?;

        if total_bytes > max_bytes {
            return Err(format!("Response too large (max {} bytes)", max_bytes));
        }

        downloaded.extend_from_slice(&chunk);
    }

    Ok(downloaded)
}

#[cfg(test)]
mod tests {
    use super::*;

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
