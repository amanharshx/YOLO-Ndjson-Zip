use crate::parser::ImageEntry;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tauri::ipc::Channel;
use tokio::sync::Mutex;
use url::Url;

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
                    if let Err(err) = validate_download_url(&url) {
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
                                if let Ok(bytes) = response.bytes().await {
                                    let mut map = downloaded.lock().await;
                                    map.insert(file.clone(), bytes.to_vec());
                                } else {
                                    failed.fetch_add(1, Ordering::SeqCst);
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

fn validate_download_url(url: &str) -> Result<(), String> {
    let parsed = Url::parse(url).map_err(|_| "Invalid URL".to_string())?;
    match parsed.scheme() {
        "http" | "https" => {}
        _ => return Err("Only HTTP/HTTPS URLs are allowed".to_string()),
    }

    if let Some(host) = parsed.host_str() {
        let host_lower = host.to_ascii_lowercase();
        if host_lower == "localhost" || host_lower.ends_with(".local") {
            return Err("Localhost addresses are not allowed".to_string());
        }
        if let Some(ip) = parsed
            .host()
            .and_then(|h| h.to_string().parse::<std::net::IpAddr>().ok())
        {
            if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
                return Err("Private or local IPs are not allowed".to_string());
            }
            match ip {
                std::net::IpAddr::V4(v4) => {
                    if v4.is_private() || v4.is_link_local() {
                        return Err("Private or local IPs are not allowed".to_string());
                    }
                }
                std::net::IpAddr::V6(v6) => {
                    if v6.is_unique_local() || v6.is_unicast_link_local() {
                        return Err("Private or local IPs are not allowed".to_string());
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_url_accepts_https() {
        let result = validate_download_url("https://example.com/image.jpg");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_url_accepts_http() {
        let result = validate_download_url("http://example.com/image.jpg");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_url_rejects_localhost() {
        let result = validate_download_url("http://127.0.0.1/image.jpg");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Private or local"));
    }

    #[test]
    fn validate_url_rejects_private_ip_10() {
        let result = validate_download_url("http://10.0.0.1/image.jpg");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Private or local"));
    }

    #[test]
    fn validate_url_rejects_private_ip_192() {
        let result = validate_download_url("http://192.168.1.1/image.jpg");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Private or local"));
    }
}
