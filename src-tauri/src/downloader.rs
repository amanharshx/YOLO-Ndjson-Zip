use crate::parser::ImageEntry;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tauri::ipc::Channel;
use tokio::sync::Mutex;

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
                            eprintln!("Failed to download {}: {}", url, e);
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
