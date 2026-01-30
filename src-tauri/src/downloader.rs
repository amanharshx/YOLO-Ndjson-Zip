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
    pub fn new(concurrency: usize) -> Self {
        let client = Client::builder()
            .pool_max_idle_per_host(concurrency)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            concurrency,
        }
    }

    pub async fn download_all(
        &self,
        images: &[ImageEntry],
        channel: &Channel<ProgressEvent>,
    ) -> HashMap<String, Vec<u8>> {
        let images_with_urls: Vec<_> = images
            .iter()
            .filter(|img| !img.url.is_empty())
            .map(|img| (img.file.clone(), img.url.clone()))
            .collect();

        let total = images_with_urls.len() as u32;

        if total == 0 {
            return HashMap::new();
        }

        let _ = channel.send(ProgressEvent {
            phase: "downloading".to_string(),
            current: 0,
            total,
            item: None,
        });

        let downloaded = Arc::new(Mutex::new(HashMap::new()));
        let counter = Arc::new(AtomicU32::new(0));
        let client = self.client.clone();

        stream::iter(images_with_urls)
            .map(|(file, url)| {
                let client = client.clone();
                let downloaded = Arc::clone(&downloaded);
                let counter = Arc::clone(&counter);
                let channel = channel.clone();

                async move {
                    match client.get(&url).send().await {
                        Ok(response) => {
                            if response.status().is_success() {
                                if let Ok(bytes) = response.bytes().await {
                                    let mut map = downloaded.lock().await;
                                    map.insert(file.clone(), bytes.to_vec());
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to download {}: {}", url, e);
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

        Arc::try_unwrap(downloaded)
            .expect("Failed to unwrap Arc")
            .into_inner()
    }
}
