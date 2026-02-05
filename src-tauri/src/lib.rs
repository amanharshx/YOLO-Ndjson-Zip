mod converter;
mod downloader;
mod parser;

use converter::get_converter;
use downloader::{DownloadResult, Downloader, ProgressEvent};
use parser::parse_ndjson;
use serde::Serialize;
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::ipc::Channel;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

#[derive(Debug, Serialize)]
pub struct ConvertResult {
    pub zip_path: String,
    pub file_count: usize,
    pub image_count: usize,
    pub download_total: u32,
    pub failed_downloads: usize,
}

fn normalize_zip_path(path: &str) -> Result<String, String> {
    if path.is_empty() {
        return Err("ZIP entry path is empty".to_string());
    }

    let normalized = path.replace('\\', "/");
    if normalized.starts_with("//") {
        return Err(format!("Invalid ZIP entry path: {}", path));
    }
    if normalized.len() >= 2 {
        let bytes = normalized.as_bytes();
        if bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
            return Err(format!("Invalid ZIP entry path: {}", path));
        }
    }
    for component in Path::new(&normalized).components() {
        match component {
            std::path::Component::Normal(_) | std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                return Err(format!("Invalid ZIP entry path: {}", path));
            }
        }
    }

    Ok(normalized)
}

#[tauri::command]
async fn convert_ndjson(
    file_path: String,
    format: String,
    output_path: String,
    include_images: bool,
    channel: Channel<ProgressEvent>,
) -> Result<ConvertResult, String> {
    // Read the NDJSON file
    let content = std::fs::read_to_string(&file_path)
        .map_err(|e| format!("Failed to read file '{}': {}", &file_path, e))?;

    // Parse NDJSON
    channel
        .send(ProgressEvent {
            phase: "parsing".to_string(),
            current: 0,
            total: 1,
            item: Some("Parsing NDJSON...".to_string()),
        })
        .ok();

    let data = parse_ndjson(&content).map_err(|e| format!("Failed to parse NDJSON: {}", e))?;

    let mut seen_files = HashSet::new();
    let mut duplicate_files = Vec::new();
    for img in &data.images {
        if !seen_files.insert(img.file.as_str()) && duplicate_files.len() < 5 {
            duplicate_files.push(img.file.clone());
        }
    }
    if !duplicate_files.is_empty() {
        return Err(format!(
            "Duplicate image filenames detected: {}. Please ensure filenames are unique.",
            duplicate_files.join(", ")
        ));
    }

    channel
        .send(ProgressEvent {
            phase: "parsing".to_string(),
            current: 1,
            total: 1,
            item: Some(format!("Parsed {} images", data.images.len())),
        })
        .ok();

    // Download images if requested
    let download_result = if include_images {
        let downloader =
            Downloader::new(100).map_err(|e| format!("Failed to init downloader: {}", e))?;
        downloader.download_all(&data.images, &channel).await
    } else {
        DownloadResult {
            files: std::collections::HashMap::new(),
            total: 0,
            failed: 0,
        }
    };

    let image_count = download_result.files.len();
    let download_total = download_result.total;
    let failed_downloads = download_result.failed;
    if include_images && download_total > 0 && image_count == 0 {
        return Err(
            "All image downloads failed. Check your network or CDN access and try again."
                .to_string(),
        );
    }

    // Get converter
    let converter = get_converter(&format).ok_or_else(|| format!("Unknown format: {}", format))?;

    // Convert
    channel
        .send(ProgressEvent {
            phase: "converting".to_string(),
            current: 0,
            total: 1,
            item: Some("Converting annotations...".to_string()),
        })
        .ok();

    let files = converter.convert(&data, &download_result.files);

    channel
        .send(ProgressEvent {
            phase: "converting".to_string(),
            current: 1,
            total: 1,
            item: Some(format!("Converted {} files", files.len())),
        })
        .ok();

    // Create ZIP
    let total_files = files.len() as u32;
    channel
        .send(ProgressEvent {
            phase: "zipping".to_string(),
            current: 0,
            total: total_files,
            item: Some("Creating ZIP...".to_string()),
        })
        .ok();

    let output_path = PathBuf::from(&output_path);
    let file = std::fs::File::create(&output_path).map_err(|e| {
        format!(
            "Failed to create output file '{}': {}",
            output_path.display(),
            e
        )
    })?;

    let zip_result = (|| -> Result<(), String> {
        let mut zip = ZipWriter::new(file);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

        for (idx, (path, content)) in files.iter().enumerate() {
            let zip_path = normalize_zip_path(path)?;
            zip.start_file(&zip_path, options)
                .map_err(|e| format!("Failed to add file to ZIP: {}", e))?;
            zip.write_all(content)
                .map_err(|e| format!("Failed to write file to ZIP: {}", e))?;

            if idx % 50 == 0 || idx == files.len() - 1 {
                channel
                    .send(ProgressEvent {
                        phase: "zipping".to_string(),
                        current: (idx + 1) as u32,
                        total: total_files,
                        item: Some(zip_path),
                    })
                    .ok();
            }
        }

        zip.finish()
            .map_err(|e| format!("Failed to finish ZIP: {}", e))?;
        Ok(())
    })();

    if let Err(err) = zip_result {
        let _ = std::fs::remove_file(&output_path);
        return Err(err);
    }

    channel
        .send(ProgressEvent {
            phase: "complete".to_string(),
            current: 1,
            total: 1,
            item: None,
        })
        .ok();

    Ok(ConvertResult {
        zip_path: output_path.to_string_lossy().to_string(),
        file_count: files.len(),
        image_count,
        download_total,
        failed_downloads,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .invoke_handler(tauri::generate_handler![convert_ndjson])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::normalize_zip_path;

    #[test]
    fn normalize_zip_path_accepts_simple_paths() {
        assert_eq!(
            normalize_zip_path("images/foo.jpg").unwrap(),
            "images/foo.jpg"
        );
        assert_eq!(
            normalize_zip_path("labels\\foo.txt").unwrap(),
            "labels/foo.txt"
        );
    }

    #[test]
    fn normalize_zip_path_rejects_parent_dirs() {
        assert!(normalize_zip_path("../evil.txt").is_err());
        assert!(normalize_zip_path("images/../../evil.txt").is_err());
    }

    #[test]
    fn normalize_zip_path_rejects_absolute_paths() {
        assert!(normalize_zip_path("/etc/passwd").is_err());
    }

    #[test]
    fn normalize_zip_path_rejects_windows_prefix() {
        assert!(normalize_zip_path("C:\\evil.txt").is_err());
    }
}
