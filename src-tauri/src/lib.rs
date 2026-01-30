mod converter;
mod downloader;
mod parser;

use converter::get_converter;
use downloader::{Downloader, ProgressEvent};
use parser::parse_ndjson;
use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;
use tauri::ipc::Channel;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

#[derive(Debug, Serialize)]
pub struct ConvertResult {
    pub zip_path: String,
    pub file_count: usize,
    pub image_count: usize,
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
    let content =
        std::fs::read_to_string(&file_path).map_err(|e| format!("Failed to read file: {}", e))?;

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

    channel
        .send(ProgressEvent {
            phase: "parsing".to_string(),
            current: 1,
            total: 1,
            item: Some(format!("Parsed {} images", data.images.len())),
        })
        .ok();

    // Download images if requested
    let downloaded_images = if include_images {
        let downloader = Downloader::new(100);
        downloader.download_all(&data.images, &channel).await
    } else {
        std::collections::HashMap::new()
    };

    let image_count = downloaded_images.len();

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

    let files = converter.convert(&data, &downloaded_images);

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
    let file = std::fs::File::create(&output_path)
        .map_err(|e| format!("Failed to create output file: {}", e))?;

    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    for (idx, (path, content)) in files.iter().enumerate() {
        zip.start_file(path, options)
            .map_err(|e| format!("Failed to add file to ZIP: {}", e))?;
        zip.write_all(content)
            .map_err(|e| format!("Failed to write file to ZIP: {}", e))?;

        if idx % 50 == 0 || idx == files.len() - 1 {
            channel
                .send(ProgressEvent {
                    phase: "zipping".to_string(),
                    current: (idx + 1) as u32,
                    total: total_files,
                    item: Some(path.clone()),
                })
                .ok();
        }
    }

    zip.finish()
        .map_err(|e| format!("Failed to finish ZIP: {}", e))?;

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
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .invoke_handler(tauri::generate_handler![convert_ndjson])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
