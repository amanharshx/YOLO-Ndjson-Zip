mod converter;
mod downloader;
mod parser;

use converter::get_converter;
use downloader::{DownloadResult, Downloader, ProgressEvent};
use parser::{normalize_split, parse_ndjson, ImageEntry};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::ipc::Channel;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

const MAX_NDJSON_BYTES: u64 = 100 * 1024 * 1024; // 100 MiB
const MAX_DOWNLOAD_CONCURRENCY: usize = 20;

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
            std::path::Component::Normal(name) => {
                let name = name.to_string_lossy();
                if is_windows_reserved_segment(&name) {
                    return Err(format!("Invalid ZIP entry path: {}", path));
                }
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                return Err(format!("Invalid ZIP entry path: {}", path));
            }
        }
    }

    Ok(normalized)
}

fn is_windows_reserved_segment(segment: &str) -> bool {
    let trimmed = segment.trim_end_matches([' ', '.']);
    if trimmed.is_empty() {
        return false;
    }

    let base = trimmed.split('.').next().unwrap_or(trimmed);
    let upper = base.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    )
}

fn is_ndjson_size_allowed(size: u64) -> bool {
    size <= MAX_NDJSON_BYTES
}

fn short_stable_hash(input: &str) -> String {
    // FNV-1a 64-bit hash, truncated for compact deterministic filenames.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:08x}", (hash & 0xffff_ffff) as u32)
}

fn file_name_with_suffix(file_name: &str, suffix: &str) -> String {
    match file_name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && !ext.is_empty() => {
            format!("{}__{}.{}", stem, suffix, ext)
        }
        _ => format!("{}__{}", file_name, suffix),
    }
}

fn next_unique_file_name(
    original_file: &str,
    hash_source: &str,
    used_names: &mut HashSet<String>,
) -> String {
    let hash = short_stable_hash(hash_source);
    let mut suffix = hash.clone();
    let mut counter = 2usize;

    loop {
        let candidate = file_name_with_suffix(original_file, &suffix);
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
        suffix = format!("{}__{}", hash, counter);
        counter += 1;
    }
}

fn prepare_images_with_unique_output_names(images: &[ImageEntry]) -> Vec<ImageEntry> {
    let mut seen_entries: HashMap<(String, String), usize> = HashMap::new();
    let mut used_names_by_split: HashMap<String, HashSet<String>> = HashMap::new();
    let mut prepared_images = Vec::with_capacity(images.len());

    for image in images {
        let split_key = normalize_split(&image.split).to_string();
        let used_names = used_names_by_split.entry(split_key.clone()).or_default();
        let dedupe_key = (split_key, image.file.clone());
        let occurrence = seen_entries.entry(dedupe_key).or_insert(0);

        let mut prepared = image.clone();
        if *occurrence == 0 {
            if !used_names.insert(image.file.clone()) {
                let hash_source = if image.url.is_empty() {
                    image.file.as_str()
                } else {
                    image.url.as_str()
                };
                prepared.output_file =
                    Some(next_unique_file_name(&image.file, hash_source, used_names));
            }
        } else {
            let hash_source = if image.url.is_empty() {
                image.file.as_str()
            } else {
                image.url.as_str()
            };
            prepared.output_file =
                Some(next_unique_file_name(&image.file, hash_source, used_names));
        }
        *occurrence += 1;
        prepared_images.push(prepared);
    }

    prepared_images
}

#[tauri::command]
async fn convert_ndjson(
    file_path: String,
    format: String,
    output_path: String,
    include_images: bool,
    channel: Channel<ProgressEvent>,
) -> Result<ConvertResult, String> {
    let metadata = std::fs::metadata(&file_path)
        .map_err(|e| format!("Failed to inspect file '{}': {}", &file_path, e))?;
    if !is_ndjson_size_allowed(metadata.len()) {
        return Err(format!(
            "NDJSON file is too large ({} bytes). Maximum allowed is {} bytes.",
            metadata.len(),
            MAX_NDJSON_BYTES
        ));
    }

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

    let mut data = parse_ndjson(&content).map_err(|e| format!("Failed to parse NDJSON: {}", e))?;
    data.images = prepare_images_with_unique_output_names(&data.images);

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
        let downloader = Downloader::new(MAX_DOWNLOAD_CONCURRENCY)
            .map_err(|e| format!("Failed to init downloader: {}", e))?;
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
    use super::{
        file_name_with_suffix, is_ndjson_size_allowed, normalize_zip_path,
        prepare_images_with_unique_output_names, short_stable_hash, MAX_NDJSON_BYTES,
    };
    use crate::parser::parse_ndjson;

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

    #[test]
    fn normalize_zip_path_rejects_windows_reserved_names() {
        assert!(normalize_zip_path("CON.txt").is_err());
        assert!(normalize_zip_path("train/NUL.jpg").is_err());
        assert!(normalize_zip_path("labels/lpt1").is_err());
    }

    #[test]
    fn ndjson_size_limit_allows_max_size() {
        assert!(is_ndjson_size_allowed(MAX_NDJSON_BYTES));
    }

    #[test]
    fn ndjson_size_limit_rejects_oversize() {
        assert!(!is_ndjson_size_allowed(MAX_NDJSON_BYTES + 1));
    }

    #[test]
    fn prepare_images_keeps_first_and_renames_same_split_duplicates() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}
{"type":"image","file":"img1.jpg","width":640,"height":480,"split":"train","url":"https://a.example/img1.jpg"}
{"type":"image","file":"img1.jpg","width":320,"height":240,"split":"val","url":"https://b.example/img1.jpg"}
{"type":"image","file":"img1.jpg","width":800,"height":600,"split":"train","url":"https://c.example/img1.jpg"}
{"type":"image","file":"img2.jpg","width":640,"height":480,"split":"test","url":"https://c.example/img2.jpg"}"#;

        let data = parse_ndjson(content).unwrap();
        let prepared = prepare_images_with_unique_output_names(&data.images);

        assert_eq!(prepared.len(), 4);
        assert_eq!(prepared[0].file, "img1.jpg");
        assert_eq!(prepared[0].effective_file_name(), "img1.jpg");
        assert_eq!(prepared[1].split, "val");
        assert_eq!(prepared[1].effective_file_name(), "img1.jpg");
        assert_eq!(
            prepared[2].effective_file_name(),
            file_name_with_suffix("img1.jpg", &short_stable_hash("https://c.example/img1.jpg"))
        );
        assert_eq!(prepared[3].effective_file_name(), "img2.jpg");
    }

    #[test]
    fn prepare_images_uses_counter_when_hash_suffix_collides() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}
{"type":"image","file":"img1.jpg","width":640,"height":480,"split":"train","url":"https://a.example/img1.jpg"}
{"type":"image","file":"img1.jpg","width":640,"height":480,"split":"train","url":"https://b.example/img1.jpg","annotations":{"boxes":[[0,0.1,0.2,0.3,0.4]]}}
{"type":"image","file":"img1.jpg","width":640,"height":480,"split":"train","url":"https://b.example/img1.jpg","annotations":{"boxes":[[1,0.2,0.3,0.3,0.4]]}}"#;

        let data = parse_ndjson(content).unwrap();
        let prepared = prepare_images_with_unique_output_names(&data.images);

        assert_eq!(prepared.len(), 3);
        let hash = short_stable_hash("https://b.example/img1.jpg");
        assert_eq!(prepared[0].effective_file_name(), "img1.jpg");
        assert_eq!(
            prepared[1].effective_file_name(),
            file_name_with_suffix("img1.jpg", &hash)
        );
        assert_eq!(
            prepared[2].effective_file_name(),
            file_name_with_suffix("img1.jpg", &format!("{}__2", hash))
        );
    }
}
