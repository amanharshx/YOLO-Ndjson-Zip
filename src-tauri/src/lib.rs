mod converter;
mod downloader;
mod parser;

use converter::get_converter;
use downloader::{DownloadResult, Downloader, ProgressEvent};
use parser::{normalize_split, parse_ndjson, ImageEntry, NDJSONData, ParseError};
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
    pub omitted_images: usize,
    pub expired_url_failures: usize,
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

/// True when a `semantic` dataset has no polygon segments in any image.
/// Per-image empty is still valid; this only triggers when the entire dataset
/// is devoid of polygons (e.g. PNG-mask-origin exports).
fn semantic_dataset_has_no_polygons(data: &NDJSONData) -> bool {
    data.metadata.task == "semantic"
        && data
            .images
            .iter()
            .all(|img| img.get_segment_annotations().is_empty())
}

/// Semantic datasets are polygon data, so they can only target polygon-capable
/// formats. `yolo_darknet`, `createml`, and `tfrecord` cannot represent polygons.
fn semantic_format_supported(format: &str) -> bool {
    matches!(
        format.to_ascii_lowercase().as_str(),
        "yolo" | "coco" | "pascal_voc" | "voc"
    )
}

fn validate_pose_dataset(data: &mut NDJSONData) -> Result<(), ParseError> {
    if data.metadata.task != "pose" {
        return Ok(());
    }

    if let Some(shape) = data.pose_kpt_shape()? {
        data.metadata.kpt_shape = Some(vec![shape.num_keypoints as i32, shape.dims as i32]);
    }
    Ok(())
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

fn filter_images_without_downloads(
    data: &mut NDJSONData,
    downloaded_images: &HashMap<String, Vec<u8>>,
    include_images: bool,
) -> usize {
    if !include_images {
        return 0;
    }

    let original_image_count = data.images.len();
    data.images
        .retain(|image| downloaded_images.contains_key(&parser::image_entry_download_key(image)));
    original_image_count - data.images.len()
}

fn validate_downloaded_image_count(
    include_images: bool,
    original_image_count: usize,
    kept_image_count: usize,
    download_total: u32,
    expired_url_failures: usize,
) -> Result<(), String> {
    if !include_images || original_image_count == 0 || kept_image_count > 0 {
        return Ok(());
    }

    if download_total == 0 {
        return Err(
            "No image URLs were found in this export. Re-export the dataset and try again."
                .to_string(),
        );
    }

    let mut message =
        "All image downloads failed. Check your network or CDN access and try again.".to_string();
    if expired_url_failures > 0 {
        message
            .push_str(" Some signed URLs may have expired; re-export the dataset and try again.");
    }
    Err(message)
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
        .map_err(|e| format!("Failed to inspect file '{}': {}", file_path, e))?;
    if !is_ndjson_size_allowed(metadata.len()) {
        return Err(format!(
            "NDJSON file is too large ({} bytes). Maximum allowed is {} bytes.",
            metadata.len(),
            MAX_NDJSON_BYTES
        ));
    }

    // Read the NDJSON file
    let content = std::fs::read_to_string(&file_path)
        .map_err(|e| format!("Failed to read file '{}': {}", file_path, e))?;

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
    validate_pose_dataset(&mut data).map_err(|e| e.to_string())?;
    data.images = prepare_images_with_unique_output_names(&data.images);
    let original_image_count = data.images.len();

    channel
        .send(ProgressEvent {
            phase: "parsing".to_string(),
            current: 1,
            total: 1,
            item: Some(format!("Parsed {} images", data.images.len())),
        })
        .ok();

    // Semantic segmentation is polygon data; only polygon-capable formats can
    // represent it. Reject other formats (e.g. CreateML/TFRecord/Darknet) up front.
    if data.metadata.task == "semantic" && !semantic_format_supported(&format) {
        return Err(format!(
            "The '{}' format does not support semantic segmentation datasets. \
             Use YOLO, COCO, or Pascal VOC.",
            format
        ));
    }

    // Semantic datasets carry polygon segments. PNG-mask-origin exports arrive
    // with no real polygons (and often junk class names), which would silently
    // produce an empty dataset. Fail fast before downloading anything.
    if semantic_dataset_has_no_polygons(&data) {
        return Err("Semantic dataset has no polygon segments in any image. \
             PNG-mask exports are not supported; provide polygon annotations."
            .to_string());
    }

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
            expired_url_failures: 0,
        }
    };

    let download_total = download_result.total;
    let failed_downloads = download_result.failed;
    let expired_url_failures = download_result.expired_url_failures;
    let omitted_images =
        filter_images_without_downloads(&mut data, &download_result.files, include_images);
    let kept_image_count = data.images.len();
    validate_downloaded_image_count(
        include_images,
        original_image_count,
        kept_image_count,
        download_total,
        expired_url_failures,
    )?;
    let image_count = download_result.files.len();

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
        omitted_images,
        expired_url_failures,
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
        file_name_with_suffix, filter_images_without_downloads, is_ndjson_size_allowed,
        normalize_zip_path, prepare_images_with_unique_output_names,
        semantic_dataset_has_no_polygons, semantic_format_supported, short_stable_hash,
        validate_downloaded_image_count, validate_pose_dataset, MAX_NDJSON_BYTES,
    };
    use crate::converter::get_converter;
    use crate::parser::{image_entry_download_key, parse_ndjson};
    use std::collections::HashMap;

    fn partial_download_data() -> crate::parser::NDJSONData {
        parse_ndjson(
            r#"{"type":"dataset","task":"detect","name":"partial","class_names":{"0":"cat"}}
{"type":"image","file":"present.jpg","width":640,"height":480,"split":"train","url":"https://cdn.example/present.jpg","annotations":{"boxes":[[0,0.5,0.5,0.2,0.2]]}}
{"type":"image","file":"missing.jpg","width":640,"height":480,"split":"train","url":"","annotations":{"boxes":[[0,0.5,0.5,0.2,0.2]]}}"#,
        )
        .unwrap()
    }

    #[test]
    fn filtering_keeps_only_records_with_downloaded_bytes() {
        let mut data = partial_download_data();
        let mut downloaded = HashMap::new();
        downloaded.insert(
            image_entry_download_key(&data.images[0]),
            b"present-image".to_vec(),
        );

        let omitted = filter_images_without_downloads(&mut data, &downloaded, true);

        assert_eq!(omitted, 1);
        assert_eq!(data.images.len(), 1);
        assert_eq!(data.images[0].file, "present.jpg");
    }

    #[test]
    fn filtering_is_disabled_for_labels_only_conversion() {
        let mut data = partial_download_data();

        let omitted = filter_images_without_downloads(&mut data, &HashMap::new(), false);

        assert_eq!(omitted, 0);
        assert_eq!(data.images.len(), 2);
    }

    #[test]
    fn filtered_records_are_absent_from_every_converter_output() {
        let mut data = partial_download_data();
        let mut downloaded = HashMap::new();
        downloaded.insert(
            image_entry_download_key(&data.images[0]),
            b"present-image".to_vec(),
        );
        filter_images_without_downloads(&mut data, &downloaded, true);

        for format in ["yolo", "yolo_darknet", "coco", "pascal_voc", "createml"] {
            let files = get_converter(format).unwrap().convert(&data, &downloaded);
            assert!(
                files.keys().all(|path| !path.contains("missing")),
                "{format} emitted path for missing image"
            );
            assert!(
                files.values().all(|content| {
                    std::str::from_utf8(content)
                        .map(|text| !text.contains("missing.jpg"))
                        .unwrap_or(true)
                }),
                "{format} emitted annotation for missing image"
            );
        }
    }

    #[test]
    fn all_missing_images_fail_even_when_no_download_was_attempted() {
        let error = validate_downloaded_image_count(true, 2, 0, 0, 0).unwrap_err();

        assert_eq!(
            error,
            "No image URLs were found in this export. Re-export the dataset and try again."
        );
    }

    #[test]
    fn all_network_failures_return_generic_retry_message() {
        let error = validate_downloaded_image_count(true, 2, 0, 2, 0).unwrap_err();

        assert_eq!(
            error,
            "All image downloads failed. Check your network or CDN access and try again."
        );
        assert!(!error.contains("signed URLs"));
    }

    #[test]
    fn all_expired_urls_include_reexport_hint() {
        let error = validate_downloaded_image_count(true, 2, 0, 2, 2).unwrap_err();

        assert!(error.contains("signed URLs may have expired"));
        assert!(error.contains("re-export"));
    }

    #[test]
    fn zero_image_and_labels_only_datasets_do_not_fail_download_validation() {
        assert!(validate_downloaded_image_count(true, 0, 0, 0, 0).is_ok());
        assert!(validate_downloaded_image_count(false, 2, 0, 0, 0).is_ok());
    }

    #[test]
    fn semantic_format_supported_allows_polygon_formats() {
        assert!(semantic_format_supported("yolo"));
        assert!(semantic_format_supported("coco"));
        assert!(semantic_format_supported("pascal_voc"));
        assert!(semantic_format_supported("voc"));
    }

    #[test]
    fn semantic_format_supported_rejects_non_polygon_formats() {
        assert!(!semantic_format_supported("yolo_darknet"));
        assert!(!semantic_format_supported("createml"));
        assert!(!semantic_format_supported("tfrecord"));
    }

    #[test]
    fn semantic_guard_flags_dataset_without_polygons() {
        // PNG-mask-origin export: semantic task, junk classes, no segments.
        let content = r#"{"type":"dataset","task":"semantic","name":"masks","class_names":{"0":"0","1":"1"}}
{"type":"image","file":"a.png","width":512,"height":512,"split":"train","annotations":{}}
{"type":"image","file":"b.png","width":512,"height":512,"split":"train","annotations":{"segments":[]}}"#;
        let data = parse_ndjson(content).unwrap();
        assert!(semantic_dataset_has_no_polygons(&data));
    }

    #[test]
    fn semantic_guard_allows_dataset_with_any_polygon() {
        // One empty image is fine as long as the dataset has polygons somewhere.
        let content = r#"{"type":"dataset","task":"semantic","name":"city","class_names":{"0":"road"}}
{"type":"image","file":"a.png","width":512,"height":512,"split":"train","annotations":{}}
{"type":"image","file":"b.png","width":512,"height":512,"split":"train","annotations":{"segments":[[0,0.1,0.1,0.2,0.1,0.2,0.2]]}}"#;
        let data = parse_ndjson(content).unwrap();
        assert!(!semantic_dataset_has_no_polygons(&data));
    }

    #[test]
    fn semantic_guard_ignores_non_semantic_tasks() {
        // Detection dataset with no polygons must not trip the semantic guard.
        let content = r#"{"type":"dataset","task":"detect","name":"d","class_names":{"0":"cat"}}
{"type":"image","file":"a.jpg","width":512,"height":512,"split":"train","annotations":{"bboxes":[[0,0.5,0.5,0.2,0.2]]}}"#;
        let data = parse_ndjson(content).unwrap();
        assert!(!semantic_dataset_has_no_polygons(&data));
    }

    #[test]
    fn pose_validation_writes_inferred_shape_into_metadata() {
        let content = r#"{"type":"dataset","task":"pose","name":"pose","class_names":{"0":"object"}}
{"type":"image","file":"pose.jpg","width":640,"height":480,"split":"train","annotations":{"pose":[[0,0.5,0.5,0.4,0.4,0.1,0.2,2,0.3,0.4,2]]}}"#;
        let mut data = parse_ndjson(content).unwrap();

        validate_pose_dataset(&mut data).unwrap();

        assert_eq!(data.metadata.kpt_shape, Some(vec![2, 3]));
    }

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
