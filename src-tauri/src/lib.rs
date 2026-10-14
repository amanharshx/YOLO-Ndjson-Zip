mod converter;
mod downloader;
mod parser;

use converter::{check_task_format, get_class_list, get_converter};
use downloader::{DownloadResult, Downloader, FailureSummary, ProgressEvent};
use parser::{normalize_split, output_stem, parse_ndjson, ImageEntry, NDJSONData, ParseError};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
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
    pub omitted_images: usize,
    pub renamed_images: usize,
    pub failure_summary: FailureSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvertErrorKind {
    ConversionFailed,
    DownloadFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConvertError {
    pub kind: ConvertErrorKind,
    pub message: String,
    pub failure_summary: Option<FailureSummary>,
}

impl ConvertError {
    fn download_failed(failure_summary: &FailureSummary) -> Self {
        Self {
            kind: ConvertErrorKind::DownloadFailed,
            message: "Images could not be downloaded.".to_string(),
            failure_summary: Some(failure_summary.clone()),
        }
    }
}

impl From<String> for ConvertError {
    fn from(message: String) -> Self {
        Self {
            kind: ConvertErrorKind::ConversionFailed,
            message,
            failure_summary: None,
        }
    }
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

/// Appends `suffix` to the output stem, so every suffix yields a new `output_name_key`.
fn file_name_with_suffix(file_name: &str, suffix: &str) -> String {
    let stem = output_stem(file_name);
    format!("{}__{}{}", stem, suffix, &file_name[stem.len()..])
}

fn next_unique_file_name(
    original_file: &str,
    hash_source: &str,
    used_keys: &mut HashSet<String>,
) -> String {
    let hash = short_stable_hash(hash_source);
    let mut suffix = hash.clone();
    let mut counter = 2usize;

    loop {
        let candidate = file_name_with_suffix(original_file, &suffix);
        if used_keys.insert(output_name_key(&candidate)) {
            return candidate;
        }
        suffix = format!("{}__{}", hash, counter);
        counter += 1;
    }
}

/// Rejects a `semantic` dataset with no polygon segments in any image, naming the
/// annotation keys it does carry. Per-image empty is still valid.
fn validate_semantic_polygons(data: &NDJSONData) -> Result<(), String> {
    if data.metadata.task != "semantic"
        || data
            .images
            .iter()
            .any(|img| !img.get_segment_annotations().is_empty())
    {
        return Ok(());
    }

    let keys: BTreeSet<&str> = data
        .images
        .iter()
        .filter_map(|img| img.annotations.as_ref()?.as_object())
        .flat_map(|annotations| annotations.keys().map(String::as_str))
        .collect();
    let keys = if keys.is_empty() {
        "none".to_string()
    } else {
        keys.into_iter().collect::<Vec<_>>().join(", ")
    };
    Err(format!(
        "Semantic dataset has no polygons under annotations.segments in any image. \
         Annotation keys found: {}.",
        keys
    ))
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

/// Key under which output names clash: label and XML files are named by stem, and
/// macOS and Windows extract ZIPs case-insensitively.
fn output_name_key(file_name: &str) -> String {
    output_stem(file_name).to_lowercase()
}

fn prepare_images_with_unique_output_names(images: &[ImageEntry]) -> Vec<ImageEntry> {
    let mut used_keys_by_split: HashMap<&str, HashSet<String>> = HashMap::new();
    let mut prepared_images = Vec::with_capacity(images.len());

    for image in images {
        let used_keys = used_keys_by_split
            .entry(normalize_split(&image.split))
            .or_default();

        let mut prepared = image.clone();
        if !used_keys.insert(output_name_key(&image.file)) {
            let hash_source = if image.url.is_empty() {
                image.file.as_str()
            } else {
                image.url.as_str()
            };
            prepared.output_file = Some(next_unique_file_name(&image.file, hash_source, used_keys));
        }
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

    // Keep classes used only by images about to be dropped.
    let class_list = get_class_list(data);
    if let Some(name) = class_list.last() {
        let max_id = class_list.len() - 1;
        data.metadata
            .class_names
            .insert(max_id.to_string(), name.clone());
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
    failure_summary: &FailureSummary,
) -> Result<(), ConvertError> {
    if !include_images || original_image_count == 0 || kept_image_count > 0 {
        return Ok(());
    }

    Err(ConvertError::download_failed(failure_summary))
}

#[tauri::command]
async fn convert_ndjson(
    file_path: String,
    format: String,
    output_path: String,
    include_images: bool,
    channel: Channel<ProgressEvent>,
) -> Result<ConvertResult, ConvertError> {
    let metadata = std::fs::metadata(&file_path)
        .map_err(|e| format!("Failed to inspect file '{}': {}", file_path, e))?;
    if !is_ndjson_size_allowed(metadata.len()) {
        return Err(format!(
            "NDJSON file is too large ({} bytes). Maximum allowed is {} bytes.",
            metadata.len(),
            MAX_NDJSON_BYTES
        )
        .into());
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
    check_task_format(&data.metadata.task, &format)?;
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

    validate_semantic_polygons(&data)?;

    // Download images if requested
    let download_result = if include_images {
        let downloader = Downloader::new(MAX_DOWNLOAD_CONCURRENCY)
            .map_err(|e| format!("Failed to init downloader: {}", e))?;
        downloader.download_all(&data.images, &channel).await
    } else {
        DownloadResult {
            files: std::collections::HashMap::new(),
            failure_summary: Default::default(),
        }
    };

    let omitted_images =
        filter_images_without_downloads(&mut data, &download_result.files, include_images);
    let kept_image_count = data.images.len();
    let renamed_images = data
        .images
        .iter()
        .filter(|image| image.output_file.is_some())
        .count();
    validate_downloaded_image_count(
        include_images,
        original_image_count,
        kept_image_count,
        &download_result.failure_summary,
    )?;
    let DownloadResult {
        files: downloaded_images,
        failure_summary,
    } = download_result;

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
        return Err(err.into());
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
        omitted_images,
        renamed_images,
        failure_summary,
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
        normalize_zip_path, prepare_images_with_unique_output_names, short_stable_hash,
        validate_downloaded_image_count, validate_pose_dataset, validate_semantic_polygons,
        ConvertErrorKind, MAX_NDJSON_BYTES,
    };
    use crate::converter::{get_class_list, get_converter};
    use crate::downloader::{FailureGroup, FailureKind, FailureSummary};
    use crate::parser::{image_entry_download_key, parse_ndjson};
    use std::collections::{HashMap, HashSet};

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
    fn filtering_keeps_classes_used_only_by_omitted_images() {
        let mut data = parse_ndjson(
            r#"{"type":"dataset","task":"detect","name":"partial"}
{"type":"image","file":"present.jpg","width":640,"height":480,"split":"train","url":"https://cdn.example/present.jpg","annotations":{"boxes":[[0,0.5,0.5,0.2,0.2]]}}
{"type":"image","file":"missing.jpg","width":640,"height":480,"split":"train","url":"","annotations":{"boxes":[[2,0.5,0.5,0.2,0.2]]}}"#,
        )
        .unwrap();
        let downloaded = HashMap::from([(
            image_entry_download_key(&data.images[0]),
            b"present-image".to_vec(),
        )]);

        filter_images_without_downloads(&mut data, &downloaded, true);

        assert_eq!(data.images.len(), 1);
        assert_eq!(get_class_list(&data), vec!["class0", "class1", "class2"]);
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
        let summary = failure_summary(FailureKind::MissingUrl, 2);
        let error = validate_downloaded_image_count(true, 2, 0, &summary).unwrap_err();

        assert_eq!(error.kind, ConvertErrorKind::DownloadFailed);
        assert_eq!(error.failure_summary, Some(summary));
        assert_eq!(error.message, "Images could not be downloaded.");
    }

    #[test]
    fn all_network_failures_return_structured_summary() {
        let summary = failure_summary(FailureKind::Connect, 2);
        let error = validate_downloaded_image_count(true, 2, 0, &summary).unwrap_err();

        assert_eq!(error.kind, ConvertErrorKind::DownloadFailed);
        assert_eq!(error.failure_summary, Some(summary));
    }

    #[test]
    fn all_expired_urls_return_structured_summary() {
        let summary = failure_summary(FailureKind::ExpiredUrl, 2);
        let error = validate_downloaded_image_count(true, 2, 0, &summary).unwrap_err();

        assert_eq!(error.kind, ConvertErrorKind::DownloadFailed);
        assert_eq!(error.failure_summary, Some(summary));
    }

    fn failure_summary(kind: FailureKind, count: usize) -> FailureSummary {
        FailureSummary {
            groups: vec![FailureGroup {
                kind,
                count,
                examples: vec!["example.jpg".to_string()],
                http_statuses: Vec::new(),
            }],
            expiry: None,
        }
    }

    #[test]
    fn zero_image_and_labels_only_datasets_do_not_fail_download_validation() {
        let summary = FailureSummary::default();
        assert!(validate_downloaded_image_count(true, 0, 0, &summary).is_ok());
        assert!(validate_downloaded_image_count(false, 2, 0, &summary).is_ok());
    }

    #[test]
    fn semantic_without_polygons_lists_annotation_keys_found() {
        let content = r#"{"type":"dataset","task":"semantic","name":"masks","class_names":{"0":"road"}}
{"type":"image","file":"a.png","width":512,"height":512,"split":"train","annotations":{"boxes":[[0,0.5,0.5,0.2,0.2]]}}
{"type":"image","file":"b.png","width":512,"height":512,"split":"train","annotations":{"segments":[],"masks":"b_mask.png"}}"#;
        let data = parse_ndjson(content).unwrap();

        assert_eq!(
            validate_semantic_polygons(&data).unwrap_err(),
            "Semantic dataset has no polygons under annotations.segments in any image. \
             Annotation keys found: boxes, masks, segments."
        );
    }

    #[test]
    fn semantic_without_any_annotations_reports_no_keys() {
        let content = r#"{"type":"dataset","task":"semantic","name":"empty","class_names":{"0":"road"}}
{"type":"image","file":"a.png","width":512,"height":512,"split":"train"}"#;
        let data = parse_ndjson(content).unwrap();

        assert_eq!(
            validate_semantic_polygons(&data).unwrap_err(),
            "Semantic dataset has no polygons under annotations.segments in any image. \
             Annotation keys found: none."
        );
    }

    #[test]
    fn semantic_guard_allows_dataset_with_any_polygon() {
        // One empty image is fine as long as the dataset has polygons somewhere.
        let content = r#"{"type":"dataset","task":"semantic","name":"city","class_names":{"0":"road"}}
{"type":"image","file":"a.png","width":512,"height":512,"split":"train","annotations":{}}
{"type":"image","file":"b.png","width":512,"height":512,"split":"train","annotations":{"segments":[[0,0.1,0.1,0.2,0.1,0.2,0.2]]}}"#;
        let data = parse_ndjson(content).unwrap();
        assert!(validate_semantic_polygons(&data).is_ok());
    }

    #[test]
    fn semantic_guard_ignores_non_semantic_tasks() {
        // Detection dataset with no polygons must not trip the semantic guard.
        let content = r#"{"type":"dataset","task":"detect","name":"d","class_names":{"0":"cat"}}
{"type":"image","file":"a.jpg","width":512,"height":512,"split":"train","annotations":{"bboxes":[[0,0.5,0.5,0.2,0.2]]}}"#;
        let data = parse_ndjson(content).unwrap();
        assert!(validate_semantic_polygons(&data).is_ok());
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

    fn prepared_names(content: &str) -> Vec<String> {
        let data = parse_ndjson(content).unwrap();
        prepare_images_with_unique_output_names(&data.images)
            .iter()
            .map(|image| image.effective_file_name().to_string())
            .collect()
    }

    #[test]
    fn names_differing_only_by_case_get_distinct_output_names() {
        let names = prepared_names(
            r#"{"type":"dataset","name":"case","class_names":{}}
{"type":"image","file":"A.jpg","width":640,"height":480,"split":"train","url":"https://cdn.example/A.jpg"}
{"type":"image","file":"a.jpg","width":640,"height":480,"split":"train","url":"https://cdn.example/a.jpg"}
{"type":"image","file":"b.JPG","width":640,"height":480,"split":"train","url":"https://cdn.example/b.JPG"}
{"type":"image","file":"B.png","width":640,"height":480,"split":"train","url":"https://cdn.example/B.png"}"#,
        );

        assert_eq!(names[0], "A.jpg");
        assert_eq!(names[2], "b.JPG");
        let names_lower: HashSet<String> = names.iter().map(|name| name.to_lowercase()).collect();
        let stems_lower: HashSet<&str> = names_lower
            .iter()
            .map(|name| name.rsplit_once('.').unwrap().0)
            .collect();
        assert_eq!((names_lower.len(), stems_lower.len()), (4, 4), "{names:?}");
    }

    #[test]
    fn same_names_in_different_splits_are_not_renamed() {
        let names = prepared_names(
            r#"{"type":"dataset","name":"splits","class_names":{}}
{"type":"image","file":"a.jpg","width":640,"height":480,"split":"train","url":"https://cdn.example/1.jpg"}
{"type":"image","file":"a.png","width":640,"height":480,"split":"val","url":"https://cdn.example/2.png"}
{"type":"image","file":"A.jpg","width":640,"height":480,"split":"test","url":"https://cdn.example/3.jpg"}"#,
        );

        assert_eq!(names, ["a.jpg", "a.png", "A.jpg"]);
    }

    #[test]
    fn val_and_valid_share_output_names() {
        let names = prepared_names(
            r#"{"type":"dataset","name":"splits","class_names":{}}
{"type":"image","file":"a.jpg","width":640,"height":480,"split":"val","url":"https://cdn.example/1.jpg"}
{"type":"image","file":"a.png","width":640,"height":480,"split":"valid","url":"https://cdn.example/2.png"}"#,
        );

        assert_eq!(names[0], "a.jpg");
        assert_eq!(
            names[1],
            file_name_with_suffix("a.png", &short_stable_hash("https://cdn.example/2.png"))
        );
    }

    #[test]
    fn output_names_are_deterministic() {
        let content = r#"{"type":"dataset","name":"repeat","class_names":{}}
{"type":"image","file":"a.jpg","width":640,"height":480,"split":"train","url":"https://cdn.example/1.jpg"}
{"type":"image","file":"A.png","width":640,"height":480,"split":"train","url":"https://cdn.example/2.png"}
{"type":"image","file":"a.jpeg","width":640,"height":480,"split":"train","url":"https://cdn.example/3.jpeg"}"#;

        let first = prepared_names(content);
        assert_eq!(first, prepared_names(content));
        assert_eq!(first[0], "a.jpg");
        assert_eq!(first[1], "A__8fc54561.png");
    }

    #[test]
    fn names_with_empty_stems_get_distinct_output_names() {
        let names = prepared_names(
            r#"{"type":"dataset","name":"dotfiles","class_names":{}}
{"type":"image","file":".jpg","width":640,"height":480,"split":"train","url":"https://cdn.example/1.jpg"}
{"type":"image","file":".png","width":640,"height":480,"split":"train","url":"https://cdn.example/2.png"}"#,
        );

        assert_eq!(names[0], ".jpg");
        assert_eq!(
            names[1],
            format!("__{}.png", short_stable_hash("https://cdn.example/2.png"))
        );
    }

    #[test]
    fn same_stem_images_get_their_own_annotation_files() {
        let content = r#"{"type":"dataset","task":"detect","name":"stems","class_names":{"0":"cat","1":"dog"}}
{"type":"image","file":"a.jpg","width":640,"height":480,"split":"train","url":"https://cdn.example/a.jpg","annotations":{"boxes":[[0,0.5,0.5,0.2,0.2]]}}
{"type":"image","file":"a.png","width":640,"height":480,"split":"train","url":"https://cdn.example/a.png","annotations":{"boxes":[[1,0.4,0.4,0.3,0.3]]}}"#;
        let mut data = parse_ndjson(content).unwrap();
        data.images = prepare_images_with_unique_output_names(&data.images);
        let renamed =
            file_name_with_suffix("a.png", &short_stable_hash("https://cdn.example/a.png"));
        assert_eq!(data.images[0].effective_file_name(), "a.jpg");
        assert_eq!(data.images[1].effective_file_name(), renamed);
        let renamed_stem = renamed.trim_end_matches(".png");

        let text = |files: &HashMap<String, Vec<u8>>, path: &str| {
            String::from_utf8(
                files
                    .get(path)
                    .unwrap_or_else(|| panic!("missing {path}"))
                    .clone(),
            )
            .unwrap()
        };
        for (format, label_dir) in [("yolo", "train/labels/"), ("yolo_darknet", "train/")] {
            let files = get_converter(format)
                .unwrap()
                .convert(&data, &HashMap::new());
            assert_eq!(
                text(&files, &format!("{label_dir}a.txt")),
                "0 0.500000 0.500000 0.200000 0.200000",
                "{format}"
            );
            assert_eq!(
                text(&files, &format!("{label_dir}{renamed_stem}.txt")),
                "1 0.400000 0.400000 0.300000 0.300000",
                "{format}"
            );
        }

        let files = get_converter("pascal_voc")
            .unwrap()
            .convert(&data, &HashMap::new());
        let first = text(&files, "train/a.xml");
        let second = text(&files, &format!("train/{renamed_stem}.xml"));
        assert!(first.contains("<filename>a.jpg</filename>") && first.contains("<name>cat</name>"));
        assert!(!first.contains("<name>dog</name>"));
        assert!(second.contains(&format!("<filename>{renamed}</filename>")));
        assert!(second.contains("<name>dog</name>") && !second.contains("<name>cat</name>"));
    }
}
