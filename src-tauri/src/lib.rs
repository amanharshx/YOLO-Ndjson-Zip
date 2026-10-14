mod converter;
mod downloader;
mod parser;

use converter::{check_task_format, get_class_list, get_converter, Converter};
use downloader::{DownloadRequest, DownloadResult, Downloader, FailureSummary, ProgressEvent};
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
/// One in this many train images moves to val when an export has none, as in Ultralytics.
const AUTO_VAL_SHARE: usize = 10;

/// Train images moved to val because no val images were left to export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutoValSplit {
    pub moved: usize,
    /// Train images before the move.
    pub train_images: usize,
}

#[derive(Debug, Serialize)]
pub struct ConvertResult {
    pub zip_path: String,
    pub omitted_images: usize,
    pub renamed_images: usize,
    /// Classify images skipped because they have no class ID.
    pub unlabeled_images: usize,
    pub auto_val_split: Option<AutoValSplit>,
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

/// FNV-1a 64-bit hash: stable across runs and platforms.
fn fnv1a_64(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn short_stable_hash(input: &str) -> String {
    // Truncated for compact deterministic filenames.
    format!("{:08x}", (fnv1a_64(input) & 0xffff_ffff) as u32)
}

/// Number of train images to move to val: `max(1, n / AUTO_VAL_SHARE)` when the converter
/// writes a `data.yaml` and `data` has train images but none in val, else 0. Ultralytics
/// training refuses a dataset YAML with no `val:` key.
fn val_images_to_move(data: &NDJSONData, converter: &dyn Converter) -> Result<usize, String> {
    if !converter.writes_data_yaml(&data.metadata.task) || !data.valid_images().is_empty() {
        return Ok(0);
    }
    match data.train_images().len() {
        // Datasets with no train split are exported as they are.
        0 => Ok(0),
        1 => Err("No val images and only 1 train image are left to export. \
                  Ultralytics training needs a val split: add val images, \
                  or at least 2 train images so one can be moved to val."
            .to_string()),
        n => Ok((n / AUTO_VAL_SHARE).max(1)),
    }
}

/// Moves train images to val when `val_images_to_move` asks for it, re-keying their
/// downloaded bytes. The moved images are the train images with the lowest FNV-1a hash
/// of their output name (name breaks ties), so the same input moves the same images
/// whatever the download order. Names stay unique: they were unique in train and val
/// was empty. An error names the `omitted_images` that failed to download.
fn move_train_images_to_val(
    data: &mut NDJSONData,
    converter: &dyn Converter,
    downloaded_images: &mut HashMap<String, Vec<u8>>,
    omitted_images: usize,
) -> Result<Option<AutoValSplit>, String> {
    let moved = val_images_to_move(data, converter).map_err(|message| match omitted_images {
        0 => message,
        1 => format!("{message} 1 image was skipped because it could not be downloaded."),
        n => format!("{message} {n} images were skipped because they could not be downloaded."),
    })?;
    if moved == 0 {
        return Ok(None);
    }

    let pairs_depth_maps = converter.pairs_depth_maps(&data.metadata.task);
    let mut train: Vec<&mut ImageEntry> = data
        .images
        .iter_mut()
        .filter(|image| image.split == "train")
        .collect();
    let train_images = train.len();
    train.sort_by_cached_key(|image| {
        let name = image.effective_file_name().to_string();
        (fnv1a_64(&name), name)
    });
    for image in train.into_iter().take(moved) {
        let train_downloads = image_downloads(image, pairs_depth_maps);
        image.split = "val".to_string();
        for (train, val) in train_downloads
            .into_iter()
            .zip(image_downloads(image, pairs_depth_maps))
        {
            if let Some(bytes) = downloaded_images.remove(&train.key) {
                downloaded_images.insert(val.key, bytes);
            }
        }
    }
    Ok(Some(AutoValSplit {
        moved,
        train_images,
    }))
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

/// Rejects a depth export Ultralytics would refuse for missing depth URLs, before download.
fn validate_depth_dataset(
    data: &NDJSONData,
    converter: &dyn Converter,
    include_images: bool,
) -> Result<(), String> {
    if !converter.pairs_depth_maps(&data.metadata.task) {
        return Ok(());
    }
    if !include_images {
        return Err("Depth exports need images: their depth maps are the labels.".to_string());
    }
    let mut missing = data
        .images
        .iter()
        .filter(|image| image.depth_url().is_none());
    let Some(first) = missing.next() else {
        return Ok(());
    };
    let affected = 1 + missing.count();
    let count = if affected > 1 {
        format!(" ({} image lines affected)", affected)
    } else {
        String::new()
    };
    Err(format!(
        "Depth image '{}' has no depth.url{}. Every depth image line needs one.",
        first.file, count
    ))
}

/// For converters that write pixel coordinates, keeps declared sizes and reads missing ones
/// from the downloaded image header, or rejects them when no images are exported.
fn resolve_image_sizes(
    data: &mut NDJSONData,
    converter: &dyn Converter,
    downloaded_images: &HashMap<String, Vec<u8>>,
    include_images: bool,
) -> Result<(), String> {
    if !converter.needs_image_sizes(&data.metadata.task) {
        return Ok(());
    }
    if !include_images {
        let mut sizeless = data.images.iter().filter(|image| image.size().is_none());
        return match sizeless.next() {
            None => Ok(()),
            Some(first) => Err(format!(
                "Width/height missing on {} of {} image lines (first: '{}'). \
                 Include images in the export so sizes can be read from them.",
                1 + sizeless.count(),
                data.images.len(),
                first.file
            )),
        };
    }
    for image in data
        .images
        .iter_mut()
        .filter(|image| image.size().is_none())
    {
        let (width, height) = downloaded_images
            .get(&parser::image_entry_download_key(image))
            .ok_or("it was not downloaded")
            .and_then(|bytes| {
                let size = imagesize::blob_size(bytes).map_err(|_| {
                    "the file is not a recognised image; add width/height to its NDJSON line"
                })?;
                match (i32::try_from(size.width), i32::try_from(size.height)) {
                    (Ok(width), Ok(height)) => Ok((width, height)),
                    _ => Err("size out of range"),
                }
            })
            .map_err(|reason| {
                format!(
                    "Can't read the pixel size of image '{}': {}",
                    image.file, reason
                )
            })?;
        image.width.get_or_insert(width);
        image.height.get_or_insert(height);
    }
    Ok(())
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

/// Drops the unlabeled images `converter` leaves out, then gives the rest unique output
/// names, so skipped images are never downloaded or counted as renamed. Returns how many
/// were dropped.
fn prepare_export_images(data: &mut NDJSONData, converter: &dyn Converter) -> usize {
    let original_image_count = data.images.len();
    if converter.skips_unlabeled_images(&data.metadata.task) {
        data.images
            .retain(|image| !image.get_classifications().is_empty());
    }
    let unlabeled_images = original_image_count - data.images.len();
    data.images = prepare_images_with_unique_output_names(&data.images);
    unlabeled_images
}

/// Files downloaded for one image: the image, then its depth map when `pairs_depth_maps`.
fn image_downloads(image: &ImageEntry, pairs_depth_maps: bool) -> Vec<DownloadRequest> {
    let mut downloads = vec![DownloadRequest::image(image)];
    if pairs_depth_maps {
        downloads.push(DownloadRequest::depth_map(image));
    }
    downloads
}

/// Every file `converter` needs downloaded for `data`, in one batch, so progress and the
/// failure summary cover depth maps too.
fn download_requests(data: &NDJSONData, converter: &dyn Converter) -> Vec<DownloadRequest> {
    let pairs_depth_maps = converter.pairs_depth_maps(&data.metadata.task);
    data.images
        .iter()
        .flat_map(|image| image_downloads(image, pairs_depth_maps))
        .collect()
}

/// Drops images any of whose files failed to download, so an image and its depth map are
/// kept or omitted together. Returns how many were dropped.
fn filter_images_without_downloads(
    data: &mut NDJSONData,
    converter: &dyn Converter,
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

    let pairs_depth_maps = converter.pairs_depth_maps(&data.metadata.task);
    let original_image_count = data.images.len();
    data.images.retain(|image| {
        image_downloads(image, pairs_depth_maps)
            .iter()
            .all(|download| downloaded_images.contains_key(&download.key))
    });
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
    let converter = get_converter(&format).ok_or_else(|| format!("Unknown format: {}", format))?;
    validate_pose_dataset(&mut data).map_err(|e| e.to_string())?;
    let unlabeled_images = prepare_export_images(&mut data, converter.as_ref());
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
    validate_depth_dataset(&data, converter.as_ref(), include_images)?;
    val_images_to_move(&data, converter.as_ref())?;

    // Download images if requested
    let download_result = if include_images {
        let downloader = Downloader::new(MAX_DOWNLOAD_CONCURRENCY)
            .map_err(|e| format!("Failed to init downloader: {}", e))?;
        downloader
            .download_all(&download_requests(&data, converter.as_ref()), &channel)
            .await
    } else {
        DownloadResult {
            files: std::collections::HashMap::new(),
            failure_summary: Default::default(),
        }
    };

    let omitted_images = filter_images_without_downloads(
        &mut data,
        converter.as_ref(),
        &download_result.files,
        include_images,
    );
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
    resolve_image_sizes(
        &mut data,
        converter.as_ref(),
        &download_result.files,
        include_images,
    )?;
    let DownloadResult {
        files: mut downloaded_images,
        failure_summary,
    } = download_result;
    // After download filtering, so the val split matches the images in the ZIP.
    let auto_val_split = move_train_images_to_val(
        &mut data,
        converter.as_ref(),
        &mut downloaded_images,
        omitted_images,
    )?;

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
        unlabeled_images,
        auto_val_split,
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
        download_requests, file_name_with_suffix, filter_images_without_downloads,
        is_ndjson_size_allowed, move_train_images_to_val, normalize_zip_path,
        prepare_export_images, prepare_images_with_unique_output_names, resolve_image_sizes,
        short_stable_hash, val_images_to_move, validate_depth_dataset,
        validate_downloaded_image_count, validate_pose_dataset, validate_semantic_polygons,
        ConvertErrorKind, MAX_NDJSON_BYTES,
    };
    use crate::converter::{get_class_list, get_converter};
    use crate::downloader::{DownloadRequest, FailureGroup, FailureKind, FailureSummary};
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

        let omitted = filter_images_without_downloads(
            &mut data,
            get_converter("yolo").unwrap().as_ref(),
            &downloaded,
            true,
        );

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

        filter_images_without_downloads(
            &mut data,
            get_converter("yolo").unwrap().as_ref(),
            &downloaded,
            true,
        );

        assert_eq!(data.images.len(), 1);
        assert_eq!(get_class_list(&data), vec!["class0", "class1", "class2"]);
    }

    #[test]
    fn filtering_is_disabled_for_labels_only_conversion() {
        let mut data = partial_download_data();

        let omitted = filter_images_without_downloads(
            &mut data,
            get_converter("yolo").unwrap().as_ref(),
            &HashMap::new(),
            false,
        );

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
        filter_images_without_downloads(
            &mut data,
            get_converter("yolo").unwrap().as_ref(),
            &downloaded,
            true,
        );

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

    /// Detect dataset whose image lines declare no `width`/`height`.
    fn sizeless_detect_data() -> crate::parser::NDJSONData {
        parse_ndjson(
            r#"{"type":"dataset","task":"detect","name":"sizeless","class_names":{"0":"cat"}}
{"file":"a.jpg","split":"train","url":"https://cdn.example/a.jpg","annotations":{"boxes":[[0,0.5,0.5,0.5,0.5]]}}
{"file":"b.png","split":"val","url":"https://cdn.example/b.png","annotations":{"boxes":[[0,0.25,0.25,0.5,0.5]]}}"#,
        )
        .unwrap()
    }

    fn downloads_for(
        data: &crate::parser::NDJSONData,
        bytes: [Vec<u8>; 2],
    ) -> HashMap<String, Vec<u8>> {
        data.images
            .iter()
            .map(image_entry_download_key)
            .zip(bytes)
            .collect()
    }

    #[test]
    fn yolo_formats_export_without_sizes_with_or_without_images() {
        for (format, label) in [
            ("yolo", "train/labels/a.txt"),
            ("yolo_darknet", "train/a.txt"),
        ] {
            for include_images in [false, true] {
                let mut data = sizeless_detect_data();
                let downloaded = if include_images {
                    downloads_for(&data, [b"not-an-image".to_vec(), b"x".to_vec()])
                } else {
                    HashMap::new()
                };

                let converter = get_converter(format).unwrap();
                resolve_image_sizes(&mut data, converter.as_ref(), &downloaded, include_images)
                    .unwrap();
                let files = converter.convert(&data, &downloaded);

                assert_eq!(
                    files.get(label).map(Vec::as_slice),
                    Some(&b"0 0.500000 0.500000 0.500000 0.500000"[..]),
                    "{format}, include_images={include_images}"
                );
            }
        }
    }

    #[test]
    fn labels_only_pixel_formats_reject_missing_sizes_naming_count_and_first_file() {
        for format in ["coco", "pascal_voc"] {
            let mut data = sizeless_detect_data();

            let converter = get_converter(format).unwrap();
            let error = resolve_image_sizes(&mut data, converter.as_ref(), &HashMap::new(), false)
                .unwrap_err();

            assert_eq!(
                error,
                "Width/height missing on 2 of 2 image lines (first: 'a.jpg'). \
                 Include images in the export so sizes can be read from them.",
                "{format}"
            );
        }
    }

    fn crc32(bytes: &[u8]) -> u32 {
        !bytes.iter().fold(!0u32, |crc, &byte| {
            (0..8).fold(crc ^ u32::from(byte), |crc, _| {
                (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg())
            })
        })
    }

    /// A complete 8-bit grayscale PNG (stored deflate block, so no compressor is needed).
    fn png_bytes(width: u16, height: u16) -> Vec<u8> {
        let chunk = |kind: &[u8], body: &[u8]| {
            let typed = [kind, body].concat();
            [
                &(body.len() as u32).to_be_bytes()[..],
                &typed,
                &crc32(&typed).to_be_bytes(),
            ]
            .concat()
        };
        let row = [&[0u8][..], &vec![0x80; usize::from(width)]].concat();
        let pixels = row.repeat(usize::from(height));
        let (a, b) = pixels.iter().fold((1u32, 0u32), |(a, b), &byte| {
            let a = (a + u32::from(byte)) % 65521;
            (a, (b + a) % 65521)
        });
        let len = pixels.len() as u16;
        let zlib = [
            &[0x78, 0x01, 0x01][..],
            &len.to_le_bytes(),
            &(!len).to_le_bytes(),
            &pixels,
            &((b << 16) | a).to_be_bytes(),
        ]
        .concat();
        let header = [
            &u32::from(width).to_be_bytes()[..],
            &u32::from(height).to_be_bytes(),
            &[8, 0, 0, 0, 0],
        ]
        .concat();
        [
            &b"\x89PNG\r\n\x1a\n"[..],
            &chunk(b"IHDR", &header),
            &chunk(b"IDAT", &zlib),
            &chunk(b"IEND", &[]),
        ]
        .concat()
    }

    /// A complete baseline grayscale JPEG of flat gray: every 8x8 block codes DC 0 then EOB.
    fn jpeg_bytes(width: u16, height: u16) -> Vec<u8> {
        let one_symbol_table = |class: u8| {
            let mut segment = vec![0xFF, 0xC4, 0, 20, class, 1];
            segment.extend([0; 15]);
            segment.push(0);
            segment
        };
        let blocks = usize::from(width.div_ceil(8)) * usize::from(height.div_ceil(8));
        let bits = 2 * blocks;
        let mut scan = vec![0u8; bits.div_ceil(8)];
        if bits % 8 != 0 {
            *scan.last_mut().unwrap() = 0xFF >> (bits % 8); // pad with 1 bits
        }
        [
            &[0xFF, 0xD8, 0xFF, 0xDB, 0, 67, 0][..],
            &[1; 64],
            &[0xFF, 0xC0, 0, 11, 8],
            &height.to_be_bytes(),
            &width.to_be_bytes(),
            &[1, 1, 0x11, 0],
            &one_symbol_table(0x00),
            &one_symbol_table(0x10),
            &[0xFF, 0xDA, 0, 8, 1, 1, 0x00, 0, 63, 0],
            &scan,
            &[0xFF, 0xD9],
        ]
        .concat()
    }

    fn coco_json(files: &HashMap<String, Vec<u8>>, split: &str) -> serde_json::Value {
        serde_json::from_slice(&files[&format!("{split}/_annotations.coco.json")]).unwrap()
    }

    #[test]
    fn coco_with_images_reads_missing_sizes_from_jpeg_and_png_headers() {
        let mut data = sizeless_detect_data();
        let downloaded = downloads_for(&data, [jpeg_bytes(100, 60), png_bytes(40, 30)]);

        let converter = get_converter("coco").unwrap();
        resolve_image_sizes(&mut data, converter.as_ref(), &downloaded, true).unwrap();
        let files = converter.convert(&data, &downloaded);

        let train = coco_json(&files, "train");
        assert_eq!(
            (&train["images"][0]["width"], &train["images"][0]["height"]),
            (&serde_json::json!(100), &serde_json::json!(60))
        );
        assert_eq!(
            train["annotations"][0]["bbox"],
            serde_json::json!([25.0, 15.0, 50.0, 30.0])
        );
        let valid = coco_json(&files, "valid");
        assert_eq!(
            (&valid["images"][0]["width"], &valid["images"][0]["height"]),
            (&serde_json::json!(40), &serde_json::json!(30))
        );
        assert_eq!(
            valid["annotations"][0]["bbox"],
            serde_json::json!([0.0, 0.0, 20.0, 15.0])
        );
    }

    #[test]
    fn voc_with_images_reads_missing_sizes_from_jpeg_and_png_headers() {
        let mut data = sizeless_detect_data();
        let downloaded = downloads_for(&data, [jpeg_bytes(100, 60), png_bytes(40, 30)]);

        let converter = get_converter("pascal_voc").unwrap();
        resolve_image_sizes(&mut data, converter.as_ref(), &downloaded, true).unwrap();
        let files = converter.convert(&data, &downloaded);

        let xml = |path: &str| String::from_utf8(files[path].clone()).unwrap();
        let train = xml("train/a.xml");
        assert!(train.contains("<width>100</width>") && train.contains("<height>60</height>"));
        assert!(train.contains(
            "<xmin>25</xmin>\n      <ymin>15</ymin>\n      <xmax>75</xmax>\n      <ymax>45</ymax>"
        ));
        let valid = xml("valid/b.xml");
        assert!(valid.contains("<width>40</width>") && valid.contains("<height>30</height>"));
        assert!(valid.contains(
            "<xmin>0</xmin>\n      <ymin>0</ymin>\n      <xmax>20</xmax>\n      <ymax>15</ymax>"
        ));
    }

    #[test]
    fn unreadable_image_without_declared_size_fails_pixel_formats_naming_the_file() {
        for format in ["coco", "pascal_voc"] {
            let mut data = sizeless_detect_data();
            let downloaded = downloads_for(&data, [jpeg_bytes(100, 60), b"<html>".to_vec()]);

            let converter = get_converter(format).unwrap();
            let error =
                resolve_image_sizes(&mut data, converter.as_ref(), &downloaded, true).unwrap_err();

            assert_eq!(
                error,
                "Can't read the pixel size of image 'b.png': the file is not a recognised \
                 image; add width/height to its NDJSON line",
                "{format}"
            );
        }
    }

    #[test]
    fn declared_sizes_are_used_unchanged_without_reading_images() {
        let mut data = parse_ndjson(
            r#"{"type":"dataset","task":"detect","name":"sized","class_names":{"0":"cat"}}
{"file":"a.jpg","width":200,"height":100,"split":"train","url":"https://cdn.example/a.jpg","annotations":{"boxes":[[0,0.5,0.5,0.5,0.5]]}}"#,
        )
        .unwrap();
        let downloaded = HashMap::from([(
            image_entry_download_key(&data.images[0]),
            jpeg_bytes(100, 60),
        )]);

        let converter = get_converter("coco").unwrap();
        resolve_image_sizes(&mut data, converter.as_ref(), &downloaded, true).unwrap();
        let files = converter.convert(&data, &downloaded);

        let train = coco_json(&files, "train");
        assert_eq!(
            (&train["images"][0]["width"], &train["images"][0]["height"]),
            (&serde_json::json!(200), &serde_json::json!(100))
        );
        assert_eq!(
            train["annotations"][0]["bbox"],
            serde_json::json!([50.0, 25.0, 100.0, 50.0])
        );
        let voc = get_converter("pascal_voc").unwrap();
        assert!(resolve_image_sizes(&mut data, voc.as_ref(), &HashMap::new(), false).is_ok());
    }

    /// `task` dataset of `train` images `t0.jpg..`, then `val` images `v0.jpg..`, each with a URL.
    fn split_data(task: &str, train: usize, val: usize) -> crate::parser::NDJSONData {
        let annotations = if task == "classify" {
            r#"{"classification":[0]}"#
        } else {
            r#"{"boxes":[[0,0.5,0.5,0.2,0.2]]}"#
        };
        let lines = (0..train)
            .map(|i| ("train", format!("t{i}.jpg")))
            .chain((0..val).map(|i| ("val", format!("v{i}.jpg"))))
            .map(|(split, file)| {
                format!(
                    r#"{{"file":"{file}","width":64,"height":64,"split":"{split}","url":"https://cdn.example/{file}","annotations":{annotations}}}"#
                )
            });
        let header = format!(
            r#"{{"type":"dataset","task":"{task}","name":"split","class_names":{{"0":"cat"}}}}"#
        );
        parse_ndjson(
            &std::iter::once(header)
                .chain(lines)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap()
    }

    fn all_downloaded(data: &crate::parser::NDJSONData) -> HashMap<String, Vec<u8>> {
        data.images
            .iter()
            .map(|image| {
                (
                    image_entry_download_key(image),
                    image.file.clone().into_bytes(),
                )
            })
            .collect()
    }

    fn paths_under(files: &HashMap<String, Vec<u8>>, prefix: &str) -> Vec<String> {
        let mut paths: Vec<String> = files
            .keys()
            .filter(|path| path.starts_with(prefix))
            .cloned()
            .collect();
        paths.sort();
        paths
    }

    #[test]
    fn yolo_export_without_val_moves_a_tenth_of_train_to_val() {
        let mut data = split_data("detect", 10, 0);
        let mut downloaded = all_downloaded(&data);
        let converter = get_converter("yolo").unwrap();

        let moved = move_train_images_to_val(&mut data, converter.as_ref(), &mut downloaded, 0)
            .unwrap()
            .unwrap();
        let files = converter.convert(&data, &downloaded);

        assert_eq!((moved.moved, moved.train_images), (1, 10));
        assert_eq!(paths_under(&files, "train/images/").len(), 9);
        assert_eq!(paths_under(&files, "train/labels/").len(), 9);
        let val_images = paths_under(&files, "valid/images/");
        assert_eq!(val_images.len(), 1);
        let file = val_images[0].trim_start_matches("valid/images/");
        assert_eq!(files[&val_images[0]], file.as_bytes());
        assert_eq!(
            files[&format!("valid/labels/{}.txt", file.trim_end_matches(".jpg"))],
            b"0 0.500000 0.500000 0.200000 0.200000"
        );
        let yaml = std::str::from_utf8(&files["data.yaml"]).unwrap();
        assert!(
            yaml.contains("train: train/images\nval: valid/images\nnc: 1\n"),
            "{yaml}"
        );
        assert!(!yaml.contains("# WARNING:"), "{yaml}");
    }

    fn val_files(data: &crate::parser::NDJSONData) -> Vec<String> {
        let mut files: Vec<String> = data
            .valid_images()
            .iter()
            .map(|image| image.file.clone())
            .collect();
        files.sort();
        files
    }

    #[test]
    fn auto_val_split_moves_floor_of_a_tenth() {
        let mut data = split_data("segment", 25, 0);
        let converter = get_converter("yolo").unwrap();

        let moved = move_train_images_to_val(&mut data, converter.as_ref(), &mut HashMap::new(), 0)
            .unwrap()
            .unwrap();

        assert_eq!((moved.moved, moved.train_images), (2, 25));
        assert_eq!((data.train_images().len(), val_files(&data).len()), (23, 2));
    }

    #[test]
    fn auto_val_split_picks_the_same_images_on_every_run() {
        let converter = get_converter("yolo").unwrap();
        let picks = |reverse: bool| {
            let mut data = split_data("detect", 25, 0);
            if reverse {
                data.images.reverse();
            }
            move_train_images_to_val(&mut data, converter.as_ref(), &mut HashMap::new(), 0)
                .unwrap();
            val_files(&data)
        };

        let first = picks(false);
        assert_eq!(first, picks(false));
        // With unique names, as here, reordering the lines changes nothing either.
        assert_eq!(first, picks(true));
        // Lowest FNV-1a 64 of t0.jpg..t24.jpg, computed independently.
        assert_eq!(first, ["t1.jpg", "t20.jpg"]);
    }

    #[test]
    fn single_train_image_without_val_fails_before_download() {
        let mut data = split_data("detect", 1, 0);
        let converter = get_converter("yolo").unwrap();
        let expected = "No val images and only 1 train image are left to export. \
                        Ultralytics training needs a val split: add val images, \
                        or at least 2 train images so one can be moved to val.";

        assert_eq!(
            val_images_to_move(&data, converter.as_ref()).unwrap_err(),
            expected
        );
        assert_eq!(
            move_train_images_to_val(&mut data, converter.as_ref(), &mut HashMap::new(), 0)
                .unwrap_err(),
            expected
        );
    }

    #[test]
    fn single_train_image_left_after_failed_downloads_names_the_skipped_count() {
        for (train, val, omitted_text) in [
            (
                2,
                0,
                "1 image was skipped because it could not be downloaded.",
            ),
            (
                2,
                2,
                "3 images were skipped because they could not be downloaded.",
            ),
        ] {
            let mut data = split_data("detect", train, val);
            let converter = get_converter("yolo").unwrap();
            // Passes before download; the problem only appears once downloads fail.
            assert!(val_images_to_move(&data, converter.as_ref()).is_ok());
            let mut downloaded = all_downloaded(&data);
            downloaded.retain(|key, _| key.ends_with(":t0.jpg"));
            let omitted = filter_images_without_downloads(
                &mut data,
                get_converter("yolo").unwrap().as_ref(),
                &downloaded,
                true,
            );

            let error =
                move_train_images_to_val(&mut data, converter.as_ref(), &mut downloaded, omitted)
                    .unwrap_err();

            assert_eq!(
                error,
                format!(
                    "No val images and only 1 train image are left to export. \
                     Ultralytics training needs a val split: add val images, \
                     or at least 2 train images so one can be moved to val. {omitted_text}"
                )
            );
        }
    }

    #[test]
    fn dataset_with_val_images_is_not_split() {
        let mut data = split_data("detect", 10, 1);
        let converter = get_converter("yolo").unwrap();

        let moved = move_train_images_to_val(&mut data, converter.as_ref(), &mut HashMap::new(), 0)
            .unwrap();

        assert_eq!(moved, None);
        assert_eq!(val_files(&data), ["v0.jpg"]);
        assert_eq!(data.train_images().len(), 10);
    }

    #[test]
    fn val_split_whose_downloads_all_failed_is_replaced_from_train() {
        let mut data = split_data("detect", 10, 2);
        let mut downloaded = all_downloaded(&data);
        downloaded.retain(|key, _| !key.contains(":valid:"));
        filter_images_without_downloads(
            &mut data,
            get_converter("yolo").unwrap().as_ref(),
            &downloaded,
            true,
        );
        let converter = get_converter("yolo").unwrap();

        let moved = move_train_images_to_val(&mut data, converter.as_ref(), &mut downloaded, 0)
            .unwrap()
            .unwrap();

        assert_eq!((moved.moved, moved.train_images), (1, 10));
        assert_eq!(val_files(&data).len(), 1);
    }

    #[test]
    fn other_formats_and_classify_are_never_split() {
        for (task, format) in [
            ("detect", "yolo_darknet"),
            ("detect", "coco"),
            ("detect", "pascal_voc"),
            ("classify", "yolo"),
            ("classify", "yolo_darknet"),
            ("classify", "pascal_voc"),
        ] {
            let converter = get_converter(format).unwrap();
            let mut data = split_data(task, 10, 0);
            let mut downloaded = all_downloaded(&data);
            let before = all_downloaded(&data);

            let moved = move_train_images_to_val(&mut data, converter.as_ref(), &mut downloaded, 0)
                .unwrap();
            assert_eq!(moved, None, "{task} + {format}");
            assert!(val_files(&data).is_empty(), "{task} + {format}");
            assert_eq!(downloaded, before, "{task} + {format}");

            let mut single = split_data(task, 1, 0);
            assert_eq!(val_images_to_move(&single, converter.as_ref()), Ok(0));
            assert_eq!(
                move_train_images_to_val(&mut single, converter.as_ref(), &mut HashMap::new(), 0),
                Ok(None),
                "{task} + {format}"
            );
        }
    }

    fn pets(image_lines: &[&str]) -> crate::parser::NDJSONData {
        let header =
            r#"{"type":"dataset","task":"classify","name":"pets","class_names":{"0":"cat"}}"#;
        parse_ndjson(&[&[header], image_lines].concat().join("\n")).unwrap()
    }

    #[test]
    fn unlabeled_classify_images_are_skipped_and_counted() {
        for unlabeled in [
            r#"{"file":"b.jpg","split":"train","url":"https://cdn.example/b.jpg"}"#,
            r#"{"file":"b.jpg","split":"train","url":"https://cdn.example/b.jpg","annotations":{"classification":[]}}"#,
        ] {
            for format in ["yolo", "yolo_darknet", "pascal_voc"] {
                let mut data = pets(&[
                    r#"{"file":"a.jpg","split":"train","url":"https://cdn.example/a.jpg","annotations":{"classification":[0]}}"#,
                    unlabeled,
                ]);
                let downloaded = all_downloaded(&data);
                let converter = get_converter(format).unwrap();

                assert_eq!(
                    prepare_export_images(&mut data, converter.as_ref()),
                    1,
                    "{format}: {unlabeled}"
                );

                let files = converter.convert(&data, &downloaded);
                assert_eq!(
                    paths_under(&files, "train/"),
                    ["train/cat/a.jpg"],
                    "{format}"
                );
            }
        }
    }

    #[test]
    fn skipped_unlabeled_image_is_not_counted_as_renamed() {
        let labeled = r#"{"file":"a.jpg","split":"train","url":"https://cdn.example/a.jpg","annotations":{"classification":[0]}}"#;
        let unlabeled = r#"{"file":"a.png","split":"train","url":"https://cdn.example/a.png"}"#;
        let converter = get_converter("yolo").unwrap();
        for lines in [[labeled, unlabeled], [unlabeled, labeled]] {
            let mut data = pets(&lines);

            prepare_export_images(&mut data, converter.as_ref());

            assert_eq!(data.images.len(), 1);
            assert_eq!(data.images[0].file, "a.jpg");
            // `renamed_images` counts the images with an `output_file`.
            assert_eq!(data.images[0].output_file, None);
        }
    }

    #[test]
    fn unlabeled_images_are_kept_by_other_tasks_and_createml() {
        let mut detect = parse_ndjson(
            r#"{"type":"dataset","task":"detect","name":"background"}
{"file":"empty.jpg","width":64,"height":64,"split":"train"}"#,
        )
        .unwrap();
        let yolo = get_converter("yolo").unwrap();
        assert_eq!(prepare_export_images(&mut detect, yolo.as_ref()), 0);
        assert_eq!(detect.images.len(), 1);

        let mut classify =
            pets(&[r#"{"file":"b.jpg","split":"train","url":"https://cdn.example/b.jpg"}"#]);
        let downloaded = all_downloaded(&classify);
        let createml = get_converter("createml").unwrap();
        assert_eq!(prepare_export_images(&mut classify, createml.as_ref()), 0);
        let files = createml.convert(&classify, &downloaded);
        assert!(files.contains_key("train/b.jpg"));
    }

    #[test]
    fn classify_exports_never_need_sizes() {
        for format in ["pascal_voc", "createml"] {
            let mut data = parse_ndjson(
                r#"{"type":"dataset","task":"classify","name":"pets","class_names":{"0":"cat"}}
{"file":"a.jpg","split":"train","annotations":{"classification":[0]}}"#,
            )
            .unwrap();
            let converter = get_converter(format).unwrap();

            resolve_image_sizes(&mut data, converter.as_ref(), &HashMap::new(), false).unwrap();

            converter.convert(&data, &HashMap::new());
        }
    }

    /// Depth dataset with `dataset_fields` appended to its record, and `train` and `val`
    /// images `a.jpg` and `b.jpg` with depth maps, as in the Ultralytics NDJSON spec.
    fn depth_data(dataset_fields: &str) -> crate::parser::NDJSONData {
        parse_ndjson(&format!(
            r#"{{"type":"dataset","task":"depth","name":"rooms","class_names":{{}}{dataset_fields}}}
{{"type":"image","file":"a.jpg","split":"train","url":"https://cdn.example/a.jpg","depth":{{"url":"https://cdn.example/a_depth.png"}}}}
{{"type":"image","file":"b.jpg","split":"val","url":"https://cdn.example/b.jpg","depth":{{"url":"https://cdn.example/b_depth.png"}}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn image_without_depth_url_fails_before_download_naming_the_file() {
        let yolo = get_converter("yolo").unwrap();
        for depth in [
            "",
            r#","depth":null"#,
            r#","depth":"https://cdn.example/b_depth.png""#,
            r#","depth":{}"#,
            r#","depth":{"url":""}"#,
            r#","depth":{"url":5}"#,
        ] {
            let mut data = depth_data("");
            let line = format!(
                r#"{{"file":"c.jpg","split":"train","url":"https://cdn.example/c.jpg"{depth}}}"#
            );
            data.images.push(serde_json::from_str(&line).unwrap());

            assert_eq!(
                validate_depth_dataset(&data, yolo.as_ref(), true).unwrap_err(),
                "Depth image 'c.jpg' has no depth.url. Every depth image line needs one.",
                "{depth}"
            );
        }
    }

    #[test]
    fn missing_depth_urls_are_counted_when_several_lines_lack_one() {
        let mut data = depth_data("");
        for image in &mut data.images {
            image.depth = None;
        }

        let yolo = get_converter("yolo").unwrap();
        assert_eq!(
            validate_depth_dataset(&data, yolo.as_ref(), true).unwrap_err(),
            "Depth image 'a.jpg' has no depth.url (2 image lines affected). \
             Every depth image line needs one."
        );
    }

    #[test]
    fn labels_only_depth_export_is_rejected() {
        let yolo = get_converter("yolo").unwrap();

        assert_eq!(
            validate_depth_dataset(&depth_data(""), yolo.as_ref(), false).unwrap_err(),
            "Depth exports need images: their depth maps are the labels."
        );
    }

    /// Bytes for every file YOLO downloads for `data`, each spelling its progress label.
    fn depth_downloads(data: &crate::parser::NDJSONData) -> HashMap<String, Vec<u8>> {
        download_requests(data, get_converter("yolo").unwrap().as_ref())
            .into_iter()
            .map(|request| (request.key, request.label.into_bytes()))
            .collect()
    }

    fn sorted_paths(files: &HashMap<String, Vec<u8>>) -> Vec<&str> {
        let mut paths: Vec<&str> = files.keys().map(String::as_str).collect();
        paths.sort();
        paths
    }

    #[test]
    fn depth_export_pairs_each_image_with_its_depth_map_and_declares_the_task() {
        let data = depth_data("");

        let files = get_converter("yolo")
            .unwrap()
            .convert(&data, &depth_downloads(&data));

        assert_eq!(
            sorted_paths(&files),
            [
                "classes.txt",
                "data.yaml",
                "train/depth/a.png",
                "train/images/a.jpg",
                "valid/depth/b.png",
                "valid/images/b.jpg",
            ]
        );
        assert_eq!(files["train/images/a.jpg"], b"train/a.jpg");
        assert_eq!(files["train/depth/a.png"], b"train/depth/a.png");
        assert_eq!(files["valid/depth/b.png"], b"valid/depth/b.png");
        assert_eq!(files["classes.txt"], b"depth");
        let yaml = std::str::from_utf8(&files["data.yaml"]).unwrap();
        assert!(
            yaml.ends_with(
                "train: train/images\nval: valid/images\nnc: 1\nnames:\n  0: \"depth\"\n\
                 task: depth\ndepth_scale: 1000\n"
            ),
            "{yaml}"
        );
    }

    #[test]
    fn declared_depth_scale_is_written_unchanged() {
        for (value, written) in [("256", "256"), ("0.5", "0.5"), ("1e-7", "0.0000001")] {
            let data = depth_data(&format!(r#","depth_scale":{value}"#));

            let files = get_converter("yolo")
                .unwrap()
                .convert(&data, &depth_downloads(&data));

            let yaml = std::str::from_utf8(&files["data.yaml"]).unwrap();
            assert!(
                yaml.ends_with(&format!("depth_scale: {written}\n")),
                "{value}: {yaml}"
            );
        }
    }

    #[test]
    fn image_or_depth_map_that_failed_to_download_omits_the_pair() {
        let yolo = get_converter("yolo").unwrap();
        let first = &depth_data("").images[0];
        for failed in [
            DownloadRequest::image(first).key,
            DownloadRequest::depth_map(first).key,
        ] {
            let mut data = depth_data("");
            let mut downloaded = depth_downloads(&data);
            downloaded.remove(&failed);

            let omitted =
                filter_images_without_downloads(&mut data, yolo.as_ref(), &downloaded, true);
            let files = yolo.convert(&data, &downloaded);

            assert_eq!(omitted, 1, "{failed}");
            assert_eq!(
                sorted_paths(&files),
                [
                    "classes.txt",
                    "data.yaml",
                    "valid/depth/b.png",
                    "valid/images/b.jpg"
                ],
                "{failed}"
            );
        }
    }

    #[test]
    fn auto_val_split_moves_each_depth_map_with_its_image() {
        let lines: Vec<String> = (0..10)
            .map(|i| {
                format!(
                    r#"{{"file":"t{i}.jpg","split":"train","url":"https://cdn.example/t{i}.jpg","depth":{{"url":"https://cdn.example/t{i}.png"}}}}"#
                )
            })
            .collect();
        let header = r#"{"type":"dataset","task":"depth","name":"rooms"}"#.to_string();
        let mut data = parse_ndjson(&[vec![header], lines].concat().join("\n")).unwrap();
        let mut downloaded = depth_downloads(&data);
        let yolo = get_converter("yolo").unwrap();

        move_train_images_to_val(&mut data, yolo.as_ref(), &mut downloaded, 0).unwrap();
        let files = yolo.convert(&data, &downloaded);

        let val = paths_under(&files, "valid/");
        assert_eq!(val.len(), 2, "{val:?}");
        let stem = val[1]
            .strip_prefix("valid/images/")
            .and_then(|name| name.strip_suffix(".jpg"))
            .unwrap();
        assert_eq!(val[0], format!("valid/depth/{stem}.png"));
        assert_eq!(files[&val[0]], format!("train/depth/{stem}.png").as_bytes());
        assert_eq!(paths_under(&files, "train/depth/").len(), 9);
    }

    #[test]
    fn same_stem_depth_images_get_their_own_depth_maps() {
        let mut data = depth_data("");
        data.images[1] = serde_json::from_str(
            r#"{"file":"a.png","split":"train","url":"https://cdn.example/a.png","depth":{"url":"https://cdn.example/a2_depth.png"}}"#,
        )
        .unwrap();
        let yolo = get_converter("yolo").unwrap();
        prepare_export_images(&mut data, yolo.as_ref());

        let files = yolo.convert(&data, &depth_downloads(&data));

        let renamed = data.images[1].effective_file_name();
        let renamed_depth = format!("train/depth/{}.png", renamed.trim_end_matches(".png"));
        assert_eq!(
            sorted_paths(&files)[2..],
            [
                "train/depth/a.png".to_string(),
                renamed_depth.clone(),
                "train/images/a.jpg".to_string(),
                format!("train/images/{renamed}"),
            ]
        );
        assert_eq!(files["train/depth/a.png"], b"train/depth/a.png");
        assert_eq!(files[&renamed_depth], renamed_depth.as_bytes());
    }

    #[test]
    fn nested_images_folder_gets_its_depth_map_where_ultralytics_looks() {
        let mut data = depth_data("");
        data.images[0].file = r"scans\images/a.jpg".to_string();

        let files = get_converter("yolo")
            .unwrap()
            .convert(&data, &depth_downloads(&data));

        // Ultralytics swaps the last `images` folder of the image path for `depth`.
        let depth = "train/images/scans/depth/a.png";
        let zip_paths: Vec<String> = sorted_paths(&files)
            .into_iter()
            .map(|path| crate::normalize_zip_path(path).unwrap())
            .collect();
        assert!(zip_paths.contains(&depth.to_string()), "{zip_paths:?}");
        assert!(zip_paths.contains(&"train/images/scans/images/a.jpg".to_string()));
        // The progress label names the same path.
        let depth_entry = files
            .iter()
            .find(|(path, _)| crate::normalize_zip_path(path).unwrap() == depth)
            .unwrap();
        assert_eq!(
            crate::normalize_zip_path(std::str::from_utf8(depth_entry.1).unwrap()).unwrap(),
            depth
        );
    }
}
