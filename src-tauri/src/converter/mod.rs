pub mod coco;
pub mod createml;
pub mod pascal_voc;
pub mod yolo;

use crate::parser::{ImageEntry, NDJSONData};
use std::collections::HashMap;

pub trait Converter {
    /// Whether `convert` writes pixel coordinates for `task`, so every image needs a size.
    fn needs_image_sizes(&self, _task: &str) -> bool {
        false
    }

    /// Whether `convert` writes a `data.yaml` for `task`.
    fn writes_data_yaml(&self, _task: &str) -> bool {
        false
    }

    fn convert(
        &self,
        data: &NDJSONData,
        downloaded_images: &HashMap<String, Vec<u8>>,
    ) -> HashMap<String, Vec<u8>>;
}

/// Canonical format ID: lowercase, with `voc` as an alias of `pascal_voc`.
fn format_id(format: &str) -> String {
    let id = format.to_ascii_lowercase();
    if id == "voc" {
        "pascal_voc".to_string()
    } else {
        id
    }
}

pub fn get_converter(format: &str) -> Option<Box<dyn Converter + Send + Sync>> {
    match format_id(format).as_str() {
        "yolo" => Some(Box::new(yolo::YoloConverter::new())),
        "yolo_darknet" => Some(Box::new(yolo::YoloConverter::new_darknet())),
        "coco" => Some(Box::new(coco::CocoConverter::new())),
        "pascal_voc" => Some(Box::new(pascal_voc::PascalVocConverter::new())),
        "createml" => Some(Box::new(createml::CreateMlConverter::new())),
        _ => None,
    }
}

/// Formats offered in the UI, as (ID, label).
const UI_FORMATS: [(&str, &str); 4] = [
    ("yolo", "YOLO"),
    ("yolo_darknet", "YOLO Darknet"),
    ("coco", "COCO JSON"),
    ("pascal_voc", "Pascal VOC XML"),
];

fn format_label(id: &str) -> Option<&'static str> {
    UI_FORMATS
        .iter()
        .find(|(format_id, _)| *format_id == id)
        .map(|(_, label)| *label)
}

/// Task × format matrix: the UI formats that can represent each exportable task.
const TASK_FORMATS: [(&str, &[&str]); 6] = [
    ("detect", &["yolo", "yolo_darknet", "coco", "pascal_voc"]),
    ("segment", &["yolo", "coco", "pascal_voc"]),
    ("semantic", &["yolo", "coco", "pascal_voc"]),
    ("pose", &["yolo", "coco"]),
    ("obb", &["yolo", "coco"]),
    ("classify", &["yolo", "yolo_darknet", "pascal_voc"]),
];

/// Rejects a task the chosen format can't represent, before anything is downloaded.
pub fn check_task_format(task: &str, format: &str) -> Result<(), String> {
    let format_id = format_id(format);
    let label = format_label(&format_id);
    // Formats outside the UI (CreateML, TFRecord, unrecognised IDs) keep their
    // existing handling, which only rejected semantic datasets.
    if label.is_none() && task != "semantic" {
        return Ok(());
    }
    if task == "depth" {
        return Err("Depth datasets are not supported yet.".to_string());
    }
    let Some(&(_, supported)) = TASK_FORMATS.iter().find(|(name, _)| *name == task) else {
        let tasks: Vec<&str> = TASK_FORMATS.iter().map(|(name, _)| *name).collect();
        return Err(format!(
            "Unsupported task '{}'. Supported tasks: {}.",
            task,
            tasks.join(", ")
        ));
    };
    if supported.contains(&format_id.as_str()) {
        return Ok(());
    }
    let format_name = label.map_or_else(|| format!("'{}'", format), String::from);
    let supported_names: Vec<&str> = supported.iter().filter_map(|id| format_label(id)).collect();
    Err(format!(
        "{} can't export '{}' datasets. Formats that support '{}': {}.",
        format_name,
        task,
        task,
        supported_names.join(", ")
    ))
}

pub fn get_class_names(data: &NDJSONData) -> HashMap<i32, String> {
    data.metadata
        .class_names
        .iter()
        .filter_map(|(k, v)| k.parse::<i32>().ok().map(|id| (id, v.clone())))
        .collect()
}

/// Name for a class ID, with the Ultralytics `class{i}` placeholder when it has none.
pub fn class_name(class_names: &HashMap<i32, String>, id: i32) -> String {
    class_names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("class{}", id))
}

/// Pixel size for converters whose `needs_image_sizes` is true; `resolve_image_sizes`
/// fills or rejects every missing size before they run.
pub fn image_size(img: &ImageEntry) -> (i32, i32) {
    img.size()
        .expect("image sizes are resolved before a size-dependent conversion")
}

/// Class names for IDs 0..=max(class_names key, annotation class ID).
pub fn get_class_list(data: &NDJSONData) -> Vec<String> {
    let class_names = get_class_names(data);
    let max_id = class_names
        .keys()
        .copied()
        .chain(
            data.images
                .iter()
                .flat_map(ImageEntry::annotation_class_ids),
        )
        .max();

    match max_id {
        Some(max_id) => (0..=max_id).map(|i| class_name(&class_names, i)).collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{DatasetMetadata, ImageEntry, NDJSONData};
    use serde_json::json;

    fn make_metadata_with_classes(class_names: HashMap<String, String>) -> NDJSONData {
        NDJSONData {
            metadata: DatasetMetadata {
                task: "detect".to_string(),
                name: "test".to_string(),
                description: String::new(),
                bytes: 0,
                url: String::new(),
                class_names,
                kpt_shape: None,
                version: "1".to_string(),
            },
            images: vec![],
        }
    }

    #[test]
    fn get_converter_returns_known_formats() {
        assert!(get_converter("yolo").is_some());
        assert!(get_converter("YOLO").is_some());
        assert!(get_converter("coco").is_some());
        assert!(get_converter("pascal_voc").is_some());
        assert!(get_converter("voc").is_some());
        assert!(get_converter("createml").is_some());
        assert!(get_converter("yolo_darknet").is_some());
    }

    #[test]
    fn get_converter_returns_none_for_unknown() {
        assert!(get_converter("unknown_format").is_none());
        assert!(get_converter("").is_none());
        assert!(get_converter("xml").is_none());
    }

    #[test]
    fn task_format_matrix_allows_and_rejects_every_cell() {
        let formats = ["yolo", "yolo_darknet", "coco", "pascal_voc"];
        let matrix = [
            ("detect", [true, true, true, true]),
            ("segment", [true, false, true, true]),
            ("semantic", [true, false, true, true]),
            ("pose", [true, false, true, false]),
            ("obb", [true, false, true, false]),
            ("classify", [true, true, false, true]),
            ("depth", [false, false, false, false]),
        ];

        for (task, allowed) in matrix {
            for (format, allowed) in formats.into_iter().zip(allowed) {
                assert_eq!(
                    check_task_format(task, format).is_ok(),
                    allowed,
                    "{task} + {format}"
                );
            }
        }
    }

    #[test]
    fn rejected_pair_names_task_format_and_supporting_formats() {
        assert_eq!(
            check_task_format("pose", "pascal_voc").unwrap_err(),
            "Pascal VOC XML can't export 'pose' datasets. \
             Formats that support 'pose': YOLO, COCO JSON."
        );
        assert_eq!(
            check_task_format("classify", "COCO").unwrap_err(),
            "COCO JSON can't export 'classify' datasets. \
             Formats that support 'classify': YOLO, YOLO Darknet, Pascal VOC XML."
        );
    }

    #[test]
    fn depth_is_not_supported_yet_in_any_format() {
        for format in ["yolo", "yolo_darknet", "coco", "pascal_voc"] {
            assert_eq!(
                check_task_format("depth", format).unwrap_err(),
                "Depth datasets are not supported yet.",
                "{format}"
            );
        }
    }

    #[test]
    fn unknown_task_is_rejected_by_name() {
        assert_eq!(
            check_task_format("keypoints", "yolo").unwrap_err(),
            "Unsupported task 'keypoints'. \
             Supported tasks: detect, segment, semantic, pose, obb, classify."
        );
    }

    #[test]
    fn formats_outside_the_ui_only_reject_semantic() {
        for format in ["createml", "tfrecord"] {
            for task in [
                "detect",
                "segment",
                "pose",
                "obb",
                "classify",
                "depth",
                "keypoints",
            ] {
                assert!(check_task_format(task, format).is_ok(), "{task} + {format}");
            }
            assert_eq!(
                check_task_format("semantic", format).unwrap_err(),
                format!(
                    "'{format}' can't export 'semantic' datasets. \
                     Formats that support 'semantic': YOLO, COCO JSON, Pascal VOC XML."
                )
            );
        }
    }

    #[test]
    fn voc_alias_follows_pascal_voc() {
        assert!(check_task_format("semantic", "voc").is_ok());
        assert!(check_task_format("pose", "voc").is_err());
    }

    #[test]
    fn get_class_list_orders_by_id() {
        let mut class_names = HashMap::new();
        class_names.insert("2".to_string(), "bird".to_string());
        class_names.insert("0".to_string(), "cat".to_string());
        class_names.insert("1".to_string(), "dog".to_string());

        let data = make_metadata_with_classes(class_names);
        let class_list = get_class_list(&data);

        assert_eq!(class_list, vec!["cat", "dog", "bird"]);
    }

    #[test]
    fn get_class_list_fills_gaps_with_placeholders() {
        let data = make_metadata_with_classes(HashMap::from([
            ("0".to_string(), "a".to_string()),
            ("2".to_string(), "c".to_string()),
        ]));

        assert_eq!(get_class_list(&data), vec!["a", "class1", "c"]);
    }

    #[test]
    fn get_class_list_covers_ids_from_every_annotation_row_kind() {
        let rows = [
            json!({ "boxes": [[2, 0.5, 0.5, 0.2, 0.2]] }),
            json!({ "bboxes": [[2, 0.5, 0.5, 0.2, 0.2]] }),
            json!({ "segments": [[2, 0.1, 0.1, 0.9, 0.1, 0.5, 0.9]] }),
            json!({ "pose": [[2, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5, 2]] }),
            json!({ "obb": [[2, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]] }),
            json!({ "classification": [2] }),
        ];

        for annotations in rows {
            let mut data = make_metadata_with_classes(HashMap::new());
            data.images.push(ImageEntry {
                file: "a.jpg".to_string(),
                output_file: None,
                url: String::new(),
                width: Some(640),
                height: Some(480),
                split: "train".to_string(),
                annotations: Some(annotations.clone()),
            });

            assert_eq!(
                get_class_list(&data),
                vec!["class0", "class1", "class2"],
                "{annotations}"
            );
        }
    }
}
