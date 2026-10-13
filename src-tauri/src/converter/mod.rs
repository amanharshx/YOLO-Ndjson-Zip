pub mod coco;
pub mod createml;
pub mod pascal_voc;
pub mod yolo;

use crate::parser::{ImageEntry, NDJSONData};
use std::collections::HashMap;

pub trait Converter {
    fn convert(
        &self,
        data: &NDJSONData,
        downloaded_images: &HashMap<String, Vec<u8>>,
    ) -> HashMap<String, Vec<u8>>;
}

pub fn get_converter(format: &str) -> Option<Box<dyn Converter + Send + Sync>> {
    match format.to_lowercase().as_str() {
        "yolo" => Some(Box::new(yolo::YoloConverter::new())),
        "yolo_darknet" => Some(Box::new(yolo::YoloConverter::new_darknet())),
        "coco" => Some(Box::new(coco::CocoConverter::new())),
        "pascal_voc" | "voc" => Some(Box::new(pascal_voc::PascalVocConverter::new())),
        "createml" => Some(Box::new(createml::CreateMlConverter::new())),
        _ => None,
    }
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
                r#type: "dataset".to_string(),
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
                r#type: "image".to_string(),
                file: "a.jpg".to_string(),
                output_file: None,
                url: String::new(),
                width: 640,
                height: 480,
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
