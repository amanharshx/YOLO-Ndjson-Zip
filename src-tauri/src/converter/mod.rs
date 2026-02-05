pub mod coco;
pub mod createml;
pub mod pascal_voc;
pub mod yolo;

use crate::parser::NDJSONData;
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

pub fn get_class_list(data: &NDJSONData) -> Vec<String> {
    let class_names = get_class_names(data);
    if class_names.is_empty() {
        return Vec::new();
    }

    let max_id = *class_names.keys().max().unwrap_or(&0);
    (0..=max_id)
        .map(|i| {
            class_names
                .get(&i)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", i))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{DatasetMetadata, NDJSONData};

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
                version: 1,
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
}
