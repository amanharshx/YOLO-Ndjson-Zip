pub mod coco;
pub mod createml;
pub mod pascal_voc;
pub mod yolo;

use crate::parser::NDJSONData;
use std::collections::HashMap;

pub trait Converter {
    fn convert(&self, data: &NDJSONData, downloaded_images: &HashMap<String, Vec<u8>>) -> HashMap<String, Vec<u8>>;
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
        .map(|i| class_names.get(&i).cloned().unwrap_or_else(|| format!("class_{}", i)))
        .collect()
}
