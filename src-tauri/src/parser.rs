use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Failed to parse JSON: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("No metadata found in NDJSON")]
    NoMetadata,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub class_id: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    #[serde(default)]
    pub r#type: String,
    #[serde(default = "default_task")]
    pub task: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub class_names: HashMap<String, String>,
    #[serde(default)]
    pub version: i32,
}

fn default_task() -> String {
    "detect".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEntry {
    #[serde(default)]
    pub r#type: String,
    pub file: String,
    #[serde(default)]
    pub url: String,
    pub width: i32,
    pub height: i32,
    #[serde(default = "default_split")]
    pub split: String,
    #[serde(default)]
    pub annotations: Option<serde_json::Value>,
}

fn default_split() -> String {
    "train".to_string()
}

impl ImageEntry {
    pub fn get_bboxes(&self) -> Vec<BoundingBox> {
        let Some(annotations) = &self.annotations else {
            return Vec::new();
        };

        let Some(bboxes) = annotations.get("bboxes") else {
            return Vec::new();
        };

        let Some(bbox_array) = bboxes.as_array() else {
            return Vec::new();
        };

        bbox_array
            .iter()
            .filter_map(|bbox_data| {
                let arr = bbox_data.as_array()?;
                if arr.len() >= 5 {
                    Some(BoundingBox {
                        class_id: arr[0].as_i64()? as i32,
                        x: arr[1].as_f64()?,
                        y: arr[2].as_f64()?,
                        width: arr[3].as_f64()?,
                        height: arr[4].as_f64()?,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct NDJSONData {
    pub metadata: DatasetMetadata,
    pub images: Vec<ImageEntry>,
}

impl NDJSONData {
    pub fn train_images(&self) -> Vec<&ImageEntry> {
        self.images.iter().filter(|img| img.split == "train").collect()
    }

    pub fn valid_images(&self) -> Vec<&ImageEntry> {
        self.images
            .iter()
            .filter(|img| img.split == "valid" || img.split == "val")
            .collect()
    }

    pub fn test_images(&self) -> Vec<&ImageEntry> {
        self.images.iter().filter(|img| img.split == "test").collect()
    }
}

pub fn parse_ndjson(content: &str) -> Result<NDJSONData, ParseError> {
    let mut metadata: Option<DatasetMetadata> = None;
    let mut images: Vec<ImageEntry> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let value: serde_json::Value = serde_json::from_str(line)?;

        if let Some(type_str) = value.get("type").and_then(|v| v.as_str()) {
            match type_str {
                "dataset" => {
                    metadata = Some(serde_json::from_value(value)?);
                }
                "image" => {
                    images.push(serde_json::from_value(value)?);
                }
                _ => {}
            }
        }
    }

    let metadata = metadata.ok_or(ParseError::NoMetadata)?;

    Ok(NDJSONData { metadata, images })
}
