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
pub struct PoseAnnotation {
    pub class_id: i32,
    pub bbox_x: f64,
    pub bbox_y: f64,
    pub bbox_w: f64,
    pub bbox_h: f64,
    pub keypoints: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentAnnotation {
    pub class_id: i32,
    pub points: Vec<(f64, f64)>,
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
    pub kpt_shape: Option<Vec<i32>>,
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

    pub fn get_classifications(&self) -> Vec<i32> {
        let Some(annotations) = &self.annotations else {
            return Vec::new();
        };

        let Some(classification) = annotations.get("classification") else {
            return Vec::new();
        };

        classification
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|n| n as i32))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_pose_annotations(&self, kpt_shape: Option<&[i32]>) -> Vec<PoseAnnotation> {
        let Some(annotations) = &self.annotations else {
            return Vec::new();
        };

        let Some(poses) = annotations.get("pose") else {
            return Vec::new();
        };

        let Some(pose_array) = poses.as_array() else {
            return Vec::new();
        };

        let num_keypoints = kpt_shape.and_then(|s| s.first()).copied().unwrap_or(17) as usize;

        pose_array
            .iter()
            .filter_map(|pose_data| {
                let arr = pose_data.as_array()?;
                let expected_len = 1 + num_keypoints * 2 + 4;
                if arr.len() < expected_len {
                    return None;
                }

                let class_id = arr[0].as_i64()? as i32;

                let mut keypoints = Vec::with_capacity(num_keypoints);
                for i in 0..num_keypoints {
                    let kp_x = arr[1 + i * 2].as_f64()?;
                    let kp_y = arr[1 + i * 2 + 1].as_f64()?;
                    keypoints.push((kp_x, kp_y));
                }

                let bbox_start = 1 + num_keypoints * 2;
                Some(PoseAnnotation {
                    class_id,
                    bbox_x: arr[bbox_start].as_f64()?,
                    bbox_y: arr[bbox_start + 1].as_f64()?,
                    bbox_w: arr[bbox_start + 2].as_f64()?,
                    bbox_h: arr[bbox_start + 3].as_f64()?,
                    keypoints,
                })
            })
            .collect()
    }

    pub fn get_segment_annotations(&self) -> Vec<SegmentAnnotation> {
        let Some(annotations) = &self.annotations else {
            return Vec::new();
        };

        let Some(segments) = annotations.get("segments") else {
            return Vec::new();
        };

        let Some(seg_array) = segments.as_array() else {
            return Vec::new();
        };

        seg_array
            .iter()
            .filter_map(|seg_data| {
                let arr = seg_data.as_array()?;
                if arr.len() < 7 {
                    return None;
                }

                let class_id = arr[0].as_i64()? as i32;
                let mut points = Vec::new();

                for i in (1..arr.len()).step_by(2) {
                    if i + 1 < arr.len() {
                        let x = arr[i].as_f64()?;
                        let y = arr[i + 1].as_f64()?;
                        points.push((x, y));
                    }
                }

                Some(SegmentAnnotation { class_id, points })
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
        self.images
            .iter()
            .filter(|img| img.split == "train")
            .collect()
    }

    pub fn valid_images(&self) -> Vec<&ImageEntry> {
        self.images
            .iter()
            .filter(|img| img.split == "valid" || img.split == "val")
            .collect()
    }

    pub fn test_images(&self) -> Vec<&ImageEntry> {
        self.images
            .iter()
            .filter(|img| img.split == "test")
            .collect()
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
