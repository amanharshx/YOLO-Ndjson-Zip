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

pub fn normalize_split(split: &str) -> &str {
    match split {
        "val" | "valid" => "valid",
        _ => split,
    }
}

pub fn image_download_key(split: &str, file: &str) -> String {
    let split = normalize_split(split);
    format!("{}:{}:{}", split.len(), split, file)
}

pub fn image_entry_download_key(image: &ImageEntry) -> String {
    image_download_key(&image.split, &image.file)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_detection_ndjson() {
        let content = r#"{"type":"dataset","name":"test","class_names":{"0":"cat","1":"dog"}}
{"type":"image","file":"img1.jpg","width":640,"height":480,"split":"train","url":"https://example.com/img1.jpg","annotations":{"bboxes":[[0,0.1,0.2,0.3,0.4]]}}"#;

        let result = parse_ndjson(content).unwrap();
        assert_eq!(result.metadata.name, "test");
        assert_eq!(result.images.len(), 1);
        assert_eq!(result.images[0].file, "img1.jpg");
        assert_eq!(result.images[0].width, 640);
        assert_eq!(result.images[0].height, 480);
    }

    #[test]
    fn parse_malformed_json_returns_error() {
        let content = r#"{"type":"dataset","name":"test"
{invalid json}"#;

        let result = parse_ndjson(content);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ParseError::JsonError(_)));
    }

    #[test]
    fn parse_missing_metadata_returns_error() {
        let content = r#"{"type":"image","file":"img1.jpg","width":640,"height":480}"#;

        let result = parse_ndjson(content);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ParseError::NoMetadata));
    }

    #[test]
    fn get_bboxes_extracts_correctly() {
        let entry = ImageEntry {
            r#type: "image".to_string(),
            file: "test.jpg".to_string(),
            url: String::new(),
            width: 640,
            height: 480,
            split: "train".to_string(),
            annotations: Some(serde_json::json!({
                "bboxes": [[0, 0.1, 0.2, 0.3, 0.4], [1, 0.5, 0.6, 0.7, 0.8]]
            })),
        };

        let bboxes = entry.get_bboxes();
        assert_eq!(bboxes.len(), 2);
        assert_eq!(bboxes[0].class_id, 0);
        assert!((bboxes[0].x - 0.1).abs() < f64::EPSILON);
        assert_eq!(bboxes[1].class_id, 1);
        assert!((bboxes[1].x - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn train_images_filters_correctly() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}
{"type":"image","file":"train1.jpg","width":640,"height":480,"split":"train","url":""}
{"type":"image","file":"valid1.jpg","width":640,"height":480,"split":"valid","url":""}
{"type":"image","file":"train2.jpg","width":640,"height":480,"split":"train","url":""}"#;

        let data = parse_ndjson(content).unwrap();
        let train = data.train_images();
        assert_eq!(train.len(), 2);
        assert!(train.iter().all(|img| img.split == "train"));
    }

    #[test]
    fn valid_images_filters_correctly() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}
{"type":"image","file":"train1.jpg","width":640,"height":480,"split":"train","url":""}
{"type":"image","file":"valid1.jpg","width":640,"height":480,"split":"valid","url":""}
{"type":"image","file":"val1.jpg","width":640,"height":480,"split":"val","url":""}"#;

        let data = parse_ndjson(content).unwrap();
        let valid = data.valid_images();
        assert_eq!(valid.len(), 2);
        assert!(valid
            .iter()
            .all(|img| img.split == "valid" || img.split == "val"));
    }
}
