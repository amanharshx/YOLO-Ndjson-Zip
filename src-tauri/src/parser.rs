use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("No metadata found in NDJSON")]
    NoMetadata,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid pose annotations: {0}")]
    InvalidPose(String),
    #[error("Invalid split: {0}")]
    InvalidSplit(String),
    #[error("Invalid record on line {line}: {message}")]
    InvalidRecord { line: usize, message: String },
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
    pub keypoints: Vec<f64>,
    pub dims: usize,
}

impl PoseAnnotation {
    pub fn num_keypoints(&self) -> usize {
        self.keypoints.len() / self.dims
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentAnnotation {
    pub class_id: i32,
    pub points: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObbAnnotation {
    pub class_id: i32,
    pub points: [(f64, f64); 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
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
    #[serde(default, deserialize_with = "deserialize_version")]
    pub version: String,
    /// Depth PNG scale, set by `parse_ndjson` from a depth dataset's `depth_scale`.
    #[serde(skip, default = "default_depth_scale")]
    pub depth_scale: f64,
}

fn default_task() -> String {
    "detect".to_string()
}

/// Depth PNG scale Ultralytics assumes when a depth dataset declares none.
fn default_depth_scale() -> f64 {
    1000.0
}

/// A depth dataset's `depth_scale`, or the default when absent. Like Ultralytics, anything
/// but a positive number is rejected; other tasks ignore the key.
fn depth_scale(
    task: &str,
    value: Option<&serde_json::Value>,
    line: usize,
) -> Result<f64, ParseError> {
    match value {
        Some(value) if task == "depth" => value
            // JSON numbers are always finite.
            .as_f64()
            .filter(|scale| *scale > 0.0)
            .ok_or_else(|| ParseError::InvalidRecord {
                line,
                message: format!("depth_scale must be a positive number, got {value}"),
            }),
        _ => Ok(default_depth_scale()),
    }
}

fn deserialize_version<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        _ => Ok(String::new()),
    }
}

/// Missing or null splits become empty; other non-strings keep their JSON text so the
/// split check can name them.
fn deserialize_split<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Null => Ok(String::new()),
        other => Ok(other.to_string()),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEntry {
    pub file: String,
    #[serde(skip)]
    pub output_file: Option<String>,
    #[serde(default)]
    pub url: String,
    /// Optional, as in Ultralytics; formats that need pixel sizes fill gaps from the image.
    #[serde(default)]
    pub width: Option<i32>,
    #[serde(default)]
    pub height: Option<i32>,
    #[serde(default, deserialize_with = "deserialize_split")]
    pub split: String,
    /// Raw JSON, so only depth datasets reject a malformed value. Read it through `depth_url`.
    #[serde(default)]
    pub depth: Option<serde_json::Value>,
    #[serde(default)]
    pub annotations: Option<serde_json::Value>,
}

const ACCEPTED_SPLITS: [&str; 4] = ["train", "val", "valid", "test"];

pub fn normalize_split(split: &str) -> &str {
    match split {
        "val" | "valid" => "valid",
        _ => split,
    }
}

/// The name an image's label, XML, or other per-image output file is built from.
pub fn output_stem(file_name: &str) -> &str {
    file_name
        .rsplit_once('.')
        .map_or(file_name, |(stem, _)| stem)
}

pub fn image_download_key(split: &str, file: &str) -> String {
    let split = normalize_split(split);
    format!("{}:{}:{}", split.len(), split, file)
}

pub fn image_entry_download_key(image: &ImageEntry) -> String {
    image_download_key(&image.split, image.effective_file_name())
}

/// Name of the depth map paired with an image: Ultralytics looks for its stem plus `.png`.
pub fn depth_file_name(image_file: &str) -> String {
    format!("{}.png", output_stem(image_file))
}

/// Download key of the depth map paired with an image. Its `{split}/depth` pseudo split is
/// never an image split, so a depth map key never collides with an image key.
pub fn depth_download_key(split: &str, image_file: &str) -> String {
    image_download_key(
        &format!("{}/depth", normalize_split(split)),
        &depth_file_name(image_file),
    )
}

impl ImageEntry {
    pub fn effective_file_name(&self) -> &str {
        self.output_file.as_deref().unwrap_or(&self.file)
    }

    /// The non-empty `depth.url` string of the depth map paired with this image.
    pub fn depth_url(&self) -> Option<&str> {
        self.depth
            .as_ref()?
            .get("url")?
            .as_str()
            .filter(|url| !url.is_empty())
    }

    pub fn size(&self) -> Option<(i32, i32)> {
        Some((self.width?, self.height?))
    }

    pub fn get_bboxes(&self) -> Vec<BoundingBox> {
        let Some(annotations) = &self.annotations else {
            return Vec::new();
        };

        let Some(bboxes) = annotations
            .get("bboxes")
            .or_else(|| annotations.get("boxes"))
        else {
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

    /// Class IDs of every annotation row, whatever the task (as Ultralytics collects them).
    pub fn annotation_class_ids(&self) -> impl Iterator<Item = i32> + '_ {
        let annotations = self.annotations.as_ref();
        let row_ids = ["boxes", "bboxes", "segments", "pose", "obb"]
            .into_iter()
            .filter_map(move |key| annotations?.get(key)?.as_array())
            .flatten()
            .filter_map(|row| row.as_array()?.first()?.as_i64());
        let classification_ids = annotations
            .and_then(|annotations| annotations.get("classification"))
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_i64);

        row_ids.chain(classification_ids).map(|id| id as i32)
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

    pub fn get_pose_annotations(&self, dims: usize) -> Vec<PoseAnnotation> {
        if !matches!(dims, 2 | 3) {
            return Vec::new();
        }

        let Some(annotations) = &self.annotations else {
            return Vec::new();
        };

        let Some(poses) = annotations.get("pose") else {
            return Vec::new();
        };

        let Some(pose_array) = poses.as_array() else {
            return Vec::new();
        };

        // Format: [class_id, bbox_cx, bbox_cy, bbox_w, bbox_h, kp1...]
        pose_array
            .iter()
            .filter_map(|pose_data| {
                let arr = pose_data.as_array()?;
                if arr.len() < 5 + dims {
                    return None;
                }

                let remaining = arr.len() - 5; // subtract class_id + bbox(4)
                if remaining % dims != 0 {
                    return None;
                }

                let class_id = arr[0].as_i64()? as i32;
                let bbox_x = arr[1].as_f64()?;
                let bbox_y = arr[2].as_f64()?;
                let bbox_w = arr[3].as_f64()?;
                let bbox_h = arr[4].as_f64()?;

                let keypoints = arr[5..]
                    .iter()
                    .map(serde_json::Value::as_f64)
                    .collect::<Option<Vec<_>>>()?;

                Some(PoseAnnotation {
                    class_id,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                    keypoints,
                    dims,
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

    pub fn get_obb_annotations(&self) -> Vec<ObbAnnotation> {
        let Some(annotations) = &self.annotations else {
            return Vec::new();
        };

        let Some(obbs) = annotations.get("obb") else {
            return Vec::new();
        };

        let Some(obb_array) = obbs.as_array() else {
            return Vec::new();
        };

        // Format: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
        obb_array
            .iter()
            .filter_map(|obb_data| {
                let arr = obb_data.as_array()?;
                if arr.len() != 9 {
                    return None;
                }

                let class_id = arr[0].as_i64()? as i32;
                let points = [
                    (arr[1].as_f64()?, arr[2].as_f64()?),
                    (arr[3].as_f64()?, arr[4].as_f64()?),
                    (arr[5].as_f64()?, arr[6].as_f64()?),
                    (arr[7].as_f64()?, arr[8].as_f64()?),
                ];

                Some(ObbAnnotation { class_id, points })
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct NDJSONData {
    pub metadata: DatasetMetadata,
    pub images: Vec<ImageEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoseKptShape {
    pub num_keypoints: usize,
    pub dims: usize,
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

    pub fn pose_kpt_shape(&self) -> Result<Option<PoseKptShape>, ParseError> {
        if self.metadata.task != "pose" {
            return Ok(None);
        }

        let metadata_shape = self
            .metadata
            .kpt_shape
            .as_ref()
            .map(|shape| {
                if shape.len() != 2 || shape[0] <= 0 || !matches!(shape[1], 2 | 3) {
                    return Err(ParseError::InvalidPose(format!(
                        "dataset kpt_shape must be [positive number_of_keypoints, 2|3], got {:?}",
                        shape
                    )));
                }
                Ok(PoseKptShape {
                    num_keypoints: shape[0] as usize,
                    dims: shape[1] as usize,
                })
            })
            .transpose()?;

        let mut payloads = Vec::new();
        for image in &self.images {
            let Some(pose_rows) = image
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.get("pose"))
            else {
                continue;
            };
            let rows = pose_rows.as_array().ok_or_else(|| {
                ParseError::InvalidPose(format!(
                    "image '{}' has a non-array 'pose' value",
                    image.file
                ))
            })?;

            for (row_index, row) in rows.iter().enumerate() {
                let values = row.as_array().ok_or_else(|| {
                    ParseError::InvalidPose(format!(
                        "image '{}', pose row {} is not an array",
                        image.file,
                        row_index + 1
                    ))
                })?;
                if values.len() <= 5 {
                    return Err(ParseError::InvalidPose(format!(
                        "image '{}', pose row {} has no keypoint payload",
                        image.file,
                        row_index + 1
                    )));
                }
                if values[0]
                    .as_i64()
                    .and_then(|class_id| i32::try_from(class_id).ok())
                    .is_none()
                {
                    return Err(ParseError::InvalidPose(format!(
                        "image '{}', pose row {} class ID must be an integer in the i32 range",
                        image.file,
                        row_index + 1
                    )));
                }
                if values.iter().any(|value| value.as_f64().is_none()) {
                    return Err(ParseError::InvalidPose(format!(
                        "image '{}', pose row {} contains a non-numeric value",
                        image.file,
                        row_index + 1
                    )));
                }

                payloads.push((image.file.as_str(), row_index + 1, &values[5..]));
            }
        }

        if let Some(shape) = metadata_shape {
            let expected_values = shape.num_keypoints * shape.dims;
            if let Some((file, row_index, payload)) = payloads
                .iter()
                .find(|(_, _, payload)| payload.len() != expected_values)
            {
                return Err(ParseError::InvalidPose(format!(
                    "image '{}', pose row {} has {} keypoint values, but dataset kpt_shape [{}, {}] requires {}",
                    file,
                    row_index,
                    payload.len(),
                    shape.num_keypoints,
                    shape.dims,
                    expected_values
                )));
            }
            // Metadata is authoritative. For dims=3, visibility values remain verbatim.
            return Ok(Some(shape));
        }

        if payloads.is_empty() {
            return Ok(None);
        }

        let payload_len = payloads[0].2.len();
        if let Some((file, row_index, payload)) = payloads
            .iter()
            .find(|(_, _, payload)| payload.len() != payload_len)
        {
            return Err(ParseError::InvalidPose(format!(
                "image '{}', pose row {} has {} keypoint values; expected {}",
                file,
                row_index,
                payload.len(),
                payload_len
            )));
        }

        if payload_len % 3 == 0
            && payloads.iter().all(|(_, _, payload)| {
                payload[2..]
                    .iter()
                    .step_by(3)
                    .all(|v| matches!(v.as_f64(), Some(0.0 | 1.0 | 2.0)))
            })
        {
            return Ok(Some(PoseKptShape {
                num_keypoints: payload_len / 3,
                dims: 3,
            }));
        }

        if payload_len % 2 == 0 && payload_len % 3 != 0 {
            return Ok(Some(PoseKptShape {
                num_keypoints: payload_len / 2,
                dims: 2,
            }));
        }

        Err(ParseError::InvalidPose(
            "cannot infer keypoint dimensions; add kpt_shape: [number_of_keypoints, 2|3] \
             to the dataset record"
                .to_string(),
        ))
    }
}

/// Like Ultralytics, the first non-empty line is the dataset record and every later
/// non-empty line is an image record, whatever its `type`.
pub fn parse_ndjson(content: &str) -> Result<NDJSONData, ParseError> {
    // Line numbers count blank lines so they match what an editor shows.
    let mut records = content
        .lines()
        .enumerate()
        .map(|(index, text)| (index + 1, text.trim()))
        .filter(|(_, text)| !text.is_empty());

    let (line, text) = records.next().ok_or(ParseError::NoMetadata)?;
    let value: serde_json::Value = serde_json::from_str(text).map_err(record_error(line))?;
    match value.get("type") {
        None | Some(serde_json::Value::Null) => {}
        Some(found) if *found == "dataset" => {}
        Some(found) => {
            return Err(ParseError::InvalidRecord {
                line,
                message: format!(
                    "the first line must be the dataset record, found type {}",
                    found
                ),
            });
        }
    }
    let raw_depth_scale = value.get("depth_scale").cloned();
    let mut metadata: DatasetMetadata =
        serde_json::from_value(value).map_err(record_error(line))?;
    metadata.depth_scale = depth_scale(&metadata.task, raw_depth_scale.as_ref(), line)?;

    let mut images: Vec<ImageEntry> = Vec::new();
    for (line, text) in records {
        let value: serde_json::Value = serde_json::from_str(text).map_err(record_error(line))?;
        if value.get("type").and_then(serde_json::Value::as_str) == Some("dataset") {
            return Err(ParseError::InvalidRecord {
                line,
                message: "only the first line may be the dataset record".to_string(),
            });
        }
        images.push(serde_json::from_value(value).map_err(record_error(line))?);
    }

    validate_splits(&images)?;

    Ok(NDJSONData { metadata, images })
}

/// serde positions are relative to the one-line record, so only the column is kept.
fn record_error(line: usize) -> impl FnOnce(serde_json::Error) -> ParseError {
    move |error| {
        let message = error.to_string();
        let position = format!(" at line {} column {}", error.line(), error.column());
        let message = match message.strip_suffix(&position) {
            Some(reason) => format!("{} at column {}", reason, error.column()),
            None => message,
        };
        ParseError::InvalidRecord { line, message }
    }
}

/// Rejects missing or unknown splits before download, so no image is fetched and then
/// silently dropped from every format's output.
fn validate_splits(images: &[ImageEntry]) -> Result<(), ParseError> {
    let mut invalid = images
        .iter()
        .filter(|image| !ACCEPTED_SPLITS.contains(&image.split.as_str()));
    let Some(image) = invalid.next() else {
        return Ok(());
    };

    let problem = if image.split.is_empty() {
        "is missing its split".to_string()
    } else {
        format!("has unknown value '{}'", image.split)
    };
    let affected = 1 + invalid.count();
    let count = if affected > 1 {
        format!(" ({} image lines affected)", affected)
    } else {
        String::new()
    };
    Err(ParseError::InvalidSplit(format!(
        "image '{}' {}{}. Accepted splits: {}.",
        image.file,
        problem,
        count,
        ACCEPTED_SPLITS.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose_data(kpt_shape: Option<Vec<i32>>, payloads: Vec<Vec<f64>>) -> NDJSONData {
        let images = payloads
            .into_iter()
            .enumerate()
            .map(|(index, payload)| {
                let mut row = vec![
                    serde_json::json!(0),
                    serde_json::json!(0.5),
                    serde_json::json!(0.5),
                    serde_json::json!(0.4),
                    serde_json::json!(0.4),
                ];
                row.extend(payload.into_iter().map(|value| serde_json::json!(value)));
                ImageEntry {
                    file: format!("pose_{index}.jpg"),
                    output_file: None,
                    url: String::new(),
                    width: Some(640),
                    height: Some(480),
                    split: "train".to_string(),
                    depth: None,
                    annotations: Some(serde_json::json!({ "pose": [row] })),
                }
            })
            .collect();

        NDJSONData {
            metadata: DatasetMetadata {
                task: "pose".to_string(),
                name: "pose-test".to_string(),
                description: String::new(),
                bytes: 0,
                url: String::new(),
                class_names: HashMap::from([("0".to_string(), "object".to_string())]),
                kpt_shape,
                version: "1".to_string(),
                depth_scale: 1000.0,
            },
            images,
        }
    }

    fn keypoint_payload(num_keypoints: usize, dims: usize) -> Vec<f64> {
        (0..num_keypoints)
            .flat_map(|index| {
                let x = 0.1 + index as f64 * 0.001;
                let y = 0.2 + index as f64 * 0.001;
                if dims == 3 {
                    vec![x, y, 2.0]
                } else {
                    vec![x, y]
                }
            })
            .collect()
    }

    #[test]
    fn pose_kpt_shape_infers_three_dimensions_for_common_templates() {
        for num_keypoints in [17, 21, 18, 68, 4] {
            let data = pose_data(None, vec![keypoint_payload(num_keypoints, 3)]);

            assert_eq!(
                data.pose_kpt_shape().unwrap(),
                Some(PoseKptShape {
                    num_keypoints,
                    dims: 3,
                })
            );
        }
    }

    #[test]
    fn pose_kpt_shape_uses_explicit_two_dimensions_for_common_templates() {
        for num_keypoints in [17, 21, 18, 68, 4] {
            let data = pose_data(
                Some(vec![num_keypoints as i32, 2]),
                vec![keypoint_payload(num_keypoints, 2)],
            );

            assert_eq!(
                data.pose_kpt_shape().unwrap(),
                Some(PoseKptShape {
                    num_keypoints,
                    dims: 2,
                })
            );
        }
    }

    #[test]
    fn pose_kpt_shape_infers_two_dimensions_when_unambiguous() {
        for num_keypoints in [17, 68, 4] {
            let data = pose_data(None, vec![keypoint_payload(num_keypoints, 2)]);

            assert_eq!(
                data.pose_kpt_shape().unwrap(),
                Some(PoseKptShape {
                    num_keypoints,
                    dims: 2,
                })
            );
        }
    }

    #[test]
    fn pose_kpt_shape_rejects_ambiguous_two_dimensional_payloads_without_metadata() {
        for num_keypoints in [21, 18] {
            let data = pose_data(None, vec![keypoint_payload(num_keypoints, 2)]);

            let error = data.pose_kpt_shape().unwrap_err().to_string();
            assert!(error.contains("cannot infer keypoint dimensions"));
            assert!(error.contains("add kpt_shape"));
        }
    }

    #[test]
    fn pose_kpt_shape_rejects_metadata_disagreement_with_image_and_row() {
        let data = pose_data(Some(vec![17, 3]), vec![keypoint_payload(21, 3)]);

        let error = data.pose_kpt_shape().unwrap_err().to_string();
        assert!(error.contains("pose_0.jpg"));
        assert!(error.contains("pose row 1"));
        assert!(error.contains("kpt_shape [17, 3]"));
        assert!(error.contains("requires 51"));
    }

    #[test]
    fn pose_kpt_shape_rejects_non_uniform_payload_lengths() {
        let data = pose_data(None, vec![keypoint_payload(17, 3), keypoint_payload(21, 3)]);

        let error = data.pose_kpt_shape().unwrap_err().to_string();
        assert!(error.contains("pose_1.jpg"));
        assert!(error.contains("pose row 1"));
        assert!(error.contains("63 keypoint values; expected 51"));
    }

    #[test]
    fn pose_kpt_shape_allows_zero_pose_rows() {
        let without_metadata = pose_data(None, Vec::new());
        assert_eq!(without_metadata.pose_kpt_shape().unwrap(), None);

        let with_metadata = pose_data(Some(vec![17, 3]), Vec::new());
        assert_eq!(
            with_metadata.pose_kpt_shape().unwrap(),
            Some(PoseKptShape {
                num_keypoints: 17,
                dims: 3,
            })
        );
    }

    #[test]
    fn pose_kpt_shape_ignores_pose_rows_for_non_pose_task() {
        let mut data = pose_data(None, vec![keypoint_payload(21, 2)]);
        data.metadata.task = "detect".to_string();

        assert_eq!(data.pose_kpt_shape().unwrap(), None);
    }

    #[test]
    fn pose_kpt_shape_rejects_malformed_metadata() {
        for malformed in [vec![], vec![17], vec![0, 3], vec![-1, 3], vec![17, 4]] {
            let data = pose_data(Some(malformed.clone()), Vec::new());

            let error = data.pose_kpt_shape().unwrap_err().to_string();
            assert!(error.contains(&format!("got {:?}", malformed)));
        }
    }

    #[test]
    fn pose_kpt_shape_rejects_non_integer_class_id_with_location() {
        let mut data = pose_data(None, vec![keypoint_payload(17, 3)]);
        data.images[0].annotations.as_mut().unwrap()["pose"][0][0] = serde_json::json!(0.5);

        let error = data.pose_kpt_shape().unwrap_err().to_string();
        assert!(error.contains("pose_0.jpg"));
        assert!(error.contains("pose row 1"));
        assert!(error.contains("class ID must be an integer"));
    }

    #[test]
    fn parse_valid_detection_ndjson() {
        let content = r#"{"type":"dataset","name":"test","class_names":{"0":"cat","1":"dog"}}
{"type":"image","file":"img1.jpg","width":640,"height":480,"split":"train","url":"https://example.com/img1.jpg","annotations":{"bboxes":[[0,0.1,0.2,0.3,0.4]]}}"#;

        let result = parse_ndjson(content).unwrap();
        assert_eq!(result.metadata.name, "test");
        assert_eq!(result.images.len(), 1);
        assert_eq!(result.images[0].file, "img1.jpg");
        assert_eq!(result.images[0].width, Some(640));
        assert_eq!(result.images[0].height, Some(480));
    }

    #[test]
    fn image_lines_are_images_whatever_their_type_value() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}
{"type":null,"file":"a.jpg","width":1,"height":1,"split":"train"}
{"type":5,"file":"b.jpg","width":1,"height":1,"split":"train"}
{"type":"video","file":"c.jpg","width":1,"height":1,"split":"train"}"#;

        let data = parse_ndjson(content).unwrap();

        let files: Vec<&str> = data.images.iter().map(|img| img.file.as_str()).collect();
        assert_eq!(files, ["a.jpg", "b.jpg", "c.jpg"]);
    }

    #[test]
    fn first_line_with_null_type_is_the_dataset_record() {
        let content = r#"{"type":null,"name":"test","class_names":{}}
{"file":"a.jpg","width":1,"height":1,"split":"train"}"#;

        let data = parse_ndjson(content).unwrap();

        assert_eq!(data.metadata.name, "test");
        assert_eq!(data.images.len(), 1);
    }

    #[test]
    fn first_line_with_non_string_type_is_rejected() {
        let error = parse_ndjson(r#"{"type":5,"name":"test","class_names":{}}"#)
            .unwrap_err()
            .to_string();

        assert_eq!(
            error,
            "Invalid record on line 1: the first line must be the dataset record, found type 5"
        );
    }

    #[test]
    fn repeated_dataset_line_is_rejected_naming_its_line() {
        let content = r#"{"type":"dataset","name":"first","class_names":{}}
{"file":"a.jpg","width":1,"height":1,"split":"train"}

{"type":"dataset","name":"second","class_names":{}}"#;

        let error = parse_ndjson(content).unwrap_err().to_string();

        assert_eq!(
            error,
            "Invalid record on line 4: only the first line may be the dataset record"
        );
    }

    #[test]
    fn image_line_without_file_is_rejected_naming_its_line() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}
{"file":"a.jpg","width":1,"height":1,"split":"train"}
{"type":"image","width":1,"height":1,"split":"train"}"#;

        let error = parse_ndjson(content).unwrap_err().to_string();

        assert_eq!(error, "Invalid record on line 3: missing field `file`");
    }

    #[test]
    fn image_sizes_are_optional_and_kept_when_declared() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}
{"file":"a.jpg","split":"train"}
{"file":"b.jpg","width":640,"split":"train"}
{"file":"c.jpg","width":640,"height":480,"split":"train"}"#;

        let data = parse_ndjson(content).unwrap();

        let sizes: Vec<_> = data
            .images
            .iter()
            .map(|img| (img.width, img.height))
            .collect();
        assert_eq!(
            sizes,
            [(None, None), (Some(640), None), (Some(640), Some(480))]
        );
    }

    #[test]
    fn parse_string_version() {
        let content = r#"{"type":"dataset","name":"test","class_names":{},"version":"latest"}"#;
        let result = parse_ndjson(content).unwrap();
        assert_eq!(result.metadata.version, "latest");
    }

    #[test]
    fn parse_integer_version() {
        let content = r#"{"type":"dataset","name":"test","class_names":{},"version":1}"#;
        let result = parse_ndjson(content).unwrap();
        assert_eq!(result.metadata.version, "1");
    }

    #[test]
    fn parse_missing_version_defaults_to_empty() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}"#;
        let result = parse_ndjson(content).unwrap();
        assert_eq!(result.metadata.version, "");
    }

    #[test]
    fn malformed_json_is_rejected_naming_its_line_and_column() {
        let content = r#"{"type":"dataset","name":"test","class_names":{}}

{"file":"a.jpg","width":1,"height":1,"split":"train"} trailing"#;

        let error = parse_ndjson(content).unwrap_err().to_string();

        assert_eq!(
            error,
            "Invalid record on line 3: trailing characters at column 55"
        );
    }

    #[test]
    fn parse_empty_content_returns_no_metadata() {
        assert!(matches!(
            parse_ndjson("\n  \n").unwrap_err(),
            ParseError::NoMetadata
        ));
    }

    #[test]
    fn first_line_typed_as_image_is_rejected() {
        let content = r#"
{"type":"image","file":"img1.jpg","width":640,"height":480,"split":"train"}"#;

        let error = parse_ndjson(content).unwrap_err().to_string();

        assert_eq!(
            error,
            "Invalid record on line 2: the first line must be the dataset record, \
             found type \"image\""
        );
    }

    #[test]
    fn get_bboxes_extracts_correctly() {
        let entry = ImageEntry {
            file: "test.jpg".to_string(),
            output_file: None,
            url: String::new(),
            width: Some(640),
            height: Some(480),
            split: "train".to_string(),
            depth: None,
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
    fn get_bboxes_extracts_from_boxes_alias() {
        let entry = ImageEntry {
            file: "test.jpg".to_string(),
            output_file: None,
            url: String::new(),
            width: Some(640),
            height: Some(480),
            split: "train".to_string(),
            depth: None,
            annotations: Some(serde_json::json!({
                "boxes": [[2, 0.25, 0.35, 0.45, 0.55]]
            })),
        };

        let bboxes = entry.get_bboxes();
        assert_eq!(bboxes.len(), 1);
        assert_eq!(bboxes[0].class_id, 2);
        assert!((bboxes[0].x - 0.25).abs() < f64::EPSILON);
        assert!((bboxes[0].y - 0.35).abs() < f64::EPSILON);
    }

    #[test]
    fn get_pose_annotations_parses_new_format() {
        // Format: [class_id, bbox_cx, bbox_cy, bbox_w, bbox_h, kp1_x, kp1_y, kp1_v, ...]
        let entry = ImageEntry {
            file: "test.jpg".to_string(),
            output_file: None,
            url: String::new(),
            width: Some(1200),
            height: Some(800),
            split: "train".to_string(),
            depth: None,
            annotations: Some(serde_json::json!({
                "pose": [[0, 0.5, 0.6, 0.3, 0.4, 0.1, 0.2, 2, 0.3, 0.4, 1, 0.5, 0.6, 0]]
            })),
        };

        let poses = entry.get_pose_annotations(3);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].class_id, 0);
        assert!((poses[0].bbox_x - 0.5).abs() < f64::EPSILON);
        assert!((poses[0].bbox_y - 0.6).abs() < f64::EPSILON);
        assert!((poses[0].bbox_w - 0.3).abs() < f64::EPSILON);
        assert!((poses[0].bbox_h - 0.4).abs() < f64::EPSILON);
        assert_eq!(poses[0].dims, 3);
        assert_eq!(poses[0].num_keypoints(), 3);
        assert_eq!(
            poses[0].keypoints,
            vec![0.1, 0.2, 2.0, 0.3, 0.4, 1.0, 0.5, 0.6, 0.0]
        );
    }

    #[test]
    fn get_pose_annotations_parses_two_dimensional_keypoints() {
        let entry = ImageEntry {
            file: "test.jpg".to_string(),
            output_file: None,
            url: String::new(),
            width: Some(640),
            height: Some(480),
            split: "train".to_string(),
            depth: None,
            annotations: Some(serde_json::json!({
                "pose": [[0, 0.5, 0.6, 0.3, 0.4, 0.1, 0.2]]
            })),
        };

        let poses = entry.get_pose_annotations(2);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].dims, 2);
        assert_eq!(poses[0].num_keypoints(), 1);
        assert_eq!(poses[0].keypoints, vec![0.1, 0.2]);
    }

    #[test]
    fn image_entry_download_key_uses_effective_file_name() {
        let entry = ImageEntry {
            file: "img1.jpg".to_string(),
            output_file: Some("img1__abcd1234.jpg".to_string()),
            url: String::new(),
            width: Some(640),
            height: Some(480),
            split: "val".to_string(),
            depth: None,
            annotations: None,
        };

        let key = image_entry_download_key(&entry);
        assert_eq!(key, image_download_key("valid", "img1__abcd1234.jpg"));
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

    const SPLIT_TEST_DATASET: &str = r#"{"type":"dataset","name":"test","class_names":{}}"#;

    fn parse_split_error(image_lines: &str) -> String {
        let error = parse_ndjson(&format!("{SPLIT_TEST_DATASET}\n{image_lines}")).unwrap_err();
        assert!(matches!(error, ParseError::InvalidSplit(_)), "{error}");
        error.to_string()
    }

    #[test]
    fn accepted_splits_are_kept_unchanged() {
        let content = format!(
            r#"{SPLIT_TEST_DATASET}
{{"type":"image","file":"a.jpg","width":1,"height":1,"split":"train"}}
{{"type":"image","file":"b.jpg","width":1,"height":1,"split":"val"}}
{{"type":"image","file":"c.jpg","width":1,"height":1,"split":"valid"}}
{{"type":"image","file":"d.jpg","width":1,"height":1,"split":"test"}}"#
        );

        let data = parse_ndjson(&content).unwrap();

        let splits: Vec<&str> = data.images.iter().map(|img| img.split.as_str()).collect();
        assert_eq!(splits, ["train", "val", "valid", "test"]);
    }

    #[test]
    fn unknown_split_is_rejected_naming_value_and_image() {
        let error = parse_split_error(
            r#"{"type":"image","file":"a.jpg","width":1,"height":1,"split":"train"}
{"type":"image","file":"b.jpg","width":1,"height":1,"split":"validation"}"#,
        );

        assert_eq!(
            error,
            "Invalid split: image 'b.jpg' has unknown value 'validation'. \
             Accepted splits: train, val, valid, test."
        );
    }

    #[test]
    fn missing_split_is_rejected_instead_of_defaulting_to_train() {
        let error = parse_split_error(r#"{"type":"image","file":"a.jpg","width":1,"height":1}"#);

        assert_eq!(
            error,
            "Invalid split: image 'a.jpg' is missing its split. \
             Accepted splits: train, val, valid, test."
        );
    }

    #[test]
    fn null_or_non_string_split_is_rejected_naming_the_image() {
        let cases = [
            (r#""split":null"#, "is missing its split"),
            (r#""split":1"#, "has unknown value '1'"),
            (r#""split":["train"]"#, r#"has unknown value '["train"]'"#),
        ];
        for (split, expected) in cases {
            let error = parse_split_error(&format!(
                r#"{{"type":"image","file":"a.jpg","width":1,"height":1,{split}}}"#
            ));

            assert_eq!(
                error,
                format!(
                    "Invalid split: image 'a.jpg' {expected}. \
                     Accepted splits: train, val, valid, test."
                ),
                "{split}"
            );
        }
    }

    #[test]
    fn invalid_split_count_is_reported_when_several_lines_are_affected() {
        let error = parse_split_error(
            r#"{"type":"image","file":"a.jpg","width":1,"height":1,"split":"train"}
{"type":"image","file":"b.jpg","width":1,"height":1,"split":"validation"}
{"type":"image","file":"c.jpg","width":1,"height":1}
{"type":"image","file":"d.jpg","width":1,"height":1,"split":"Train"}"#,
        );

        assert_eq!(
            error,
            "Invalid split: image 'b.jpg' has unknown value 'validation' (3 image lines affected). \
             Accepted splits: train, val, valid, test."
        );
    }

    #[test]
    fn parse_semantic_ndjson_reads_polygon_segments() {
        let content = r#"{"type":"dataset","task":"semantic","name":"city","class_names":{"0":"road"}}
{"type":"image","file":"street.jpg","width":640,"height":640,"split":"train","annotations":{"segments":[[0,0.1,0.1,0.2,0.1,0.2,0.2]]}}"#;

        let data = parse_ndjson(content).unwrap();
        assert_eq!(data.metadata.task, "semantic");
        let segments = data.images[0].get_segment_annotations();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].class_id, 0);
        assert_eq!(segments[0].points.len(), 3);
    }

    #[test]
    fn get_obb_annotations_parses_correctly() {
        let entry = ImageEntry {
            file: "test.jpg".to_string(),
            output_file: None,
            url: String::new(),
            width: Some(640),
            height: Some(640),
            split: "train".to_string(),
            depth: None,
            annotations: Some(serde_json::json!({
                "obb": [[0, 0.3, 0.6, 0.2, 0.7, 0.25, 0.7, 0.29, 0.63]]
            })),
        };

        let obbs = entry.get_obb_annotations();
        assert_eq!(obbs.len(), 1);
        assert_eq!(obbs[0].class_id, 0);
        assert!((obbs[0].points[0].0 - 0.3).abs() < f64::EPSILON);
        assert!((obbs[0].points[0].1 - 0.6).abs() < f64::EPSILON);
        assert!((obbs[0].points[3].0 - 0.29).abs() < f64::EPSILON);
        assert!((obbs[0].points[3].1 - 0.63).abs() < f64::EPSILON);
    }

    #[test]
    fn get_obb_annotations_rejects_wrong_length() {
        let entry = ImageEntry {
            file: "test.jpg".to_string(),
            output_file: None,
            url: String::new(),
            width: Some(640),
            height: Some(640),
            split: "train".to_string(),
            depth: None,
            annotations: Some(serde_json::json!({
                "obb": [[0, 0.3, 0.6, 0.2, 0.7, 0.25]]
            })),
        };

        let obbs = entry.get_obb_annotations();
        assert!(obbs.is_empty());
    }

    fn depth_dataset(fields: &str) -> Result<NDJSONData, ParseError> {
        parse_ndjson(&format!(
            r#"{{"type":"dataset","task":"depth","name":"rooms"{fields}}}"#
        ))
    }

    #[test]
    fn depth_scale_defaults_to_1000_and_keeps_any_positive_number() {
        for (fields, scale) in [
            ("", 1000.0),
            (r#","depth_scale":256"#, 256.0),
            (r#","depth_scale":0.5"#, 0.5),
        ] {
            assert_eq!(
                depth_dataset(fields).unwrap().metadata.depth_scale,
                scale,
                "{fields}"
            );
        }
    }

    #[test]
    fn invalid_depth_scale_is_rejected_naming_the_value() {
        for value in ["0", "-1", "true", r#""1000""#, "null", "[1000]"] {
            let error = depth_dataset(&format!(r#","depth_scale":{value}"#))
                .unwrap_err()
                .to_string();

            assert_eq!(
                error,
                format!(
                    "Invalid record on line 1: depth_scale must be a positive number, got {value}"
                )
            );
        }
    }

    #[test]
    fn depth_scale_is_ignored_by_other_tasks() {
        let data = parse_ndjson(r#"{"type":"dataset","task":"detect","depth_scale":true}"#);

        assert!(data.is_ok());
    }
}
