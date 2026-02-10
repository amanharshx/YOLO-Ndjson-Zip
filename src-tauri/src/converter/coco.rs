use super::{get_class_list, Converter};
use crate::parser::{image_download_key, ImageEntry, NDJSONData};
use chrono::Utc;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
struct CocoInfo {
    description: String,
    url: String,
    version: String,
    year: i32,
    contributor: String,
    date_created: String,
}

#[derive(Serialize)]
struct CocoLicense {
    id: i32,
    name: String,
    url: String,
}

#[derive(Serialize)]
struct CocoCategory {
    id: i32,
    name: String,
    supercategory: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    keypoints: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    skeleton: Option<Vec<[i32; 2]>>,
}

#[derive(Serialize)]
struct CocoImage {
    id: i32,
    file_name: String,
    width: i32,
    height: i32,
    license: i32,
    date_captured: String,
}

#[derive(Serialize)]
struct CocoAnnotation {
    id: i32,
    image_id: i32,
    category_id: i32,
    bbox: [f64; 4],
    area: f64,
    iscrowd: i32,
    segmentation: Vec<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keypoints: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_keypoints: Option<i32>,
}

#[derive(Serialize)]
struct CocoFormat {
    info: CocoInfo,
    licenses: Vec<CocoLicense>,
    categories: Vec<CocoCategory>,
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
}

pub struct CocoConverter;

impl CocoConverter {
    pub fn new() -> Self {
        Self
    }

    fn create_coco_json(&self, images: &[&ImageEntry], data: &NDJSONData, _split: &str) -> String {
        let class_names = get_class_list(data);
        let now = Utc::now();
        let task = &data.metadata.task;
        let kpt_shape = data.metadata.kpt_shape.as_deref();
        let is_pose = task == "pose";
        let num_kpts = kpt_shape.and_then(|s| s.first()).copied().unwrap_or(17) as usize;

        let mut coco = CocoFormat {
            info: CocoInfo {
                description: if data.metadata.name.is_empty() {
                    "Converted from NDJSON".to_string()
                } else {
                    data.metadata.name.clone()
                },
                url: data.metadata.url.clone(),
                version: data.metadata.version.to_string(),
                year: now.format("%Y").to_string().parse().unwrap_or(2024),
                contributor: "YOLO NDJSON Converter".to_string(),
                date_created: now.to_rfc3339(),
            },
            licenses: vec![CocoLicense {
                id: 1,
                name: "Unknown".to_string(),
                url: String::new(),
            }],
            categories: class_names
                .iter()
                .enumerate()
                .map(|(i, name)| CocoCategory {
                    id: i as i32,
                    name: name.clone(),
                    supercategory: String::new(),
                    keypoints: if is_pose {
                        Some((0..num_kpts).map(|k| format!("keypoint_{}", k)).collect())
                    } else {
                        None
                    },
                    skeleton: if is_pose { Some(Vec::new()) } else { None },
                })
                .collect(),
            images: Vec::new(),
            annotations: Vec::new(),
        };

        let mut annotation_id = 1;

        for (img_idx, img) in images.iter().enumerate() {
            let img_id = (img_idx + 1) as i32;

            coco.images.push(CocoImage {
                id: img_id,
                file_name: img.file.clone(),
                width: img.width,
                height: img.height,
                license: 1,
                date_captured: now.to_rfc3339(),
            });

            match task.as_str() {
                "segment" => {
                    for seg in img.get_segment_annotations() {
                        if seg.points.is_empty() {
                            continue;
                        }
                        let mut abs_points: Vec<f64> = Vec::new();
                        let mut min_x = f64::MAX;
                        let mut min_y = f64::MAX;
                        let mut max_x = f64::MIN;
                        let mut max_y = f64::MIN;

                        for (x, y) in &seg.points {
                            let abs_x = x * img.width as f64;
                            let abs_y = y * img.height as f64;
                            abs_points.push(abs_x);
                            abs_points.push(abs_y);
                            min_x = min_x.min(abs_x);
                            min_y = min_y.min(abs_y);
                            max_x = max_x.max(abs_x);
                            max_y = max_y.max(abs_y);
                        }

                        let w = max_x - min_x;
                        let h = max_y - min_y;

                        coco.annotations.push(CocoAnnotation {
                            id: annotation_id,
                            image_id: img_id,
                            category_id: seg.class_id,
                            bbox: [min_x, min_y, w, h],
                            area: w * h,
                            iscrowd: 0,
                            segmentation: vec![abs_points],
                            keypoints: None,
                            num_keypoints: None,
                        });
                        annotation_id += 1;
                    }
                }
                "pose" => {
                    for pose in img.get_pose_annotations(kpt_shape) {
                        let x_min = (pose.bbox_x - pose.bbox_w / 2.0) * img.width as f64;
                        let y_min = (pose.bbox_y - pose.bbox_h / 2.0) * img.height as f64;
                        let w = pose.bbox_w * img.width as f64;
                        let h = pose.bbox_h * img.height as f64;

                        let mut kps: Vec<f64> = Vec::new();
                        let mut visible_count = 0;
                        for (kp_x, kp_y) in &pose.keypoints {
                            let abs_x = kp_x * img.width as f64;
                            let abs_y = kp_y * img.height as f64;
                            let v = if *kp_x > 0.0 || *kp_y > 0.0 { 2.0 } else { 0.0 };
                            if v > 0.0 {
                                visible_count += 1;
                            }
                            kps.push(abs_x);
                            kps.push(abs_y);
                            kps.push(v);
                        }

                        coco.annotations.push(CocoAnnotation {
                            id: annotation_id,
                            image_id: img_id,
                            category_id: pose.class_id,
                            bbox: [x_min, y_min, w, h],
                            area: w * h,
                            iscrowd: 0,
                            segmentation: Vec::new(),
                            keypoints: Some(kps),
                            num_keypoints: Some(visible_count),
                        });
                        annotation_id += 1;
                    }
                }
                _ => {
                    // Detection (default)
                    for bbox in img.get_bboxes() {
                        let x_min = (bbox.x - bbox.width / 2.0) * img.width as f64;
                        let y_min = (bbox.y - bbox.height / 2.0) * img.height as f64;
                        let w = bbox.width * img.width as f64;
                        let h = bbox.height * img.height as f64;

                        coco.annotations.push(CocoAnnotation {
                            id: annotation_id,
                            image_id: img_id,
                            category_id: bbox.class_id,
                            bbox: [x_min, y_min, w, h],
                            area: w * h,
                            iscrowd: 0,
                            segmentation: Vec::new(),
                            keypoints: None,
                            num_keypoints: None,
                        });
                        annotation_id += 1;
                    }
                }
            }
        }

        serde_json::to_string_pretty(&coco).unwrap_or_default()
    }
}

impl Converter for CocoConverter {
    fn convert(
        &self,
        data: &NDJSONData,
        downloaded_images: &HashMap<String, Vec<u8>>,
    ) -> HashMap<String, Vec<u8>> {
        let mut files: HashMap<String, Vec<u8>> = HashMap::new();

        let splits = [
            ("train", data.train_images()),
            ("valid", data.valid_images()),
            ("test", data.test_images()),
        ];

        for (split, images) in &splits {
            if images.is_empty() {
                continue;
            }

            // Add images to {split}/ directory
            for img in images {
                if let Some(image_data) =
                    downloaded_images.get(&image_download_key(split, &img.file))
                {
                    files.insert(format!("{}/{}", split, img.file), image_data.clone());
                }
            }

            // Create JSON at {split}/_annotations.coco.json
            let coco_json = self.create_coco_json(images, data, split);
            files.insert(
                format!("{}/_annotations.coco.json", split),
                coco_json.into_bytes(),
            );
        }

        files
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{image_download_key, DatasetMetadata};
    use serde_json::json;

    #[test]
    fn convert_uses_split_aware_download_keys() {
        let data = NDJSONData {
            metadata: DatasetMetadata {
                r#type: "dataset".to_string(),
                task: "detect".to_string(),
                name: "test".to_string(),
                description: String::new(),
                bytes: 0,
                url: String::new(),
                class_names: HashMap::from([("0".to_string(), "animal".to_string())]),
                kpt_shape: None,
                version: 1,
            },
            images: vec![
                ImageEntry {
                    r#type: "image".to_string(),
                    file: "img1.jpg".to_string(),
                    url: String::new(),
                    width: 640,
                    height: 480,
                    split: "train".to_string(),
                    annotations: Some(json!({
                        "bboxes": [[0, 0.5, 0.5, 0.2, 0.2]]
                    })),
                },
                ImageEntry {
                    r#type: "image".to_string(),
                    file: "img1.jpg".to_string(),
                    url: String::new(),
                    width: 640,
                    height: 480,
                    split: "val".to_string(),
                    annotations: Some(json!({
                        "bboxes": [[0, 0.4, 0.4, 0.3, 0.3]]
                    })),
                },
            ],
        };

        let converter = CocoConverter::new();
        let mut downloaded_images = HashMap::new();
        downloaded_images.insert(image_download_key("train", "img1.jpg"), vec![1]);
        downloaded_images.insert(image_download_key("valid", "img1.jpg"), vec![2]);

        let files = converter.convert(&data, &downloaded_images);

        assert_eq!(files.get("train/img1.jpg"), Some(&vec![1]));
        assert_eq!(files.get("valid/img1.jpg"), Some(&vec![2]));
    }
}
