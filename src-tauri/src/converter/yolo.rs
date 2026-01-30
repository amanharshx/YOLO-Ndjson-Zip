use super::{get_class_list, get_class_names, Converter};
use crate::parser::{ImageEntry, NDJSONData};
use std::collections::HashMap;

pub struct YoloConverter {
    darknet: bool,
}

impl YoloConverter {
    pub fn new() -> Self {
        Self { darknet: false }
    }

    pub fn new_darknet() -> Self {
        Self { darknet: true }
    }

    fn create_data_yaml(&self, data: &NDJSONData) -> String {
        let class_names = get_class_list(data);
        let task = &data.metadata.task;

        let mut yaml = String::new();
        yaml.push_str("path: .\n");
        yaml.push_str("train: train/images\n");
        yaml.push_str("val: valid/images\n");
        yaml.push_str("test: test/images\n");
        yaml.push_str(&format!("nc: {}\n", class_names.len()));
        yaml.push_str("names:\n");

        for (i, name) in class_names.iter().enumerate() {
            yaml.push_str(&format!("  {}: {}\n", i, name));
        }

        if task == "pose" {
            if let Some(kpt_shape) = &data.metadata.kpt_shape {
                yaml.push_str(&format!(
                    "kpt_shape: [{}, {}]\n",
                    kpt_shape[0],
                    kpt_shape.get(1).unwrap_or(&2)
                ));
            }
        }

        yaml
    }

    fn create_detection_label(&self, img: &ImageEntry) -> String {
        img.get_bboxes()
            .iter()
            .map(|bbox| {
                format!(
                    "{} {:.6} {:.6} {:.6} {:.6}",
                    bbox.class_id, bbox.x, bbox.y, bbox.width, bbox.height
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn create_pose_label(&self, img: &ImageEntry, kpt_shape: Option<&[i32]>) -> String {
        img.get_pose_annotations(kpt_shape)
            .iter()
            .map(|pose| {
                let mut parts = vec![
                    pose.class_id.to_string(),
                    format!("{:.6}", pose.bbox_x),
                    format!("{:.6}", pose.bbox_y),
                    format!("{:.6}", pose.bbox_w),
                    format!("{:.6}", pose.bbox_h),
                ];

                for (kp_x, kp_y) in &pose.keypoints {
                    let visibility = if *kp_x > 0.0 || *kp_y > 0.0 { 2 } else { 0 };
                    parts.push(format!("{:.6}", kp_x));
                    parts.push(format!("{:.6}", kp_y));
                    parts.push(visibility.to_string());
                }

                parts.join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn create_segment_label(&self, img: &ImageEntry) -> String {
        img.get_segment_annotations()
            .iter()
            .map(|seg| {
                let mut parts = vec![seg.class_id.to_string()];
                for (x, y) in &seg.points {
                    parts.push(format!("{:.6}", x));
                    parts.push(format!("{:.6}", y));
                }
                parts.join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Converter for YoloConverter {
    fn convert(
        &self,
        data: &NDJSONData,
        downloaded_images: &HashMap<String, Vec<u8>>,
    ) -> HashMap<String, Vec<u8>> {
        let mut files: HashMap<String, Vec<u8>> = HashMap::new();
        let task = &data.metadata.task;

        if self.darknet {
            // Darknet mode: _darknet.labels instead of data.yaml/classes.txt
            let class_list = get_class_list(data);
            files.insert(
                "_darknet.labels".to_string(),
                class_list.join("\n").into_bytes(),
            );
        } else {
            // Standard YOLO mode
            files.insert(
                "data.yaml".to_string(),
                self.create_data_yaml(data).into_bytes(),
            );
            let class_list = get_class_list(data);
            files.insert(
                "classes.txt".to_string(),
                class_list.join("\n").into_bytes(),
            );
        }

        let kpt_shape = data.metadata.kpt_shape.as_deref();

        // Process images by split
        let splits = [
            ("train", data.train_images()),
            ("valid", data.valid_images()),
            ("test", data.test_images()),
        ];

        for (split, images) in splits {
            for img in images {
                // Create label file
                let label_content = match task.as_str() {
                    "pose" => self.create_pose_label(img, kpt_shape),
                    "segment" => self.create_segment_label(img),
                    "classify" => {
                        // For classification, we use folder structure
                        let classifications = img.get_classifications();
                        if let Some(&class_id) = classifications.first() {
                            let class_names = get_class_names(data);
                            let class_name = class_names
                                .get(&class_id)
                                .cloned()
                                .unwrap_or_else(|| format!("class_{}", class_id));

                            if let Some(image_data) = downloaded_images.get(&img.file) {
                                files.insert(
                                    format!("{}/{}/{}", split, class_name, img.file),
                                    image_data.clone(),
                                );
                            }
                        }
                        continue;
                    }
                    _ => self.create_detection_label(img),
                };

                let label_filename = img
                    .file
                    .rsplit_once('.')
                    .map(|(name, _)| name)
                    .unwrap_or(&img.file);

                if self.darknet {
                    // Darknet: flat structure, images + labels side by side in {split}/
                    files.insert(
                        format!("{}/{}.txt", split, label_filename),
                        label_content.into_bytes(),
                    );
                    if let Some(image_data) = downloaded_images.get(&img.file) {
                        files.insert(format!("{}/{}", split, img.file), image_data.clone());
                    }
                } else {
                    // Standard YOLO: {split}/labels/ and {split}/images/
                    files.insert(
                        format!("{}/labels/{}.txt", split, label_filename),
                        label_content.into_bytes(),
                    );
                    if let Some(image_data) = downloaded_images.get(&img.file) {
                        files.insert(format!("{}/images/{}", split, img.file), image_data.clone());
                    }
                }
            }
        }

        files
    }
}
