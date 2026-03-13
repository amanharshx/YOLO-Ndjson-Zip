use super::{get_class_names, Converter};
use crate::parser::{image_download_key, ImageEntry, NDJSONData};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
struct CreateMlCoordinates {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

#[derive(Serialize)]
struct CreateMlAnnotation {
    label: String,
    coordinates: CreateMlCoordinates,
}

#[derive(Serialize)]
struct CreateMlImage {
    image: String,
    #[serde(rename = "imageURL")]
    image_url: String,
    annotations: Vec<CreateMlAnnotation>,
}

#[derive(Serialize)]
struct CreateMlClassification {
    image: String,
    label: String,
}

pub struct CreateMlConverter;

impl CreateMlConverter {
    pub fn new() -> Self {
        Self
    }

    fn create_createml_json(
        &self,
        images: &[&ImageEntry],
        class_names: &HashMap<i32, String>,
    ) -> String {
        let result: Vec<CreateMlImage> = images
            .iter()
            .map(|img| {
                let annotations = img
                    .get_bboxes()
                    .iter()
                    .map(|bbox| {
                        let class_name = class_names
                            .get(&bbox.class_id)
                            .cloned()
                            .unwrap_or_else(|| format!("class_{}", bbox.class_id));

                        CreateMlAnnotation {
                            label: class_name,
                            coordinates: CreateMlCoordinates {
                                x: bbox.x * img.width as f64,
                                y: bbox.y * img.height as f64,
                                width: bbox.width * img.width as f64,
                                height: bbox.height * img.height as f64,
                            },
                        }
                    })
                    .collect();

                CreateMlImage {
                    image: img.effective_file_name().to_string(),
                    image_url: img.url.clone(),
                    annotations,
                }
            })
            .collect();

        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    fn create_createml_obb_json(
        &self,
        images: &[&ImageEntry],
        class_names: &HashMap<i32, String>,
    ) -> String {
        let result: Vec<CreateMlImage> = images
            .iter()
            .map(|img| {
                let annotations = img
                    .get_obb_annotations()
                    .iter()
                    .map(|obb| {
                        let class_name = class_names
                            .get(&obb.class_id)
                            .cloned()
                            .unwrap_or_else(|| format!("class_{}", obb.class_id));

                        // Derive axis-aligned bbox from OBB corners
                        let xs: Vec<f64> = obb.points.iter().map(|(x, _)| *x).collect();
                        let ys: Vec<f64> = obb.points.iter().map(|(_, y)| *y).collect();
                        let min_x = xs.iter().cloned().fold(f64::MAX, f64::min);
                        let max_x = xs.iter().cloned().fold(f64::MIN, f64::max);
                        let min_y = ys.iter().cloned().fold(f64::MAX, f64::min);
                        let max_y = ys.iter().cloned().fold(f64::MIN, f64::max);
                        let cx = (min_x + max_x) / 2.0 * img.width as f64;
                        let cy = (min_y + max_y) / 2.0 * img.height as f64;
                        let w = (max_x - min_x) * img.width as f64;
                        let h = (max_y - min_y) * img.height as f64;

                        CreateMlAnnotation {
                            label: class_name,
                            coordinates: CreateMlCoordinates {
                                x: cx,
                                y: cy,
                                width: w,
                                height: h,
                            },
                        }
                    })
                    .collect();

                CreateMlImage {
                    image: img.effective_file_name().to_string(),
                    image_url: img.url.clone(),
                    annotations,
                }
            })
            .collect();

        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    fn create_createml_classification_json(
        &self,
        images: &[&ImageEntry],
        class_names: &HashMap<i32, String>,
    ) -> String {
        let result: Vec<CreateMlClassification> = images
            .iter()
            .filter_map(|img| {
                let classifications = img.get_classifications();
                let class_id = classifications.first()?;
                let class_name = class_names
                    .get(class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", class_id));
                Some(CreateMlClassification {
                    image: img.effective_file_name().to_string(),
                    label: class_name,
                })
            })
            .collect();

        serde_json::to_string_pretty(&result).unwrap_or_default()
    }
}

impl Converter for CreateMlConverter {
    fn convert(
        &self,
        data: &NDJSONData,
        downloaded_images: &HashMap<String, Vec<u8>>,
    ) -> HashMap<String, Vec<u8>> {
        let mut files: HashMap<String, Vec<u8>> = HashMap::new();
        let class_names = get_class_names(data);
        let task = &data.metadata.task;

        let splits = [
            ("train", data.train_images()),
            ("valid", data.valid_images()),
            ("test", data.test_images()),
        ];

        for (split, images) in &splits {
            if images.is_empty() {
                continue;
            }

            let json = match task.as_str() {
                "classify" => self.create_createml_classification_json(images, &class_names),
                "obb" => self.create_createml_obb_json(images, &class_names),
                _ => self.create_createml_json(images, &class_names),
            };
            files.insert(format!("{}.json", split), json.into_bytes());

            // Add images
            for img in images {
                let image_file = img.effective_file_name();
                if let Some(image_data) =
                    downloaded_images.get(&image_download_key(split, image_file))
                {
                    files.insert(format!("{}/{}", split, image_file), image_data.clone());
                }
            }
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
                version: "1".to_string(),
            },
            images: vec![
                ImageEntry {
                    r#type: "image".to_string(),
                    file: "img1.jpg".to_string(),
                    output_file: None,
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
                    output_file: None,
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

        let converter = CreateMlConverter::new();
        let mut downloaded_images = HashMap::new();
        downloaded_images.insert(image_download_key("train", "img1.jpg"), vec![1]);
        downloaded_images.insert(image_download_key("valid", "img1.jpg"), vec![2]);

        let files = converter.convert(&data, &downloaded_images);

        assert_eq!(files.get("train/img1.jpg"), Some(&vec![1]));
        assert_eq!(files.get("valid/img1.jpg"), Some(&vec![2]));
    }

    #[test]
    fn convert_uses_effective_file_name_in_images_and_json() {
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
                version: "1".to_string(),
            },
            images: vec![ImageEntry {
                r#type: "image".to_string(),
                file: "img1.jpg".to_string(),
                output_file: Some("img1__abcd1234.jpg".to_string()),
                url: String::new(),
                width: 640,
                height: 480,
                split: "train".to_string(),
                annotations: Some(json!({
                    "boxes": [[0, 0.5, 0.5, 0.2, 0.2]]
                })),
            }],
        };

        let converter = CreateMlConverter::new();
        let mut downloaded_images = HashMap::new();
        downloaded_images.insert(image_download_key("train", "img1__abcd1234.jpg"), vec![1]);

        let files = converter.convert(&data, &downloaded_images);

        assert_eq!(files.get("train/img1__abcd1234.jpg"), Some(&vec![1]));
        let data: serde_json::Value =
            serde_json::from_slice(files.get("train.json").unwrap()).unwrap();
        assert_eq!(
            data.as_array()
                .and_then(|arr| arr.first())
                .and_then(|item| item.get("image"))
                .and_then(|v| v.as_str()),
            Some("img1__abcd1234.jpg")
        );
    }
}
