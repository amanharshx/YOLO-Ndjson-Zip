use super::{get_class_names, Converter};
use crate::parser::{ImageEntry, NDJSONData};
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
                    image: img.file.clone(),
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
                    image: img.file.clone(),
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

            let json = if task == "classify" {
                self.create_createml_classification_json(images, &class_names)
            } else {
                self.create_createml_json(images, &class_names)
            };
            files.insert(format!("{}.json", split), json.into_bytes());

            // Add images
            for img in images {
                if let Some(image_data) = downloaded_images.get(&img.file) {
                    files.insert(format!("{}/{}", split, img.file), image_data.clone());
                }
            }
        }

        files
    }
}
