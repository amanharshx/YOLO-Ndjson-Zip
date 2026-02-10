use super::{get_class_names, Converter};
use crate::parser::{image_download_key, ImageEntry, NDJSONData};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use std::collections::HashMap;
use std::io::Cursor;

pub struct PascalVocConverter;

impl PascalVocConverter {
    pub fn new() -> Self {
        Self
    }

    fn create_voc_xml(
        &self,
        img: &ImageEntry,
        class_names: &HashMap<i32, String>,
        task: &str,
    ) -> String {
        let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 2);

        // XML declaration
        writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("utf-8"), None)))
            .ok();

        // Root element
        writer
            .write_event(Event::Start(BytesStart::new("annotation")))
            .ok();

        let image_file = img.effective_file_name();

        // folder (empty)
        Self::write_element(&mut writer, "folder", "");

        // filename
        Self::write_element(&mut writer, "filename", image_file);

        // path (just the filename)
        Self::write_element(&mut writer, "path", image_file);

        // source
        writer
            .write_event(Event::Start(BytesStart::new("source")))
            .ok();
        Self::write_element(&mut writer, "database", "NDJSON Convert");
        writer.write_event(Event::End(BytesEnd::new("source"))).ok();

        // size
        writer
            .write_event(Event::Start(BytesStart::new("size")))
            .ok();
        Self::write_element(&mut writer, "width", &img.width.to_string());
        Self::write_element(&mut writer, "height", &img.height.to_string());
        Self::write_element(&mut writer, "depth", "3");
        writer.write_event(Event::End(BytesEnd::new("size"))).ok();

        // segmented flag
        Self::write_element(
            &mut writer,
            "segmented",
            if task == "segment" { "1" } else { "0" },
        );

        if task == "segment" {
            // Segmentation: derive bounding boxes from polygon vertices
            for seg in img.get_segment_annotations() {
                if seg.points.is_empty() {
                    continue;
                }

                writer
                    .write_event(Event::Start(BytesStart::new("object")))
                    .ok();

                let class_name = class_names
                    .get(&seg.class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", seg.class_id));

                Self::write_element(&mut writer, "name", &class_name);
                Self::write_element(&mut writer, "pose", "Unspecified");
                Self::write_element(&mut writer, "truncated", "0");
                Self::write_element(&mut writer, "difficult", "0");

                // Compute bounding box from polygon vertices
                let mut min_x = f64::MAX;
                let mut min_y = f64::MAX;
                let mut max_x = f64::MIN;
                let mut max_y = f64::MIN;

                for (x, y) in &seg.points {
                    let abs_x = x * img.width as f64;
                    let abs_y = y * img.height as f64;
                    min_x = min_x.min(abs_x);
                    min_y = min_y.min(abs_y);
                    max_x = max_x.max(abs_x);
                    max_y = max_y.max(abs_y);
                }

                let xmin = min_x.round() as i32;
                let ymin = min_y.round() as i32;
                let xmax = max_x.round() as i32;
                let ymax = max_y.round() as i32;

                writer
                    .write_event(Event::Start(BytesStart::new("bndbox")))
                    .ok();
                Self::write_element(&mut writer, "xmin", &xmin.max(0).to_string());
                Self::write_element(&mut writer, "ymin", &ymin.max(0).to_string());
                Self::write_element(&mut writer, "xmax", &xmax.min(img.width).to_string());
                Self::write_element(&mut writer, "ymax", &ymax.min(img.height).to_string());
                writer.write_event(Event::End(BytesEnd::new("bndbox"))).ok();

                writer.write_event(Event::End(BytesEnd::new("object"))).ok();
            }
        } else {
            // Detection (default): use bounding boxes directly
            for bbox in img.get_bboxes() {
                writer
                    .write_event(Event::Start(BytesStart::new("object")))
                    .ok();

                let class_name = class_names
                    .get(&bbox.class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", bbox.class_id));

                Self::write_element(&mut writer, "name", &class_name);
                Self::write_element(&mut writer, "pose", "Unspecified");
                Self::write_element(&mut writer, "truncated", "0");
                Self::write_element(&mut writer, "difficult", "0");

                // Convert normalized coords to absolute Pascal VOC format
                // VOC uses [xmin, ymin, xmax, ymax] in pixels
                let xmin = ((bbox.x - bbox.width / 2.0) * img.width as f64).round() as i32;
                let ymin = ((bbox.y - bbox.height / 2.0) * img.height as f64).round() as i32;
                let xmax = ((bbox.x + bbox.width / 2.0) * img.width as f64).round() as i32;
                let ymax = ((bbox.y + bbox.height / 2.0) * img.height as f64).round() as i32;

                writer
                    .write_event(Event::Start(BytesStart::new("bndbox")))
                    .ok();
                Self::write_element(&mut writer, "xmin", &xmin.max(0).to_string());
                Self::write_element(&mut writer, "ymin", &ymin.max(0).to_string());
                Self::write_element(&mut writer, "xmax", &xmax.min(img.width).to_string());
                Self::write_element(&mut writer, "ymax", &ymax.min(img.height).to_string());
                writer.write_event(Event::End(BytesEnd::new("bndbox"))).ok();

                writer.write_event(Event::End(BytesEnd::new("object"))).ok();
            }
        }

        writer
            .write_event(Event::End(BytesEnd::new("annotation")))
            .ok();

        let result = writer.into_inner().into_inner();
        String::from_utf8(result).unwrap_or_default()
    }

    fn write_element(writer: &mut Writer<Cursor<Vec<u8>>>, name: &str, value: &str) {
        writer.write_event(Event::Start(BytesStart::new(name))).ok();
        writer.write_event(Event::Text(BytesText::new(value))).ok();
        writer.write_event(Event::End(BytesEnd::new(name))).ok();
    }
}

impl Converter for PascalVocConverter {
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

            if task == "classify" {
                // Classification: folder structure {split}/{class_name}/{file}
                for img in images.iter() {
                    let image_file = img.effective_file_name();
                    let classifications = img.get_classifications();
                    if let Some(&class_id) = classifications.first() {
                        let class_name = class_names
                            .get(&class_id)
                            .cloned()
                            .unwrap_or_else(|| format!("class_{}", class_id));

                        if let Some(image_data) =
                            downloaded_images.get(&image_download_key(split, image_file))
                        {
                            files.insert(
                                format!("{}/{}/{}", split, class_name, image_file),
                                image_data.clone(),
                            );
                        }
                    }
                }
            } else {
                // Detection or Segmentation: create XML annotations
                for img in images.iter() {
                    let image_file = img.effective_file_name();
                    let xml_content = self.create_voc_xml(img, &class_names, task);
                    let xml_filename = image_file
                        .rsplit_once('.')
                        .map(|(name, _)| name)
                        .unwrap_or(image_file);
                    files.insert(
                        format!("{}/{}.xml", split, xml_filename),
                        xml_content.into_bytes(),
                    );

                    if let Some(image_data) =
                        downloaded_images.get(&image_download_key(split, image_file))
                    {
                        files.insert(format!("{}/{}", split, image_file), image_data.clone());
                    }
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
                version: 1,
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

        let converter = PascalVocConverter::new();
        let mut downloaded_images = HashMap::new();
        downloaded_images.insert(image_download_key("train", "img1.jpg"), vec![1]);
        downloaded_images.insert(image_download_key("valid", "img1.jpg"), vec![2]);

        let files = converter.convert(&data, &downloaded_images);

        assert_eq!(files.get("train/img1.jpg"), Some(&vec![1]));
        assert_eq!(files.get("valid/img1.jpg"), Some(&vec![2]));
    }

    #[test]
    fn convert_uses_effective_file_names_for_duplicate_outputs() {
        let data = NDJSONData {
            metadata: DatasetMetadata {
                r#type: "dataset".to_string(),
                task: "detect".to_string(),
                name: "test".to_string(),
                description: String::new(),
                bytes: 0,
                url: String::new(),
                class_names: HashMap::from([("0".to_string(), "heic".to_string())]),
                kpt_shape: None,
                version: 1,
            },
            images: vec![
                ImageEntry {
                    r#type: "image".to_string(),
                    file: "Frame_98.jpg".to_string(),
                    output_file: None,
                    url: "https://cdn.example/a.jpg".to_string(),
                    width: 640,
                    height: 480,
                    split: "train".to_string(),
                    annotations: Some(json!({
                        "boxes": [[0, 0.5, 0.5, 0.2, 0.2]]
                    })),
                },
                ImageEntry {
                    r#type: "image".to_string(),
                    file: "Frame_98.jpg".to_string(),
                    output_file: Some("Frame_98__abcd1234.jpg".to_string()),
                    url: "https://cdn.example/b.jpg".to_string(),
                    width: 640,
                    height: 480,
                    split: "train".to_string(),
                    annotations: Some(json!({
                        "boxes": [[0, 0.4, 0.4, 0.3, 0.3]]
                    })),
                },
            ],
        };

        let converter = PascalVocConverter::new();
        let mut downloaded_images = HashMap::new();
        downloaded_images.insert(image_download_key("train", "Frame_98.jpg"), vec![1]);
        downloaded_images.insert(
            image_download_key("train", "Frame_98__abcd1234.jpg"),
            vec![2],
        );

        let files = converter.convert(&data, &downloaded_images);

        assert_eq!(files.get("train/Frame_98.jpg"), Some(&vec![1]));
        assert_eq!(files.get("train/Frame_98__abcd1234.jpg"), Some(&vec![2]));
        assert!(files.contains_key("train/Frame_98.xml"));
        assert!(files.contains_key("train/Frame_98__abcd1234.xml"));
    }
}
