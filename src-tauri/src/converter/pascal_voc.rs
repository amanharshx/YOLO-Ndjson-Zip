use super::{class_name, classify_folder_names, get_class_names, image_size, Converter};
use crate::parser::{image_download_key, output_stem, ImageEntry, NDJSONData};
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
        let (width, height) = image_size(img);

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
        Self::write_element(&mut writer, "width", &width.to_string());
        Self::write_element(&mut writer, "height", &height.to_string());
        Self::write_element(&mut writer, "depth", "3");
        writer.write_event(Event::End(BytesEnd::new("size"))).ok();

        // No mask PNGs are exported, so never claim segmentation.
        Self::write_element(&mut writer, "segmented", "0");

        if task == "obb" {
            // OBB: derive axis-aligned bounding boxes from oriented corners
            for obb in img.get_obb_annotations() {
                writer
                    .write_event(Event::Start(BytesStart::new("object")))
                    .ok();

                Self::write_element(&mut writer, "name", &class_name(class_names, obb.class_id));
                Self::write_element(&mut writer, "pose", "Unspecified");
                Self::write_element(&mut writer, "truncated", "0");
                Self::write_element(&mut writer, "difficult", "0");

                let mut min_x = f64::MAX;
                let mut min_y = f64::MAX;
                let mut max_x = f64::MIN;
                let mut max_y = f64::MIN;

                for (x, y) in &obb.points {
                    let abs_x = x * width as f64;
                    let abs_y = y * height as f64;
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
                Self::write_element(&mut writer, "xmax", &xmax.min(width).to_string());
                Self::write_element(&mut writer, "ymax", &ymax.min(height).to_string());
                writer.write_event(Event::End(BytesEnd::new("bndbox"))).ok();

                writer.write_event(Event::End(BytesEnd::new("object"))).ok();
            }
        } else if task == "segment" || task == "semantic" {
            // Segmentation: derive bounding boxes from polygon vertices
            for seg in img.get_segment_annotations() {
                if seg.points.is_empty() {
                    continue;
                }

                writer
                    .write_event(Event::Start(BytesStart::new("object")))
                    .ok();

                Self::write_element(&mut writer, "name", &class_name(class_names, seg.class_id));
                Self::write_element(&mut writer, "pose", "Unspecified");
                Self::write_element(&mut writer, "truncated", "0");
                Self::write_element(&mut writer, "difficult", "0");

                // Compute bounding box from polygon vertices
                let mut min_x = f64::MAX;
                let mut min_y = f64::MAX;
                let mut max_x = f64::MIN;
                let mut max_y = f64::MIN;

                for (x, y) in &seg.points {
                    let abs_x = x * width as f64;
                    let abs_y = y * height as f64;
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
                Self::write_element(&mut writer, "xmax", &xmax.min(width).to_string());
                Self::write_element(&mut writer, "ymax", &ymax.min(height).to_string());
                writer.write_event(Event::End(BytesEnd::new("bndbox"))).ok();

                writer.write_event(Event::End(BytesEnd::new("object"))).ok();
            }
        } else {
            // Detection (default): use bounding boxes directly
            for bbox in img.get_bboxes() {
                writer
                    .write_event(Event::Start(BytesStart::new("object")))
                    .ok();

                Self::write_element(&mut writer, "name", &class_name(class_names, bbox.class_id));
                Self::write_element(&mut writer, "pose", "Unspecified");
                Self::write_element(&mut writer, "truncated", "0");
                Self::write_element(&mut writer, "difficult", "0");

                // Convert normalized coords to absolute Pascal VOC format
                // VOC uses [xmin, ymin, xmax, ymax] in pixels
                let xmin = ((bbox.x - bbox.width / 2.0) * width as f64).round() as i32;
                let ymin = ((bbox.y - bbox.height / 2.0) * height as f64).round() as i32;
                let xmax = ((bbox.x + bbox.width / 2.0) * width as f64).round() as i32;
                let ymax = ((bbox.y + bbox.height / 2.0) * height as f64).round() as i32;

                writer
                    .write_event(Event::Start(BytesStart::new("bndbox")))
                    .ok();
                Self::write_element(&mut writer, "xmin", &xmin.max(0).to_string());
                Self::write_element(&mut writer, "ymin", &ymin.max(0).to_string());
                Self::write_element(&mut writer, "xmax", &xmax.min(width).to_string());
                Self::write_element(&mut writer, "ymax", &ymax.min(height).to_string());
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
    fn skips_unlabeled_images(&self, task: &str) -> bool {
        task == "classify"
    }

    fn needs_image_sizes(&self, task: &str) -> bool {
        task != "classify"
    }

    fn convert(
        &self,
        data: &NDJSONData,
        downloaded_images: &HashMap<String, Vec<u8>>,
    ) -> HashMap<String, Vec<u8>> {
        let mut files: HashMap<String, Vec<u8>> = HashMap::new();
        let class_names = get_class_names(data);
        let task = &data.metadata.task;
        let class_folders = classify_folder_names(data);

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
                // Classification: folder structure {split}/{class_folder}/{file}
                for img in images.iter() {
                    let image_file = img.effective_file_name();
                    let classifications = img.get_classifications();
                    if let Some(class_dir) = classifications
                        .first()
                        .and_then(|class_id| class_folders.get(class_id))
                    {
                        if let Some(image_data) =
                            downloaded_images.get(&image_download_key(split, image_file))
                        {
                            files.insert(
                                format!("{}/{}/{}", split, class_dir, image_file),
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
                    let xml_filename = output_stem(image_file);
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
    use crate::parser::{image_download_key, parse_ndjson, DatasetMetadata};
    use serde_json::json;

    #[test]
    fn convert_uses_split_aware_download_keys() {
        let data = NDJSONData {
            metadata: DatasetMetadata {
                task: "detect".to_string(),
                name: "test".to_string(),
                description: String::new(),
                bytes: 0,
                url: String::new(),
                class_names: HashMap::from([("0".to_string(), "animal".to_string())]),
                kpt_shape: None,
                version: "1".to_string(),
                depth_scale: 1000.0,
            },
            images: vec![
                ImageEntry {
                    file: "img1.jpg".to_string(),
                    output_file: None,
                    url: String::new(),
                    width: Some(640),
                    height: Some(480),
                    split: "train".to_string(),
                    depth: None,
                    annotations: Some(json!({
                        "bboxes": [[0, 0.5, 0.5, 0.2, 0.2]]
                    })),
                },
                ImageEntry {
                    file: "img1.jpg".to_string(),
                    output_file: None,
                    url: String::new(),
                    width: Some(640),
                    height: Some(480),
                    split: "val".to_string(),
                    depth: None,
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
    fn object_names_use_placeholder_for_unnamed_class_ids() {
        let data = NDJSONData {
            metadata: DatasetMetadata {
                task: "detect".to_string(),
                name: "test".to_string(),
                description: String::new(),
                bytes: 0,
                url: String::new(),
                class_names: HashMap::from([("0".to_string(), "cat".to_string())]),
                kpt_shape: None,
                version: "1".to_string(),
                depth_scale: 1000.0,
            },
            images: vec![ImageEntry {
                file: "a.jpg".to_string(),
                output_file: None,
                url: String::new(),
                width: Some(640),
                height: Some(480),
                split: "train".to_string(),
                depth: None,
                annotations: Some(json!({
                    "boxes": [[0, 0.5, 0.5, 0.2, 0.2], [2, 0.4, 0.4, 0.1, 0.1]]
                })),
            }],
        };

        let files = PascalVocConverter::new().convert(&data, &HashMap::new());
        let xml = std::str::from_utf8(&files["train/a.xml"]).unwrap();

        assert!(xml.contains("<name>cat</name>"));
        assert!(xml.contains("<name>class2</name>"));
    }

    #[test]
    fn xml_never_claims_segmentation_masks() {
        for (task, annotations) in [
            ("detect", r#"{"boxes":[[0,0.5,0.5,0.2,0.2]]}"#),
            ("segment", r#"{"segments":[[0,0.1,0.1,0.5,0.1,0.3,0.5]]}"#),
            ("semantic", r#"{"segments":[[0,0.1,0.1,0.5,0.1,0.3,0.5]]}"#),
        ] {
            let data = parse_ndjson(&format!(
                r#"{{"type":"dataset","task":"{task}","name":"test","class_names":{{"0":"car"}}}}
{{"type":"image","file":"a.jpg","width":640,"height":480,"split":"train","annotations":{annotations}}}"#
            ))
            .unwrap();

            let files = PascalVocConverter::new().convert(&data, &HashMap::new());
            let xml = std::str::from_utf8(&files["train/a.xml"]).unwrap();

            assert!(xml.contains("<segmented>0</segmented>"), "{task}");
            assert!(xml.contains("<object>"), "{task}");
        }
    }

    #[test]
    fn convert_uses_effective_file_names_for_duplicate_outputs() {
        let data = NDJSONData {
            metadata: DatasetMetadata {
                task: "detect".to_string(),
                name: "test".to_string(),
                description: String::new(),
                bytes: 0,
                url: String::new(),
                class_names: HashMap::from([("0".to_string(), "heic".to_string())]),
                kpt_shape: None,
                version: "1".to_string(),
                depth_scale: 1000.0,
            },
            images: vec![
                ImageEntry {
                    file: "Frame_98.jpg".to_string(),
                    output_file: None,
                    url: "https://cdn.example/a.jpg".to_string(),
                    width: Some(640),
                    height: Some(480),
                    split: "train".to_string(),
                    depth: None,
                    annotations: Some(json!({
                        "boxes": [[0, 0.5, 0.5, 0.2, 0.2]]
                    })),
                },
                ImageEntry {
                    file: "Frame_98.jpg".to_string(),
                    output_file: Some("Frame_98__abcd1234.jpg".to_string()),
                    url: "https://cdn.example/b.jpg".to_string(),
                    width: Some(640),
                    height: Some(480),
                    split: "train".to_string(),
                    depth: None,
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
