export interface ProgressEvent {
  phase: string;
  current: number;
  total: number;
  item: string | null;
}

export interface ConvertResult {
  zip_path: string;
  file_count: number;
  image_count: number;
}

export interface Format {
  id: string;
  name: string;
  available: boolean;
  desc: string;
  highlight: string;
}

export const formats: Format[] = [
  { id: "yolo", name: "YOLO26", available: true, desc: "TXT annotations and YAML config", highlight: "YOLO26" },
  { id: "yolo", name: "YOLOv12", available: true, desc: "TXT annotations and YAML config", highlight: "YOLOv12" },
  { id: "yolo", name: "YOLOv11", available: true, desc: "TXT annotations and YAML config", highlight: "YOLO11" },
  { id: "yolo", name: "YOLOv9", available: true, desc: "TXT annotations and YAML config", highlight: "YOLOv9" },
  { id: "yolo", name: "YOLOv8", available: true, desc: "TXT annotations and YAML config", highlight: "YOLOv8" },
  { id: "yolo", name: "YOLOv5", available: true, desc: "TXT annotations and YAML config", highlight: "YOLOv5" },
  { id: "yolo", name: "YOLOv7", available: true, desc: "TXT annotations and YAML config", highlight: "YOLOv7" },
  { id: "coco", name: "COCO JSON", available: true, desc: "COCO JSON annotations", highlight: "EfficientDet Pytorch and Detectron 2" },
  { id: "yolo_darknet", name: "YOLO Darknet", available: true, desc: "Darknet TXT annotations", highlight: "YOLO Darknet (both v3 and v4) and YOLOv3 PyTorch" },
  { id: "pascal_voc", name: "Pascal VOC XML", available: true, desc: "Common XML annotation format for local data munging (pioneered by", highlight: "ImageNet" },
  { id: "tfrecord", name: "TFRecord", available: false, desc: "TFRecord binary format", highlight: "Tensorflow 1.5 and Tensorflow 2.0 Object Detection models" },
  { id: "createml", name: "CreateML JSON", available: false, desc: "CreateML JSON format", highlight: "Apple's CreateML and Turi Create tools" },
];

export const formatBadges = formats.map((f) => f.name);
