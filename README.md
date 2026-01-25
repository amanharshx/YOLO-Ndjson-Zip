# YOLO NDJSON Converter

> Convert YOLO NDJSON annotation exports to popular ML formats - fast, private, and cross-platform.

## Features

- **12 Output Formats** - YOLO26 through Darknet, COCO JSON, Pascal VOC, TFRecord, CreateML
- **4 Task Types** - Detection, Segmentation, Pose Estimation, Classification
- **Parallel Downloads** - 100 concurrent connections for fast image fetching
- **Privacy-First** - Everything runs locally; your data never leaves your device
- **Cross-Platform** - macOS, Windows, and Linux

## Supported Formats

| Format | Status | Compatible Tasks |
|--------|:------:|------------------|
| YOLO26 | ✅ | Detection, Segmentation, Pose, Classification |
| YOLOv12 | ✅ | Detection, Segmentation, Pose, Classification |
| YOLO11 | ✅ | Detection, Segmentation, Pose, Classification |
| YOLOv9 | ✅ | Detection, Segmentation |
| YOLOv8 | ✅ | Detection, Segmentation, Pose, Classification |
| YOLOv7 | ✅ | Detection |
| YOLOv5 | ✅ | Detection, Segmentation, Classification |
| YOLO Darknet | ✅ | Detection, Classification |
| COCO JSON | ✅ | Detection, Segmentation, Pose |
| Pascal VOC XML | ✅ | Detection, Segmentation, Classification |
| CreateML JSON | 🔜 | Detection, Classification |
| TFRecord | 🔜 | Detection |

## Tech Stack

- **Frontend** - React 19 + TypeScript, Tailwind CSS, Vite
- **Backend** - Rust + Tauri v2
- **Package Manager** - Bun

## Development

```bash
bun install
bun run tauri dev
```

## License

MIT
