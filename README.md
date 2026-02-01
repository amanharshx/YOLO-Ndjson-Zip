<div align="center">

# YOLO NDJSON Converter

**Convert YOLO NDJSON annotation exports to popular ML formats - fast, private, and cross-platform.**

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)](#installation)
[![Homebrew](https://img.shields.io/badge/Homebrew-tap-FBB040?logo=homebrew)](https://github.com/amanharshx/homebrew-tap)
[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri%20v2-ffc131.svg)](https://v2.tauri.app)
[![Rust](https://img.shields.io/badge/Rust-%23dea584.svg?logo=rust&logoColor=black)](#tech-stack)
[![TypeScript](https://img.shields.io/badge/TypeScript-%23007ACC.svg?logo=typescript&logoColor=white)](#tech-stack)

<br>

<img src="assets/screenshot.png" width="720" alt="YOLO NDJSON Converter welcome screen">

</div>
<br>

[**YOLO NDJSON Converter**](https://yolondjson.zip) is a desktop app that converts [Ultralytics](https://www.ultralytics.com/) YOLO's NDJSON annotation exports into ready-to-use datasets for YOLO, COCO, Pascal VOC, and other ML formats. Select your file, pick a format, and get a ZIP with images, labels, and config files.

## Features

- **12 Output Formats** - YOLO26 through Darknet, COCO JSON, Pascal VOC, TFRecord, CreateML
- **4 Task Types** - Detection, Segmentation, Pose Estimation, Classification
- **Parallel Downloads** - 100 concurrent connections for fast image fetching
- **Privacy-First** - Everything runs locally; your data never leaves your device
- **Cross-Platform** - macOS, Windows, and Linux
- **~5 MB Binary** - Tauri + Rust keeps the app tiny compared to Electron alternatives

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

## Installation

### Quick Install

**Windows (PowerShell):**

```powershell
irm https://yolondjson.zip/install.ps1 | iex
```

**macOS / Linux:**

```bash
curl -fsSL https://yolondjson.zip/install.sh | sh
```

**macOS (Homebrew):**

```bash
brew tap amanharshx/tap
brew install --cask yolo-ndjson-converter
```

Or download the latest release directly from [GitHub Releases](https://github.com/amanharshx/yolo-ndjson-zip/releases).

### Troubleshooting

> **Note:** The app is not yet code-signed (Apple Developer account costs $99/year, Windows EV certificate ~$300/year). I'm planning to get these when I can afford them. For now, you may see security warnings:

<details>
<summary><b>macOS</b> — "App is damaged and can't be opened"</summary>

Run this command in Terminal after installing:
```bash
xattr -cr "/Applications/YOLO NDJSON Converter.app"
```
Then open the app again.

</details>

<details>
<summary><b>Windows</b> — "Windows protected your PC" (SmartScreen)</summary>

1. Click **"More info"**
2. Click **"Run anyway"**

Or: Right-click the `.exe` → **Properties** → Check **"Unblock"** → **Apply**

</details>

### Auto Updates

The app supports automatic updates.

When a new version is released:

1. Open the app
2. Click **Updates** (top right)
3. If a new version is available, click **Update**
4. Restart the app to finish installing

Updates are downloaded securely from GitHub Releases and verified using cryptographic signatures.

### Build from Source

**Prerequisites:** [Rust](https://rustup.rs/), [Bun](https://bun.sh/), [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/)

```bash
git clone https://github.com/amanharshx/yolo-ndjson-zip.git
cd yolo-ndjson-zip
bun install
bun run tauri dev        # development
bun run tauri build      # local production build
```

> **Note:** Local builds are unsigned and intended for development only. Official signed builds are generated automatically via GitHub Releases.

## NDJSON Input Format

The app expects newline-delimited JSON with this structure:

```jsonl
{"type":"dataset","task":"detect","name":"My Dataset","class_names":{"0":"cat","1":"dog"}}
{"type":"image","file":"img1.jpg","url":"https://...","width":640,"height":480,"split":"train","annotations":{"bboxes":[[0,0.5,0.5,0.2,0.3]]}}
{"type":"image","file":"img2.jpg","url":"https://...","width":640,"height":480,"split":"valid","annotations":{"bboxes":[[1,0.3,0.4,0.1,0.2]]}}
```

## Tech Stack

- **Frontend** - React 19 + TypeScript, Tailwind CSS, Vite
- **Backend** - Rust + Tauri v2
- **Package Manager** - Bun

## Releases & Versioning

This project uses automated releases.

- Versions follow semantic versioning (`MAJOR.MINOR.PATCH`)
- Merging changes into `main` automatically prepares the next release
- Releases are built and published via GitHub Actions

All official binaries are available on the [GitHub Releases](https://github.com/amanharshx/yolo-ndjson-zip/releases) page.

## Update Security

All releases are cryptographically signed.

The built-in updater verifies signatures before installing updates to ensure authenticity and prevent tampering.

## Contributing

Contributions are welcome! Whether it's a bug fix, new format, or documentation improvement - every bit helps. Please read the [Contributing Guide](CONTRIBUTING.md) before opening a pull request.

## Security

To report a security vulnerability, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the [MIT License](LICENSE).
