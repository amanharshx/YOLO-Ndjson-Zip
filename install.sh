#!/usr/bin/env bash
set -euo pipefail

REPO="amanharshx/YOLO-Ndjson-Zip"
APP_NAME="YOLO NDJSON Converter"

info()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
error() { printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin) PLATFORM="macos" ;;
  Linux)  PLATFORM="linux" ;;
  *)      error "Unsupported OS: $OS. Please download manually from https://github.com/$REPO/releases" ;;
esac

case "$ARCH" in
  x86_64|amd64) ARCH_LABEL="x86_64" ;;
  arm64|aarch64) ARCH_LABEL="aarch64" ;;
  *)             error "Unsupported architecture: $ARCH" ;;
esac

info "Detected platform: $PLATFORM ($ARCH_LABEL)"

# Fetch latest release tag from GitHub API
info "Fetching latest release..."
RELEASE_JSON="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest")" \
  || error "No releases found. Please check https://github.com/$REPO/releases"

TAG="$(echo "$RELEASE_JSON" | grep '"tag_name"' | head -1 | sed 's/.*: "\(.*\)".*/\1/')"
[ -n "$TAG" ] || error "Could not determine latest release tag."
info "Latest release: $TAG"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

download_asset() {
  local pattern="$1"
  local url
  url="$(echo "$RELEASE_JSON" | grep '"browser_download_url"' | grep "$pattern" | head -1 | sed 's/.*"\(https[^"]*\)".*/\1/')"
  [ -n "$url" ] || error "Could not find asset matching '$pattern' in release $TAG"
  info "Downloading: $url"
  curl -fsSL -o "$TMPDIR/$(basename "$url")" "$url"
  echo "$TMPDIR/$(basename "$url")"
}

if [ "$PLATFORM" = "macos" ]; then
  ASSET_PATH="$(download_asset "${ARCH_LABEL}.*\.dmg")"
  info "Mounting disk image..."
  MOUNT_POINT="$(hdiutil attach "$ASSET_PATH" -nobrowse -quiet | tail -1 | awk '{print $NF}')" \
    || error "Failed to mount DMG"

  APP_SRC="$(find "$MOUNT_POINT" -name '*.app' -maxdepth 1 | head -1)"
  [ -n "$APP_SRC" ] || { hdiutil detach "$MOUNT_POINT" -quiet 2>/dev/null; error "No .app found in DMG"; }

  info "Installing to /Applications..."
  rm -rf "/Applications/$(basename "$APP_SRC")" 2>/dev/null || true
  cp -R "$APP_SRC" /Applications/
  hdiutil detach "$MOUNT_POINT" -quiet 2>/dev/null

  info "Done! $APP_NAME has been installed to /Applications."
  info "You can launch it from Spotlight or the Applications folder."

elif [ "$PLATFORM" = "linux" ]; then
  # Try .deb first, fall back to .AppImage
  if echo "$RELEASE_JSON" | grep -q '"browser_download_url".*\.deb"'; then
    ASSET_PATH="$(download_asset '\.deb"')"
    info "Installing .deb package..."
    sudo dpkg -i "$ASSET_PATH" || { sudo apt-get install -f -y && sudo dpkg -i "$ASSET_PATH"; }
    info "Done! $APP_NAME has been installed."
  elif echo "$RELEASE_JSON" | grep -q '"browser_download_url".*\.AppImage"'; then
    ASSET_PATH="$(download_asset '\.AppImage"')"
    INSTALL_DIR="${HOME}/.local/bin"
    mkdir -p "$INSTALL_DIR"
    APPIMAGE_NAME="yolo-ndjson-converter.AppImage"
    mv "$ASSET_PATH" "$INSTALL_DIR/$APPIMAGE_NAME"
    chmod +x "$INSTALL_DIR/$APPIMAGE_NAME"
    info "Done! AppImage installed to $INSTALL_DIR/$APPIMAGE_NAME"
    info "Make sure $INSTALL_DIR is in your PATH."
  else
    error "No .deb or .AppImage found in release $TAG"
  fi
fi
