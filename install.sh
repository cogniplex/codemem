#!/bin/sh
# Codemem installer — https://github.com/cogniplex/codemem
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/cogniplex/codemem/main/install.sh | sh
#
# Options (via env vars):
#   CODEMEM_VERSION   — version to install (default: latest)
#   CODEMEM_INSTALL   — install directory (default: /usr/local/bin)

set -e

REPO="cogniplex/codemem"
INSTALL_DIR="${CODEMEM_INSTALL:-/usr/local/bin}"

# --- Detect platform ---

detect_platform() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case "$OS" in
        linux)  PLATFORM="linux" ;;
        darwin) PLATFORM="macos" ;;
        *)
            echo "Error: unsupported OS: $OS" >&2
            echo "Codemem supports Linux and macOS. See https://github.com/$REPO/releases" >&2
            exit 1
            ;;
    esac

    case "$ARCH" in
        x86_64|amd64)   ARCH_NAME="amd64" ;;
        aarch64|arm64)   ARCH_NAME="arm64" ;;
        *)
            echo "Error: unsupported architecture: $ARCH" >&2
            echo "Codemem supports x86_64 and ARM64. See https://github.com/$REPO/releases" >&2
            exit 1
            ;;
    esac

    # macOS only ships ARM64 binaries
    if [ "$PLATFORM" = "macos" ] && [ "$ARCH_NAME" = "amd64" ]; then
        echo "Error: macOS x86_64 binaries are not available." >&2
        echo "Use 'cargo install codemem-cli' or run under Rosetta." >&2
        exit 1
    fi

    ASSET_NAME="codemem-${PLATFORM}-${ARCH_NAME}"
}

# --- Resolve version ---

resolve_version() {
    if [ -n "$CODEMEM_VERSION" ]; then
        VERSION="$CODEMEM_VERSION"
    else
        VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
            | grep '"tag_name"' | head -1 | cut -d'"' -f4)
        if [ -z "$VERSION" ]; then
            echo "Error: could not determine latest version." >&2
            echo "Set CODEMEM_VERSION or visit https://github.com/$REPO/releases" >&2
            exit 1
        fi
    fi
}

# --- Download and install ---

install() {
    DOWNLOAD_URL="https://github.com/$REPO/releases/download/$VERSION/${ASSET_NAME}.tar.gz"
    CHECKSUM_URL="${DOWNLOAD_URL}.sha256"

    TMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TMP_DIR"' EXIT

    echo "Downloading codemem $VERSION ($PLATFORM/$ARCH_NAME)..."
    curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/codemem.tar.gz"
    curl -fsSL "$CHECKSUM_URL" -o "$TMP_DIR/codemem.tar.gz.sha256"

    # Verify checksum
    echo "Verifying checksum..."
    cd "$TMP_DIR"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c codemem.tar.gz.sha256
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c codemem.tar.gz.sha256
    else
        echo "Warning: no checksum tool found, skipping verification." >&2
    fi

    # Extract
    tar xzf codemem.tar.gz

    # Install
    if [ -w "$INSTALL_DIR" ]; then
        mv codemem "$INSTALL_DIR/codemem"
    else
        echo "Installing to $INSTALL_DIR (requires sudo)..."
        sudo mv codemem "$INSTALL_DIR/codemem"
    fi

    echo ""
    echo "codemem $VERSION installed to $INSTALL_DIR/codemem"
    echo ""
    echo "Get started:"
    echo "  codemem init        # Initialize in your project"
    echo "  codemem --help      # See all commands"
    echo ""
}

# --- Main ---

detect_platform
resolve_version
install
