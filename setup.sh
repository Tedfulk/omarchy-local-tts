#!/bin/bash
# Omarchy Local TTS - Setup Script
# Sets up index-tts, downloads models, and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Omarchy Local TTS Setup ==="
echo ""

# Check for required tools
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is required but not installed."
        echo "Please install $1 and run this script again."
        exit 1
    fi
}

check_command git
check_command python3
check_command curl

# Check for git-lfs
if ! git lfs version &> /dev/null; then
    echo "Error: git-lfs is required but not installed."
    echo "Please install git-lfs: https://git-lfs.com/"
    exit 1
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    pip install -U uv
fi

# Clone index-tts if not present
if [ ! -d "index-tts" ]; then
    echo ""
    echo "=== Cloning index-tts ==="
    git clone https://github.com/index-tts/index-tts.git
    cd index-tts
    git lfs pull
    cd ..
else
    echo "index-tts already exists, skipping clone..."
fi

# Set up index-tts virtual environment and dependencies
echo ""
echo "=== Setting up index-tts dependencies ==="
cd index-tts
uv sync --extra webui
cd ..

# Download models if not present
if [ ! -f "index-tts/checkpoints/config.yaml" ]; then
    echo ""
    echo "=== Downloading models ==="
    echo "This will download ~2GB of model files..."
    cd index-tts

    # Use huggingface-cli to download models
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints
    else
        # Fallback: use uv run to access huggingface-cli from venv
        uv run huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints
    fi
    cd ..
else
    echo "Models already downloaded, skipping..."
fi

# Install server dependencies
echo ""
echo "=== Installing server dependencies ==="
cd index-tts
uv pip install gradio fastapi uvicorn
cd ..

# Migrate config.json to ~/.config/omarchy-local-tts if it exists in root
CONFIG_DIR="$HOME/.config/omarchy-local-tts"
mkdir -p "$CONFIG_DIR"

if [ -f "$SCRIPT_DIR/config.json" ]; then
    echo ""
    echo "=== Migrating config.json ==="
    echo "Moving config.json to $CONFIG_DIR"
    mv "$SCRIPT_DIR/config.json" "$CONFIG_DIR/config.json"
    echo "  Config migrated successfully"
fi

# Create symlinks for bin scripts
echo ""
echo "=== Setting up command symlinks ==="
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

for script in "$SCRIPT_DIR/bin/omarchy-tts-"*; do
    script_name=$(basename "$script")
    target="$BIN_DIR/$script_name"

    if [ -L "$target" ]; then
        rm "$target"
    fi

    ln -sf "$script" "$target"
    echo "  Linked: $script_name"
done

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "NOTE: ~/.local/bin is not in your PATH."
    echo "Add this to your shell config (~/.bashrc or ~/.zshrc):"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the TTS server:"
echo "  ./start-server.sh"
echo ""
echo "Server will be available at:"
echo "  API: http://127.0.0.1:5858"
echo "  UI:  http://127.0.0.1:5858/ui"
echo ""
echo "Commands available:"
echo "  omarchy-tts-speak        - Speak highlighted text (bind to a hotkey)"
echo "  omarchy-tts-stop         - Stop playback"
echo "  omarchy-tts-toggle-pause - Toggle pause/resume"
echo "  omarchy-tts-status       - Waybar status (for status bar integration)"
echo ""
echo "See README.md for Waybar integration and keybinding setup."
