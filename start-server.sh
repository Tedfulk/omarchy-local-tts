#!/bin/bash
# start-server.sh - Start the omarchy-local-tts server
#
# This script activates the index-tts virtual environment and starts the FastAPI server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_TTS_DIR="${SCRIPT_DIR}/index-tts"

# Change to index-tts directory (where the venv and checkpoints are)
cd "${INDEX_TTS_DIR}"

# Activate the virtual environment
source .venv/bin/activate

# Run the server
exec python "${SCRIPT_DIR}/server/main.py"
