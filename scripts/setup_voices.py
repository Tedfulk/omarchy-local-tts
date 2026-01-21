#!/usr/bin/env python3
"""
Setup script to copy voice files from Downloads and generate metadata.
"""

import json
import re
import shutil
from pathlib import Path

# Paths
DOWNLOADS_DIR = Path.home() / "Downloads"
PROJECT_ROOT = Path(__file__).parent.parent
VOICES_DIR = PROJECT_ROOT / "voices"
VOICES_JSON = VOICES_DIR / "voices.json"
CONFIG_JSON = PROJECT_ROOT / "config.json"


def parse_voice_filename(filename: str) -> tuple[str, str]:
    """Extract voice name and description from filename.

    Example: 'voice_preview_rachelle - well-spoken, wise and clear.mp3'
    Returns: ('rachelle', 'well-spoken, wise and clear')
    """
    # Remove extension and 'voice_preview_' prefix
    stem = Path(filename).stem
    stem = stem.replace("voice_preview_", "")

    # Split on ' - ' to get name and description
    parts = stem.split(" - ", 1)
    name = parts[0].strip().lower().replace(" ", "_")
    description = parts[1].strip() if len(parts) > 1 else ""

    return name, description


def main():
    # Ensure voices directory exists
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Find all voice preview MP3s in Downloads
    voice_files = list(DOWNLOADS_DIR.glob("voice_preview_*.mp3"))

    if not voice_files:
        print("No voice files found in Downloads matching 'voice_preview_*.mp3'")
        return

    print(f"Found {len(voice_files)} voice files in Downloads")

    voices = []

    for src_file in sorted(voice_files):
        name, description = parse_voice_filename(src_file.name)
        clean_filename = f"{name}.mp3"
        dest_file = VOICES_DIR / clean_filename

        # Copy file
        print(f"  Copying: {src_file.name} -> {clean_filename}")
        shutil.copy2(src_file, dest_file)

        # Add to voice list
        voices.append({
            "id": name,
            "name": name.replace("_", " ").title(),
            "description": description,
            "file": clean_filename,
        })

    # Write voices.json
    with open(VOICES_JSON, "w") as f:
        json.dump({"voices": voices}, f, indent=2)
    print(f"\nCreated {VOICES_JSON}")

    # Create initial config.json with first voice as default
    if voices:
        default_voice = voices[0]["id"]
        # Use rachelle if available (the original default)
        for v in voices:
            if v["id"] == "rachelle":
                default_voice = "rachelle"
                break

        config = {"active_voice": default_voice}
        with open(CONFIG_JSON, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Created {CONFIG_JSON} with default voice: {default_voice}")

    print(f"\nSetup complete! {len(voices)} voices ready.")


if __name__ == "__main__":
    main()
