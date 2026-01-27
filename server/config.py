"""
Voice configuration management for Omarchy Local TTS.
"""

import json
import shutil
from pathlib import Path
from typing import TypedDict


class VoiceInfo(TypedDict):
    id: str
    name: str
    description: str
    file: str


class VoiceWithStatus(VoiceInfo):
    is_active: bool


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
VOICES_DIR = PROJECT_ROOT / "voices"
VOICES_JSON = VOICES_DIR / "voices.json"
CONFIG_DIR = Path.home() / ".config" / "omarchy-local-tts"
CONFIG_JSON = CONFIG_DIR / "config.json"


class VoiceConfig:
    """Manages voice configuration and selection."""

    def __init__(self):
        self._voices: list[VoiceInfo] = []
        self._active_voice: str = ""
        self._load()

    def _load(self):
        """Load configuration from files."""
        # Load voices
        if VOICES_JSON.exists():
            with open(VOICES_JSON) as f:
                data = json.load(f)
                self._voices = data.get("voices", [])

        # Load config
        if CONFIG_JSON.exists():
            with open(CONFIG_JSON) as f:
                config = json.load(f)
                self._active_voice = config.get("active_voice", "")

        # Default to first voice if no active voice set
        if not self._active_voice and self._voices:
            self._active_voice = self._voices[0]["id"]
            self._save_config()

    def _save_config(self):
        """Save active voice to config.json."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_JSON, "w") as f:
            json.dump({"active_voice": self._active_voice}, f, indent=2)

    def _save_voices(self):
        """Save voices list to voices.json."""
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        with open(VOICES_JSON, "w") as f:
            json.dump({"voices": self._voices}, f, indent=2)

    def get_active_voice_id(self) -> str:
        """Get the ID of the currently active voice."""
        return self._active_voice

    def get_active_voice_path(self) -> Path:
        """Get the file path of the currently active voice."""
        for voice in self._voices:
            if voice["id"] == self._active_voice:
                return VOICES_DIR / voice["file"]
        # Fallback: return first voice if active not found
        if self._voices:
            return VOICES_DIR / self._voices[0]["file"]
        raise FileNotFoundError("No voices configured")

    def get_voice_path(self, voice_id: str) -> Path:
        """Get the file path for a specific voice."""
        for voice in self._voices:
            if voice["id"] == voice_id:
                return VOICES_DIR / voice["file"]
        raise ValueError(f"Voice not found: {voice_id}")

    def set_active_voice(self, voice_id: str) -> bool:
        """Set the active voice by ID."""
        for voice in self._voices:
            if voice["id"] == voice_id:
                self._active_voice = voice_id
                self._save_config()
                return True
        return False

    def list_voices(self) -> list[VoiceWithStatus]:
        """List all available voices with active status."""
        result: list[VoiceWithStatus] = []
        for voice in self._voices:
            result.append({
                **voice,
                "is_active": voice["id"] == self._active_voice,
            })
        return result

    def get_voice(self, voice_id: str) -> VoiceInfo | None:
        """Get a single voice by ID."""
        for voice in self._voices:
            if voice["id"] == voice_id:
                return voice
        return None

    def add_voice(self, file_path: Path, name: str, description: str = "") -> VoiceInfo:
        """Add a new voice from an uploaded file."""
        # Generate ID from name
        voice_id = name.lower().replace(" ", "_")

        # Ensure unique ID
        base_id = voice_id
        counter = 1
        while any(v["id"] == voice_id for v in self._voices):
            voice_id = f"{base_id}_{counter}"
            counter += 1

        # Determine filename
        suffix = file_path.suffix or ".mp3"
        filename = f"{voice_id}{suffix}"
        dest_path = VOICES_DIR / filename

        # Copy file to voices directory
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)

        # Add to voices list
        voice: VoiceInfo = {
            "id": voice_id,
            "name": name,
            "description": description,
            "file": filename,
        }
        self._voices.append(voice)
        self._save_voices()

        return voice


# Global instance
voice_config = VoiceConfig()

# Shared TTS model reference (set by main.py after loading)
tts_model = None
