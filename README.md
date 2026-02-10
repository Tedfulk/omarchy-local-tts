# Omarchy Local TTS

A local text-to-speech server with voice cloning capabilities, built on [index-tts](https://github.com/index-tts/index-tts). Designed for integration with [Omarchy](https://omarchy.org/) Linux but works on any Linux system with PipeWire audio.

## Features

- **Voice Cloning**: Clone any voice from a short audio sample
- **Web UI**: Easy voice management at `http://127.0.0.1:5858/ui`
- **REST API**: Simple HTTP endpoints for integration
- **Waybar Integration**: Status indicator showing playback state
- **Playback Controls**: Pause, resume, and stop via API or Waybar
- **Streaming**: Producer-consumer pattern for responsive playback

## Requirements

- Linux with PipeWire audio (`pw-play` command)
- NVIDIA GPU with CUDA support (for fast inference)
- Python 3.10+
- git and git-lfs
- ~4GB disk space for models

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/omarchy-local-tts.git
cd omarchy-local-tts

# Run setup (installs index-tts, downloads models, creates symlinks)
./setup.sh

# Start the server
./start-server.sh
```

The server will be available at:
- **API**: http://127.0.0.1:5858
- **Web UI**: http://127.0.0.1:5858/ui

## Usage

### Speak Highlighted Text

Bind `omarchy-tts-speak` to a hotkey (e.g., Super+D):

**Hyprland** (`~/.config/hypr/bindings.conf`):
```
bind = SUPER, D, exec, omarchy-tts-speak
```

Then highlight any text and press the hotkey to hear it spoken.

### Repeat Last Spoken Text

Bind `omarchy-tts-repeat` to a hotkey (e.g., Super+Shift+D):

**Hyprland** (`~/.config/hypr/bindings.conf`):
```
bind = SUPER SHIFT, D, exec, omarchy-tts-repeat
```

Press the hotkey to re-speak the last spoken text using the same voice.

### Web UI

Open http://127.0.0.1:5858/ui to:
- Select the active voice
- Preview voice samples
- Test text-to-speech
- Upload new voice samples

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/speak` | POST | Generate and play speech `{"text": "Hello world"}` |
| `/repeat` | POST | Re-speak the last spoken text (same voice) |
| `/stop` | POST | Stop playback immediately |
| `/pause` | POST | Pause playback |
| `/resume` | POST | Resume playback |
| `/toggle-pause` | POST | Toggle pause/resume |
| `/status` | GET | Get playback status |
| `/voices` | GET | List available voices |
| `/voices/select/{id}` | POST | Set active voice |
| `/voices/test` | POST | Test a voice `{"voice_id": "alex", "text": "Hello"}` |
| `/history` | GET | Query command history (params: `days`, `voice_id`, `limit`, `offset`) |
| `/health` | GET | Health check |

Example:
```bash
curl -X POST http://127.0.0.1:5858/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test."}'
```

## Waybar Integration

Add a TTS status indicator to your Waybar.

### 1. Add module to config

Edit `~/.config/waybar/config.jsonc`:

```jsonc
{
  "modules-right": [
    // ... other modules
    "custom/tts"
  ],
  "custom/tts": {
    "exec": "~/.local/bin/omarchy-tts-status",
    "return-type": "json",
    "interval": "once",
    "signal": 9,
    "tooltip": true,
    "on-click": "~/.local/bin/omarchy-tts-toggle-pause",
    "on-click-right": "~/.local/bin/omarchy-tts-stop"
  }
}
```

### 2. Add styles

Edit `~/.config/waybar/style.css`:

```css
#custom-tts {
  min-width: 12px;
  margin: 0 10px 0 7.5px;
}

#custom-tts.speaking {
  color: #55a5a5;
}

#custom-tts.paused {
  color: #a5a555;
}

#custom-tts.idle {
  opacity: 0;
}
```

### 3. Restart Waybar

```bash
omarchy-restart-waybar  # or: pkill waybar && waybar &
```

Now you'll see:
- **Cyan speaker icon**: Playing
- **Yellow pause icon**: Paused
- **Hidden**: Idle

Click to pause/resume, right-click to stop.

## Adding Voice Samples

### Via Web UI

1. Open http://127.0.0.1:5858/ui
2. Expand "Add New Voice"
3. Upload an audio file (MP3 or WAV, 5-30 seconds recommended)
4. Enter a name and description
5. Click "Add Voice"

### Manually

1. Add audio file to `voices/` directory
2. Edit `voices/voices.json`:

```json
{
  "voices": [
    {
      "id": "my_voice",
      "name": "My Voice",
      "description": "A custom voice",
      "file": "my_voice.mp3"
    }
  ]
}
```

## Voice Sample Tips

For best results, voice samples should:
- Be 5-30 seconds of clear speech
- Have minimal background noise
- Feature natural, expressive speaking
- Be in MP3 or WAV format

## Autostart

To start the TTS server automatically:

**Hyprland** (`~/.config/hypr/autostart.conf`):
```
exec-once = /path/to/omarchy-local-tts/start-server.sh
```

**Systemd user service** (`~/.config/systemd/user/omarchy-tts.service`):
```ini
[Unit]
Description=Omarchy Local TTS Server
After=network.target

[Service]
Type=simple
ExecStart=/path/to/omarchy-local-tts/start-server.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Then enable: `systemctl --user enable --now omarchy-tts`

## Troubleshooting

### Server won't start

1. Check if port 5858 is in use: `lsof -i :5858`
2. Verify models downloaded: `ls index-tts/checkpoints/`
3. Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### No audio

1. Verify PipeWire: `pw-play --help`
2. Check audio output: `wpctl status`

### Slow inference

1. Ensure CUDA is being used (check server logs for "cuda:0")
2. First inference is slower due to model loading

## License

This project wraps [index-tts](https://github.com/index-tts/index-tts) which has its own license. See index-tts repository for details.

## Credits

- [index-tts](https://github.com/index-tts/index-tts) - The underlying TTS model
- [Omarchy](https://omarchy.org/) - Linux distribution this was built for
