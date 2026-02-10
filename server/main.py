#!/usr/bin/env python3
"""
Omarchy Local TTS Server

FastAPI server wrapping index-tts for text-to-speech with voice cloning.
Plays audio directly through pipewire.
"""

import os
import queue
import re
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add index-tts and server to path for direct execution
import sys
SERVER_PATH = Path(__file__).parent
PROJECT_ROOT = SERVER_PATH.parent
INDEX_TTS_PATH = PROJECT_ROOT / "index-tts"
sys.path.insert(0, str(INDEX_TTS_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

import server.config as config
import server.db as db
from server.config import voice_config

from indextts.infer_v2 import IndexTTS2

# Configuration
CHECKPOINTS_DIR = INDEX_TTS_PATH / "checkpoints"
CONFIG_PATH = CHECKPOINTS_DIR / "config.yaml"
MAX_WORDS_PER_CHUNK = 16

# Global TTS instance
tts_model: IndexTTS2 | None = None


class SpeakRequest(BaseModel):
    text: str


class TestVoiceRequest(BaseModel):
    voice_id: str
    text: str


class SelectVoiceRequest(BaseModel):
    voice_id: str


class VoiceResponse(BaseModel):
    id: str
    name: str
    description: str
    file: str
    is_active: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    voice_file: str
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TTS model on startup."""
    global tts_model

    voice_path = voice_config.get_active_voice_path()
    print(f"Loading index-tts model from {CHECKPOINTS_DIR}...")
    print(f"Using voice file: {voice_path}")

    if not voice_path.exists():
        print(f"WARNING: Voice file not found at {voice_path}")

    if not CONFIG_PATH.exists():
        print(f"WARNING: Config not found at {CONFIG_PATH}")
        print("Please download models first. See README for instructions.")
    else:
        # Initialize TTS with FP16 for RTX 3090 optimization
        # use_cuda_kernel=False to avoid compile errors; PyTorch fallback works fine
        tts_model = IndexTTS2(
            cfg_path=str(CONFIG_PATH),
            model_dir=str(CHECKPOINTS_DIR),
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )

        # Share model with UI module
        config.tts_model = tts_model

        # Pre-cache the voice reference audio
        if voice_path.exists():
            print("Pre-caching voice reference audio...")
            print(f"Model loaded on device: {tts_model.device}")

        print("TTS model ready!")

    # Initialize command history database
    db.init_db()
    print("Command history database ready.")

    yield

    # Cleanup
    if tts_model is not None:
        config.tts_model = None
        del tts_model
        torch.cuda.empty_cache()


app = FastAPI(
    title="Omarchy Local TTS",
    description="Text-to-speech server with voice cloning using index-tts",
    lifespan=lifespan,
)


# Status file for Waybar indicator
TTS_STATUS_FILE = Path("/tmp/omarchy-tts-playing")
TTS_PAUSED_FILE = Path("/tmp/omarchy-tts-paused")

# Track number of active playback threads
_playback_count = 0
_playback_lock = threading.Lock()

# Stop flag for cancelling playback
_stop_requested = False
_stop_lock = threading.Lock()

# Pause state
_paused = False
_pause_lock = threading.Lock()
_current_player: subprocess.Popen | None = None
_player_lock = threading.Lock()


def _update_tts_status(playing: bool = None, paused: bool = None):
    """Update TTS status files and signal Waybar."""
    global _playback_count

    if playing is not None:
        with _playback_lock:
            if playing:
                _playback_count += 1
            else:
                _playback_count = max(0, _playback_count - 1)

            if _playback_count > 0:
                TTS_STATUS_FILE.touch()
            else:
                TTS_STATUS_FILE.unlink(missing_ok=True)
                TTS_PAUSED_FILE.unlink(missing_ok=True)

    if paused is not None:
        if paused:
            TTS_PAUSED_FILE.touch()
        else:
            TTS_PAUSED_FILE.unlink(missing_ok=True)

    # Signal Waybar to update (SIGRTMIN+9 = signal 9)
    subprocess.run(["pkill", "-RTMIN+9", "waybar"], capture_output=True)


def play_audio_pipewire(audio_path: str) -> bool:
    """Play audio file using pw-play (pipewire).

    Returns True if played completely, False if stopped early.
    """
    global _stop_requested, _paused, _current_player
    import signal
    import time

    _update_tts_status(playing=True)
    try:
        proc = subprocess.Popen(
            ["pw-play", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Track current player for pause/resume
        with _player_lock:
            _current_player = proc

        # Poll for completion while checking stop/pause flags
        while proc.poll() is None:
            with _stop_lock:
                if _stop_requested:
                    proc.terminate()
                    proc.wait(timeout=1)
                    with _player_lock:
                        _current_player = None
                    return False

            # Check pause state
            with _pause_lock:
                is_paused = _paused

            if is_paused:
                # Send SIGSTOP to pause the process
                try:
                    proc.send_signal(signal.SIGSTOP)
                    _update_tts_status(paused=True)
                    # Wait while paused (check flag with lock each iteration)
                    while proc.poll() is None:
                        with _pause_lock:
                            if not _paused:
                                break
                        with _stop_lock:
                            if _stop_requested:
                                proc.send_signal(signal.SIGCONT)
                                proc.terminate()
                                proc.wait(timeout=1)
                                with _player_lock:
                                    _current_player = None
                                return False
                        time.sleep(0.05)
                    # Resume
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGCONT)
                        _update_tts_status(paused=False)
                except (ProcessLookupError, OSError):
                    pass  # Process already finished

            time.sleep(0.05)

        with _player_lock:
            _current_player = None
        return proc.returncode == 0
    except FileNotFoundError:
        print("pw-play not found. Please install pipewire.")
        raise
    finally:
        with _player_lock:
            _current_player = None
        _update_tts_status(playing=False)


def split_long_sentence(sentence: str) -> list[str]:
    """Split a sentence in half if it exceeds MAX_WORDS_PER_CHUNK words."""
    words = sentence.split()
    if len(words) <= MAX_WORDS_PER_CHUNK:
        return [sentence]

    # Split in half
    mid = len(words) // 2
    first_half = " ".join(words[:mid])
    second_half = " ".join(words[mid:])

    # Recursively split if still too long
    return split_long_sentence(first_half) + split_long_sentence(second_half)


def chunk_text(text: str) -> list[str]:
    """Split text into sentences, then split long sentences in half."""
    # Split on sentence-ending punctuation, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            chunks.extend(split_long_sentence(sentence))

    return chunks


import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool for running blocking TTS operations
_tts_executor = ThreadPoolExecutor(max_workers=1)


def _run_speak(text: str, voice_file: Path, voice_id: str, voice_name: str) -> dict:
    """Run the TTS generation and playback synchronously."""
    global tts_model, _stop_requested

    chunks = chunk_text(text)
    if not chunks:
        return {"status": "success", "text": text, "chunks": 0}

    # Queue for passing generated audio files from producer to consumer
    audio_queue: queue.Queue[str | None] = queue.Queue()
    generation_error: list[Exception] = []

    def producer():
        """Generate audio for all chunks and put file paths in queue."""
        global _stop_requested
        try:
            for chunk in chunks:
                # Check if stop was requested
                with _stop_lock:
                    if _stop_requested:
                        break

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name

                tts_model.infer(
                    spk_audio_prompt=str(voice_file),
                    text=chunk,
                    output_path=tmp_path,
                    verbose=False,
                )

                # Check again after generation
                with _stop_lock:
                    if _stop_requested:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        break

                audio_queue.put(tmp_path)
        except Exception as e:
            generation_error.append(e)
        finally:
            # Signal end of generation
            audio_queue.put(None)

    def consumer():
        """Play audio files from queue in order."""
        global _stop_requested
        while True:
            # Check if stop was requested
            with _stop_lock:
                if _stop_requested:
                    # Drain remaining items from queue
                    while not audio_queue.empty():
                        try:
                            path = audio_queue.get_nowait()
                            if path and os.path.exists(path):
                                os.unlink(path)
                        except queue.Empty:
                            break
                    break

            # Use timeout to periodically check stop flag
            try:
                audio_path = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if audio_path is None:
                break
            try:
                played = play_audio_pipewire(audio_path)
                if not played:
                    # Stopped early, drain queue and exit
                    while not audio_queue.empty():
                        try:
                            path = audio_queue.get_nowait()
                            if path and os.path.exists(path):
                                os.unlink(path)
                        except queue.Empty:
                            break
                    break
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

    # Start producer thread (generates audio)
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    # Run consumer (plays audio)
    try:
        consumer()
    finally:
        producer_thread.join()

    if generation_error:
        raise generation_error[0]

    # Log to command history
    try:
        db.log_command(
            text=text,
            voice_id=voice_id,
            voice_name=voice_name,
            char_count=len(text),
            chunk_count=len(chunks),
        )
    except Exception as e:
        print(f"Warning: failed to log command history: {e}")

    return {"status": "success", "text": text, "chunks": len(chunks)}


@app.post("/speak")
async def speak(request: SpeakRequest):
    """Generate speech from text and play it with streaming chunks.

    Uses producer-consumer pattern: generates all chunks as fast as possible
    while playing them in order. This eliminates pauses when short sentences
    are followed by long ones.
    """
    global tts_model, _stop_requested, _paused

    # Reset stop and pause flags at start of new speech
    with _stop_lock:
        _stop_requested = False
    with _pause_lock:
        _paused = False

    if tts_model is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded. Check server logs for errors."
        )

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    voice_file = voice_config.get_active_voice_path()
    if not voice_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Voice file not found: {voice_file}"
        )

    # Resolve active voice info for history logging
    active_id = voice_config.get_active_voice_id()
    voice_info = voice_config.get_voice(active_id)
    active_name = voice_info["name"] if voice_info else active_id

    # Run TTS in thread pool to not block the event loop
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _tts_executor,
            _run_speak,
            request.text.strip(),
            voice_file,
            active_id,
            active_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop")
async def stop_playback():
    """Stop current TTS playback and cancel pending generation."""
    global _stop_requested, _paused

    with _stop_lock:
        _stop_requested = True
    with _pause_lock:
        _paused = False

    # Kill any running pw-play processes
    subprocess.run(["pkill", "-f", "pw-play"], capture_output=True)

    # Clear the status files and update waybar
    TTS_STATUS_FILE.unlink(missing_ok=True)
    TTS_PAUSED_FILE.unlink(missing_ok=True)
    subprocess.run(["pkill", "-RTMIN+9", "waybar"], capture_output=True)

    return {"status": "stopped"}


@app.post("/pause")
async def pause_playback():
    """Pause current TTS playback. Generation continues in background."""
    global _paused

    with _pause_lock:
        _paused = True

    _update_tts_status(paused=True)
    return {"status": "paused"}


@app.post("/resume")
async def resume_playback():
    """Resume paused TTS playback."""
    global _paused

    with _pause_lock:
        _paused = False

    _update_tts_status(paused=False)
    return {"status": "resumed"}


@app.post("/toggle-pause")
async def toggle_pause():
    """Toggle pause/resume state."""
    global _paused

    with _pause_lock:
        _paused = not _paused
        is_paused = _paused

    _update_tts_status(paused=is_paused)
    return {"status": "paused" if is_paused else "playing"}


@app.get("/status")
async def get_status():
    """Get current TTS status."""
    with _pause_lock:
        is_paused = _paused
    is_playing = TTS_STATUS_FILE.exists()
    return {
        "playing": is_playing,
        "paused": is_paused,
    }


@app.post("/repeat")
async def repeat_last():
    """Re-speak the most recent command from history using its original voice."""
    global tts_model, _stop_requested, _paused

    # Reset stop and pause flags at start of new speech
    with _stop_lock:
        _stop_requested = False
    with _pause_lock:
        _paused = False

    if tts_model is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded. Check server logs for errors."
        )

    last = db.get_last_command()
    if last is None:
        raise HTTPException(status_code=404, detail="No command history found")

    # Try to use the original voice; fall back to active voice if it no longer exists
    voice_id = last["voice_id"]
    voice_name = last["voice_name"]
    try:
        voice_file = voice_config.get_voice_path(voice_id)
        if not voice_file.exists():
            raise ValueError("file missing")
    except (ValueError, KeyError):
        voice_file = voice_config.get_active_voice_path()
        voice_id = voice_config.get_active_voice_id()
        voice_info = voice_config.get_voice(voice_id)
        voice_name = voice_info["name"] if voice_info else voice_id

    if not voice_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Voice file not found: {voice_file}"
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _tts_executor,
            _run_speak,
            last["text"],
            voice_file,
            voice_id,
            voice_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(
    days: int = 30,
    voice_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Query command history with optional filters."""
    return db.get_history(days=days, voice_id=voice_id, limit=limit, offset=offset)


@app.get("/voices", response_model=list[VoiceResponse])
async def list_voices():
    """List all available voices."""
    return voice_config.list_voices()


@app.post("/voices/select/{voice_id}")
async def select_voice(voice_id: str):
    """Set a voice as the active/default voice."""
    if voice_config.set_active_voice(voice_id):
        return {"status": "success", "active_voice": voice_id}
    raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice by ID."""
    try:
        if voice_config.delete_voice(voice_id):
            return {"status": "success", "deleted": voice_id}
        raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/voices/test")
async def test_voice(request: TestVoiceRequest):
    """Test a voice without changing the default.

    Returns the generated audio file path for playback.
    """
    global tts_model

    if tts_model is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded."
        )

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        voice_path = voice_config.get_voice_path(request.voice_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Voice not found: {request.voice_id}")

    if not voice_path.exists():
        raise HTTPException(status_code=500, detail=f"Voice file not found: {voice_path}")

    # Generate to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    try:
        tts_model.infer(
            spk_audio_prompt=str(voice_path),
            text=request.text.strip(),
            output_path=output_path,
            verbose=False,
        )
        return {"status": "success", "audio_path": output_path}
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    voice_file = voice_config.get_active_voice_path()
    return HealthResponse(
        status="healthy" if tts_model is not None else "model_not_loaded",
        model_loaded=tts_model is not None,
        voice_file=str(voice_file),
        device=tts_model.device if tts_model else "N/A",
    )


# Mount Gradio UI after model is loaded (called from lifespan)
def mount_gradio_ui():
    """Mount Gradio UI to FastAPI app. Called after lifespan initializes model."""
    global app
    from server.ui import create_ui
    gradio_app = create_ui(tts_model)
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")


if __name__ == "__main__":
    import uvicorn

    # Mount Gradio UI before starting server
    # Note: tts_model is None here, UI will use it after lifespan loads it
    from server.ui import create_ui
    gradio_app = create_ui(tts_model)
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")

    print("Starting server...")
    print("API: http://127.0.0.1:5858")
    print("UI:  http://127.0.0.1:5858/ui")
    uvicorn.run(app, host="127.0.0.1", port=5858)
