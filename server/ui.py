"""
Gradio UI for voice selection and testing.
"""

import tempfile
from pathlib import Path

import gradio as gr

import server.config as config
from server.config import voice_config, VOICES_DIR


def get_tts_model():
    """Get the TTS model from shared config at runtime."""
    return config.tts_model


def create_ui(tts_model=None):
    """Create the Gradio UI for voice management.

    Args:
        tts_model: Deprecated, kept for compatibility. Model is fetched at runtime.
    """

    def get_voice_choices():
        """Get voice choices for dropdown."""
        voices = voice_config.list_voices()
        return [(f"{v['name']} - {v['description']}", v["id"]) for v in voices]

    def get_active_voice_id():
        """Get currently active voice ID."""
        return voice_config.get_active_voice_id()

    def get_voice_preview_path(voice_id: str) -> str | None:
        """Get the original voice file path for preview."""
        if not voice_id:
            return None
        try:
            path = voice_config.get_voice_path(voice_id)
            return str(path) if path.exists() else None
        except ValueError:
            return None

    def set_default_voice(voice_id: str) -> str:
        """Set the selected voice as default."""
        if not voice_id:
            return "Please select a voice first."
        if voice_config.set_active_voice(voice_id):
            voice = voice_config.get_voice(voice_id)
            name = voice["name"] if voice else voice_id
            return f"Default voice set to: {name}"
        return "Failed to set default voice."

    def test_voice(voice_id: str, text: str) -> tuple[str | None, str]:
        """Generate speech with selected voice."""
        model = get_tts_model()

        if not voice_id:
            return None, "Please select a voice."
        if not text.strip():
            return None, "Please enter some text."
        if model is None:
            return None, "TTS model not loaded. Please wait for server to finish loading."

        try:
            voice_path = voice_config.get_voice_path(voice_id)
            if not voice_path.exists():
                return None, f"Voice file not found: {voice_path}"

            # Generate audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

            model.infer(
                spk_audio_prompt=str(voice_path),
                text=text,
                output_path=output_path,
                verbose=False,
            )

            return output_path, "Audio generated successfully."
        except Exception as e:
            return None, f"Error: {str(e)}"

    def add_new_voice(file, name: str, description: str) -> tuple[str, dict]:
        """Add a new voice from uploaded file."""
        if file is None:
            return "Please upload an audio file.", gr.update()
        if not name.strip():
            return "Please enter a voice name.", gr.update()

        try:
            file_path = Path(file.name if hasattr(file, 'name') else file)
            voice = voice_config.add_voice(file_path, name.strip(), description.strip())
            new_choices = get_voice_choices()
            return f"Voice '{voice['name']}' added successfully!", gr.update(choices=new_choices)
        except Exception as e:
            return f"Error adding voice: {str(e)}", gr.update()

    def on_voice_change(voice_id: str):
        """Update preview when voice changes."""
        preview_path = get_voice_preview_path(voice_id)
        return preview_path

    # Build the UI
    with gr.Blocks(title="Omarchy Local TTS", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Omarchy Local TTS - Voice Manager")

        with gr.Row():
            with gr.Column(scale=2):
                voice_dropdown = gr.Dropdown(
                    choices=get_voice_choices(),
                    value=get_active_voice_id(),
                    label="Select Voice",
                    interactive=True,
                )

                set_default_btn = gr.Button("Set as Default", variant="secondary")
                status_text = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Test Voice")
                test_input = gr.Textbox(
                    label="Text to speak",
                    placeholder="Enter text to test the voice...",
                    lines=3,
                )
                test_btn = gr.Button("Test Voice", variant="primary")
                test_output = gr.Audio(label="Generated Audio", type="filepath")

            with gr.Column(scale=1):
                gr.Markdown("### Voice Preview")
                gr.Markdown("*Original voice sample*")
                preview_audio = gr.Audio(
                    label="Original Voice Sample",
                    value=get_voice_preview_path(get_active_voice_id()),
                    type="filepath",
                    interactive=False,
                )

        with gr.Accordion("Add New Voice", open=False):
            with gr.Row():
                with gr.Column():
                    upload_file = gr.File(
                        label="Upload Voice File",
                        file_types=["audio"],
                    )
                    new_voice_name = gr.Textbox(
                        label="Voice Name",
                        placeholder="e.g., John",
                    )
                    new_voice_desc = gr.Textbox(
                        label="Description",
                        placeholder="e.g., warm and friendly narrator",
                    )
                    add_voice_btn = gr.Button("Add Voice", variant="secondary")
                    add_status = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        voice_dropdown.change(
            fn=on_voice_change,
            inputs=[voice_dropdown],
            outputs=[preview_audio],
        )

        set_default_btn.click(
            fn=set_default_voice,
            inputs=[voice_dropdown],
            outputs=[status_text],
        )

        test_btn.click(
            fn=test_voice,
            inputs=[voice_dropdown, test_input],
            outputs=[test_output, status_text],
        )

        add_voice_btn.click(
            fn=add_new_voice,
            inputs=[upload_file, new_voice_name, new_voice_desc],
            outputs=[add_status, voice_dropdown],
        )

    return ui
