from __future__ import annotations

import io
import logging
import wave

import numpy as np
import gradio as gr
from PIL import Image

from pipeline.session import create_session
from pipeline.tts import synthesize
from server.agent import run_agent_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _numpy_to_wav_bytes(audio_np: tuple[int, np.ndarray] | None) -> bytes | None:
    if audio_np is None:
        return None
    sample_rate, data = audio_np
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if data.ndim > 1:
            data = data[:, 0]
        wf.writeframes(data.astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()


def _tts_to_numpy(text: str) -> tuple[int, np.ndarray] | None:
    try:
        audio_bytes = synthesize(text)
        buf = io.BytesIO(audio_bytes)
        try:
            with wave.open(buf, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                return sr, np.frombuffer(frames, dtype=np.int16)
        except Exception:
            # gTTS outputs MP3 — return as raw uint8 cast at 24kHz
            buf.seek(0)
            return 24000, np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16)
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
        return None


def process(
    image: Image.Image | None,
    audio_input: tuple[int, np.ndarray] | None,
    text_question: str,
) -> tuple[str, tuple[int, np.ndarray] | None, Image.Image | None]:

    # ── Input validation ─────────────────────────────────────────────────────
    if image is None:
        msg = "Please upload an image before submitting."
        return msg, _tts_to_numpy(msg), None

    audio_bytes = _numpy_to_wav_bytes(audio_input)
    question_text = text_question.strip() if text_question else ""

    if audio_bytes is None and not question_text:
        msg = "Please record a question or type one before submitting."
        return msg, _tts_to_numpy(msg), None

    if audio_bytes is not None and len(audio_bytes) < 1000:
        if not question_text:
            msg = "I couldn't hear a question. Please try recording again or type your question."
            return msg, _tts_to_numpy(msg), None
        audio_bytes = None  # Too short — fall through to text question

    # ── Image quality check ──────────────────────────────────────────────────
    img_array = np.array(image.convert("RGB"))
    if img_array.mean() < 20:
        msg = (
            "The image appears very dark. "
            "Please take a photo in better lighting for accurate distance measurements."
        )
        return msg, _tts_to_numpy(msg), None

    # ── Determine question payload ────────────────────────────────────────────
    # Prefer audio (native Gemma multimodal); fall back to text if audio absent
    question: str | bytes = audio_bytes if audio_bytes is not None else question_text

    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        session = create_session(image, question)
        response_text = run_agent_loop(session)

        if session.intrinsics.source == "assumed" and "degree" in response_text.lower():
            response_text += (
                " (Note: bearing is approximate — no camera focal length data was available.)"
            )

        depth_display = session.depth_colormap
        session.release()

    except ConnectionRefusedError:
        response_text = (
            "Could not connect to the Gemma model server. "
            "Please make sure the vLLM server is running — see quickstart.md."
        )
        depth_display = None
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        response_text = (
            "Something went wrong during analysis. "
            "Please try again with a different image or question."
        )
        depth_display = None

    return response_text, _tts_to_numpy(response_text), depth_display


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="SpatialSense — Blind Navigation Assistant") as demo:
    gr.Markdown(
        "## SpatialSense\n"
        "Upload a photo, record or type a question, and receive a spoken navigation response."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Camera Image")
            audio_input = gr.Audio(
                sources=["microphone"], type="numpy", label="Spoken Question"
            )
            text_input = gr.Textbox(
                label="Typed Question (optional)",
                placeholder="e.g. how far is the chair?",
            )
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            text_output = gr.Textbox(label="Response", lines=4)
            audio_output = gr.Audio(type="numpy", label="Spoken Response", autoplay=True)
            depth_output = gr.Image(type="pil", label="Depth Map (demo display)")

    submit_btn.click(
        fn=process,
        inputs=[image_input, audio_input, text_input],
        outputs=[text_output, audio_output, depth_output],
    )

if __name__ == "__main__":
    demo.launch()
