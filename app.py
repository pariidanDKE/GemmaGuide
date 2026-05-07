from __future__ import annotations

import hashlib
import io
import logging
import os
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


def _strip_markdown(text: str) -> str:
    import re
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def _tts_to_numpy(text: str) -> tuple[int, np.ndarray] | None:
    try:
        from pydub import AudioSegment
        audio_bytes = synthesize(_strip_markdown(text))
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        seg = seg.set_channels(1).set_sample_width(2)
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
        return seg.frame_rate, samples
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
        return None


def _image_hash(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def process(
    image: Image.Image | None,
    audio_input: tuple[int, np.ndarray] | None,
    text_question: str,
    history: list | None,
    last_image_hash: str | None,
) -> tuple[str, tuple[int, np.ndarray] | None, Image.Image | None, list | None, str | None]:

    # ── Input validation ─────────────────────────────────────────────────────
    if image is None:
        msg = "Please upload an image before submitting."
        return msg, _tts_to_numpy(msg), None, history, last_image_hash

    audio_bytes = _numpy_to_wav_bytes(audio_input)
    question_text = text_question.strip() if text_question else ""

    if audio_bytes is None and not question_text:
        msg = "Please record a question or type one before submitting."
        return msg, _tts_to_numpy(msg), None, history, last_image_hash

    if audio_bytes is not None and len(audio_bytes) < 1000:
        if not question_text:
            msg = "I couldn't hear a question. Please try recording again or type your question."
            return msg, _tts_to_numpy(msg), None, history, last_image_hash
        audio_bytes = None  # Too short — fall through to text question

    # ── Image quality check ──────────────────────────────────────────────────
    img_array = np.array(image.convert("RGB"))
    if img_array.mean() < 20:
        msg = (
            "The image appears very dark. "
            "Please take a photo in better lighting for accurate distance measurements."
        )
        return msg, _tts_to_numpy(msg), None, history, last_image_hash

    # ── Determine question payload ────────────────────────────────────────────
    # Prefer audio (native Gemma multimodal); fall back to text if audio absent
    question: str | bytes = audio_bytes if audio_bytes is not None else question_text

    # ── Conversation history ──────────────────────────────────────────────────
    current_hash = _image_hash(image)
    image_changed = last_image_hash is not None and current_hash != last_image_hash
    is_first_turn = history is None
    send_image = is_first_turn or image_changed

    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        session = create_session(image, question)
        response_text, updated_history = run_agent_loop(
            session,
            history=None if is_first_turn else history,
            send_image=send_image,
        )

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
        updated_history = history
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        response_text = (
            "Something went wrong during analysis. "
            "Please try again with a different image or question."
        )
        depth_display = None
        updated_history = history

    return response_text, _tts_to_numpy(response_text), depth_display, updated_history, current_hash


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="SpatialSense — Blind Navigation Assistant") as demo:
    history_state = gr.State(None)
    image_hash_state = gr.State(None)

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
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                new_scene_btn = gr.Button("New Scene", variant="secondary")
        with gr.Column():
            text_output = gr.Textbox(label="Response", lines=4)
            audio_output = gr.Audio(type="numpy", label="Spoken Response", autoplay=True)
            depth_output = gr.Image(type="pil", label="Depth Map (demo display)")

    submit_btn.click(
        fn=process,
        inputs=[image_input, audio_input, text_input, history_state, image_hash_state],
        outputs=[text_output, audio_output, depth_output, history_state, image_hash_state],
    )
    new_scene_btn.click(
        fn=lambda: (None, None),
        outputs=[history_state, image_hash_state],
    )

if __name__ == "__main__":
    import httpx
    _orig_httpx_get = httpx.get
    def _httpx_get_no_verify_localhost(url, **kwargs):
        if isinstance(url, str) and any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
            kwargs["verify"] = False
        return _orig_httpx_get(url, **kwargs)
    httpx.get = _httpx_get_no_verify_localhost

    demo.launch(
        server_name="0.0.0.0",
        ssl_certfile="/home/dan-parii/ml-workstation.tail9deb72.ts.net.crt",
        ssl_keyfile="/home/dan-parii/ml-workstation.tail9deb72.ts.net.key",
    )
