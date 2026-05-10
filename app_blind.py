"""Blind-first FastAPI server — serves custom HTML UI and pipeline API."""
from __future__ import annotations

import base64
import io
import logging
import os
import re

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from PIL import Image
from pydub import AudioSegment

from pipeline.session import create_session
from pipeline.tts import synthesize
from server.agent import run_agent_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Server-side conversation history keyed by session_id
_sessions: dict[str, list | None] = {}

_HTML = os.path.join(os.path.dirname(__file__), "designs", "blind_first_phone_v2.html")


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


@app.get("/", response_class=HTMLResponse)
@app.get("/blind", response_class=HTMLResponse)
async def blind_ui():
    with open(_HTML, encoding="utf-8") as f:
        return f.read()


@app.post("/api/query")
async def api_query(
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    session_id: str = Form(""),
):
    # ── Load image ────────────────────────────────────────────
    pil_image: Image.Image | None = None
    if image and image.filename:
        raw = await image.read()
        if raw:
            try:
                pil_image = Image.open(io.BytesIO(raw))
            except Exception:
                logger.warning("Could not decode uploaded image")

    # ── Load and convert audio to WAV ─────────────────────────
    question: str | bytes = "What do you see around me?"
    if audio and audio.filename:
        raw_audio = await audio.read()
        if raw_audio and len(raw_audio) > 500:
            try:
                seg = AudioSegment.from_file(io.BytesIO(raw_audio))
                seg = seg.set_channels(1).set_sample_width(2)
                wav_buf = io.BytesIO()
                seg.export(wav_buf, format="wav")
                question = wav_buf.getvalue()
            except Exception as exc:
                logger.warning("Audio conversion failed: %s", exc)

    # ── Retrieve session history ──────────────────────────────
    history = _sessions.get(session_id)
    is_first = history is None
    # Only send the image on the first turn — subsequent turns already have
    # it in conversation history, and resending accumulates copies that hit
    # the model's per-prompt image limit.
    send_image = pil_image is not None and is_first

    # ── Run pipeline ──────────────────────────────────────────
    response_text = ""
    depth_b64: str | None = None

    try:
        session = create_session(pil_image, question)
        response_text, updated_history = run_agent_loop(
            session,
            history=None if is_first else history,
            send_image=send_image,
        )

        if session.depth_colormap:
            buf = io.BytesIO()
            session.depth_colormap.save(buf, format="JPEG", quality=85)
            depth_b64 = base64.b64encode(buf.getvalue()).decode()

        session.release()

        if session_id:
            _sessions[session_id] = updated_history

    except ConnectionRefusedError:
        response_text = (
            "Could not connect to the Gemma model server. "
            "Please make sure the vLLM server is running."
        )
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        response_text = "Something went wrong. Please try again."

    # ── TTS ───────────────────────────────────────────────────
    audio_b64: str | None = None
    try:
        tts_bytes = synthesize(_strip_markdown(response_text))
        audio_b64 = base64.b64encode(tts_bytes).decode()
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)

    return JSONResponse({
        "response": response_text,
        "audio_b64": audio_b64,
        "depth_b64": depth_b64,
    })


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


if __name__ == "__main__":
    import httpx

    _orig_get = httpx.get

    def _httpx_no_verify_localhost(url, **kwargs):
        if isinstance(url, str) and any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
            kwargs["verify"] = False
        return _orig_get(url, **kwargs)

    httpx.get = _httpx_no_verify_localhost

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7862,
        ssl_certfile="/home/dan-parii/ml-workstation.tail9deb72.ts.net.crt",
        ssl_keyfile="/home/dan-parii/ml-workstation.tail9deb72.ts.net.key",
    )
