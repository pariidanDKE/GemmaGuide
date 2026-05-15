"""Blind-first FastAPI server — serves custom HTML UI and pipeline API."""
from __future__ import annotations

import base64
import io
import logging
import os
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
from pydub import AudioSegment

from pipeline.session import (
    add_metric_sample,
    create_session,
    merge_metrics,
    summarize_metrics,
)
from pipeline.debug_render import render_annotated_image
from pipeline.tts import synthesize
from server.media import image_to_jpeg_b64
from server.messages import build_turn_user_content
from server.runtime import patch_httpx_localhost_verify, strip_markdown
from server.session_store import InMemorySessionStore, SessionState
from server.agent import run_mapper_loop, run_navigator_loop, run_scout_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

_session_store = InMemorySessionStore()

_HTML = os.path.join(os.path.dirname(__file__), "designs", "gemma_guide.html")
_DESIGNS_DIR = os.path.join(os.path.dirname(__file__), "designs")

app.mount("/designs", StaticFiles(directory=_DESIGNS_DIR), name="designs")

_EMPTY_RESPONSE_FALLBACK = (
    "I could not produce a spoken response for that request. Please try again."
)
_MISSING_IMAGE_SPATIAL_FALLBACK = (
    "I don't currently have a scene photo to analyze. Please take the photo again, then ask that question once more."
)

def _history_for_debug(history: list[dict] | None) -> list[dict]:
    debug_items: list[dict] = []
    for message in history or []:
        role = str(message.get("role", "unknown"))
        content = message.get("content", "")
        if isinstance(content, str):
            preview = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text", "")))
                elif block_type == "image_url":
                    parts.append("[image]")
                elif block_type == "input_audio":
                    parts.append("[audio]")
                else:
                    parts.append(f"[{block_type or 'content'}]")
            preview = " ".join(part for part in parts if part).strip()
        else:
            preview = str(content)
        debug_items.append({"role": role, "text": preview})
    return debug_items

def _finalize_turn_metrics(session_metrics_state: dict | None, turn_metrics: dict, request_t0: float) -> dict:
    add_metric_sample(turn_metrics, "request.total", time.monotonic() - request_t0)
    return merge_metrics(session_metrics_state, turn_metrics)


def _record_tts_metrics(turn_metrics: dict, session_metrics_state: dict | None, tts_seconds: float) -> None:
    add_metric_sample(turn_metrics, "tts.total", tts_seconds)
    if session_metrics_state is not None:
        add_metric_sample(session_metrics_state, "tts.total", tts_seconds, include_timing=False)
        if session_metrics_state.get("turns"):
            latest_turn = session_metrics_state["turns"][-1]
            add_metric_sample(latest_turn, "tts.total", tts_seconds, include_timing=False)


def _persist_session_state(
    session_id: str,
    *,
    history: list[dict],
    measurements: list[dict] | None,
    image: Image.Image | None,
    metrics: dict,
) -> None:
    _session_store.set(session_id, SessionState(
        history=history,
        measurements=measurements,
        image=image,
        metrics=metrics,
    ))


def _append_turn_to_history(
    history: list[dict],
    *,
    active_image: Image.Image | None,
    question: str | bytes,
    send_image: bool,
    response_text: str,
) -> list[dict]:
    updated_history = list(history)
    turn_user_content = build_turn_user_content(active_image, question, send_image=send_image)
    updated_history.append({"role": "user", "content": turn_user_content})
    updated_history.append({"role": "assistant", "content": response_text})
    return updated_history


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

    # ── Retrieve session data ─────────────────────────────────
    session_state = _session_store.get(session_id)
    shared_history = session_state.history if session_state is not None else None
    prior_measurements = session_state.measurements if session_state is not None else None
    cached_image = session_state.image if session_state is not None else None
    active_image = pil_image if pil_image is not None else cached_image
    prior_turn_count = len(shared_history or [])
    logger.info(
        "api_query session_id=%s uploaded_image=%s cached_image=%s active_image=%s history=%s prior_measurements=%s",
        session_id or "<empty>",
        pil_image is not None,
        cached_image is not None,
        active_image is not None,
        shared_history is not None,
        prior_measurements is not None,
    )

    # Send an image on the first turn and whenever the user uploads a fresh one.
    # Relying only on prior conversation history causes follow-up turns with a new
    # image to reference stale visual context.
    send_image = active_image is not None and (shared_history is None or pil_image is not None)

    # ── Run pipeline ──────────────────────────────────────────
    response_text = ""
    depth_b64: str | None = None
    active_image_b64: str | None = None
    navigator_image_b64: str | None = None
    measurement_state: list[dict] | None = None
    response_route = "direct"
    turn_metrics: dict = {"timings": [], "summary": {}, "counts": {}}
    session_metrics_state = session_state.metrics if session_state is not None else None
    request_t0 = time.monotonic()

    try:
        scout_session = create_session(active_image, question)
        route, scout_text, _scout_trace = run_scout_loop(
            scout_session,
            history=shared_history,
            send_image=send_image,
            has_active_image=active_image is not None,
        )
        response_route = route

        if route == "navigator":
            if active_image is None:
                logger.warning(
                    "scout requested navigator without active image; forcing direct retry message session_id=%s",
                    session_id or "<empty>",
                )
                response_route = "direct"
                response_text = _MISSING_IMAGE_SPATIAL_FALLBACK
                turn_metrics = scout_session.export_metrics()
                active_image_b64 = None
                measurement_state = prior_measurements
                next_image = None
                next_history = list(shared_history or [])
                scout_session.release()
                session_metrics_state = _finalize_turn_metrics(session_metrics_state, turn_metrics, request_t0)
                _persist_session_state(
                    session_id,
                    history=_append_turn_to_history(
                        next_history,
                        active_image=active_image,
                        question=question,
                        send_image=send_image,
                        response_text=response_text,
                    ),
                    measurements=measurement_state,
                    image=next_image,
                    metrics=session_metrics_state,
                )
            else:
                session = create_session(
                    active_image,
                    question,
                    intrinsics=scout_session.intrinsics,
                    metrics=scout_session.metrics,
                )

                # Mapper: populates session.measurements via tool calls.
                run_mapper_loop(
                    session,
                    history=None,
                    prior_measurements=prior_measurements,
                    prior_turn_count=prior_turn_count,
                    fresh_image_attached=pil_image is not None,
                )

                # Render annotated image for Navigator (always, so summary boxes are current)
                annotated = render_annotated_image(session)

                if session.depth_colormap:
                    depth_b64 = image_to_jpeg_b64(
                        session.depth_colormap,
                        resize_to=active_image.size,
                    )

                # Navigator: produces all user-facing text; receives annotated image + scene summary
                nav_image = annotated
                active_image_b64 = image_to_jpeg_b64(active_image)
                navigator_image_b64 = image_to_jpeg_b64(nav_image)
                response_text, _nav_trace = run_navigator_loop(
                    session,
                    annotated_image=nav_image,
                    history=shared_history,
                    send_image=send_image,
                )
                turn_metrics = session.export_metrics()
                session.release()
                measurement_state = [
                    {k: v for k, v in m.items() if k != "mask_dpt"}
                    for m in session.measurements
                ]
                next_image = active_image
                next_history = list(shared_history or [])
        else:
            response_text = scout_text
            turn_metrics = scout_session.export_metrics()
            active_image_b64 = image_to_jpeg_b64(active_image)
            if route == "restart":
                measurement_state = None
                next_image = None
                next_history = []
            else:
                measurement_state = prior_measurements
                next_image = active_image
                next_history = list(shared_history or [])

        scout_session.release()
        session_metrics_state = _finalize_turn_metrics(session_metrics_state, turn_metrics, request_t0)

        updated_shared_history = list(next_history)
        if route != "restart":
            updated_shared_history = _append_turn_to_history(
                updated_shared_history,
                active_image=active_image,
                question=question,
                send_image=send_image,
                response_text=response_text,
            )
        _persist_session_state(
            session_id,
            history=updated_shared_history,
            measurements=measurement_state,
            image=next_image,
            metrics=session_metrics_state,
        )

    except ConnectionRefusedError:
        response_text = (
            "Could not connect to the Gemma model server. "
            "Please make sure the vLLM server is running."
        )
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        response_text = "Something went wrong. Please try again."

    if not response_text.strip():
        logger.warning("empty response_text for route=%s; using fallback", response_route)
        response_text = _EMPTY_RESPONSE_FALLBACK

    # ── TTS ───────────────────────────────────────────────────
    audio_b64: str | None = None
    try:
        tts_t0 = time.monotonic()
        tts_bytes = synthesize(strip_markdown(response_text))
        audio_b64 = base64.b64encode(tts_bytes).decode()
        tts_seconds = round(time.monotonic() - tts_t0, 3)
        _record_tts_metrics(turn_metrics, session_metrics_state, tts_seconds)
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
        if not audio_b64:
            response_text = _EMPTY_RESPONSE_FALLBACK if not response_text.strip() else response_text

    metrics_payload = {
        "turn_top": summarize_metrics(turn_metrics.get("summary", {}), turn_metrics.get("counts", {})),
        "session_top": summarize_metrics(
            (session_metrics_state or {}).get("summary", {}),
            (session_metrics_state or {}).get("counts", {}),
        ),
    }

    logger.info(
        "api_query route=%s response_len=%s audio_b64=%s depth_b64=%s",
        response_route,
        len(response_text),
        bool(audio_b64),
        bool(depth_b64),
    )

    debug_session_state = _session_store.get(session_id)
    debug_history = _history_for_debug(debug_session_state.history if debug_session_state is not None else None)
    debug_payload = {
        "active_image_b64": active_image_b64,
        "depth_b64": depth_b64,
        "navigator_image_b64": navigator_image_b64,
        "measurements": (debug_session_state.measurements if debug_session_state is not None else None) or measurement_state or [],
        "history": debug_history,
    }

    return JSONResponse({
        "response": response_text,
        "route": response_route,
        "audio_b64": audio_b64,
        "depth_b64": depth_b64,
        "metrics": metrics_payload,
        "debug": debug_payload,
    })


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    _session_store.delete(session_id)
    return {"ok": True}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_server() -> None:
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "7862"))
    ssl_certfile = os.getenv("APP_SSL_CERTFILE")
    ssl_keyfile = os.getenv("APP_SSL_KEYFILE")

    # Default to the existing Tailscale cert paths for local workstation use.
    # Notebook/tunnel mode sets APP_DISABLE_SSL=1 so uvicorn stays on plain HTTP.
    if not ssl_certfile:
        ssl_certfile = "/home/dan-parii/ml-workstation.tail9deb72.ts.net.crt"
    if not ssl_keyfile:
        ssl_keyfile = "/home/dan-parii/ml-workstation.tail9deb72.ts.net.key"
    if _env_flag("APP_DISABLE_SSL"):
        ssl_certfile = None
        ssl_keyfile = None

    logger.info(
        "Starting blind app on %s:%s ssl=%s",
        host,
        port,
        bool(ssl_certfile and ssl_keyfile),
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


if __name__ == "__main__":
    patch_httpx_localhost_verify()
    _run_server()
