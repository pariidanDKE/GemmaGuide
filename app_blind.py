"""Blind-first FastAPI server — serves custom HTML UI and pipeline API."""
from __future__ import annotations

import base64
import io
import logging
import os
import re
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
from pydub import AudioSegment

from pipeline.session import create_session
from pipeline.tts import synthesize
from server.agent import build_turn_user_content, run_mapper_loop, run_navigator_loop, run_scout_loop
from pipeline.tools import render_annotated_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Server-side session data keyed by session_id.
# Each entry stores:
# - shared user/assistant conversation history
# - the last set of measurements (tensors stripped)
# - the last image so follow-up turns can still enter the spatial pipeline
_sessions: dict[str, dict] = {}

_HTML = os.path.join(os.path.dirname(__file__), "designs", "blind_first_phone_v2.html")
_DESIGNS_DIR = os.path.join(os.path.dirname(__file__), "designs")

app.mount("/designs", StaticFiles(directory=_DESIGNS_DIR), name="designs")


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def _summarize_metrics(summary: dict[str, float], counts: dict[str, int], limit: int = 10) -> list[dict]:
    ranked = sorted(summary.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "stage": stage,
            "seconds": round(seconds, 3),
            "count": int(counts.get(stage, 0)),
        }
        for stage, seconds in ranked[:limit]
    ]


def _merge_metrics(existing: dict | None, turn_metrics: dict) -> dict:
    merged = {
        "turns": list((existing or {}).get("turns", [])),
        "summary": dict((existing or {}).get("summary", {})),
        "counts": dict((existing or {}).get("counts", {})),
    }
    merged["turns"].append({
        "summary": dict(turn_metrics.get("summary", {})),
        "counts": dict(turn_metrics.get("counts", {})),
    })
    for stage, seconds in turn_metrics.get("summary", {}).items():
        merged["summary"][stage] = round(float(merged["summary"].get(stage, 0.0)) + float(seconds), 3)
    for stage, count in turn_metrics.get("counts", {}).items():
        merged["counts"][stage] = int(merged["counts"].get(stage, 0) + int(count))
    return merged


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
    session_data = _sessions.get(session_id) or {}
    shared_history = session_data.get("history")
    prior_measurements = session_data.get("measurements")
    cached_image = session_data.get("image")
    active_image = pil_image if pil_image is not None else cached_image

    # Only send the image on the first turn — subsequent turns already have
    # it in the shared conversation history.
    send_image = active_image is not None and shared_history is None

    # ── Run pipeline ──────────────────────────────────────────
    response_text = ""
    depth_b64: str | None = None
    response_route = "direct"
    turn_metrics: dict = {"timings": [], "summary": {}, "counts": {}}
    session_metrics_state = session_data.get("metrics")
    request_t0 = time.monotonic()

    try:
        scout_session = create_session(active_image, question)
        route, scout_text, _scout_trace = run_scout_loop(
            scout_session,
            history=shared_history,
            send_image=send_image,
        )
        response_route = route

        if route == "navigator":
            session = create_session(
                active_image,
                question,
                intrinsics=scout_session.intrinsics,
                metrics=scout_session.metrics,
            )

            # Mapper: populates session.measurements via tool calls.
            if active_image is not None:
                run_mapper_loop(
                    session,
                    history=shared_history,
                    prior_measurements=prior_measurements,
                )

            # Render annotated image for Navigator (always, so summary boxes are current)
            annotated = render_annotated_image(session) if active_image is not None else None

            if session.depth_colormap:
                buf = io.BytesIO()
                session.depth_colormap.save(buf, format="JPEG", quality=85)
                depth_b64 = base64.b64encode(buf.getvalue()).decode()

            # Navigator: produces all user-facing text; receives annotated image + scene summary
            nav_image = annotated if annotated is not None else active_image
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
            if route == "restart":
                measurement_state = None
                next_image = None
                next_history = []
            else:
                measurement_state = prior_measurements
                next_image = active_image
                next_history = list(shared_history or [])

        scout_session.release()
        total_request_seconds = time.monotonic() - request_t0
        turn_metrics["timings"].append({"stage": "request.total", "seconds": round(total_request_seconds, 3)})
        turn_metrics["summary"]["request.total"] = round(total_request_seconds, 3)
        turn_metrics["counts"]["request.total"] = 1
        session_metrics_state = _merge_metrics(session_metrics_state, turn_metrics)

        if session_id:
            updated_shared_history = next_history
            if route != "restart":
                turn_user_content = build_turn_user_content(active_image, question, send_image=send_image)
                updated_shared_history.append({"role": "user", "content": turn_user_content})
                updated_shared_history.append({"role": "assistant", "content": response_text})
            _sessions[session_id] = {
                "history": updated_shared_history,
                "measurements": measurement_state,
                "image": next_image,
                "metrics": session_metrics_state,
            }

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
        tts_t0 = time.monotonic()
        tts_bytes = synthesize(_strip_markdown(response_text))
        audio_b64 = base64.b64encode(tts_bytes).decode()
        tts_seconds = round(time.monotonic() - tts_t0, 3)
        turn_metrics["timings"].append({"stage": "tts.total", "seconds": tts_seconds})
        turn_metrics["summary"]["tts.total"] = round(float(turn_metrics["summary"].get("tts.total", 0.0)) + tts_seconds, 3)
        turn_metrics["counts"]["tts.total"] = int(turn_metrics["counts"].get("tts.total", 0) + 1)
        if session_metrics_state is not None:
            session_metrics_state["summary"]["tts.total"] = round(float(session_metrics_state["summary"].get("tts.total", 0.0)) + tts_seconds, 3)
            session_metrics_state["counts"]["tts.total"] = int(session_metrics_state["counts"].get("tts.total", 0) + 1)
            if session_metrics_state.get("turns"):
                latest_turn = session_metrics_state["turns"][-1]
                latest_turn["summary"]["tts.total"] = round(float(latest_turn["summary"].get("tts.total", 0.0)) + tts_seconds, 3)
                latest_turn["counts"]["tts.total"] = int(latest_turn["counts"].get("tts.total", 0) + 1)
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)

    metrics_payload = {
        "turn_top": _summarize_metrics(turn_metrics.get("summary", {}), turn_metrics.get("counts", {})),
        "session_top": _summarize_metrics(
            (session_metrics_state or {}).get("summary", {}),
            (session_metrics_state or {}).get("counts", {}),
        ),
    }

    return JSONResponse({
        "response": response_text,
        "route": response_route,
        "audio_b64": audio_b64,
        "depth_b64": depth_b64,
        "metrics": metrics_payload,
    })


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


def _patch_httpx_localhost_verify() -> None:
    import httpx

    orig_get = httpx.get

    def _httpx_no_verify_localhost(url, **kwargs):
        if isinstance(url, str) and any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
            kwargs["verify"] = False
        return orig_get(url, **kwargs)

    httpx.get = _httpx_no_verify_localhost


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
    _patch_httpx_localhost_verify()
    _run_server()
