from __future__ import annotations

import io
from pathlib import Path
import logging

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydub import AudioSegment

from server.contracts import QueryInput
from server.orchestrator import AppOrchestrator


logger = logging.getLogger(__name__)


class BlindEndpoints:
    def __init__(self, orchestrator: AppOrchestrator, html_path: str) -> None:
        self._orchestrator = orchestrator
        self._html_path = html_path

    def register(self, app: FastAPI) -> None:
        app.add_api_route("/", self.blind_ui, methods=["GET"], response_class=HTMLResponse)
        app.add_api_route("/api/query", self.api_query, methods=["POST"])
        app.add_api_route("/api/session/{session_id}", self.clear_session, methods=["DELETE"])

    async def blind_ui(self) -> str:
        return Path(self._html_path).read_text(encoding="utf-8")

    async def api_query(
        self,
        image: UploadFile | None = File(None),
        audio: UploadFile | None = File(None),
        session_id: str = Form(""),
    ) -> JSONResponse:
        query = QueryInput(
            session_id=session_id,
            uploaded_image=await self._load_image(image),
            question=await self._load_question(audio),
        )
        result = self._orchestrator.handle_query(query)
        return JSONResponse(
            {
                "response": result.response,
                "route": result.route,
                "metrics": result.metrics,
                "debug": {
                    "active_image_b64": result.debug.active_image_b64,
                    "depth_b64": result.debug.depth_b64,
                    "navigator_image_b64": result.debug.navigator_image_b64,
                    "measurements": result.debug.measurements,
                    "history": result.debug.history,
                },
            }
        )

    async def clear_session(self, session_id: str) -> dict[str, bool]:
        self._orchestrator.clear_session(session_id)
        return {"ok": True}

    async def _load_image(self, image: UploadFile | None) -> Image.Image | None:
        if image is None or not image.filename:
            return None
        raw = await image.read()
        if not raw:
            return None
        try:
            return Image.open(io.BytesIO(raw))
        except Exception:
            logger.warning("Could not decode uploaded image")
            return None

    async def _load_question(self, audio: UploadFile | None) -> str | bytes:
        question: str | bytes = "What do you see around me?"
        if audio is None or not audio.filename:
            return question

        raw_audio = await audio.read()
        if not raw_audio or len(raw_audio) <= 500:
            return question
        try:
            seg = AudioSegment.from_file(io.BytesIO(raw_audio))
            seg = seg.set_channels(1).set_sample_width(2)
            wav_buf = io.BytesIO()
            seg.export(wav_buf, format="wav")
            return wav_buf.getvalue()
        except Exception as exc:
            logger.warning("Audio conversion failed: %s", exc)
            return question
