from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.endpoints import BlindEndpoints
from server.orchestrator import AppOrchestrator
from server.session_store import InMemorySessionStore


def create_app() -> FastAPI:
    app = FastAPI()

    repo_root = os.path.dirname(os.path.dirname(__file__))
    html_path = os.path.join(repo_root, "designs", "gemma_guide.html")
    designs_dir = os.path.join(repo_root, "designs")

    app.mount("/designs", StaticFiles(directory=designs_dir), name="designs")

    session_store = InMemorySessionStore()
    orchestrator = AppOrchestrator(session_store)
    BlindEndpoints(orchestrator, html_path).register(app)
    return app
