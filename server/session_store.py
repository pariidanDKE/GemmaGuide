from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class SessionState:
    history: list[dict]
    measurements: list[dict] | None
    image: Image.Image | None
    metrics: dict[str, Any]


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._last_session_id: str | None = None

    def get(self, session_id: str) -> SessionState | None:
        if not session_id:
            return None
        return self._sessions.get(session_id)

    def set(self, session_id: str, state: SessionState) -> None:
        if not session_id:
            return
        self._sessions[session_id] = state
        self._last_session_id = session_id

    def get_latest(self) -> tuple[str | None, SessionState | None]:
        if not self._last_session_id:
            return None, None
        state = self._sessions.get(self._last_session_id)
        if state is None:
            return None, None
        return self._last_session_id, state

    def delete(self, session_id: str) -> None:
        if not session_id:
            return
        self._sessions.pop(session_id, None)
        if self._last_session_id == session_id:
            self._last_session_id = None
