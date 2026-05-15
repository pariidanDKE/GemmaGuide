from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class QueryInput:
    session_id: str
    uploaded_image: Image.Image | None
    question: str | bytes


@dataclass
class SessionContext:
    history: list[dict] | None
    prior_measurements: list[dict] | None
    cached_image: Image.Image | None
    metrics: dict[str, Any] | None


@dataclass
class DebugPayload:
    active_image_b64: str | None
    depth_b64: str | None
    navigator_image_b64: str | None
    measurements: list[dict]
    history: list[dict]


@dataclass
class QueryResult:
    response: str
    route: str
    metrics: dict[str, Any]
    debug: DebugPayload
