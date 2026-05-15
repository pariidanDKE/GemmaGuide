from __future__ import annotations

from PIL import Image

from pipeline.session import Session
from server.agents import MapperAgent, NavigatorAgent, ScoutAgent


def run_scout_loop(
    session: Session,
    history: list[dict] | None = None,
    send_image: bool = True,
    has_active_image: bool = True,
    image_source: str = "none",
) -> tuple[str, str, list[dict]]:
    return ScoutAgent(session).run(
        history=history,
        send_image=send_image,
        has_active_image=has_active_image,
        image_source=image_source,
    )


def run_mapper_loop(
    session: Session,
    history: list[dict] | None = None,
    prior_measurements: list[dict] | None = None,
    prior_turn_count: int = 0,
    fresh_image_attached: bool = False,
    image_source: str = "none",
) -> None:
    MapperAgent(session).run(
        history=history,
        prior_measurements=prior_measurements,
        prior_turn_count=prior_turn_count,
        fresh_image_attached=fresh_image_attached,
        image_source=image_source,
    )


def run_navigator_loop(
    session: Session,
    annotated_image: Image.Image | None,
    original_image: Image.Image | None = None,
    history: list[dict] | None = None,
    send_image: bool = True,
    image_source: str = "none",
) -> tuple[str, list[dict]]:
    return NavigatorAgent(session).run(
        annotated_image=annotated_image,
        original_image=original_image,
        history=history,
        send_image=send_image,
        image_source=image_source,
    )
