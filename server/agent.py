from __future__ import annotations

from PIL import Image

from pipeline.session import Session
from server.agents import MapperAgent, NavigatorAgent, ScoutAgent


def run_scout_loop(
    session: Session,
    history: list[dict] | None = None,
    send_image: bool = True,
    has_active_image: bool = True,
) -> tuple[str, str, list[dict]]:
    return ScoutAgent(session).run(
        history=history,
        send_image=send_image,
        has_active_image=has_active_image,
    )


def run_mapper_loop(
    session: Session,
    history: list[dict] | None = None,
    prior_measurements: list[dict] | None = None,
    prior_turn_count: int = 0,
    fresh_image_attached: bool = False,
) -> None:
    MapperAgent(session).run(
        history=history,
        prior_measurements=prior_measurements,
        prior_turn_count=prior_turn_count,
        fresh_image_attached=fresh_image_attached,
    )


def run_navigator_loop(
    session: Session,
    annotated_image: Image.Image | None,
    history: list[dict] | None = None,
    send_image: bool = True,
) -> tuple[str, list[dict]]:
    return NavigatorAgent(session).run(
        annotated_image=annotated_image,
        history=history,
        send_image=send_image,
    )
