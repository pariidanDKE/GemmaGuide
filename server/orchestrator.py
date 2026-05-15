from __future__ import annotations

import logging
import time
from typing import Any

from PIL import Image

from pipeline.debug_render import render_annotated_image
from pipeline.session import add_metric_sample, create_session, merge_metrics, summarize_metrics
from server.agent import run_mapper_loop, run_navigator_loop, run_scout_loop
from server.contracts import DebugPayload, QueryInput, QueryResult, SessionContext
from server.media import image_to_jpeg_b64
from server.messages import append_turn_to_history, history_for_debug
from server.session_store import InMemorySessionStore, SessionState


logger = logging.getLogger(__name__)

_EMPTY_RESPONSE_FALLBACK = (
    "I could not produce a spoken response for that request. Please try again."
)
_MISSING_IMAGE_SPATIAL_FALLBACK = (
    "I don't currently have a scene photo to analyze. Please take the photo again, then ask that question once more."
)


class AppOrchestrator:
    def __init__(self, session_store: InMemorySessionStore) -> None:
        self._session_store = session_store

    def load_context(self, session_id: str) -> SessionContext:
        session_state = self._session_store.get(session_id)
        if session_state is None:
            return SessionContext(
                history=None,
                prior_measurements=None,
                cached_image=None,
                metrics=None,
            )
        return SessionContext(
            history=session_state.history,
            prior_measurements=session_state.measurements,
            cached_image=session_state.image,
            metrics=session_state.metrics,
        )

    def clear_session(self, session_id: str) -> None:
        self._session_store.delete(session_id)

    def handle_query(self, query: QueryInput) -> QueryResult:
        context = self.load_context(query.session_id)
        active_image = query.uploaded_image if query.uploaded_image is not None else context.cached_image
        prior_turn_count = len(context.history or [])
        logger.info(
            "api_query session_id=%s uploaded_image=%s cached_image=%s active_image=%s history=%s prior_measurements=%s",
            query.session_id or "<empty>",
            query.uploaded_image is not None,
            context.cached_image is not None,
            active_image is not None,
            context.history is not None,
            context.prior_measurements is not None,
        )

        send_image = active_image is not None and (context.history is None or query.uploaded_image is not None)

        response_text = ""
        depth_b64: str | None = None
        active_image_b64: str | None = None
        navigator_image_b64: str | None = None
        measurement_state: list[dict] | None = None
        response_route = "direct"
        turn_metrics: dict[str, Any] = {"timings": [], "summary": {}, "counts": {}}
        session_metrics_state = context.metrics
        request_t0 = time.monotonic()

        try:
            scout_session = create_session(active_image, query.question)
            route, scout_text, _scout_trace = run_scout_loop(
                scout_session,
                history=context.history,
                send_image=send_image,
                has_active_image=active_image is not None,
            )
            response_route = route

            if route == "navigator":
                (
                    response_route,
                    response_text,
                    turn_metrics,
                    active_image_b64,
                    depth_b64,
                    navigator_image_b64,
                    measurement_state,
                    next_image,
                    next_history,
                ) = self._run_navigation_pipeline(
                    scout_session=scout_session,
                    query=query,
                    context=context,
                    active_image=active_image,
                    send_image=send_image,
                    prior_turn_count=prior_turn_count,
                )
            else:
                response_text = scout_text
                turn_metrics = scout_session.export_metrics()
                active_image_b64 = image_to_jpeg_b64(active_image)
                if route == "restart":
                    measurement_state = None
                    next_image = None
                    next_history: list[dict] = []
                else:
                    measurement_state = context.prior_measurements
                    next_image = active_image
                    next_history = list(context.history or [])
                scout_session.release()

            session_metrics_state = self._finalize_turn_metrics(session_metrics_state, turn_metrics, request_t0)
            debug = self._build_debug_payload(
                active_image_b64=active_image_b64,
                depth_b64=depth_b64,
                navigator_image_b64=navigator_image_b64,
            )
            return self._finalize_result(
                query=query,
                active_image=active_image,
                send_image=send_image,
                response_text=response_text,
                response_route=response_route,
                turn_metrics=turn_metrics,
                session_metrics_state=session_metrics_state,
                measurement_state=measurement_state,
                next_image=next_image,
                next_history=next_history,
                debug=debug,
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

        metrics_payload = {
            "turn_top": summarize_metrics(turn_metrics.get("summary", {}), turn_metrics.get("counts", {})),
            "session_top": summarize_metrics(
                (session_metrics_state or {}).get("summary", {}),
                (session_metrics_state or {}).get("counts", {}),
            ),
        }
        return QueryResult(
            response=response_text,
            route=response_route,
            metrics=metrics_payload,
            debug=DebugPayload(
                active_image_b64=None,
                depth_b64=None,
                navigator_image_b64=None,
                measurements=[],
                history=[],
            ),
        )

    def _run_navigation_pipeline(
        self,
        *,
        scout_session: Any,
        query: QueryInput,
        context: SessionContext,
        active_image: Image.Image | None,
        send_image: bool,
        prior_turn_count: int,
    ) -> tuple[str, str, dict[str, Any], str | None, str | None, str | None, list[dict] | None, Image.Image | None, list[dict]]:
        if active_image is None:
            logger.warning(
                "scout requested navigator without active image; forcing direct retry message session_id=%s",
                query.session_id or "<empty>",
            )
            response_text = _MISSING_IMAGE_SPATIAL_FALLBACK
            turn_metrics = scout_session.export_metrics()
            next_history = list(context.history or [])
            scout_session.release()
            return (
                "direct",
                response_text,
                turn_metrics,
                None,
                None,
                None,
                context.prior_measurements,
                None,
                next_history,
            )

        session = create_session(
            active_image,
            query.question,
            intrinsics=scout_session.intrinsics,
            metrics=scout_session.metrics,
        )
        run_mapper_loop(
            session,
            history=None,
            prior_measurements=context.prior_measurements,
            prior_turn_count=prior_turn_count,
            fresh_image_attached=query.uploaded_image is not None,
        )
        annotated = render_annotated_image(session)

        depth_b64: str | None = None
        if session.depth_colormap:
            depth_b64 = image_to_jpeg_b64(session.depth_colormap, resize_to=active_image.size)

        active_image_b64 = image_to_jpeg_b64(active_image)
        navigator_image_b64 = image_to_jpeg_b64(annotated)
        response_text, _nav_trace = run_navigator_loop(
            session,
            annotated_image=annotated,
            history=context.history,
            send_image=send_image,
        )
        turn_metrics = session.export_metrics()
        session.release()
        measurement_state = [
            {k: v for k, v in measurement.items() if k != "mask_dpt"}
            for measurement in session.measurements
        ]
        scout_session.release()
        return (
            "navigator",
            response_text,
            turn_metrics,
            active_image_b64,
            depth_b64,
            navigator_image_b64,
            measurement_state,
            active_image,
            list(context.history or []),
        )

    def persist_session_state(
        self,
        session_id: str,
        *,
        history: list[dict],
        measurements: list[dict] | None,
        image: Image.Image | None,
        metrics: dict[str, Any],
    ) -> None:
        self._session_store.set(
            session_id,
            SessionState(
                history=history,
                measurements=measurements,
                image=image,
                metrics=metrics,
            ),
        )

    def _finalize_turn_metrics(
        self,
        session_metrics_state: dict[str, Any] | None,
        turn_metrics: dict[str, Any],
        request_t0: float,
    ) -> dict[str, Any]:
        add_metric_sample(turn_metrics, "request.total", time.monotonic() - request_t0)
        return merge_metrics(session_metrics_state, turn_metrics)

    def _build_debug_payload(
        self,
        *,
        active_image_b64: str | None,
        depth_b64: str | None,
        navigator_image_b64: str | None,
    ) -> DebugPayload:
        return DebugPayload(
            active_image_b64=active_image_b64,
            depth_b64=depth_b64,
            navigator_image_b64=navigator_image_b64,
            measurements=[],
            history=[],
        )

    def _finalize_result(
        self,
        *,
        query: QueryInput,
        active_image: Image.Image | None,
        send_image: bool,
        response_text: str,
        response_route: str,
        turn_metrics: dict[str, Any],
        session_metrics_state: dict[str, Any],
        measurement_state: list[dict] | None,
        next_image: Image.Image | None,
        next_history: list[dict],
        debug: DebugPayload,
    ) -> QueryResult:
        if not response_text.strip():
            logger.warning("empty response_text for route=%s; using fallback", response_route)
            response_text = _EMPTY_RESPONSE_FALLBACK

        updated_shared_history = list(next_history)
        if response_route != "restart":
            updated_shared_history = append_turn_to_history(
                updated_shared_history,
                active_image=active_image,
                question=query.question,
                send_image=send_image,
                response_text=response_text,
            )
        self.persist_session_state(
            query.session_id,
            history=updated_shared_history,
            measurements=measurement_state,
            image=next_image,
            metrics=session_metrics_state,
        )
        debug.history = history_for_debug(updated_shared_history)
        debug.measurements = measurement_state or []

        metrics_payload = {
            "turn_top": summarize_metrics(turn_metrics.get("summary", {}), turn_metrics.get("counts", {})),
            "session_top": summarize_metrics(
                session_metrics_state.get("summary", {}),
                session_metrics_state.get("counts", {}),
            ),
        }

        logger.info(
            "api_query route=%s response_len=%s depth_b64=%s",
            response_route,
            len(response_text),
            bool(debug.depth_b64),
        )
        return QueryResult(
            response=response_text,
            route=response_route,
            metrics=metrics_payload,
            debug=debug,
        )
