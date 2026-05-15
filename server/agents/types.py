from __future__ import annotations

import json
import logging
import re
from typing import Any

from PIL import Image

from pipeline.session import Session
from . import runtime
from .base import SingleCallAgent, ToolLoopAgent
from .prompts import (
    MAPPER_SYSTEM_PROMPT,
    NAVIGATOR_SYSTEM_PROMPT,
    SCOUT_RESPONSE_SCHEMA,
    SCOUT_SYSTEM_PROMPT,
)
from server.media import image_to_data_url
from server.messages import build_question_content, build_turn_user_content


logger = logging.getLogger(__name__)

GEMMA_IMAGE_MULTIPLE = 48


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start:end + 1]
    return stripped


def _parse_scout_response(text: str) -> tuple[str, str, str]:
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError:
        logger.warning("scout returned non-JSON response; defaulting to direct answer")
        return "direct", text.strip(), ""

    response_type = payload.get("type")
    response_text = payload.get("text", "")
    reason = payload.get("reason", "")

    if response_type == "handoff_navigator":
        logger.info("scout route=navigator text_len=%s reason_len=%s", len(str(response_text)), len(str(reason)))
        return "navigator", "", str(reason).strip()
    if response_type == "restart_conversation":
        restart_text = str(response_text).strip()
        if not restart_text:
            restart_text = "Starting a new scene. Please show me what is in front of you."
            logger.warning("scout returned empty restart text; using fallback")
        logger.info("scout route=restart text_len=%s reason_len=%s", len(restart_text), len(str(reason)))
        return "restart", restart_text, str(reason).strip()
    if response_type == "direct":
        direct_text = str(response_text).strip()
        if not direct_text:
            logger.warning("scout returned empty direct text; using fallback")
            direct_text = "I understood your request, but I could not form a spoken answer. Please try asking again."
        logger.info("scout route=direct text_len=%s reason_len=%s", len(direct_text), len(str(reason)))
        return "direct", direct_text, str(reason).strip()

    logger.warning("scout returned unknown type=%r; defaulting to direct answer", response_type)
    return "direct", str(response_text or text).strip(), str(reason).strip()


class ScoutAgent(SingleCallAgent):
    name = "scout"
    system_prompt = SCOUT_SYSTEM_PROMPT
    llm_timing_stage = "scout.llm"
    total_timing_stage = "scout.total"

    def build_messages(
        self,
        *,
        history: list[dict] | None = None,
        send_image: bool = True,
        has_active_image: bool = True,
        image_source: str = "none",
    ) -> list[dict]:
        user_content = build_turn_user_content(self.session.image, self.session.question, send_image=send_image)
        image_source_note = {
            "fresh": "A fresh image is attached with this request.",
            "cached": "No new image was uploaded for this request. The currently attached image is the cached image from the previous session state and is still likely the image the user is referring to unless they clearly indicate otherwise.",
            "none": "No active image is available for this request.",
        }.get(image_source, "Image source is unknown.")
        runtime_prompt = (
            f"{self.system_prompt}\n\n"
            "Runtime state:\n"
            f"- active_image_available: {'true' if has_active_image else 'false'}\n"
            f"- image_source: {image_source}\n"
            f"- {image_source_note}\n"
            "- If active_image_available is false and the user is asking a spatial or navigation question, "
            'respond with type="direct" and ask them to take the photo again.'
        )
        if history is not None:
            return [{"role": "system", "content": runtime_prompt}] + history + [{"role": "user", "content": user_content}]
        return [
            {"role": "system", "content": runtime_prompt},
            {"role": "user", "content": user_content},
        ]

    def run(
        self,
        *,
        history: list[dict] | None = None,
        send_image: bool = True,
        has_active_image: bool = True,
        image_source: str = "none",
    ) -> tuple[str, str, list[dict]]:
        messages = self.build_messages(
            history=history,
            send_image=send_image,
            has_active_image=has_active_image,
            image_source=image_source,
        )
        response = self.run_single_call(
            messages=messages,
            request_overrides={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": SCOUT_RESPONSE_SCHEMA,
                },
            },
        )
        choice = response.choices[0]
        raw_content = choice.message.content or ""
        logger.info("scout raw_content_len=%s finish_reason=%s", len(raw_content), choice.finish_reason)
        route, final_text, _reason = _parse_scout_response(raw_content)
        messages.append({"role": "assistant", "content": final_text if route == "direct" else raw_content})
        return route, final_text, messages[1:]


class MapperAgent(ToolLoopAgent):
    name = "mapper"
    system_prompt = MAPPER_SYSTEM_PROMPT
    llm_timing_stage = "mapper.llm_round"
    total_timing_stage = "mapper.total"

    def build_request_overrides(self, _round_idx: int) -> dict[str, Any]:
        return {
            "tools": runtime.active_tools(),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

    def handle_no_tool_calls(self, messages: list[dict], choice: Any, round_idx: int) -> tuple[bool, bool]:
        content = choice.message.content or ""
        if runtime.looks_like_raw_tool_protocol(content):
            logger.warning("mapper round=%s: raw tool protocol text, sending correction", round_idx)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Use the structured tool call format, not text. Call call_dpt_head now."})
            return True, False

        if not self.session.measurements:
            logger.warning(
                "mapper round=%s produced prose with no measurements; requesting tool-based retry",
                round_idx,
            )
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Do not answer in prose yet. Use tools to create measured scene state first. "
                        "Call call_dpt_head, then measure the target and any relevant obstacles."
                    ),
                }
            )
            return True, False

        logger.info(
            "mapper round=%s done: finish_reason=%s measurements=%s content=%r",
            round_idx,
            choice.finish_reason,
            len(self.session.measurements),
            content[:200],
        )
        return False, True

    def build_prior_measurements_context(self, prior_measurements: list[dict] | None) -> str | None:
        if not prior_measurements:
            return None
        lines = [
            "Prior measured scene from the earlier conversation. Reuse this context when relevant, but remeasure if the user's current request needs a fresh or more specific measurement:",
        ]
        for i, measurement in enumerate(prior_measurements, start=1):
            lines.append(
                f"{i}. {measurement['class_name']} — {measurement['tips_distance_m']} m, {measurement['direction']}"
            )
        return "\n".join(lines)

    def build_messages(
        self,
        *,
        history: list[dict] | None = None,
        prior_measurements: list[dict] | None = None,
        prior_turn_count: int = 0,
        fresh_image_attached: bool = False,
        image_source: str = "none",
        send_image: bool = True,
    ) -> list[dict]:
        user_content = build_turn_user_content(self.session.image, self.session.question, send_image=send_image)

        prefix_parts: list[str] = []
        prior_context = self.build_prior_measurements_context(prior_measurements)
        if prior_context:
            prefix_parts.append(prior_context)
        if prior_measurements is None:
            prefix_parts.append(
                "I am blind. Measure every object around me that could affect my navigation or safety — scan the full scene. Also specifically:"
            )
        else:
            followup_notes: list[str] = [
                "This is a follow-up request.",
                f"There have been {prior_turn_count} prior user/assistant messages since the session started."
                if prior_turn_count
                else "There has been no prior user/assistant message in this session.",
            ]
            if fresh_image_attached:
                followup_notes.append(
                    "A new photo is attached for this turn. Treat prior measurements as stale context only. "
                    "Measure from the current image again before relying on any prior distance."
                )
            elif image_source == "cached":
                followup_notes.append(
                    "No new photo is attached for this turn, but the active image provided with this request is the cached image from the previous session state. "
                    "The user is likely still referring to that image unless they clearly indicate a different scene."
                )
            else:
                followup_notes.append(
                    "No new photo is attached for this turn. You may use prior measurements as context, "
                    "but still remeasure if the current request needs a fresh or more specific measurement."
                )
            followup_notes.append(
                "Use prior measurements only as background context. Do not report an old distance as the current answer unless you remeasure it for this turn."
            )
            followup_notes.append(
                "If the user is asking about the same target after moving, prefer remeasuring that target and nearby obstacles from the current image."
            )
            followup_notes.append("Also specifically:")
            prefix_parts.append(" ".join(followup_notes))

        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": "\n\n".join(prefix_parts)}, *user_content],
        }
        if history is not None:
            return [self.build_system_message()] + history + [user_message]
        return [self.build_system_message(), user_message]

    def run(
        self,
        *,
        history: list[dict] | None = None,
        prior_measurements: list[dict] | None = None,
        prior_turn_count: int = 0,
        fresh_image_attached: bool = False,
        image_source: str = "none",
    ) -> None:
        if self.session.image is None:
            return
        messages = self.build_messages(
            history=history,
            prior_measurements=prior_measurements,
            prior_turn_count=prior_turn_count,
            fresh_image_attached=fresh_image_attached,
            image_source=image_source,
            send_image=True,
        )
        self.run_tool_loop(messages)


class NavigatorAgent(SingleCallAgent):
    name = "navigator"
    system_prompt = NAVIGATOR_SYSTEM_PROMPT
    llm_timing_stage = "navigator.llm"
    total_timing_stage = "navigator.total"

    def build_scene_summary(self) -> str:
        if not self.session.measurements:
            return "No objects were measured in this scene."
        lines = ["Scene measurements:"]
        for i, measurement in enumerate(self.session.measurements, start=1):
            class_name = measurement["class_name"]
            requested_class_name = measurement.get("requested_class_name")
            if requested_class_name and requested_class_name != class_name:
                label = f"{requested_class_name} (measured as {class_name})"
            else:
                label = class_name
            lines.append(f"{i}. {label} — {measurement['tips_distance_m']} m, {measurement['direction']}")
        return "\n".join(lines)

    def build_messages(
        self,
        *,
        annotated_image: Image.Image | None,
        original_image: Image.Image | None = None,
        history: list[dict] | None = None,
        send_image: bool = True,
        image_source: str = "none",
    ) -> list[dict]:
        scene_summary = self.build_scene_summary()
        is_followup = history is not None
        image_part: list[dict] = []
        if send_image:
            if original_image is not None:
                orig_url = image_to_data_url(original_image, multiple=GEMMA_IMAGE_MULTIPLE)
                image_part.append({"type": "image_url", "image_url": {"url": orig_url}})
            ann_url = image_to_data_url(annotated_image, multiple=GEMMA_IMAGE_MULTIPLE)
            image_part.append({"type": "image_url", "image_url": {"url": ann_url}})

        ref = "both images and the measurements" if (send_image and original_image is not None) else ("the image and the measurements" if send_image else "the measurements")
        if is_followup:
            if image_source == "cached":
                framing = (
                    "The attached image is the cached image from the previous session state and is still likely the scene the user is referring to. "
                    f"Using {ref}, answer my follow-up question: "
                )
            else:
                framing = f"Using {ref}, answer my follow-up question: "
        else:
            framing = (
                "I am blind and navigating this space. "
                "Never state a distance that is not in the measurements above. "
                "Address my question and also mention all measurements relevant to safely navigating "
                "toward the answer — including any obstacles or hazards along the way. "
                f"Using {ref}, answer my question: "
            )

        question_content: list[dict] = [
            *image_part,
            {"type": "text", "text": scene_summary},
            {"type": "text", "text": framing},
            *build_question_content(self.session.question),
        ]

        if history is not None:
            return [self.build_system_message()] + history + [{"role": "user", "content": question_content}]
        return [
            self.build_system_message(),
            {"role": "user", "content": question_content},
        ]

    def run(
        self,
        *,
        annotated_image: Image.Image | None,
        original_image: Image.Image | None = None,
        history: list[dict] | None = None,
        send_image: bool = True,
        image_source: str = "none",
    ) -> tuple[str, list[dict]]:
        messages = self.build_messages(
            annotated_image=annotated_image,
            original_image=original_image,
            history=history,
            send_image=send_image,
            image_source=image_source,
        )
        response = self.run_single_call(messages=messages, request_overrides={"max_tokens": 2048})
        final_text = response.choices[0].message.content or ""
        self.session.spatial_report = final_text
        messages.append({"role": "assistant", "content": final_text})
        return final_text, messages[1:]
