from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import logging
import os
import re
import time
from typing import Any

from openai import OpenAI
from PIL import Image

from pipeline.session import Session
from pipeline import tools as pipeline_tools
from server.schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
MODEL_ID = os.getenv("VLLM_MODEL_ID", "gemma-4-e4b-it")
MAX_TOOL_ROUNDS = 10
MAX_TOKENS = 1024
DUMP_REQUEST_PATH = os.getenv("SPATIALSENSE_DUMP_REQUEST_PATH", "")
DUMP_RESPONSE_PATH = os.getenv("SPATIALSENSE_DUMP_RESPONSE_PATH", "")
TOOL_MODE = os.getenv("SPATIALSENSE_TOOL_MODE", "full").strip().lower()
GEMMA_IMAGE_MULTIPLE = 48

SYSTEM_PROMPT = """\
You are SpatialSense, a navigation assistant for a blind user.
Primary goal: provide safe, actionable guidance using the current image and tool results.

Instruction hierarchy (highest to lowest):
1) Safety and factuality
2) Tool-use requirements
3) Response-format requirements
4) Conversation style

Safety and factuality:
- Never invent objects, distances, or directions.
- Never state a numeric distance unless measure_object returned distance_m for that instance.
- If an object is visible but unmeasured, say it is visible but unmeasured.

Tool-use requirements:
1. For object distance/location questions, numeric distance is valid only after a successful measure_object call.
2. call_dpt_head and call_encoder_zero_shot prepare detections only; by themselves they do not provide distance.
3. In-vocabulary path: call_dpt_head first, then measure_object for each requested instance.
4. Out-of-vocabulary/fallback path: use search_seg_classes; if there is no useful ADE match, or if the requested class is not in detected_classes and no close ADE label is reasonable, call call_encoder_zero_shot, then measure_object.
5. After call_dpt_head, treat detected_classes as the preferred ADE labels for measurement.
6. If user wording and ADE label differ, only allow near-synonym substitutions (for example chair vs armchair). Never substitute semantically unrelated labels (for example fire extinguisher -> painting).
7. In multi-instance questions, do not stop after one instance; measure up to 4 clearly visible instances with separate boxes.
8. Prefer parallel measure_object calls for multi-instance cases.
9. If measure_object indicates class mismatch/substitution or returns error, do not report distance for the requested object. Retry with a tighter box or use search_seg_classes/call_encoder_zero_shot.
10. For scene description questions (intent C), you MUST call call_dpt_head and then call measure_object for every navigation-relevant detected class (skip wall, floor, ceiling, sky) before generating the scene summary. Do not describe the scene from the image alone.

Direction and reporting rules:
- measure_object returns direction text. Use that direction phrase verbatim.
- If multiple relevant instances are visible, explicitly say there are multiple instances.

Response format requirements by intent:

A) Direct object distance question (example: "How far is the chair?")
- Single clear instance:
    "<object> is <distance_m> meters away, <direction>."
- Multiple instances:
    "I can see <N> <object_plural>. Nearest is <distance_m>, <direction>. I also see <brief summary of others>. Are you asking about a specific one?"

B) Location/direction question
- Give direction first, then distance if measured.
- If distance is unavailable:
    "<object> is visible <direction>, but I could not measure distance reliably."

C) Scene question (example: "What is around me?")
- Describe the scene in concise, practical terms (layout, main nearby objects, pathway/opening cues).
- Flag the closest objects straight ahead as potential hazards.
- Include one optional next-step question.
- Keep concise and navigation-first.

D) Navigation question (example: "How do I get there?")
- First identify and measure the destination object instance using measure_object.
- Then call measure_object on any objects that appear between the user and the destination.
- Provide practical step-by-step guidance: for each closer object in roughly the same direction, state its distance and direction, then tell the user which side to step around it before continuing toward the destination.
- If destination is ambiguous or not visible, ask one clarifying question and provide the safest immediate next step.

Behavior examples (follow structure, do not copy verbatim):
- User: "How far is the table?"
    Assistant: "The table is 2.1 meters away, about 12 degrees to your right."
- User: "How far is the chair?" (multiple chairs visible)
    Assistant: "I can see three chairs. The nearest chair is 1.4 meters away, straight ahead. I also see one at 2.3 meters to your left and one at 3.0 meters to your right. Are you asking about one near another object?"
- User: "What is around me?"
    Assistant: "You are in a room with a clear opening slightly left and several objects spread across the center and right side. Safety cue: there is a chair directly ahead at about 1.8 meters — obstacle. Would you like step-by-step guidance toward the opening?"
- User: "How do I get there?"
    Assistant: "The door is 3.6 meters away, about 9 degrees to your right. There is a chair 1.4 meters ahead near center-left and a small obstacle 0.9 meters almost straight ahead. Path: shift about half a step left, walk forward 1.2 meters, then turn slightly right and continue about 2.4 meters to the door. If you want, I can guide this one step at a time."

Tool-call examples (illustrative few-shot flow):
- User: "What is around me?"
    Assistant tool flow: call_dpt_head -> measure_object on up to 5 navigation-relevant classes (including near-forward hazards) -> final scene summary with explicit close-front safety cue.
- User: "How do I get to the door?"
    Assistant tool flow: call_dpt_head -> measure_object(door) + measure_object on any objects between user and door -> final distance-aware path guidance.
- User: "How do I get to the shopping cart?" (OOV class)
    Assistant tool flow: search_seg_classes("shopping cart") -> call_encoder_zero_shot(["shopping cart"]) -> measure_object("shopping cart", box_2d) -> measure_object(intervening obstacles) -> final distance-aware path guidance.

Conversation style:
- Friendly, calm, concise, and practical.
- Natural language only; never output raw tool JSON.
"""
SYSTEM_PROMPT = os.getenv("SPATIALSENSE_SYSTEM_PROMPT", SYSTEM_PROMPT)

SCOUT_SYSTEM_PROMPT = """\
You are Scout for Gemma Guide, a visual assistant for a blind user.
Primary goal: answer the user's question directly when general visual understanding is enough, and hand off to the spatial navigation pipeline only when metric spatial reasoning is required.

Instruction hierarchy (highest to lowest):
1) Safety and factuality
2) Routing requirements
3) Response-format requirements
4) Conversation style

Safety and factuality:
- Never invent text, prices, titles, brands, objects, distances, or directions.
- If text is unreadable or the image is insufficient, say so briefly.
- If the answer requires metric distance, direction, obstacle awareness, path guidance, or safe navigation, do not guess. Hand off to the navigator pipeline.

Routing requirements:
1. Answer directly when the question is about identity, text, title, price, label, brand, color, simple scene description, or other general visual understanding.
2. Choose type="handoff_navigator" when the user is asking for distance, direction, relative location, obstacle awareness, pathing, scene safety, or navigation guidance.
3. Choose type="restart_conversation" when the user explicitly asks to start over, clear the conversation, reset the scene, or begin a new scene.
4. If the user asks a mixed question, hand off whenever the spatial part is necessary for a safe or complete answer.
5. Do not choose type="handoff_navigator" for pure reading or recognition questions.
6. If you can answer directly, do not hand off.

Response-format requirements:
- Return exactly one JSON object matching this schema:
  {"type":"direct"|"handoff_navigator"|"restart_conversation","text":"...","reason":"..."}
- If type is "direct", text must contain the user-facing answer and reason may be empty.
- If type is "handoff_navigator", text must be empty and reason must briefly explain why spatial analysis is required.
- If type is "restart_conversation", text should briefly tell the user the scene was reset and what to do next. Reason may be empty.
- Do not return markdown, code fences, or any text outside the JSON object.

Conversation style:
- Friendly, calm, concise, and practical.
- The text field should use natural spoken language.
"""

SCOUT_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "scout_response",
    "schema": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["direct", "handoff_navigator", "restart_conversation"],
            },
            "text": {
                "type": "string",
                "description": "User-facing answer when type is direct; otherwise empty.",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason why spatial navigation analysis is required when handing off.",
            },
        },
        "required": ["type", "text", "reason"],
        "additionalProperties": False,
    },
    "strict": True,
}

MAPPER_SYSTEM_PROMPT = """\
You are SpatialSense, a navigation assistant for a blind user.
Primary goal: build a rich measured scene for the navigator using the current image and tool results.

Instruction hierarchy (highest to lowest):
1) Safety and factuality
2) Tool-use requirements
3) Mapper coverage requirements
4) Conversation style

Safety and factuality:
- Never invent objects, distances, or directions.
- Never state a numeric distance unless measure_object returned distance_m for that instance.
- If an object is visible but unmeasured, leave it unmeasured rather than guessing.

Tool-use requirements:
1. For object distance/location questions, numeric distance is valid only after a successful measure_object call.
2. call_dpt_head and call_encoder_zero_shot prepare detections only; by themselves they do not provide distance.
3. In-vocabulary path: call_dpt_head first, then measure_object for each requested instance.
4. Out-of-vocabulary/fallback path: use search_seg_classes; if there is no useful ADE match, or if the requested class is not in detected_classes and no close ADE label is reasonable, call call_encoder_zero_shot, then measure_object.
5. After call_dpt_head, treat detected_classes as the preferred ADE labels for measurement.
6. If user wording and ADE label differ, only allow near-synonym substitutions (for example chair vs armchair). Never substitute semantically unrelated labels (for example fire extinguisher -> painting).
7. In multi-instance questions, do not stop after one instance; measure up to 4 clearly visible instances with separate boxes.
8. Prefer parallel measure_object calls for multi-instance cases.
9. If measure_object indicates class mismatch/substitution or returns error, do not treat that as a reliable measurement for the requested object. Retry with a tighter box or use search_seg_classes/call_encoder_zero_shot.
10. For scene description questions, you MUST call call_dpt_head and then call measure_object for every navigation-relevant detected class that matters for safe movement or for the user's question. Do not build the scene from the image alone.

Mapper coverage requirements:
- Your job is not to answer the user. Your job is to leave behind a measured scene in the tool state for the Navigator.
- Measure the objects most relevant to the user's question first.
- Also measure nearby obstacles, close-front hazards, and any objects likely to matter for reaching the target safely.
- Skip structural background unless it is directly relevant to navigation in this scene. In general skip wall, floor, ceiling, and sky.
- Prefer a bounded, useful scene over an exhaustive one. Usually 3 to 8 strong measurements are enough.
- When the user asks about a destination or route, measure the destination first, then measure any closer objects in roughly the same direction that could block or affect the path.
- When the user asks what is around them, measure the main nearby objects, pathway or opening cues, and the closest straight-ahead hazard.
- Once you have enough reliable measurements for the Navigator to answer safely, stop calling tools.

Direction and reporting rules:
- measure_object returns direction text. Use that direction phrase verbatim when reasoning about whether another object is in roughly the same direction.
- If multiple relevant instances are visible, explicitly measure multiple instances when needed to resolve the user's question safely.

Conversation style:
- Use tools, not prose, to do the work.
- If you produce text after finishing tool use, keep it brief. Do not answer the user in detail.
"""

NAVIGATOR_SYSTEM_PROMPT = """\
You are the Navigator for SpatialSense, a navigation assistant for blind users. You receive:
- An image of the scene with numbered bounding boxes labeled "N: class distance_m"
- A scene summary listing every measurement: box number, class, distance in meters, and direction
- The user's spoken question

Your job is to answer using only the data in the scene summary. Never state a distance not in the summary. If an object is visible in the image but absent from the summary, say it is visible but unmeasured. Never guess distances from visual appearance.

Direction rule: use the direction phrase from the scene summary verbatim. Do not paraphrase it.

Safety and factuality:
- Never invent objects, distances, or directions.
- If an object is visible but not in the summary, say it is visible but unmeasured.

Response by intent:

A) Object distance:
- Single instance: "[class] is [distance] meters away, [direction]."
- Multiple instances: "I can see [N] [class_plural]. The nearest is [distance] meters, [direction]. I also see [brief summary of others]. Are you asking about a specific one?"

B) Direction/location:
- Direction first, distance second. If unmeasured: "[class] is visible [direction], but distance could not be measured."

C) Scene description:
- Practical layout: main nearby objects, any open path or gap visible.
- Flag the closest object straight ahead as a potential hazard.
- One follow-up offer at the end.

D) Navigation:
- Target's distance and direction first.
- For each object between the user and the target (closer and roughly in the same direction): name it, give its distance and direction, say which side to step around it.
- If no objects are in the way, say the path looks clear.
- Step-by-step, distance-first.

Response examples (follow structure, do not copy verbatim):
- "The table is 2.1 meters away, about 12 degrees to your right."
- "I can see three chairs. The nearest is 1.4 meters away, straight ahead. I also see one at 2.3 meters to your left and one at 3.0 meters to your right. Are you asking about a specific one?"
- "You are in a room with a clear opening slightly to your left. Several objects are spread across the center and right side. There is a chair directly ahead at about 1.8 meters — watch out for that. Would you like guidance toward the opening?"
- "The door is 3.6 meters away, about 9 degrees to your right. There is a chair 1.4 meters ahead near center-left and a table 0.9 meters almost straight ahead. Step slightly left, walk forward about 1.2 meters, then shift right and continue about 2.4 meters to the door. Want me to guide this one step at a time?"

Style:
- Spoken natural language only. No markdown, no bullet points, no numbered lists.
- Short and practical — a blind user may be moving while listening.
- Reference box numbers only when two instances of the same class need to be distinguished (for example: "the chair at box 1 is closer than the one at box 3").
- Friendly, calm, direct.
"""

def _looks_like_raw_tool_protocol(text: str) -> bool:
    lowered = text.lower()
    return "<|tool_call" in lowered or "<tool_call" in lowered


def _maybe_dump_request(round_idx: int, payload: dict[str, Any]) -> None:
    if not DUMP_REQUEST_PATH:
        return
    record = {
        "round": round_idx,
        "timestamp": time.time(),
        "payload": payload,
    }
    try:
        with open(DUMP_REQUEST_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")
    except Exception:
        logger.exception("Failed to write request dump")


def _maybe_dump_response(round_idx: int, response: Any) -> None:
    if not DUMP_RESPONSE_PATH:
        return
    try:
        choice = response.choices[0]
        message = choice.message
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )

        record = {
            "round": round_idx,
            "timestamp": time.time(),
            "finish_reason": choice.finish_reason,
            "content": message.content,
            "tool_calls": tool_calls,
        }
        with open(DUMP_RESPONSE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")
    except Exception:
        logger.exception("Failed to write response dump")


def _active_tools() -> list[dict[str, Any]]:
    if TOOL_MODE == "dpt_only":
        return [
            schema
            for schema in TOOL_SCHEMAS
            if schema.get("function", {}).get("name") == "call_dpt_head"
        ]
    return TOOL_SCHEMAS


def _resize_for_gemma(image: Image.Image, multiple: int) -> Image.Image:
    """Resize image so both dimensions are multiples of `multiple`."""
    if multiple <= 1:
        return image

    w, h = image.size
    new_w = max(multiple, int(round(w / multiple) * multiple))
    new_h = max(multiple, int(round(h / multiple) * multiple))

    if new_w == w and new_h == h:
        return image

    return image.resize((new_w, new_h), Image.BILINEAR)


def _image_to_data_url(image: Image.Image) -> str:
    gemma_image = _resize_for_gemma(image, GEMMA_IMAGE_MULTIPLE)
    buf = io.BytesIO()
    gemma_image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_question_content(question: str | bytes) -> list[dict]:
    if isinstance(question, bytes):
        b64_audio = base64.b64encode(question).decode("utf-8")
        return [{"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}]
    return [{"type": "text", "text": question}]


def build_turn_user_content(image: Image.Image | None, question: str | bytes, send_image: bool = True) -> list[dict]:
    content: list[dict] = []
    if send_image and image is not None:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})
    content.extend(_build_question_content(question))
    return content


def build_messages(image: Image.Image | None, question: str | bytes) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_turn_user_content(image, question)},
    ]


def _dispatch_tool(name: str, args: dict[str, Any], session: Session) -> str:
    t0 = time.monotonic()
    print_tool_io = os.getenv("SPATIALSENSE_PRINT_TOOL_IO", "0") == "1" or os.getenv("SPATIALSENSE_PRINT_TOOL_RETURNS", "0") == "1"
    if print_tool_io:
        print(f"TOOL_INPUT {name}: {json.dumps(args, ensure_ascii=True)}")

    try:
        if name == "search_seg_classes":
            result = pipeline_tools.search_seg_classes(args["query"])
        elif name == "call_dpt_head":
            result = pipeline_tools.call_dpt_head(session)
        elif name == "call_encoder_zero_shot":
            result = pipeline_tools.call_encoder_zero_shot(
                args["class_list"],
                session,
            )
        elif name == "measure_object":
            result = pipeline_tools.measure_object(
                args["class_name"],
                args["box_2d"],
                session,
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc)}

    latency = round(time.monotonic() - t0, 3)
    session.add_timing(f"tool.{name}", latency)
    logger.info("tool=%s latency=%.3fs", name, latency)
    if print_tool_io:
        print(f"TOOL_OUTPUT {name}: {json.dumps(result, ensure_ascii=True)}")
    return json.dumps(result)


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
    """Parse Scout JSON into (route, text, reason) with a safe fallback."""
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError:
        logger.warning("scout returned non-JSON response; defaulting to direct answer")
        return "direct", text.strip(), ""

    response_type = payload.get("type")
    response_text = payload.get("text", "")
    reason = payload.get("reason", "")

    if response_type == "handoff_navigator":
        return "navigator", "", str(reason).strip()
    if response_type == "restart_conversation":
        return "restart", str(response_text).strip(), str(reason).strip()
    if response_type == "direct":
        return "direct", str(response_text).strip(), str(reason).strip()

    logger.warning("scout returned unknown type=%r; defaulting to direct answer", response_type)
    return "direct", str(response_text or text).strip(), str(reason).strip()


def _dispatch_tool_calls(
    tool_calls: list[Any],
    session: Session,
) -> list[dict[str, str]]:
    """Dispatch tool calls and return tool-role messages in the same call order."""
    parsed_calls: list[tuple[str, str, dict[str, Any]]] = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        parsed_calls.append((tc.id, tc.function.name, args))

    # Safely parallelize only when all calls are read-only measurement calls.
    all_measure = bool(parsed_calls) and all(name == "measure_object" for _, name, _ in parsed_calls)
    if all_measure and len(parsed_calls) > 1:
        max_workers = min(4, len(parsed_calls))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_dispatch_tool, name, args, session)
                for _, name, args in parsed_calls
            ]
            results = [f.result() for f in futures]
    else:
        results = [_dispatch_tool(name, args, session) for _, name, args in parsed_calls]

    return [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        }
        for (call_id, _, _), result in zip(parsed_calls, results)
    ]


def run_agent_loop(
    session: Session,
    history: list[dict] | None = None,
    send_image: bool = True,
) -> tuple[str, list[dict]]:
    t_total = time.monotonic()
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    has_image = session.image is not None

    if history is not None:
        if send_image and has_image:
            image_url = _image_to_data_url(session.image)
            user_content = [{"type": "image_url", "image_url": {"url": image_url}}, *_build_question_content(session.question)]
        else:
            user_content = _build_question_content(session.question)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_content}]
    else:
        messages = build_messages(session.image, session.question)

    for round_idx in range(MAX_TOOL_ROUNDS):
        tools = _active_tools() if has_image else None
        request_payload: dict[str, Any] = {
            "model": MODEL_ID,
            "messages": messages,
            "parallel_tool_calls": True,
            "max_tokens": MAX_TOKENS,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                }
            },
        }
        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = "auto"
        _maybe_dump_request(round_idx, request_payload)

        t0 = time.monotonic()
        response = client.chat.completions.create(
            **request_payload,
        )
        _maybe_dump_response(round_idx, response)
        llm_latency = time.monotonic() - t0
        session.add_timing("agent_loop.llm_round", llm_latency, round=round_idx)
        logger.info("round=%s llm_latency=%.3fs", round_idx, llm_latency)

        choice = response.choices[0]

        if choice.message.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            }
            messages.append(assistant_msg)

            tool_messages = _dispatch_tool_calls(
                choice.message.tool_calls,
                session,
            )
            messages.extend(tool_messages)
            continue

        content = choice.message.content or ""
        if _looks_like_raw_tool_protocol(content):
            logger.warning("Model emitted raw tool-call protocol text instead of structured tool_calls")
            err = "You emitted the wrong format for the tool call. Please retry."
            return err, messages[1:]

        # Final text response
        final_text = content
        session.spatial_report = final_text
        messages.append({"role": "assistant", "content": final_text})
        total_latency = time.monotonic() - t_total
        session.add_timing("agent_loop.total", total_latency, rounds=round_idx + 1)
        logger.info("agent_loop_total_latency=%.3fs rounds=%s", total_latency, round_idx + 1)
        return final_text, messages[1:]

    err = "I was unable to complete the spatial analysis. Please try again."
    total_latency = time.monotonic() - t_total
    session.add_timing("agent_loop.total", total_latency, rounds=MAX_TOOL_ROUNDS)
    logger.info("agent_loop_total_latency=%.3fs rounds=%s", total_latency, MAX_TOOL_ROUNDS)
    return err, messages[1:]


def run_scout_loop(
    session: Session,
    history: list[dict] | None = None,
    send_image: bool = True,
) -> tuple[str, str, list[dict]]:
    """Run Scout: answer directly or hand off to the spatial navigator pipeline.

    Returns (route, text, updated_history) where route is "direct" or "navigator".
    """
    t_total = time.monotonic()
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    user_content = build_turn_user_content(session.image, session.question, send_image=send_image)
    if history is not None:
        messages = [{"role": "system", "content": SCOUT_SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_content}]
    else:
        messages = [
            {"role": "system", "content": SCOUT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    request_payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": SCOUT_RESPONSE_SCHEMA,
        },
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    _maybe_dump_request(0, request_payload)

    t0 = time.monotonic()
    response = client.chat.completions.create(**request_payload)
    _maybe_dump_response(0, response)
    llm_latency = time.monotonic() - t0
    total_latency = time.monotonic() - t_total
    session.add_timing("scout.llm", llm_latency)
    session.add_timing("scout.total", total_latency)
    logger.info(
        "scout llm_latency=%.3fs total_latency=%.3fs",
        llm_latency,
        total_latency,
    )

    choice = response.choices[0]
    raw_content = choice.message.content or ""
    route, final_text, _reason = _parse_scout_response(raw_content)
    messages.append({"role": "assistant", "content": final_text if route == "direct" else raw_content})
    return route, final_text, messages[1:]


def run_mapper_loop(
    session: Session,
    history: list[dict] | None = None,
    prior_measurements: list[dict] | None = None,
) -> None:
    """Run the Mapper agent to populate session.measurements via tool calls.

    The mapper's text output is discarded — only session.measurements matters.
    On the first turn (prior_measurements is None) the user message includes the full
    safety-scan framing. On follow-up turns it sends just the image and the question.
    """
    if session.image is None:
        return

    t_total = time.monotonic()
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    messages = build_mapper_messages(
        session,
        history=history,
        prior_measurements=prior_measurements,
        send_image=True,
    )

    for round_idx in range(MAX_TOOL_ROUNDS):
        request_payload = {
            "model": MODEL_ID,
            "messages": messages,
            "tools": _active_tools(),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "max_tokens": MAX_TOKENS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }
        _maybe_dump_request(round_idx, request_payload)

        t0 = time.monotonic()
        response = client.chat.completions.create(**request_payload)
        _maybe_dump_response(round_idx, response)
        llm_latency = time.monotonic() - t0
        session.add_timing("mapper.llm_round", llm_latency, round=round_idx)
        logger.info("mapper round=%s llm_latency=%.3fs", round_idx, llm_latency)

        choice = response.choices[0]

        if choice.message.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ],
            }
            messages.append(assistant_msg)
            messages.extend(_dispatch_tool_calls(choice.message.tool_calls, session))
            continue

        content = choice.message.content or ""
        if _looks_like_raw_tool_protocol(content):
            logger.warning("mapper round=%s: raw tool protocol text, sending correction", round_idx)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Use the structured tool call format, not text. Call call_dpt_head now."})
            continue

        # No tool calls and no raw protocol — mapper is done
        logger.info(
            "mapper round=%s done: finish_reason=%s measurements=%s content=%r",
            round_idx,
            choice.finish_reason,
            len(session.measurements),
            content[:200],
        )
        break

    total_latency = time.monotonic() - t_total
    session.add_timing("mapper.total", total_latency, rounds=round_idx + 1, measurements=len(session.measurements))
    logger.info("mapper total_latency=%.3fs rounds=%s measurements=%s", total_latency, round_idx + 1, len(session.measurements))


def _build_prior_measurements_context(prior_measurements: list[dict] | None) -> str | None:
    if not prior_measurements:
        return None
    lines = [
        "Prior measured scene from the earlier conversation. Reuse this context when relevant, but remeasure if the user's current request needs a fresh or more specific measurement:",
    ]
    for i, m in enumerate(prior_measurements, start=1):
        lines.append(
            f"{i}. {m['class_name']} — {m['tips_distance_m']} m, {m['direction']}"
        )
    return "\n".join(lines)


def build_mapper_messages(
    session: Session,
    history: list[dict] | None = None,
    prior_measurements: list[dict] | None = None,
    send_image: bool = True,
) -> list[dict]:
    user_content = build_turn_user_content(session.image, session.question, send_image=send_image)

    prefix_parts: list[str] = []
    prior_context = _build_prior_measurements_context(prior_measurements)
    if prior_context:
        prefix_parts.append(prior_context)
    if prior_measurements is None:
        prefix_parts.append(
            "I am blind. Measure every object around me that could affect my navigation or safety — scan the full scene. Also specifically:"
        )
    else:
        prefix_parts.append("Use the conversation and prior measurements to answer this follow-up safely. Also specifically:")

    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": "\n\n".join(prefix_parts)}, *user_content],
    }
    if history is not None:
        return [{"role": "system", "content": MAPPER_SYSTEM_PROMPT}] + history + [user_message]
    return [
        {"role": "system", "content": MAPPER_SYSTEM_PROMPT},
        user_message,
    ]


def _build_scene_summary(session: Session) -> str:
    """Build a numbered plain-text scene summary from session.measurements."""
    if not session.measurements:
        return "No objects were measured in this scene."
    lines = ["Scene measurements:"]
    for i, m in enumerate(session.measurements, start=1):
        class_name = m["class_name"]
        requested_class_name = m.get("requested_class_name")
        if requested_class_name and requested_class_name != class_name:
            label = f"{requested_class_name} (measured as {class_name})"
        else:
            label = class_name
        lines.append(f"{i}. {label} — {m['tips_distance_m']} m, {m['direction']}")
    return "\n".join(lines)


def run_navigator_loop(
    session: Session,
    annotated_image: Image.Image | None,
    history: list[dict] | None = None,
    send_image: bool = True,
) -> tuple[str, list[dict]]:
    """Run the Navigator agent: single LLM call, no tools, produces all user-facing text.

    Receives the annotated image (numbered bounding boxes), a plain-text scene
    summary built from session.measurements, and the original user question.
    """
    t_total = time.monotonic()
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    scene_summary = _build_scene_summary(session)

    is_followup = history is not None
    image_part: list[dict] = []
    if send_image:
        image_url = _image_to_data_url(annotated_image)
        image_part = [{"type": "image_url", "image_url": {"url": image_url}}]

    ref = "the image and the measurements" if send_image else "the measurements"
    if is_followup:
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
        *_build_question_content(session.question),
    ]

    if history is not None:
        messages: list[dict] = (
            [{"role": "system", "content": NAVIGATOR_SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": question_content}]
        )
    else:
        messages = [
            {"role": "system", "content": NAVIGATOR_SYSTEM_PROMPT},
            {"role": "user", "content": question_content},
        ]

    request_payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    _maybe_dump_request(0, request_payload)

    t0 = time.monotonic()
    response = client.chat.completions.create(**request_payload)
    _maybe_dump_response(0, response)
    llm_latency = time.monotonic() - t0
    total_latency = time.monotonic() - t_total
    session.add_timing("navigator.llm", llm_latency)
    session.add_timing("navigator.total", total_latency)
    logger.info(
        "navigator llm_latency=%.3fs total_latency=%.3fs",
        llm_latency,
        total_latency,
    )

    final_text = response.choices[0].message.content or ""
    session.spatial_report = final_text
    messages.append({"role": "assistant", "content": final_text})
    return final_text, messages[1:]
