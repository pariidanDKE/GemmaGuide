from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import logging
import os
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
- First identify and measure the destination object instance using measure_object with include_obstacles=true.
- If the result contains an "obstacles" list, every entry is a confirmed object blocking the direct path — do not skip or ignore them.
- Provide practical step-by-step guidance: for each obstacle state its distance and direction, then tell the user which side to step around it before continuing toward the destination.
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
    Assistant tool flow: call_dpt_head -> measure_object(destination door, include_obstacles=true) -> final distance-aware path guidance using the returned obstacles list.
- User: "How do I get to the shopping cart?" (OOV class)
    Assistant tool flow: search_seg_classes("shopping cart") -> call_encoder_zero_shot(["shopping cart"]) -> measure_object("shopping cart", box_2d) -> measure_object(intervening obstacles) -> final distance-aware path guidance.

Conversation style:
- Friendly, calm, concise, and practical.
- Natural language only; never output raw tool JSON.
"""
SYSTEM_PROMPT = os.getenv("SPATIALSENSE_SYSTEM_PROMPT", SYSTEM_PROMPT)


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


def build_messages(image: Image.Image | None, question: str | bytes) -> list[dict]:
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})
    content.extend(_build_question_content(question))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
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
                include_obstacles=args.get("include_obstacles", False),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc)}

    latency = round(time.monotonic() - t0, 3)
    logger.info("tool=%s latency=%.3fs", name, latency)
    if print_tool_io:
        print(f"TOOL_OUTPUT {name}: {json.dumps(result, ensure_ascii=True)}")
    return json.dumps(result)


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
        tools = _active_tools() if has_image else []
        request_payload = {
            "model": MODEL_ID,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto" if has_image else "none",
            "parallel_tool_calls": True,
            "max_tokens": MAX_TOKENS,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                }
            },
        }
        _maybe_dump_request(round_idx, request_payload)

        t0 = time.monotonic()
        response = client.chat.completions.create(
            **request_payload,
        )
        _maybe_dump_response(round_idx, response)
        logger.info("round=%s llm_latency=%.3fs", round_idx, time.monotonic() - t0)

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
        logger.info("agent_loop_total_latency=%.3fs rounds=%s", time.monotonic() - t_total, round_idx + 1)
        return final_text, messages[1:]

    err = "I was unable to complete the spatial analysis. Please try again."
    logger.info("agent_loop_total_latency=%.3fs rounds=%s", time.monotonic() - t_total, MAX_TOOL_ROUNDS)
    return err, messages[1:]
