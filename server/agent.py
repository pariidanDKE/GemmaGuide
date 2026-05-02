from __future__ import annotations

import base64
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
Use structured tool calls only. Never emit raw tool protocol tokens in text.

Tool policy:
1. For any spatial/object query, always call search_seg_classes first using the target noun from the user request.
2. Then call call_dpt_head.
3. Then call measure_object with the best class_name from search results and a tight box_2d for the target.
4. If measure_object returns a no-overlap error, choose a tighter box around the same target and retry once.

Response policy:
- Prioritize actionable navigation language.
- Include BOTH distance and direction whenever available.
- Direction should be plain language (left/right/ahead) and can include relative bearing if helpful.
- Keep responses concise and safety-oriented.
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


def build_messages(image: Image.Image, question: str | bytes) -> list[dict]:
    image_url = _image_to_data_url(image)
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    if isinstance(question, bytes):
        # Audio bytes: encode as base64 audio/wav for vLLM audio input
        b64_audio = base64.b64encode(question).decode("utf-8")
        content.append({
            "type": "input_audio",
            "input_audio": {"data": b64_audio, "format": "wav"},
        })
    else:
        content.append({"type": "text", "text": question})

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
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc)}

    latency = round(time.monotonic() - t0, 3)
    if print_tool_io:
        print(f"TOOL_OUTPUT {name}: {json.dumps(result, ensure_ascii=True)}")
    return json.dumps(result)


def run_agent_loop(session: Session) -> str:
    t_total = time.monotonic()
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    messages = build_messages(session.image, session.question)

    for round_idx in range(MAX_TOOL_ROUNDS):
        tools = _active_tools()
        request_payload = {
            "model": MODEL_ID,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
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

            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_result = _dispatch_tool(tc.function.name, args, session)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
            continue

        content = choice.message.content or ""
        if _looks_like_raw_tool_protocol(content):
            logger.warning("Model emitted raw tool-call protocol text instead of structured tool_calls")
            return (
                "You emitted the wrong format for the tool call. "
                "Please retry."
            )

        # Final text response
        final_text = content
        session.spatial_report = final_text
        logger.info("agent_loop_total_latency=%.3fs rounds=%s", time.monotonic() - t_total, round_idx + 1)
        return final_text

    logger.info("agent_loop_total_latency=%.3fs rounds=%s", time.monotonic() - t_total, MAX_TOOL_ROUNDS)
    return "I was unable to complete the spatial analysis. Please try again."
