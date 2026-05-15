from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from typing import Any

from pipeline import tools as pipeline_tools
from pipeline.session import Session
from server.runtime import create_vllm_client
from server.schemas import TOOL_SCHEMAS


logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("VLLM_MODEL_ID", "gemma-4-e4b-it")
MAX_TOOL_ROUNDS = 10
MAX_TOKENS = 1024
DUMP_REQUEST_PATH = os.getenv("SPATIALSENSE_DUMP_REQUEST_PATH", "")
DUMP_RESPONSE_PATH = os.getenv("SPATIALSENSE_DUMP_RESPONSE_PATH", "")
TOOL_MODE = os.getenv("SPATIALSENSE_TOOL_MODE", "full").strip().lower()


def create_agent_client() -> Any:
    return create_vllm_client()


def maybe_dump_request(round_idx: int, payload: dict[str, Any]) -> None:
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


def maybe_dump_response(round_idx: int, response: Any) -> None:
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


def active_tools() -> list[dict[str, Any]]:
    if TOOL_MODE == "dpt_only":
        return [
            schema
            for schema in TOOL_SCHEMAS
            if schema.get("function", {}).get("name") == "call_dpt_head"
        ]
    return TOOL_SCHEMAS


def make_request_payload(messages: list[dict], **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    payload.update(overrides)
    return payload


def execute_model_call(
    *,
    client: Any,
    session: Session,
    request_payload: dict[str, Any],
    round_idx: int,
    timing_stage: str,
    timing_meta: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    maybe_dump_request(round_idx, request_payload)
    t0 = time.monotonic()
    response = client.chat.completions.create(**request_payload)
    maybe_dump_response(round_idx, response)
    llm_latency = time.monotonic() - t0
    session.add_timing(timing_stage, llm_latency, **(timing_meta or {}))
    return response, llm_latency


def tool_call_to_message(tc: Any) -> dict[str, Any]:
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }


def assistant_tool_message(choice: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": choice.message.content or "",
        "tool_calls": [tool_call_to_message(tc) for tc in choice.message.tool_calls],
    }


def looks_like_raw_tool_protocol(text: str) -> bool:
    lowered = text.lower()
    return "<|tool_call" in lowered or "<tool_call" in lowered


def dispatch_tool(name: str, args: dict[str, Any], session: Session) -> str:
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
            result = pipeline_tools.call_encoder_zero_shot(args["class_list"], session)
        elif name == "measure_object":
            result = pipeline_tools.measure_object(args["class_name"], args["box_2d"], session)
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


def dispatch_tool_calls(tool_calls: list[Any], session: Session) -> list[dict[str, str]]:
    parsed_calls: list[tuple[str, str, dict[str, Any]]] = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        parsed_calls.append((tc.id, tc.function.name, args))

    all_measure = bool(parsed_calls) and all(name == "measure_object" for _, name, _ in parsed_calls)
    if all_measure and len(parsed_calls) > 1:
        max_workers = min(4, len(parsed_calls))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(dispatch_tool, name, args, session)
                for _, name, args in parsed_calls
            ]
            results = [f.result() for f in futures]
    else:
        results = [dispatch_tool(name, args, session) for _, name, args in parsed_calls]

    return [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        }
        for (call_id, _, _), result in zip(parsed_calls, results)
    ]
