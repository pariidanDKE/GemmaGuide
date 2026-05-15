from __future__ import annotations

import base64
from typing import Any

from PIL import Image

from server.media import image_to_data_url


def build_question_content(question: str | bytes) -> list[dict]:
    if isinstance(question, bytes):
        b64_audio = base64.b64encode(question).decode("utf-8")
        return [{"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}]
    return [{"type": "text", "text": question}]


def build_turn_user_content(
    image: Image.Image | None,
    question: str | bytes,
    *,
    send_image: bool = True,
    image_multiple: int = 48,
) -> list[dict]:
    content: list[dict] = []
    if send_image and image is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(image, multiple=image_multiple)},
        })
    content.extend(build_question_content(question))
    return content


def append_turn_to_history(
    history: list[dict],
    *,
    active_image: Image.Image | None,
    question: str | bytes,
    send_image: bool,
    response_text: str,
) -> list[dict]:
    updated_history = list(history)
    turn_user_content = build_turn_user_content(active_image, question, send_image=send_image)
    updated_history.append({"role": "user", "content": turn_user_content})
    updated_history.append({"role": "assistant", "content": response_text})
    return updated_history


def history_for_debug(history: list[dict] | None) -> list[dict[str, str]]:
    debug_items: list[dict[str, str]] = []
    for message in history or []:
        role = str(message.get("role", "unknown"))
        content = message.get("content", "")
        preview = _message_preview(content)
        debug_items.append({"role": role, "text": preview})
    return debug_items


def _message_preview(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text", "")))
            elif block_type == "image_url":
                parts.append("[image]")
            elif block_type == "input_audio":
                parts.append("[audio]")
            else:
                parts.append(f"[{block_type or 'content'}]")
        return " ".join(part for part in parts if part).strip()
    return str(content)
