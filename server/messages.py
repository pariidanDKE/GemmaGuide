from __future__ import annotations

import base64

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
