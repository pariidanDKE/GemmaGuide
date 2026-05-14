from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw

from server.runtime import create_vllm_client
MODEL_ID = "gemma-4-e2b-it"


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def parse_box_2d(text: str) -> list[int] | None:
    text = text.strip()

    # First try full JSON payload parsing.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("box_2d"), list):
            vals = obj["box_2d"]
            if len(vals) == 4:
                return [int(v) for v in vals]
    except json.JSONDecodeError:
        pass

    # Fallback: extract from box_2d-like bracket pattern.
    m = re.search(r"box_2d\s*[:=]\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
    if m:
        return [int(m.group(i)) for i in range(1, 5)]

    # Final fallback: first 4 ints in output.
    ints = re.findall(r"-?\d+", text)
    if len(ints) >= 4:
        return [int(ints[0]), int(ints[1]), int(ints[2]), int(ints[3])]

    return None


def clamp_box(box_2d: list[int], w: int, h: int) -> list[int]:
    ymin, xmin, ymax, xmax = box_2d
    ymin = max(0, min(ymin, h - 1))
    xmin = max(0, min(xmin, w - 1))
    ymax = max(0, min(ymax, h))
    xmax = max(0, min(xmax, w))
    return [ymin, xmin, ymax, xmax]


def draw_and_save(image: Image.Image, box_2d: list[int], label: str, out_path: Path) -> None:
    ymin, xmin, ymax, xmax = box_2d
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=5)
    draw.text((xmin + 6, max(0, ymin - 22)), label, fill=(255, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="BBox-only Gemma debug runner")
    parser.add_argument("--image", default="data/media/images/original_highres.jpeg")
    parser.add_argument("--class-name", default="desk")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--split-instructions", action="store_true", help="Send image and bbox instruction in separate user messages")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    image_url = image_to_data_url(image)

    bbox_instruction = (
        f"Return ONLY JSON like {{\"class_name\":\"{args.class_name}\",\"box_2d\":[ymin,xmin,ymax,xmax]}} "
        f"for the {args.class_name} in this image. Coordinates must be in original image pixels."
    )

    messages: list[dict] = [
        {
            "role": "system",
            "content": "You are a bounding-box detector. Output only JSON. No explanation.",
        }
    ]

    if args.split_instructions:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": bbox_instruction},
                ],
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": bbox_instruction},
                ],
            }
        )

    client = create_vllm_client()
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    content = (response.choices[0].message.content or "").strip()
    box_2d = parse_box_2d(content)

    if box_2d is None:
        print("RAW_RESPONSE_START")
        print(content)
        print("RAW_RESPONSE_END")
        raise SystemExit("Could not parse box_2d from model output")

    box_2d = clamp_box(box_2d, w, h)

    debug_dir = Path("data/media/debug_boxes")
    ts = int(time.time() * 1000)
    split_tag = "split" if args.split_instructions else "adjacent"
    out_path = debug_dir / f"bbox_only_{args.class_name}_{split_tag}_{ts}.png"
    draw_and_save(image, box_2d, f"{args.class_name} box", out_path)

    print("MODEL_OUTPUT_START")
    print(content)
    print("MODEL_OUTPUT_END")
    print(f"BOX_2D={box_2d}")
    print(f"SAVED_DEBUG_IMAGE={out_path}")


if __name__ == "__main__":
    main()
