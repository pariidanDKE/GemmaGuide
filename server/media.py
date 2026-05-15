from __future__ import annotations

import base64
import io

from PIL import Image


def resize_for_multiple(image: Image.Image, multiple: int) -> Image.Image:
    if multiple <= 1:
        return image

    w, h = image.size
    new_w = max(multiple, int(round(w / multiple) * multiple))
    new_h = max(multiple, int(round(h / multiple) * multiple))

    if new_w == w and new_h == h:
        return image

    return image.resize((new_w, new_h), Image.BILINEAR)


def image_to_jpeg_b64(
    image: Image.Image | None,
    *,
    resize_to: tuple[int, int] | None = None,
    quality: int = 85,
) -> str | None:
    if image is None:
        return None
    out = image.convert("RGB")
    if resize_to is not None and out.size != resize_to:
        out = out.resize(resize_to, Image.BILINEAR)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def image_to_data_url(
    image: Image.Image,
    *,
    multiple: int = 1,
    quality: int = 90,
) -> str:
    out = resize_for_multiple(image, multiple).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
