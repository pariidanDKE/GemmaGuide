from __future__ import annotations

import torch

from pipeline.tips_runner import ADE20K_CLASSES


def normalize_box_to_original_coords(box_2d: list[int], image_w: int, image_h: int) -> list[int]:
    """Map Gemma's normalized 0..1000 box coordinates to original image pixels."""
    if len(box_2d) != 4:
        return box_2d

    ymin, xmin, ymax, xmax = box_2d
    return [
        int(round(ymin * (image_h / 1000.0))),
        int(round(xmin * (image_w / 1000.0))),
        int(round(ymax * (image_h / 1000.0))),
        int(round(xmax * (image_w / 1000.0))),
    ]


def top_classes_in_box(
    seg_mask: torch.Tensor,
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
    top_k: int = 5,
) -> list[dict[str, int | str]]:
    """Return top ADE classes by pixel count within the selected DPT-space bbox."""
    box = seg_mask[ymin:ymax, xmin:xmax]
    if box.numel() == 0:
        return []

    values, counts = torch.unique(box, return_counts=True)
    pairs = sorted(
        ((int(v.item()), int(c.item())) for v, c in zip(values, counts)),
        key=lambda t: t[1],
        reverse=True,
    )
    result: list[dict[str, int | str]] = []
    for cls_idx, pixel_count in pairs[:top_k]:
        if 0 <= cls_idx < len(ADE20K_CLASSES):
            cls_name = ADE20K_CLASSES[cls_idx]
        else:
            cls_name = f"class_{cls_idx}"
        result.append({"class_name": cls_name, "pixel_count": pixel_count, "class_idx": cls_idx})
    return result


def scale_box_to_tensor_space(
    box_original: list[int],
    *,
    orig_w: int,
    orig_h: int,
    tensor_w: int,
    tensor_h: int,
) -> tuple[list[int] | None, str | None]:
    ymin, xmin, ymax, xmax = box_original

    scale_x = tensor_w / float(orig_w)
    scale_y = tensor_h / float(orig_h)

    ymin = int(round(ymin * scale_y))
    xmin = int(round(xmin * scale_x))
    ymax = int(round(ymax * scale_y))
    xmax = int(round(xmax * scale_x))

    if ymax <= ymin or xmax <= xmin:
        return None, "ymax/xmax must be greater than ymin/xmin."

    ymin = max(0, min(ymin, tensor_h - 1))
    xmin = max(0, min(xmin, tensor_w - 1))
    ymax = max(0, min(ymax, tensor_h))
    xmax = max(0, min(xmax, tensor_w))
    return [ymin, xmin, ymax, xmax], None
