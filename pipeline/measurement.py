from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
import os

import torch

from pipeline.box_utils import normalize_box_to_original_coords, scale_box_to_tensor_space, top_classes_in_box
from pipeline.debug_render import save_measurement_debug_overlay
from pipeline.session import Session
from pipeline.tips_runner import ADE20K_CLASSES


_MIN_CLASS_PIXELS_IN_BOX = int(os.getenv("SPATIALSENSE_MIN_CLASS_PIXELS_IN_BOX", "500"))


@dataclass
class MeasurementComputation:
    response: dict
    measurement_entry: dict | None


def compute_measurement(session: Session, class_name: str, box_2d: list[int]) -> MeasurementComputation:
    orig_w, orig_h = session.image.size
    box_original = normalize_box_to_original_coords(box_2d, orig_w, orig_h)

    if session.depth_tensor is None or session.seg_mask is None:
        return MeasurementComputation(
            response={"error": "call_dpt_head or call_encoder_zero_shot must be called first"},
            measurement_entry=None,
        )

    class_idx: int | None = None
    cosine_map: torch.Tensor | None = None

    try:
        class_idx = ADE20K_CLASSES.index(class_name)
    except ValueError:
        pass

    if class_name in session.zero_shot_maps:
        cosine_map = session.zero_shot_maps[class_name]

    if class_idx is None and cosine_map is None:
        return MeasurementComputation(
            response={"error": f"Unknown class '{class_name}'. Run search_seg_classes or call_encoder_zero_shot first."},
            measurement_entry=None,
        )

    if len(box_2d) != 4:
        return MeasurementComputation(response={"error": "box_2d must be [ymin, xmin, ymax, xmax]"}, measurement_entry=None)

    H, W = session.seg_mask.shape
    box_dpt, error = scale_box_to_tensor_space(
        box_original,
        orig_w=orig_w,
        orig_h=orig_h,
        tensor_w=W,
        tensor_h=H,
    )
    if box_dpt is None:
        return MeasurementComputation(
            response={"error": f"Invalid bounding box {box_2d}: {error}"},
            measurement_entry=None,
        )

    ymin, xmin, ymax, xmax = box_dpt
    box_top_classes: list[dict[str, int | str]] = []
    requested_class_in_box_pixels: int | None = None
    used_class_in_box_pixels: int | None = None
    used_class_name = class_name
    class_substituted = False
    box_region = torch.zeros(H, W, dtype=torch.bool)
    box_region[ymin:ymax, xmin:xmax] = True

    if cosine_map is not None:
        if cosine_map.shape != (H, W):
            cosine_map_resized = torch.nn.functional.interpolate(
                cosine_map.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        else:
            cosine_map_resized = cosine_map.float()
        mask = box_region & (cosine_map_resized > 0.2)
        class_region = cosine_map_resized > 0.2

    if cosine_map is None or mask.sum() < 500:
        seg_mask = session.seg_mask
        box_top_classes = top_classes_in_box(seg_mask, ymin, xmin, ymax, xmax, top_k=5)
        requested_class_in_box_pixels = int((seg_mask[ymin:ymax, xmin:xmax] == class_idx).sum().item()) if class_idx is not None else 0
        used_class_in_box_pixels = requested_class_in_box_pixels

        used_class_idx = class_idx
        if (requested_class_in_box_pixels < _MIN_CLASS_PIXELS_IN_BOX or class_idx is None) and box_top_classes:
            top = box_top_classes[0]
            used_class_idx = int(top["class_idx"])
            used_class_name = str(top["class_name"])
            used_class_in_box_pixels = int(top["pixel_count"])
            class_substituted = used_class_name != class_name

        class_region = seg_mask == used_class_idx if used_class_idx is not None else box_region
        mask = box_region & class_region

    depth_2d = session.depth_tensor.squeeze()
    save_measurement_debug_overlay(
        session=session,
        class_name=used_class_name,
        requested_class_name=class_name,
        class_substituted=class_substituted,
        box_original=box_original,
        original_size=(orig_w, orig_h),
        box_dpt=box_dpt,
        mask=mask,
        class_region=class_region,
        depth_2d=depth_2d,
    )

    pixel_count = int(mask.sum().item())
    class_pixel_total = int(class_region.sum().item())
    selection_ratio = float(pixel_count / max(class_pixel_total, 1))
    if pixel_count == 0:
        if class_substituted:
            err = (
                f"No '{used_class_name}' pixels found within the bounding box {box_original} "
                f"after substituting requested class '{class_name}' with dominant box class '{used_class_name}'. "
                "Please review the image and provide a tighter bounding box around the object."
            )
        else:
            err = (
                f"No '{class_name}' pixels found within the bounding box {box_original}. "
                "Please review the image and provide a tighter bounding box around the object."
            )
        out = {
            "error": err,
            "applied_box_2d": box_original,
            "class_pixels_total": class_pixel_total,
            "selected_pixels": pixel_count,
            "selection_ratio": round(selection_ratio, 4),
        }
        if box_top_classes:
            out["box_top_classes"] = box_top_classes
        if requested_class_in_box_pixels is not None:
            out["requested_class_in_box_pixels"] = requested_class_in_box_pixels
            out["requested_class_name"] = class_name
            out["used_class_name"] = used_class_name
            out["used_class_in_box_pixels"] = used_class_in_box_pixels
            out["class_substituted"] = class_substituted
        return MeasurementComputation(response=out, measurement_entry=None)

    distance_m = float(depth_2d[mask].median().item())
    ys, xs = torch.where(mask)
    px = float(xs.float().mean().item())
    py = float(ys.float().mean().item())

    px_orig = px * (orig_w / float(W))
    _py_orig = py * (orig_h / float(H))
    bearing_deg = degrees(atan2((px_orig - session.intrinsics.cx) / session.intrinsics.fx, 1.0))

    bearing_rounded = round(bearing_deg, 2)
    abs_bearing = abs(bearing_rounded)
    if abs_bearing <= 10:
        direction = "straight ahead"
    elif bearing_rounded < 0:
        direction = f"about {abs_bearing:.0f} degrees to your left"
    else:
        direction = f"about {abs_bearing:.0f} degrees to your right"

    response: dict = {"class_name": used_class_name, "distance_m": round(distance_m, 3), "direction": direction}
    if class_substituted:
        response["class_substituted"] = True
        response["requested_class_name"] = class_name
        if box_top_classes:
            response["box_top_classes"] = box_top_classes
        if requested_class_in_box_pixels is not None and requested_class_in_box_pixels > 0:
            response["note"] = (
                f"Only {requested_class_in_box_pixels} pixel(s) of '{class_name}' found in box "
                f"(threshold {_MIN_CLASS_PIXELS_IN_BOX}) — used dominant class '{used_class_name}' instead. "
                "Do NOT use this distance for the requested object. "
                "Check box_top_classes and retry measure_object with the correct class label and a tighter box."
            )
        else:
            response["note"] = (
                f"'{class_name}' not found in box — used dominant class '{used_class_name}' instead. "
                "Do NOT use this distance for the requested object. "
                "Check box_top_classes and retry measure_object with the correct class label and a tighter box."
            )

    measurement_entry = {
        "class_name": used_class_name,
        "requested_class_name": class_name,
        "used_class_name": used_class_name,
        "class_substituted": class_substituted,
        "box_original": box_original,
        "tips_distance_m": round(distance_m, 3),
        "direction": direction,
        "mask_dpt": mask,
    }
    return MeasurementComputation(response=response, measurement_entry=measurement_entry)
