from __future__ import annotations

import logging
from math import atan2, degrees
import os
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib.cm as cm
from PIL import Image, ImageDraw

from pipeline.session import Session
from pipeline.tips_runner import ADE20K_CLASSES, get_detected_classes, run_dpt_inference


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOX_DEBUG_DIR = os.getenv(
    "SPATIALSENSE_BOX_DEBUG_DIR",
    str(_REPO_ROOT / "data" / "media" / "debug_boxes"),
)


def _session_debug_dir(session: Session) -> Path | None:
    if not _BOX_DEBUG_DIR:
        return None
    out_dir = Path(_BOX_DEBUG_DIR) / session.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _normalize_box_to_original_coords(box_2d: list[int], image_w: int, image_h: int) -> list[int]:
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


def _save_box_debug_image(
    session: Session,
    class_name: str,
    box_2d: list[int],
) -> str | None:
    """Save the original image with the tool-call box drawn for debugging."""
    out_dir = _session_debug_dir(session)
    if out_dir is None:
        return None

    try:
        ymin, xmin, ymax, xmax = box_2d
        vis = session.image.copy()
        draw = ImageDraw.Draw(vis)
        draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=5)
        label = f"{class_name} box"
        label_y = max(0, ymin - 22)
        draw.text((xmin + 6, label_y), label, fill=(255, 0, 0))

        ts = int(time.time() * 1000)
        out_path = out_dir / f"{session.session_id}_{class_name}_{ts}.png"
        vis.save(out_path)
        return str(out_path)
    except Exception:
        return None


def _save_depth_and_seg_debug_images(session: Session) -> None:
    out_dir = _session_debug_dir(session)
    if out_dir is None or session.seg_mask is None or session.depth_colormap is None:
        return

    try:
        ts = int(time.time() * 1000)

        # Depth debug image
        depth_vis = session.depth_colormap.convert("RGB")
        if depth_vis.size != session.image.size:
            depth_vis = depth_vis.resize(session.image.size, resample=Image.BILINEAR)
        depth_out = out_dir / f"{session.session_id}_depth_{ts}.png"
        depth_vis.save(depth_out)

        # Segmentation overlay debug image
        seg = session.seg_mask.detach().cpu().to(torch.int64)
        r = ((seg * 37) % 255).to(torch.uint8)
        g = ((seg * 73) % 255).to(torch.uint8)
        b = ((seg * 151) % 255).to(torch.uint8)
        seg_rgb = torch.stack([r, g, b], dim=-1).contiguous().numpy()
        seg_img = Image.fromarray(seg_rgb, mode="RGB")
        if seg_img.size != session.image.size:
            seg_img = seg_img.resize(session.image.size, resample=Image.NEAREST)
        seg_overlay = Image.blend(session.image, seg_img, alpha=0.45)
        seg_out = out_dir / f"{session.session_id}_seg_overlay_{ts}.png"
        seg_overlay.save(seg_out)
    except Exception:
        return


def _save_measurement_debug_overlay(
    session: Session,
    class_name: str,
    box_dpt: list[int],
    mask: torch.Tensor,
    class_region: torch.Tensor,
    depth_2d: torch.Tensor,
) -> str | None:
    """Save DPT-space overlay showing pixels used for depth extraction."""
    out_dir = _session_debug_dir(session)
    if out_dir is None or session.depth_colormap is None:
        return None

    try:
        H, W = depth_2d.shape
        ymin, xmin, ymax, xmax = box_dpt

        base = session.image.resize((W, H), resample=Image.BILINEAR).convert("RGB")
        base_np = np.array(base, dtype=np.uint8)

        class_np = class_region.detach().cpu().numpy().astype(bool)
        mask_np = mask.detach().cpu().numpy().astype(bool)

        # Show class region faintly so overlap errors are easy to inspect.
        class_tint = np.array([0, 180, 255], dtype=np.uint8)
        base_np[class_np] = (
            0.75 * base_np[class_np].astype(np.float32) + 0.25 * class_tint.astype(np.float32)
        ).astype(np.uint8)

        depth_heat = session.depth_colormap.convert("RGB")
        if depth_heat.size != (W, H):
            depth_heat = depth_heat.resize((W, H), resample=Image.BILINEAR)
        heat_np = np.array(depth_heat, dtype=np.uint8)

        # Use heat colors only where pixels are actually used for depth.
        base_np[mask_np] = (
            0.2 * base_np[mask_np].astype(np.float32) + 0.8 * heat_np[mask_np].astype(np.float32)
        ).astype(np.uint8)

        vis = Image.fromarray(base_np, mode="RGB")
        draw = ImageDraw.Draw(vis)
        draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=3)

        d_min = float(depth_2d.min().item())
        d_max = float(depth_2d.max().item())
        selected_count = int(mask.sum().item())
        draw.text((8, 8), f"class={class_name} selected_px={selected_count}", fill=(255, 255, 255))

        # Draw a color legend that matches the plasma depth mapping.
        legend_w = 64
        legend_pad = 12
        legend = np.zeros((H, legend_w, 3), dtype=np.uint8)
        for y in range(H):
            t = 1.0 - (y / max(H - 1, 1))
            rgb = (np.array(cm.plasma(t)[:3]) * 255.0).astype(np.uint8)
            legend[y, :, :] = rgb
        legend_img = Image.fromarray(legend, mode="RGB")

        canvas = Image.new("RGB", (W + legend_w + legend_pad + 8, H), (20, 20, 20))
        canvas.paste(vis, (0, 0))
        canvas.paste(legend_img, (W + legend_pad, 0))

        cdraw = ImageDraw.Draw(canvas)
        cdraw.text((W + legend_pad + 4, 4), f"{d_max:.2f} m", fill=(255, 255, 255))
        cdraw.text((W + legend_pad + 4, max(H - 18, 0)), f"{d_min:.2f} m", fill=(255, 255, 255))
        cdraw.text((W + legend_pad + 4, max(H // 2 - 8, 0)), "depth", fill=(255, 255, 255))

        ts = int(time.time() * 1000)
        out_path = out_dir / f"{session.session_id}_{class_name}_measure_overlay_{ts}.png"
        canvas.save(out_path)
        return str(out_path)
    except Exception:
        return None


# ── T008: search_seg_classes ─────────────────────────────────────────────────

def search_seg_classes(query: str) -> dict:
    query_lower = query.lower()
    matches = [cls for cls in ADE20K_CLASSES if query_lower in cls.lower()]
    return {"matches": matches}


# ── T009: call_dpt_head ──────────────────────────────────────────────────────

def call_dpt_head(session: Session) -> dict:
    t0 = time.monotonic()
    depth_tensor, seg_mask, depth_colormap = run_dpt_inference(session.image)
    session.depth_tensor = depth_tensor
    session.seg_mask = seg_mask
    session.depth_colormap = depth_colormap
    _save_depth_and_seg_debug_images(session)
    detected = get_detected_classes(seg_mask)
    logger.info(
        "tool=call_dpt_head depth_shape=%s seg_shape=%s detected=%s latency=%.3fs",
        tuple(depth_tensor.shape),
        tuple(seg_mask.shape),
        len(detected),
        time.monotonic() - t0,
    )
    return {"status": "ready", "detected_classes": detected}


# ── T010: measure_object ─────────────────────────────────────────────────────

def measure_object(class_name: str, box_2d: list[int], session: Session) -> dict:
    orig_w, orig_h = session.image.size
    box_original = _normalize_box_to_original_coords(box_2d, orig_w, orig_h)
    _save_box_debug_image(session, class_name, box_original)

    if session.depth_tensor is None or session.seg_mask is None:
        return {"error": "call_dpt_head or call_encoder_zero_shot must be called first"}

    # Resolve class index — check both ADE20K vocab and zero-shot cosine maps
    class_idx: int | None = None
    cosine_map: torch.Tensor | None = None

    try:
        class_idx = ADE20K_CLASSES.index(class_name)
    except ValueError:
        pass

    # Detect whether this is a zero-shot session with stored cosine maps
    zero_shot_maps: dict[str, torch.Tensor] = getattr(session, "_zero_shot_maps", {})
    if class_name in zero_shot_maps:
        cosine_map = zero_shot_maps[class_name]

    if class_idx is None and cosine_map is None:
        return {"error": f"Unknown class '{class_name}'. Run search_seg_classes or call_encoder_zero_shot first."}

    if len(box_2d) != 4:
        return {"error": "box_2d must be [ymin, xmin, ymax, xmax]"}

    ymin, xmin, ymax, xmax = box_original
    H, W = session.seg_mask.shape

    # The model runs on a resized image; map user/model box coordinates from
    # original image space into the segmentation/depth tensor space.
    scale_x = W / float(orig_w)
    scale_y = H / float(orig_h)

    ymin = int(round(ymin * scale_y))
    xmin = int(round(xmin * scale_x))
    ymax = int(round(ymax * scale_y))
    xmax = int(round(xmax * scale_x))

    if ymax <= ymin or xmax <= xmin:
        return {
            "error": f"Invalid bounding box {box_2d}: ymax/xmax must be greater than ymin/xmin.",
        }

    # Clamp box to image bounds
    ymin = max(0, min(ymin, H - 1))
    xmin = max(0, min(xmin, W - 1))
    ymax = max(0, min(ymax, H))
    xmax = max(0, min(xmax, W))
    box_dpt = [ymin, xmin, ymax, xmax]

    # Build pixel mask
    if cosine_map is not None:
        # Zero-shot path: threshold cosine similarity within the box
        # Resize cosine_map to seg_mask spatial resolution if needed
        if cosine_map.shape != (H, W):
            cosine_map_resized = torch.nn.functional.interpolate(
                cosine_map.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        else:
            cosine_map_resized = cosine_map.float()
        box_region = torch.zeros(H, W, dtype=torch.bool)
        box_region[ymin:ymax, xmin:xmax] = True
        mask = box_region & (cosine_map_resized > 0.2)
        class_region = cosine_map_resized > 0.2
    else:
        seg_mask = session.seg_mask
        box_region = torch.zeros(H, W, dtype=torch.bool)
        box_region[ymin:ymax, xmin:xmax] = True
        class_region = seg_mask == class_idx
        mask = box_region & class_region

    # depth_tensor: (1,1,H,W)
    depth_2d = session.depth_tensor.squeeze()  # (H, W)
    _save_measurement_debug_overlay(session, class_name, box_dpt, mask, class_region, depth_2d)

    pixel_count = int(mask.sum().item())
    if pixel_count == 0:
        class_pixel_total = int(class_region.sum().item())
        return {
            "error": (
                f"No '{class_name}' pixels found within the bounding box {box_original}. "
                "Please review the image and provide a tighter bounding box around the object."
            ),
            "applied_box_2d": box_original,
            "class_pixels_total": class_pixel_total,
        }
    else:
        distance_m = float(depth_2d[mask].median().item())
        ys, xs = torch.where(mask)
        px = float(xs.float().mean().item())
        py = float(ys.float().mean().item())

        # Confidence scoring
        near_clip = (distance_m < 0.01) or (distance_m > 9.9)
        out_of_range = distance_m < 0.5 or distance_m > 8.0
        if near_clip or pixel_count < 500 or out_of_range:
            confidence = "low"
        elif cosine_map is not None:
            confidence = "medium"
        else:
            confidence = "high"

    # Convert centroid back to original image coordinates for intrinsics math.
    px_orig = px * (orig_w / float(W))
    _py_orig = py * (orig_h / float(H))
    bearing_deg = degrees(atan2((px_orig - session.intrinsics.cx) / session.intrinsics.fx, 1.0))

    bearing_rounded = round(bearing_deg, 2)
    return {
        "distance_m": round(distance_m, 3),
        "bearing_deg": bearing_rounded,
        "confidence": confidence,
        "bearing_interpretation": "bearing_deg is camera-relative from the current POV centerline: negative is counterclockwise from center, positive is clockwise, near zero is near center.",
    }


# ── T013: call_encoder_zero_shot ─────────────────────────────────────────────

def call_encoder_zero_shot(class_list: list[str], session: Session) -> dict:
    from pipeline.tips_runner import run_zero_shot_inference

    t0 = time.monotonic()
    depth_tensor, cosine_maps, detected = run_zero_shot_inference(session.image, class_list)
    session.depth_tensor = depth_tensor
    # Store cosine maps on the session for use by measure_object
    session._zero_shot_maps = cosine_maps  # type: ignore[attr-defined]
    # Also populate seg_mask with a zero tensor so precondition checks pass
    if session.seg_mask is None:
        H, W = depth_tensor.shape[-2], depth_tensor.shape[-1]
        session.seg_mask = torch.zeros(H, W, dtype=torch.long)
    logger.info(
        "tool=call_encoder_zero_shot class_count=%s detected=%s cosine_maps=%s depth_shape=%s latency=%.3fs",
        len(class_list),
        len(detected),
        len(cosine_maps),
        tuple(depth_tensor.shape),
        time.monotonic() - t0,
    )
    return {"status": "ready", "detected_classes": detected}
