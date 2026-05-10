from __future__ import annotations

import logging
from math import atan2, degrees, radians, tan
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from pipeline.session import Session
from pipeline.tips_runner import ADE20K_CLASSES, get_detected_classes, run_dpt_inference


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOX_DEBUG_DIR = os.getenv(
    "SPATIALSENSE_BOX_DEBUG_DIR",
    str(_REPO_ROOT / "data" / "media" / "debug_boxes"),
)

# How many top-coverage ADE classes to surface to Gemma via call_dpt_head.
# Fringe classes (e.g. "armchair" with 5 stray pixels) are excluded; safety-cue
# computation inside call_dpt_head still uses the full class list.
_DETECTED_CLASSES_TOP_K = int(os.getenv("SPATIALSENSE_DETECTED_CLASSES_TOP_K", "5"))

# Minimum number of requested-class pixels that must exist inside the bounding box
# before we trust that class for depth measurement.  Below this threshold the
# fallback kicks in: we measure the dominant (highest-pixel-count) class in the
# box instead.  Set via SPATIALSENSE_MIN_CLASS_PIXELS_IN_BOX.
_MIN_CLASS_PIXELS_IN_BOX = int(os.getenv("SPATIALSENSE_MIN_CLASS_PIXELS_IN_BOX", "500"))


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return slug.strip("_") or "unknown"


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


def _top_classes_in_box(seg_mask: torch.Tensor, ymin: int, xmin: int, ymax: int, xmax: int, top_k: int = 5) -> list[dict[str, int | str]]:
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


def _draw_text_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    """Draw high-contrast text with a dark background for readability."""
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text)
        pad = 3
        draw.rectangle(
            (
                bbox[0] - pad,
                bbox[1] - pad,
                bbox[2] + pad,
                bbox[3] + pad,
            ),
            fill=(0, 0, 0),
        )
    except Exception:
        # Fallback for older Pillow versions without textbbox.
        w = int(len(text) * 7.5)
        h = 14
        draw.rectangle((x - 3, y - 3, x + w + 3, y + h + 3), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))


def _add_depth_legend_panel(image: Image.Image, d_min: float, d_max: float) -> Image.Image:
    """Attach a readable depth legend panel to the right side of an image."""
    base = image.convert("RGB")
    W, H = base.size

    panel_w = 180
    outer_pad = 10
    bar_w = 32
    bar_top = outer_pad + 20
    bar_bottom = H - outer_pad - 20
    bar_h = max(40, bar_bottom - bar_top)
    bar_left = W + outer_pad + 14
    bar_right = bar_left + bar_w

    canvas = Image.new("RGB", (W + panel_w, H), (18, 18, 18))
    canvas.paste(base, (0, 0))

    draw = ImageDraw.Draw(canvas)
    _draw_text_badge(draw, (W + outer_pad, outer_pad), "Depth (m)")

    # Build a vertical plasma color bar: top=max depth, bottom=min depth.
    bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    for y in range(bar_h):
        t = 1.0 - (y / max(bar_h - 1, 1))
        rgb = (np.array(cm.plasma(t)[:3]) * 255.0).astype(np.uint8)
        bar[y, :, :] = rgb
    bar_img = Image.fromarray(bar, mode="RGB")
    canvas.paste(bar_img, (bar_left, bar_top))

    draw.rectangle((bar_left - 1, bar_top - 1, bar_right + 1, bar_top + bar_h + 1), outline=(230, 230, 230), width=1)

    # Tick labels with background badges for high contrast.
    ticks = [1.0, 0.75, 0.5, 0.25, 0.0]
    for frac in ticks:
        y = int(round(bar_top + (1.0 - frac) * (bar_h - 1)))
        value = d_min + frac * (d_max - d_min)
        draw.line((bar_right + 5, y, bar_right + 13, y), fill=(240, 240, 240), width=1)
        _draw_text_badge(draw, (bar_right + 18, max(0, y - 8)), f"{value:.2f}")

    return canvas


def _save_depth_colormap_with_matplotlib(depth_2d: torch.Tensor, out_path: Path) -> None:
    """Save depth map with a standard matplotlib colorbar for clear, readable values."""
    depth_np = depth_2d.detach().cpu().numpy().astype(np.float32)
    h, w = depth_np.shape

    # Keep figure size proportional to the map while staying readable.
    fig_w = max(6.0, w / 180.0)
    fig_h = max(4.0, h / 180.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    im = ax.imshow(depth_np, cmap="plasma")
    ax.set_title("Depth Map", fontsize=14)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (m)", rotation=270, labelpad=16, fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _save_depth_and_seg_debug_images(session: Session) -> None:
    out_dir = _session_debug_dir(session)
    if out_dir is None or session.seg_mask is None or session.depth_colormap is None:
        return

    try:
        ts = int(time.time() * 1000)

        # Depth debug image: plain matplotlib colorbar for maximum readability.
        if session.depth_tensor is not None:
            depth_2d = session.depth_tensor.squeeze()
            depth_out = out_dir / f"depth_colormap_with_legend__ts-{ts}.png"
            _save_depth_colormap_with_matplotlib(depth_2d, depth_out)

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
        seg_out = out_dir / f"segmentation_overlay__ts-{ts}.png"
        seg_overlay.save(seg_out)
    except Exception:
        return


def _save_measurement_debug_overlay(
    session: Session,
    class_name: str,
    requested_class_name: str,
    class_substituted: bool,
    box_original: list[int],
    original_size: tuple[int, int],
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
        draw = ImageDraw.Draw(vis)
        _draw_text_badge(draw, (8, 8), f"class={class_name} selected_px={selected_count}")
        _draw_text_badge(draw, (8, 30), f"box_dpt=[{ymin},{xmin},{ymax},{xmax}] on {W}x{H}")
        if class_substituted:
            _draw_text_badge(draw, (8, 52), f"FALLBACK: requested={requested_class_name} -> used={class_name}")
            badge_y = 74
        else:
            _draw_text_badge(draw, (8, 52), "FALLBACK: no")
            badge_y = 74
        oymin, oxmin, oymax, oxmax = box_original
        orig_w, orig_h = original_size
        _draw_text_badge(
            draw,
            (8, badge_y),
            f"box_original=[{oymin},{oxmin},{oymax},{oxmax}] on {orig_w}x{orig_h}",
        )

        canvas = _add_depth_legend_panel(vis, d_min, d_max)

        class_slug = _safe_slug(class_name)

        # Exact image grid used for extraction (DPT-space RGB with box)
        source_grid = session.image.resize((W, H), resample=Image.BILINEAR).convert("RGB")
        sdraw = ImageDraw.Draw(source_grid)
        sdraw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=3)
        _draw_text_badge(sdraw, (8, 8), "EXACT GRID USED FOR EXTRACTION (DPT SPACE)")
        _draw_text_badge(sdraw, (8, 30), f"class={class_name}")
        if class_substituted:
            _draw_text_badge(sdraw, (8, 52), f"FALLBACK: requested={requested_class_name} -> used={class_name}")
            src_box_y = 74
        else:
            _draw_text_badge(sdraw, (8, 52), "FALLBACK: no")
            src_box_y = 74
        _draw_text_badge(sdraw, (8, src_box_y), f"box_dpt=[{ymin},{xmin},{ymax},{xmax}] on {W}x{H}")

        # Combined mask: class region + selected-pixel intersection in one image.
        class_count = int(class_np.sum())
        combined_mask_np = np.zeros((H, W, 3), dtype=np.uint8)
        combined_mask_np[class_np] = np.array([0, 220, 255], dtype=np.uint8)  # class region
        combined_mask_np[mask_np] = np.array([255, 255, 255], dtype=np.uint8)  # selected intersection
        combined_mask = Image.fromarray(combined_mask_np, mode="RGB")
        cmask_draw = ImageDraw.Draw(combined_mask)
        _draw_text_badge(cmask_draw, (8, 8), "CLASS REGION=CYAN, SELECTED INTERSECTION=WHITE")
        _draw_text_badge(cmask_draw, (8, 30), f"class={class_name}")
        if class_substituted:
            _draw_text_badge(cmask_draw, (8, 52), f"FALLBACK: requested={requested_class_name} -> used={class_name}")
            stat_y0 = 74
        else:
            _draw_text_badge(cmask_draw, (8, 52), "FALLBACK: no")
            stat_y0 = 74
        _draw_text_badge(cmask_draw, (8, stat_y0), f"class_region_px={class_count}")
        _draw_text_badge(cmask_draw, (8, stat_y0 + 22), f"selected_px={selected_count}")
        _draw_text_badge(
            cmask_draw,
            (8, stat_y0 + 44),
            f"selection_ratio={selected_count / max(class_count, 1):.4f}",
        )

        ts = int(time.time() * 1000)
        out_path = out_dir / f"measure_overlay_with_legend__class-{class_slug}__ts-{ts}.png"
        canvas.save(out_path)
        source_grid.save(out_dir / f"measure_source_grid_dpt__class-{class_slug}__ts-{ts}.png")
        combined_mask.save(out_dir / f"measure_combined_mask__class-{class_slug}__ts-{ts}.png")
        return str(out_path)
    except Exception:
        return None


# ── T008: search_seg_classes ─────────────────────────────────────────────────

def search_seg_classes(query: str) -> dict:
    query_lower = query.lower()
    matches = [cls for cls in ADE20K_CLASSES if query_lower in cls.lower()]
    return {"matches": matches}


_OBSTACLE_HALF_ANGLE_DEG = 7.5
_MIN_CLASS_PIXELS_CORRIDOR = 200
_SKIP_CLASSES = {"wall", "floor", "ceiling"}


def _obstacles_in_corridor(
    depth_tensor: torch.Tensor,
    seg_mask: torch.Tensor,
    session: Session,
    center_bearing_deg: float,
    max_distance_m: float,
    half_angle_deg: float = _OBSTACLE_HALF_ANGLE_DEG,
    exclude_class: str | None = None,
) -> list[dict]:
    """Return objects within a bearing corridor that are closer than max_distance_m."""
    H, W = seg_mask.shape
    orig_w, _ = session.image.size
    scale_x = W / float(orig_w)
    cx_dpt = session.intrinsics.cx * scale_x
    fx_dpt = session.intrinsics.fx * scale_x

    center_x = cx_dpt + fx_dpt * tan(radians(center_bearing_deg))
    half_width = fx_dpt * tan(radians(half_angle_deg))
    x_lo = max(0, int(round(center_x - half_width)))
    x_hi = min(W, int(round(center_x + half_width)))
    if x_hi <= x_lo:
        return []

    depth_2d = depth_tensor.squeeze()
    corridor_mask = torch.zeros(H, W, dtype=torch.bool)
    corridor_mask[:, x_lo:x_hi] = True

    detected_all = get_detected_classes(seg_mask)
    cues = []
    for cls_name in detected_all:
        if cls_name in _SKIP_CLASSES:
            continue
        if exclude_class and cls_name == exclude_class:
            continue
        try:
            cls_idx = ADE20K_CLASSES.index(cls_name)
        except ValueError:
            continue
        class_mask = (seg_mask == cls_idx) & corridor_mask
        if class_mask.sum() < _MIN_CLASS_PIXELS_CORRIDOR:
            continue
        dist_m = float(depth_2d[class_mask].median().item())
        if dist_m < max_distance_m:
            _, xs = torch.where(class_mask)
            px = float(xs.float().mean().item())
            px_orig = px * (orig_w / float(W))
            bearing_deg = degrees(atan2((px_orig - session.intrinsics.cx) / session.intrinsics.fx, 1.0))
            bearing_rounded = round(bearing_deg, 2)
            abs_bearing = abs(bearing_rounded)
            if abs_bearing <= 10:
                direction = "straight ahead"
            elif bearing_rounded < 0:
                direction = f"about {abs_bearing:.0f} degrees to your left"
            else:
                direction = f"about {abs_bearing:.0f} degrees to your right"
            cues.append({"class_name": cls_name, "distance_m": round(dist_m, 2), "direction": direction, "_bearing": bearing_rounded})

    sorted_cues = sorted(cues, key=lambda c: c["distance_m"])

    # Deduplicate co-located objects (e.g. monitor + computer): same physical
    # entity if within 0.3m depth AND within 15° bearing. Keep nearest.
    deduplicated: list[dict] = []
    for cue in sorted_cues:
        if not any(
            abs(cue["distance_m"] - kept["distance_m"]) < 0.3
            and abs(cue["_bearing"] - kept["_bearing"]) < 15.0
            for kept in deduplicated
        ):
            deduplicated.append(cue)

    for cue in deduplicated:
        del cue["_bearing"]
    return deduplicated


# ── T009: call_dpt_head ──────────────────────────────────────────────────────

def call_dpt_head(session: Session) -> dict:
    t0 = time.monotonic()
    depth_tensor, seg_mask, depth_colormap = run_dpt_inference(session.image)
    session.depth_tensor = depth_tensor
    session.seg_mask = seg_mask
    session.depth_colormap = depth_colormap
    _save_depth_and_seg_debug_images(session)
    debug_dir = _session_debug_dir(session)
    if debug_dir:
        logger.info("debug_dir=%s", debug_dir)
    detected_all = get_detected_classes(seg_mask)
    top_detected = get_detected_classes(seg_mask, top_k=_DETECTED_CLASSES_TOP_K)
    logger.info(
        "tool=call_dpt_head depth_shape=%s seg_shape=%s detected_all=%s top_k=%s latency=%.3fs",
        tuple(depth_tensor.shape),
        tuple(seg_mask.shape),
        len(detected_all),
        len(top_detected),
        time.monotonic() - t0,
    )
    return {"status": "ready", "detected_classes": top_detected}


# ── T010: measure_object ─────────────────────────────────────────────────────

def measure_object(class_name: str, box_2d: list[int], session: Session, include_obstacles: bool = False) -> dict:
    orig_w, orig_h = session.image.size
    box_original = _normalize_box_to_original_coords(box_2d, orig_w, orig_h)

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
    box_top_classes: list[dict[str, int | str]] = []
    requested_class_in_box_pixels: int | None = None
    used_class_in_box_pixels: int | None = None
    used_class_name = class_name
    class_substituted = False
    box_region = torch.zeros(H, W, dtype=torch.bool)
    box_region[ymin:ymax, xmin:xmax] = True

    if cosine_map is not None:
        # Zero-shot path: threshold cosine similarity within the box
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
        # ADE20K path (or zero-shot fallback when cosine mask is empty):
        # use dominant segmented class in the bounding box.
        seg_mask = session.seg_mask
        box_top_classes = _top_classes_in_box(seg_mask, ymin, xmin, ymax, xmax, top_k=5)
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

    # depth_tensor: (1,1,H,W)
    depth_2d = session.depth_tensor.squeeze()  # (H, W)
    _save_measurement_debug_overlay(
        session,
        used_class_name,
        class_name,
        class_substituted,
        box_original,
        (orig_w, orig_h),
        box_dpt,
        mask,
        class_region,
        depth_2d,
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
        return out
    else:
        distance_m = float(depth_2d[mask].median().item())
        ys, xs = torch.where(mask)
        px = float(xs.float().mean().item())
        py = float(ys.float().mean().item())

    # Convert centroid back to original image coordinates for intrinsics math.
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

    out: dict = {"class_name": used_class_name, "distance_m": round(distance_m, 3), "direction": direction}
    if class_substituted:
        if requested_class_in_box_pixels is not None and requested_class_in_box_pixels > 0:
            out["note"] = (
                f"Only {requested_class_in_box_pixels} pixel(s) of '{class_name}' found in box "
                f"(threshold {_MIN_CLASS_PIXELS_IN_BOX}) — used dominant class '{used_class_name}' instead."
            )
        else:
            out["note"] = f"'{class_name}' not found in box — used dominant class '{used_class_name}' instead."

    if include_obstacles and session.seg_mask is not None:
        obstacles = _obstacles_in_corridor(
            session.depth_tensor, session.seg_mask, session,
            center_bearing_deg=bearing_rounded,
            max_distance_m=distance_m,
            exclude_class=used_class_name,
        )
        out["obstacles"] = obstacles

    session.measurements.append({
        "class_name": used_class_name,
        "box_original": box_original,
        "tips_distance_m": round(distance_m, 3),
        "mask_dpt": mask,  # (H_dpt, W_dpt) bool tensor — same pixels used for TIPS median
    })
    return out


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
