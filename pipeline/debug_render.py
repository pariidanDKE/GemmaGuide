from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from pipeline.session import Session


logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOX_DEBUG_DIR = os.getenv(
    "SPATIALSENSE_BOX_DEBUG_DIR",
    str(_REPO_ROOT / "data" / "media" / "debug_boxes"),
)

_BOX_COLORS = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 120, 255),
    (255, 200, 50),
    (200, 80, 255),
    (50, 220, 220),
    (255, 140, 0),
    (200, 200, 200),
]


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return slug.strip("_") or "unknown"


def session_debug_dir(session: Session) -> Path | None:
    if not _BOX_DEBUG_DIR:
        return None
    out_dir = Path(_BOX_DEBUG_DIR) / session.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def draw_text_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text)
        pad = 3
        draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    except Exception:
        w = int(len(text) * 7.5)
        h = 14
        draw.rectangle((x - 3, y - 3, x + w + 3, y + h + 3), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))


def add_depth_legend_panel(image: Image.Image, d_min: float, d_max: float) -> Image.Image:
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
    draw_text_badge(draw, (W + outer_pad, outer_pad), "Depth (m)")

    bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    for y in range(bar_h):
        t = 1.0 - (y / max(bar_h - 1, 1))
        rgb = (np.array(cm.plasma(t)[:3]) * 255.0).astype(np.uint8)
        bar[y, :, :] = rgb
    canvas.paste(Image.fromarray(bar, mode="RGB"), (bar_left, bar_top))

    draw.rectangle((bar_left - 1, bar_top - 1, bar_right + 1, bar_top + bar_h + 1), outline=(230, 230, 230), width=1)
    for frac in [1.0, 0.75, 0.5, 0.25, 0.0]:
        y = int(round(bar_top + (1.0 - frac) * (bar_h - 1)))
        value = d_min + frac * (d_max - d_min)
        draw.line((bar_right + 5, y, bar_right + 13, y), fill=(240, 240, 240), width=1)
        draw_text_badge(draw, (bar_right + 18, max(0, y - 8)), f"{value:.2f}")

    return canvas


def save_depth_colormap_with_matplotlib(depth_2d: torch.Tensor, out_path: Path) -> None:
    depth_np = depth_2d.detach().cpu().numpy().astype(np.float32)
    h, w = depth_np.shape
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


def save_depth_and_seg_debug_images(session: Session) -> None:
    out_dir = session_debug_dir(session)
    if out_dir is None or session.seg_mask is None or session.depth_colormap is None:
        return

    try:
        ts = int(time.time() * 1000)
        if session.depth_tensor is not None:
            save_depth_colormap_with_matplotlib(
                session.depth_tensor.squeeze(),
                out_dir / f"depth_colormap_with_legend__ts-{ts}.png",
            )

        seg = session.seg_mask.detach().cpu().to(torch.int64)
        r = ((seg * 37) % 255).to(torch.uint8)
        g = ((seg * 73) % 255).to(torch.uint8)
        b = ((seg * 151) % 255).to(torch.uint8)
        seg_rgb = torch.stack([r, g, b], dim=-1).contiguous().numpy()
        seg_img = Image.fromarray(seg_rgb, mode="RGB")
        if seg_img.size != session.image.size:
            seg_img = seg_img.resize(session.image.size, resample=Image.NEAREST)
        Image.blend(session.image, seg_img, alpha=0.45).save(out_dir / f"segmentation_overlay__ts-{ts}.png")
    except Exception:
        return


def save_measurement_debug_overlay(
    *,
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
    out_dir = session_debug_dir(session)
    if out_dir is None or session.depth_colormap is None:
        return None

    try:
        H, W = depth_2d.shape
        ymin, xmin, ymax, xmax = box_dpt
        base = session.image.resize((W, H), resample=Image.BILINEAR).convert("RGB")
        base_np = np.array(base, dtype=np.uint8)

        class_np = class_region.detach().cpu().numpy().astype(bool)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        class_tint = np.array([0, 180, 255], dtype=np.uint8)
        base_np[class_np] = (0.75 * base_np[class_np].astype(np.float32) + 0.25 * class_tint.astype(np.float32)).astype(np.uint8)

        depth_heat = session.depth_colormap.convert("RGB")
        if depth_heat.size != (W, H):
            depth_heat = depth_heat.resize((W, H), resample=Image.BILINEAR)
        heat_np = np.array(depth_heat, dtype=np.uint8)
        base_np[mask_np] = (0.2 * base_np[mask_np].astype(np.float32) + 0.8 * heat_np[mask_np].astype(np.float32)).astype(np.uint8)

        vis = Image.fromarray(base_np, mode="RGB")
        draw = ImageDraw.Draw(vis)
        draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=3)

        d_min = float(depth_2d.min().item())
        d_max = float(depth_2d.max().item())
        selected_count = int(mask.sum().item())
        draw_text_badge(draw, (8, 8), f"class={class_name} selected_px={selected_count}")
        draw_text_badge(draw, (8, 30), f"box_dpt=[{ymin},{xmin},{ymax},{xmax}] on {W}x{H}")
        draw_text_badge(
            draw,
            (8, 52),
            f"FALLBACK: requested={requested_class_name} -> used={class_name}" if class_substituted else "FALLBACK: no",
        )
        oymin, oxmin, oymax, oxmax = box_original
        orig_w, orig_h = original_size
        draw_text_badge(draw, (8, 74), f"box_original=[{oymin},{oxmin},{oymax},{oxmax}] on {orig_w}x{orig_h}")

        canvas = add_depth_legend_panel(vis, d_min, d_max)

        source_grid = session.image.resize((W, H), resample=Image.BILINEAR).convert("RGB")
        sdraw = ImageDraw.Draw(source_grid)
        sdraw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=3)
        draw_text_badge(sdraw, (8, 8), "EXACT GRID USED FOR EXTRACTION (DPT SPACE)")
        draw_text_badge(sdraw, (8, 30), f"class={class_name}")
        draw_text_badge(
            sdraw,
            (8, 52),
            f"FALLBACK: requested={requested_class_name} -> used={class_name}" if class_substituted else "FALLBACK: no",
        )
        draw_text_badge(sdraw, (8, 74), f"box_dpt=[{ymin},{xmin},{ymax},{xmax}] on {W}x{H}")

        class_count = int(class_np.sum())
        combined_mask_np = np.zeros((H, W, 3), dtype=np.uint8)
        combined_mask_np[class_np] = np.array([0, 220, 255], dtype=np.uint8)
        combined_mask_np[mask_np] = np.array([255, 255, 255], dtype=np.uint8)
        combined_mask = Image.fromarray(combined_mask_np, mode="RGB")
        cmask_draw = ImageDraw.Draw(combined_mask)
        draw_text_badge(cmask_draw, (8, 8), "CLASS REGION=CYAN, SELECTED INTERSECTION=WHITE")
        draw_text_badge(cmask_draw, (8, 30), f"class={class_name}")
        draw_text_badge(
            cmask_draw,
            (8, 52),
            f"FALLBACK: requested={requested_class_name} -> used={class_name}" if class_substituted else "FALLBACK: no",
        )
        draw_text_badge(cmask_draw, (8, 74), f"class_region_px={class_count}")
        draw_text_badge(cmask_draw, (8, 96), f"selected_px={selected_count}")
        draw_text_badge(cmask_draw, (8, 118), f"selection_ratio={selected_count / max(class_count, 1):.4f}")

        ts = int(time.time() * 1000)
        class_slug = safe_slug(class_name)
        out_path = out_dir / f"measure_overlay_with_legend__class-{class_slug}__ts-{ts}.png"
        canvas.save(out_path)
        source_grid.save(out_dir / f"measure_source_grid_dpt__class-{class_slug}__ts-{ts}.png")
        combined_mask.save(out_dir / f"measure_combined_mask__class-{class_slug}__ts-{ts}.png")
        return str(out_path)
    except Exception:
        return None


def render_annotated_image(session: Session) -> Image.Image:
    from PIL import ImageFont

    img = session.image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    font_size = max(16, int(min(W, H) * 0.018))
    box_thickness = max(2, int(min(W, H) * 0.003))
    pad = max(4, font_size // 4)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for i, m in enumerate(session.measurements):
        ymin, xmin, ymax, xmax = m["box_original"]
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=box_thickness)
        label = str(i + 1)
        try:
            bbox = draw.textbbox((xmin + pad, ymin + pad), label, font=font)
            draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
            draw.text((xmin + pad, ymin + pad), label, fill=color, font=font)
        except Exception:
            draw_text_badge(draw, (xmin + 4, ymin + 4), label)

    out_dir = session_debug_dir(session)
    if out_dir is not None:
        ts = int(time.time() * 1000)
        img.save(out_dir / f"navigator_annotated__ts-{ts}.png")
    return img
