from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

# Input resolution for TIPSv2 — both dims rounded to multiples of 14 (ViT-B/14 patch size)
# 448 = training resolution, 672 = T4-friendly default, 896 = 2x for finer depth.
# Can be overridden per run with env var SPATIALSENSE_TIPSV2_SHORT_SIDE.
TIPSV2_SHORT_SIDE = int(os.getenv("SPATIALSENSE_TIPSV2_SHORT_SIDE", "672"))

# Model selection for experiments.
# DPT model examples: google/tipsv2-b14-dpt, google/tipsv2-l14-dpt
# Base model examples: google/tipsv2-b14, google/tipsv2-l14
TIPSV2_DPT_MODEL_ID = os.getenv("SPATIALSENSE_TIPSV2_DPT_MODEL", "google/tipsv2-b14-dpt")
TIPSV2_BASE_MODEL_ID = os.getenv("SPATIALSENSE_TIPSV2_BASE_MODEL", "google/tipsv2-b14")

# Colormap used for demo depth display — set after empirical comparison (T019)
DEFAULT_COLORMAP = "plasma"

# Segmentation threshold approach — set after validation (T020)
SEGMENTATION_THRESHOLD = "argmax"

_ADE20K_CSV = Path(__file__).parent.parent / "data" / "ade20k" / "objectInfo150.csv"


def _load_ade20k_classes() -> list[str]:
    classes: list[str] = []
    with open(_ADE20K_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Primary name is the part before the first semicolon
            primary = row["Name"].split(";")[0].strip()
            classes.append(primary)
    assert len(classes) == 150, f"Expected 150 ADE20K classes, got {len(classes)}"
    return classes


ADE20K_CLASSES: list[str] = _load_ade20k_classes()

_model = None
_transform = transforms.ToTensor()


def load_model():
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(TIPSV2_DPT_MODEL_ID, trust_remote_code=True)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _model


def _resize_for_tips(image: Image.Image, short_side: int) -> Image.Image:
    """Resize preserving aspect ratio; round both dims to multiples of 14 (ViT-B/14 patch size)."""
    w, h = image.size
    if w < h:
        new_w = short_side
        new_h = int(round(h * short_side / w / 14) * 14)
        new_w = int(round(new_w / 14) * 14)
    else:
        new_h = short_side
        new_w = int(round(w * short_side / h / 14) * 14)
        new_h = int(round(new_h / 14) * 14)
    return image.resize((new_w, new_h), Image.BILINEAR)


def run_dpt_inference(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    model = load_model()
    resized = _resize_for_tips(image, TIPSV2_SHORT_SIDE)
    pixel_values = _transform(resized).unsqueeze(0)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()

    with torch.no_grad():
        outputs = model(pixel_values)

    # outputs.depth: (1, 1, H, W) float32 meters
    depth_tensor = outputs.depth.cpu()

    # outputs.segmentation: (1, 150, H, W) — argmax to get (H, W) class indices
    seg_mask = outputs.segmentation.cpu().squeeze(0).argmax(dim=0)

    depth_colormap = _make_colormap(depth_tensor.squeeze().numpy())
    return depth_tensor, seg_mask, depth_colormap


def _make_colormap(depth_np: np.ndarray) -> Image.Image:
    d_min, d_max = depth_np.min(), depth_np.max()
    normalized = (depth_np - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_np)
    colormap = getattr(cm, DEFAULT_COLORMAP)
    colored = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)


def get_detected_classes(seg_mask: torch.Tensor, top_k: int | None = None) -> list[str]:
    """Return ADE20K class names present in seg_mask, sorted by pixel coverage descending.

    Pass top_k to cap the list — only the top_k most-covered classes are returned.
    When top_k is None all present classes are returned (sorted by coverage).
    """
    unique_indices, counts = seg_mask.unique(return_counts=True)
    pairs = sorted(
        ((int(i.item()), int(c.item())) for i, c in zip(unique_indices, counts)),
        key=lambda t: t[1],
        reverse=True,
    )
    if top_k is not None:
        pairs = pairs[:top_k]
    return [ADE20K_CLASSES[cls_idx] for cls_idx, _ in pairs if 0 <= cls_idx < len(ADE20K_CLASSES)]


def run_zero_shot_inference(
    image: Image.Image,
    class_list: list[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[str]]:
    """Zero-shot segmentation using TIPSv2 backbone encode_image + encode_text cosine similarity.

    The DPT forward loads and caches the base backbone in model._backbone. We reuse it
    here to get per-patch embeddings and text embeddings — no second model load needed.
    """
    model = load_model()
    resized = _resize_for_tips(image, TIPSV2_SHORT_SIDE)
    pixel_values = _transform(resized).unsqueeze(0)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()

    with torch.no_grad():
        outputs = model(pixel_values)

    depth_tensor = outputs.depth.cpu()  # (1,1,H,W)

    # Trigger lazy backbone load if DPT forward didn't do it (safety)
    if getattr(model, "_backbone", None) is None:
        model._get_backbone()
    backbone = model._backbone

    cosine_maps: dict[str, torch.Tensor] = {}
    detected: list[str] = []

    try:
        with torch.no_grad():
            img_out = backbone.encode_image(pixel_values)

        # patch_tokens: (1, N_patches, D) — per-patch spatial features
        patch_tokens = img_out.patch_tokens.squeeze(0).cpu()  # (N, D)
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)

        # Spatial grid: ViT patch size = 14 for both b14 and l14
        H_patch = resized.height // 14
        W_patch = resized.width // 14
        N_expected = H_patch * W_patch
        if patch_tokens.shape[0] != N_expected:
            patch_tokens = patch_tokens[:N_expected]

        _ANCHORS = ["wall", "floor", "ceiling", "background"]
        all_classes = list(class_list) + _ANCHORS
        prompts = [f"a photo of a {cls}" for cls in all_classes]
        with torch.no_grad():
            text_embs = backbone.encode_text(prompts)  # (num_classes + anchors, D)
        text_embs = torch.nn.functional.normalize(text_embs.cpu(), dim=-1)

        # softmax across all classes+anchors per patch — anchors absorb background probability mass
        all_sims = patch_tokens @ text_embs.T
        all_probs = torch.softmax(all_sims, dim=-1)  # (N_patches, num_classes + anchors)

        for i, cls_name in enumerate(class_list):
            sim_map = all_probs[:, i].reshape(H_patch, W_patch)
            cosine_maps[cls_name] = sim_map
            max_sim = sim_map.max().item()
            logger.info("zero_shot cls=%s max_prob=%.4f", cls_name, max_sim)
            detected.append(cls_name)  # always include — measure_object will use box fallback if similarity is low

    except Exception:
        logger.exception("zero_shot backbone inference failed")

    return depth_tensor, cosine_maps, detected
