from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

# Input resolution for TIPSv2 — both dims rounded to multiples of 14 (ViT-B/14 patch size)
# 448 = training resolution, 896 = 2x for finer depth. Reduce to 448 if OOM.
TIPSV2_SHORT_SIDE = 896

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
        _model = AutoModel.from_pretrained("google/tipsv2-b14-dpt", trust_remote_code=True)
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


def get_detected_classes(seg_mask: torch.Tensor) -> list[str]:
    unique_indices = seg_mask.unique().tolist()
    return [ADE20K_CLASSES[int(i)] for i in unique_indices if 0 <= int(i) < len(ADE20K_CLASSES)]


def run_zero_shot_inference(
    image: Image.Image,
    class_list: list[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[str]]:
    """Zero-shot segmentation using TIPSv2 base encoder + text encoder cosine similarity."""
    from transformers import AutoTokenizer

    model = load_model()
    resized = _resize_for_tips(image, TIPSV2_SHORT_SIDE)
    pixel_values = _transform(resized).unsqueeze(0)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()

    with torch.no_grad():
        outputs = model(pixel_values)

    depth_tensor = outputs.depth.cpu()  # (1,1,H,W)

    cosine_maps: dict[str, torch.Tensor] = {}
    detected: list[str] = []

    # Extract patch-level image embeddings from vision encoder hidden states
    hidden = None
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        hidden = outputs.hidden_states[-1]

    if hidden is not None:
        num_patches = hidden.shape[1]
        H_patch = W_patch = int(num_patches ** 0.5)
        image_feats = hidden.squeeze(0).cpu()  # (num_patches, hidden_dim)
        image_feats = torch.nn.functional.normalize(image_feats, dim=-1)

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/tipsv2-b14")
            text_model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
            text_model.eval()

            for cls_name in class_list:
                text_inputs = tokenizer(
                    [f"a photo of a {cls_name}"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                with torch.no_grad():
                    text_out = text_model(**text_inputs)
                text_feat = text_out.last_hidden_state[:, 0, :]
                text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

                sim = (image_feats @ text_feat.T).squeeze(-1)  # (num_patches,)
                sim_map = sim.reshape(H_patch, W_patch)
                cosine_maps[cls_name] = sim_map

                if sim_map.max().item() > 0.2:
                    detected.append(cls_name)
        except Exception:
            pass

    return depth_tensor, cosine_maps, detected
