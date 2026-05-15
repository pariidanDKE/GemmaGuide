from __future__ import annotations

import logging
import os
import time

import torch

from pipeline.debug_render import save_depth_and_seg_debug_images, session_debug_dir
from pipeline.measurement import compute_measurement
from pipeline.session import Session
from pipeline.tips_runner import ADE20K_CLASSES, get_detected_classes, run_dpt_inference


logger = logging.getLogger(__name__)

# How many top-coverage ADE classes to surface to Gemma via call_dpt_head.
# Background and outdoor-only classes are filtered out before the k selection
# so slots go to navigable obstacle classes.
_DETECTED_CLASSES_TOP_K = int(os.getenv("SPATIALSENSE_DETECTED_CLASSES_TOP_K", "6"))

# ADE20K primary names (first token before ";") to strip from the top-k list.
# These are either above the user (ceiling/sky), outdoor terrain (grass/mountain/
# sea/river/…), or venue-specific (runway/grandstand/swimming pool) — none are
# meaningful navigation obstacles in everyday pedestrian use.
_TOP_K_EXCLUDE: frozenset[str] = frozenset({
    "sky",
    "ceiling",
    "grass",
    "mountain",
    "sea",
    "field",
    "sand",
    "skyscraper",
    "grandstand",
    "runway",
    "river",
    "hill",
    "hovel",
    "dirt",       # primary name of "dirt;track"
    "land",       # primary name of "land;ground;soil"
    "swimming",   # primary name of "swimming;pool;..."
    "waterfall",
    "lake",
    "earth",      # primary name of "earth;ground"
})

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
    save_depth_and_seg_debug_images(session)
    debug_dir = session_debug_dir(session)
    if debug_dir:
        logger.info("debug_dir=%s", debug_dir)
    detected_all = get_detected_classes(seg_mask)
    filtered = [c for c in detected_all if c not in _TOP_K_EXCLUDE]
    top_detected = filtered[:_DETECTED_CLASSES_TOP_K]
    logger.info(
        "tool=call_dpt_head depth_shape=%s seg_shape=%s detected_all=%s filtered=%s top_k=%s latency=%.3fs",
        tuple(depth_tensor.shape),
        tuple(seg_mask.shape),
        len(detected_all),
        len(filtered),
        len(top_detected),
        time.monotonic() - t0,
    )
    return {"status": "ready", "detected_classes": top_detected}


# ── T010: measure_object ─────────────────────────────────────────────────────

def measure_object(class_name: str, box_2d: list[int], session: Session) -> dict:
    computation = compute_measurement(session, class_name, box_2d)
    if computation.measurement_entry is not None:
        session.measurements.append(computation.measurement_entry)
    return computation.response


# ── T013: call_encoder_zero_shot ─────────────────────────────────────────────

def call_encoder_zero_shot(class_list: list[str], session: Session) -> dict:
    from pipeline.tips_runner import run_zero_shot_inference

    t0 = time.monotonic()
    depth_tensor, cosine_maps, detected = run_zero_shot_inference(session.image, class_list)
    session.depth_tensor = depth_tensor
    # Store cosine maps on the session for use by measure_object
    session.zero_shot_maps = cosine_maps
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
