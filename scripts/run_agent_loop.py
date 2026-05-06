#!/usr/bin/env python3
from __future__ import annotations

# Usage examples:
#   Basic run:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?"
#
#   Audio question from WAV file:
#     python scripts/run_agent_loop.py \
#       --audio data/media/audio/question.wav
#
#   Use a minimal system prompt override:
#      --quantization fp8 \
#
#   Restrict tools to call_dpt_head only:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --tool-mode dpt_only
#
#   Dump request/response JSONL for debugging:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --dump-requests /tmp/spatialsense_req.jsonl \
#       --dump-responses /tmp/spatialsense_resp.jsonl
#
#   Show newly generated box debug images:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --show-new-box-images
#
#   Print each tool input/output payload inline:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --print-tool-io

import argparse
import logging
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pipeline.intrinsics import nyuv2_intrinsics
from pipeline.session import Session, create_session
from server.agent import run_agent_loop

NYUV2_DIR = REPO_ROOT / "data" / "nyuv2_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one SpatialSense agent-loop turn against a local vLLM server."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image.",
    )
    parser.add_argument(
        "--nyuv2-sample",
        type=int,
        default=None,
        metavar="N",
        help="Load NYUv2 val sample N from data/nyuv2_samples/ (uses known Kinect intrinsics).",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="User question text to run through the agent loop.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to WAV audio question. If provided, audio is sent instead of text.",
    )
    parser.add_argument(
        "--tool-mode",
        choices=["full", "dpt_only"],
        default=None,
        help="Override SPATIALSENSE_TOOL_MODE for this run.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override SPATIALSENSE_SYSTEM_PROMPT for this run.",
    )
    parser.add_argument(
        "--dump-requests",
        default=None,
        help="Path to write JSONL request dumps (SPATIALSENSE_DUMP_REQUEST_PATH).",
    )
    parser.add_argument(
        "--dump-responses",
        default=None,
        help="Path to write JSONL response dumps (SPATIALSENSE_DUMP_RESPONSE_PATH).",
    )
    parser.add_argument(
        "--show-new-box-images",
        action="store_true",
        help="Print newly created files in data/media/debug_boxes after the run.",
    )
    parser.add_argument(
        "--print-tool-io",
        action="store_true",
        help="Print each tool input and output payload during the agent loop.",
    )
    parser.add_argument(
        "--print-tool-returns",
        action="store_true",
        help="Deprecated alias for --print-tool-io.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for latency and debug logs (default: INFO).",
    )
    return parser.parse_args()


def set_env_if(name: str, value: str | None) -> None:
    if value is not None:
        os.environ[name] = value


def _suppress_noisy_third_party_logs() -> None:
    # Keep SpatialSense logs visible while reducing noisy dependency output.
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "urllib3",
        "transformers",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message="xFormers is not available.*",
        category=UserWarning,
    )


def _compute_depth_metrics(session: Session, sample_idx: int) -> None:
    """Per-object GT comparison: TIPS distance vs NYUv2 median depth at the same segmented pixels."""
    if not session.measurements:
        print("DEPTH_METRICS=skipped (no measure_object calls)")
        return

    depth_path = NYUV2_DIR / "depth" / f"{sample_idx:04d}.npy"
    if not depth_path.exists():
        return

    gt = np.load(depth_path)[0].astype(np.float32)   # (H_gt, W_gt) meters
    H_gt, W_gt = gt.shape

    print("DEPTH_GT_COMPARISON:")
    for m in session.measurements:
        mask_dpt = m.get("mask_dpt")
        if mask_dpt is not None:
            # Resize segmentation mask from DPT space → GT space (nearest-neighbor preserves bool boundaries)
            mask_np = mask_dpt.detach().cpu().numpy().astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_np, mode="L")
            mask_gt = np.array(mask_pil.resize((W_gt, H_gt), Image.NEAREST)) > 127
            valid = gt[mask_gt & (gt > 0)]
        else:
            # Fallback: use full bounding box (less accurate — includes background)
            ymin, xmin, ymax, xmax = m["box_original"]
            ymin = max(0, min(ymin, H_gt - 1))
            xmin = max(0, min(xmin, W_gt - 1))
            ymax = max(ymin + 1, min(ymax, H_gt))
            xmax = max(xmin + 1, min(xmax, W_gt))
            valid = gt[ymin:ymax, xmin:xmax]
            valid = valid[valid > 0]

        if valid.size < 10:
            print(f"  {m['class_name']}: TIPS={m['tips_distance_m']:.2f}m  GT=n/a (too few valid pixels in mask)")
            continue
        gt_median = float(np.median(valid))
        err = m["tips_distance_m"] - gt_median
        print(f"  {m['class_name']}: TIPS={m['tips_distance_m']:.2f}m  GT={gt_median:.2f}m  err={err:+.2f}m")


def _save_nyuv2_gt_depth(sample_idx: int, debug_dir: Path) -> None:
    depth_path = NYUV2_DIR / "depth" / f"{sample_idx:04d}.npy"
    if not depth_path.exists():
        return
    d = np.load(depth_path)[0]
    valid = d[d > 0]
    vmin, vmax = np.percentile(valid, 5), np.percentile(valid, 95)
    masked = np.ma.masked_where(d == 0, d)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="grey")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"NYUv2 GT Depth — sample {sample_idx}")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (m)")
    debug_dir.mkdir(parents=True, exist_ok=True)
    out = debug_dir / f"nyuv2_{sample_idx:04d}_gt_depth.png"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"GT_DEPTH_SAVED={out}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _suppress_noisy_third_party_logs()

    set_env_if("SPATIALSENSE_TOOL_MODE", args.tool_mode)
    set_env_if("SPATIALSENSE_SYSTEM_PROMPT", args.system_prompt)
    set_env_if("SPATIALSENSE_DUMP_REQUEST_PATH", args.dump_requests)
    set_env_if("SPATIALSENSE_DUMP_RESPONSE_PATH", args.dump_responses)
    if args.print_tool_io or args.print_tool_returns:
        os.environ["SPATIALSENSE_PRINT_TOOL_IO"] = "1"

    # Resolve image path and intrinsics
    nyuv2_sample: int | None = args.nyuv2_sample
    intrinsics = None
    if nyuv2_sample is not None:
        image_path = NYUV2_DIR / "images" / f"{nyuv2_sample:04d}.png"
        if not image_path.exists():
            print(f"ERROR: NYUv2 sample {nyuv2_sample} not found at {image_path}")
            print("Run fetch_nyuv2_samples.py first.")
            return 1
    else:
        image_path = Path(args.image or "data/media/images/office_back.jpeg")
        if not image_path.exists():
            print(f"ERROR: image not found: {image_path}")
            return 1

    audio_bytes: bytes | None = None
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"ERROR: audio not found: {audio_path}")
            return 1
        audio_bytes = audio_path.read_bytes()

    if audio_bytes is None and not args.question:
        print("ERROR: provide either --question or --audio")
        return 1

    debug_dir = REPO_ROOT / "data" / "media" / "debug_boxes"
    before = set(p.name for p in debug_dir.glob("*.png")) if debug_dir.exists() else set()

    image = Image.open(image_path).convert("RGB")
    if nyuv2_sample is not None:
        intrinsics = nyuv2_intrinsics(image.width, image.height)
        _save_nyuv2_gt_depth(nyuv2_sample, debug_dir)

    question: str | bytes = audio_bytes if audio_bytes is not None else args.question
    session = create_session(image, question, intrinsics=intrinsics)

    try:
        response_text = run_agent_loop(session)
        if nyuv2_sample is not None:
            _compute_depth_metrics(session, nyuv2_sample)
    finally:
        session.release()

    print("AGENT_RESPONSE_START")
    print(response_text)
    print("AGENT_RESPONSE_END")

    if args.show_new_box_images:
        after = set(p.name for p in debug_dir.glob("*.png")) if debug_dir.exists() else set()
        new_files = sorted(after - before)
        print(f"NEW_DEBUG_BOX_FILES={len(new_files)}")
        for name in new_files:
            print(f"NEW_DEBUG_BOX_FILE={debug_dir / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
