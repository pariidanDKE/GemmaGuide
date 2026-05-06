#!/usr/bin/env python3
"""
Fetch and inspect the first N samples from tanganke/nyuv2 val split.

Saves to data/nyuv2_samples/:
  images/   — RGB PNGs
  depth/    — raw depth as .npy + a false-color PNG for visual inspection
  manifest.json — per-sample stats (depth range, shape, dtypes, scene label)

Run:
  pip install datasets pillow numpy matplotlib
  python scripts/fetch_nyuv2_samples.py
  python scripts/fetch_nyuv2_samples.py --n 10 --out data/nyuv2_samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="Number of samples to fetch (default: 100)")
    p.add_argument("--out", default="data/nyuv2_samples", help="Output directory")
    p.add_argument("--split", default="val", help="Dataset split (default: val)")
    return p.parse_args()


def to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert any float/int array to HWC uint8 for saving as PNG."""
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW → HWC
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]  # single-channel → 2D
    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    return arr


def field_summary(val) -> dict:
    """Return dtype/shape/range info for any field value."""
    if isinstance(val, Image.Image):
        arr = np.array(val)
        return {"type": "PIL.Image", "mode": val.mode, "size": val.size, "array_shape": arr.shape, "dtype": str(arr.dtype)}
    if isinstance(val, np.ndarray):
        info = {"type": "np.ndarray", "shape": list(val.shape), "dtype": str(val.dtype)}
        if np.issubdtype(val.dtype, np.floating) or np.issubdtype(val.dtype, np.integer):
            info["min"] = float(val.min())
            info["max"] = float(val.max())
        return info
    return {"type": type(val).__name__, "value": str(val)[:120]}


def depth_to_colormap(depth: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm
    if depth.ndim == 3:
        depth = depth[0] if depth.shape[0] == 1 else depth[:, :, 0]
    lo, hi = depth.min(), depth.max()
    norm = (depth - lo) / (hi - lo + 1e-8)
    colored = (cm.plasma(norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)


def coerce_to_numpy(val) -> np.ndarray:
    if isinstance(val, Image.Image):
        return np.array(val)
    if isinstance(val, np.ndarray):
        return val
    # datasets may return lists for some fields
    return np.array(val)


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        raise SystemExit(1)

    out = Path(args.out)
    img_dir = out / "images"
    depth_dir = out / "depth"
    depth_vis_dir = out / "depth_vis"
    for d in (img_dir, depth_dir, depth_vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading tanganke/nyuv2 split={args.split} ...")
    ds = load_dataset("tanganke/nyuv2", split=args.split)
    total = len(ds)
    n = min(args.n, total)
    print(f"Dataset size: {total}  — fetching first {n} samples")

    # Print schema from first sample before saving anything
    first = ds[0]
    print("\n--- Field schema (sample 0) ---")
    for k, v in first.items():
        v_np = coerce_to_numpy(v) if not isinstance(v, (str, int, float, bool)) else v
        if isinstance(v_np, np.ndarray):
            summary = field_summary(v_np)
        elif isinstance(v, Image.Image):
            summary = field_summary(v)
        else:
            summary = {"type": type(v).__name__, "value": str(v)[:120]}
        print(f"  {k}: {summary}")
    print("---\n")

    manifest: list[dict] = []

    for i in range(n):
        sample = ds[i]

        # --- image ---
        img_val = sample["image"]
        img_arr = coerce_to_numpy(img_val)
        img_uint8 = to_hwc_uint8(img_arr.copy())
        if img_uint8.ndim == 2:
            pil_img = Image.fromarray(img_uint8, mode="L")
        elif img_uint8.shape[2] == 3:
            pil_img = Image.fromarray(img_uint8, mode="RGB")
        else:
            pil_img = Image.fromarray(img_uint8)
        pil_img.save(img_dir / f"{i:04d}.png")

        # --- depth ---
        depth_arr = coerce_to_numpy(sample["depth"])
        np.save(depth_dir / f"{i:04d}.npy", depth_arr)
        depth_to_colormap(depth_arr).save(depth_vis_dir / f"{i:04d}.png")

        # --- per-sample manifest entry ---
        entry: dict = {"index": i}
        for k, v in sample.items():
            if k in ("image", "depth"):
                arr = coerce_to_numpy(v)
                entry[k] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                }
            elif isinstance(v, (str, int, float, bool)):
                entry[k] = v
            else:
                arr = coerce_to_numpy(v)
                entry[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

        manifest.append(entry)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  saved {i + 1}/{n}")

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Output: {out}/")
    print(f"  {img_dir.name}/      — {n} RGB PNGs")
    print(f"  {depth_dir.name}/    — {n} .npy depth arrays")
    print(f"  {depth_vis_dir.name}/ — {n} false-color depth PNGs")
    print(f"  manifest.json — per-sample stats")

    # Summary stats across all samples
    depths = [m["depth"] for m in manifest if "depth" in m]
    if depths:
        all_mins = [d["min"] for d in depths]
        all_maxs = [d["max"] for d in depths]
        print(f"\nDepth stats across {len(depths)} samples:")
        print(f"  global min: {min(all_mins):.4f}")
        print(f"  global max: {max(all_maxs):.4f}")
        print(f"  mean of per-sample means: {sum(d['mean'] for d in depths)/len(depths):.4f}")


if __name__ == "__main__":
    main()
