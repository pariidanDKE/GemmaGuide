#!/usr/bin/env python3
"""
Save a GT depth map visualization for a NYUv2 sample, matching the TIPS depth map style.

Usage:
  python scripts/save_nyuv2_depth_vis.py 5
  python scripts/save_nyuv2_depth_vis.py 5 10 23   # multiple samples
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

indices = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [0]

for idx in indices:
    path = Path(f"data/nyuv2_samples/depth/{idx:04d}.npy")
    if not path.exists():
        print(f"SKIP {idx}: {path} not found")
        continue

    d = np.load(path)[0]        # (H, W) float64, meters
    valid = d[d > 0]
    vmin, vmax = np.percentile(valid, 5), np.percentile(valid, 95)

    # Mask zero/invalid pixels so they render as grey, not as "close"
    masked = np.ma.masked_where(d == 0, d)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="grey")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Depth Map (NYUv2 GT)")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (m)")

    out = Path(f"data/nyuv2_samples/depth_vis/{idx:04d}_gt.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"saved {out}  (depth {vmin:.2f}–{vmax:.2f} m)")
