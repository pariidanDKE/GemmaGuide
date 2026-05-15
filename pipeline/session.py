from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from PIL import Image, ImageOps

from pipeline.intrinsics import CameraIntrinsics, extract_intrinsics


def empty_metrics() -> dict[str, Any]:
    return {
        "timings": [],
        "summary": {},
        "counts": {},
    }


def add_metric_sample(metrics: dict[str, Any], stage: str, seconds: float, *, include_timing: bool = True) -> None:
    seconds_rounded = round(float(seconds), 3)
    if include_timing:
        metrics.setdefault("timings", []).append({"stage": stage, "seconds": seconds_rounded})
    summary = metrics.setdefault("summary", {})
    counts = metrics.setdefault("counts", {})
    summary[stage] = round(float(summary.get(stage, 0.0)) + seconds_rounded, 3)
    counts[stage] = int(counts.get(stage, 0) + 1)


def merge_metrics(existing: dict[str, Any] | None, turn_metrics: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "turns": list((existing or {}).get("turns", [])),
        "summary": dict((existing or {}).get("summary", {})),
        "counts": dict((existing or {}).get("counts", {})),
    }
    merged["turns"].append({
        "summary": dict(turn_metrics.get("summary", {})),
        "counts": dict(turn_metrics.get("counts", {})),
    })
    for stage, seconds in turn_metrics.get("summary", {}).items():
        merged["summary"][stage] = round(float(merged["summary"].get(stage, 0.0)) + float(seconds), 3)
    for stage, count in turn_metrics.get("counts", {}).items():
        merged["counts"][stage] = int(merged["counts"].get(stage, 0) + int(count))
    return merged


def summarize_metrics(summary: dict[str, float], counts: dict[str, int], limit: int = 10) -> list[dict[str, int | float | str]]:
    ranked = sorted(summary.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "stage": stage,
            "seconds": round(seconds, 3),
            "count": int(counts.get(stage, 0)),
        }
        for stage, seconds in ranked[:limit]
    ]


@dataclass
class Session:
    session_id: str
    image: Optional[Image.Image]
    question: str | bytes
    intrinsics: Optional[CameraIntrinsics]
    depth_tensor: Optional[torch.Tensor] = field(default=None)
    seg_mask: Optional[torch.Tensor] = field(default=None)
    depth_colormap: Optional[Image.Image] = field(default=None)
    zero_shot_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    spatial_report: Optional[str] = field(default=None)

    measurements: list[dict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=empty_metrics)

    def add_timing(self, stage: str, seconds: float, **meta: Any) -> None:
        add_metric_sample(self.metrics, stage, seconds)
        if meta:
            self.metrics["timings"][-1].update(meta)

    def export_metrics(self) -> dict[str, Any]:
        return {
            "timings": list(self.metrics["timings"]),
            "summary": dict(self.metrics["summary"]),
            "counts": dict(self.metrics["counts"]),
        }

    def release(self) -> None:
        self.depth_tensor = None
        self.seg_mask = None
        self.zero_shot_maps.clear()


def create_session(
    image: Optional[Image.Image],
    question: str | bytes,
    intrinsics: CameraIntrinsics | None = None,
    metrics: dict[str, Any] | None = None,
) -> Session:
    if image is not None:
        # Extract focal length from EXIF before any conversion strips it
        if intrinsics is None:
            intrinsics = extract_intrinsics(image)
        # Apply EXIF orientation (iPhone stores raw sensor orientation in tag)
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
    return Session(
        session_id=str(uuid.uuid4()),
        image=image,
        question=question,
        intrinsics=intrinsics,
        metrics=metrics if metrics is not None else empty_metrics(),
    )
