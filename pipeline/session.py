from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

from pipeline.intrinsics import CameraIntrinsics, extract_intrinsics


@dataclass
class Session:
    session_id: str
    image: Image.Image
    question: str | bytes
    intrinsics: CameraIntrinsics
    depth_tensor: Optional[torch.Tensor] = field(default=None)
    seg_mask: Optional[torch.Tensor] = field(default=None)
    depth_colormap: Optional[Image.Image] = field(default=None)
    spatial_report: Optional[str] = field(default=None)

    measurements: list[dict] = field(default_factory=list)

    def release(self) -> None:
        self.depth_tensor = None
        self.seg_mask = None


def create_session(
    image: Image.Image,
    question: str | bytes,
    intrinsics: CameraIntrinsics | None = None,
) -> Session:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return Session(
        session_id=str(uuid.uuid4()),
        image=image,
        question=question,
        intrinsics=intrinsics if intrinsics is not None else extract_intrinsics(image),
    )
