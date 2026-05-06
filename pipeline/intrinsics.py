from dataclasses import dataclass
from math import atan, degrees, radians, tan
from PIL import Image


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    source: str  # "exif" or "assumed"


def nyuv2_intrinsics(width: int, height: int) -> CameraIntrinsics:
    # Kinect calibration at 640x480, scaled proportionally to any resolution
    sx, sy = width / 640.0, height / 480.0
    return CameraIntrinsics(
        fx=518.8579 * sx, fy=519.4696 * sy,
        cx=325.5824 * sx, cy=253.7362 * sy,
        source="nyuv2",
    )


def extract_intrinsics(image: Image.Image) -> CameraIntrinsics:
    w, h = image.size
    try:
        exif = image._getexif()
        if exif is None:
            raise ValueError("No EXIF")
        # Tag 37386 = FocalLength, Tag 41989 = FocalLengthIn35mmFilm
        focal_mm = exif.get(37386)
        focal_35mm = exif.get(41989)
        if focal_mm is None or focal_35mm is None:
            raise ValueError("Missing focal length tags")
        # IFDRational → float
        focal_mm = float(focal_mm)
        focal_35mm = float(focal_35mm)
        if focal_mm == 0 or focal_35mm == 0:
            raise ValueError("Zero focal length")
        crop_factor = focal_35mm / focal_mm
        sensor_width = 36.0 / crop_factor
        fx = (focal_mm / sensor_width) * w
        fy = (focal_mm / sensor_width) * h
        return CameraIntrinsics(fx=fx, fy=fy, cx=w / 2, cy=h / 2, source="exif")
    except Exception:
        # 65° horizontal FOV fallback
        fx = w / (2 * tan(radians(32.5)))
        return CameraIntrinsics(fx=fx, fy=fx, cx=w / 2, cy=h / 2, source="assumed")
