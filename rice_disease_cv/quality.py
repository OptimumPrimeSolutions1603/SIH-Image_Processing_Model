from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .config import QualityConfig


@dataclass
class QualityResult:
    accepted: bool
    issues: list[str]
    metrics: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def assess_image(image_bgr: np.ndarray, config: QualityConfig | None = None) -> QualityResult:
    """Reject images that are too small, blurred, or badly exposed."""
    config = config or QualityConfig()
    if image_bgr is None or image_bgr.size == 0:
        return QualityResult(False, ["unreadable_image"], {})

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    dark_fraction = float(np.mean(gray <= 5))
    bright_fraction = float(np.mean(gray >= 250))

    issues: list[str] = []
    if width < config.min_width or height < config.min_height:
        issues.append("resolution_too_low")
    if blur_score < config.blur_threshold:
        issues.append("image_blurred")
    if brightness < config.dark_mean_threshold:
        issues.append("image_too_dark")
    if brightness > config.bright_mean_threshold:
        issues.append("image_too_bright")
    if dark_fraction > config.clipped_fraction_threshold:
        issues.append("excessive_black_clipping")
    if bright_fraction > config.clipped_fraction_threshold:
        issues.append("excessive_white_clipping")

    return QualityResult(
        accepted=not issues,
        issues=issues,
        metrics={
            "width": float(width),
            "height": float(height),
            "blur_score": blur_score,
            "mean_brightness": brightness,
            "dark_fraction": dark_fraction,
            "bright_fraction": bright_fraction,
        },
    )
