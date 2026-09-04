"""Image I/O and luminance-channel conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .exceptions import MediaIOError


@dataclass(slots=True)
class LuminanceContext:
    mode: str
    color: NDArray[np.float64] | None
    alpha: NDArray[np.uint8] | None


def split_luminance(image: NDArray[np.generic]) -> tuple[NDArray[np.float64], LuminanceContext]:
    """Extract luminance from grayscale, BGR, or BGRA image arrays."""

    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64), LuminanceContext("gray", None, None)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError("image must be grayscale, BGR, or BGRA")
    u8 = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    alpha: NDArray[np.uint8] | None = None
    bgr = u8
    mode = "bgr"
    if u8.shape[2] == 4:
        alpha = u8[..., 3].copy()
        bgr = u8[..., :3]
        mode = "bgra"
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    return ycrcb[..., 0].copy(), LuminanceContext(mode, ycrcb, alpha)


def merge_luminance(
    luminance: NDArray[np.generic], context: LuminanceContext
) -> NDArray[np.uint8]:
    """Reinsert a processed luminance plane and reconstruct the original mode."""

    y = np.clip(np.rint(luminance), 0, 255).astype(np.uint8)
    if context.mode == "gray":
        return y
    if context.color is None:
        raise ValueError("color context is missing")
    ycrcb = np.clip(np.rint(context.color), 0, 255).astype(np.uint8)
    ycrcb[..., 0] = y
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    if context.mode == "bgra":
        if context.alpha is None:
            raise ValueError("BGRA context is missing alpha")
        return np.dstack((bgr, context.alpha))
    return bgr


def read_image(path: str | Path) -> NDArray[np.uint8]:
    """Read an image with Unicode-path support."""

    source = Path(path)
    try:
        data = np.fromfile(source, dtype=np.uint8)
    except OSError as error:
        raise MediaIOError(f"unable to read image: {source}") from error
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise MediaIOError(f"OpenCV could not decode image: {source}")
    return image


def write_image(path: str | Path, image: NDArray[np.generic]) -> None:
    """Write an image with Unicode-path support and high-quality JPEG defaults."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower() or ".png"
    parameters: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 100]
    success, encoded = cv2.imencode(suffix, np.asarray(image), parameters)
    if not success:
        raise MediaIOError(f"OpenCV could not encode image as {suffix}")
    try:
        encoded.tofile(destination)
    except OSError as error:
        raise MediaIOError(f"unable to write image: {destination}") from error
