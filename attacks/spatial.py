"""Deterministic spatial-domain image attacks for robustness evaluation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def _u8(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def awgn(
    image: NDArray[np.generic], sigma: float = 10.0, *, seed: int = 0
) -> NDArray[np.uint8]:
    """Add zero-mean white Gaussian noise with standard deviation ``sigma``."""

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    source = _u8(image).astype(np.float64)
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=source.shape)
    return np.clip(np.rint(source + noise), 0, 255).astype(np.uint8)


def gaussian_blur(
    image: NDArray[np.generic], kernel_size: int = 5, sigma: float = 1.2
) -> NDArray[np.uint8]:
    """Apply Gaussian low-pass filtering."""

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return cv2.GaussianBlur(_u8(image), (kernel_size, kernel_size), sigma)


def median_filter(
    image: NDArray[np.generic], kernel_size: int = 3
) -> NDArray[np.uint8]:
    """Apply nonlinear median filtering."""

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    return cv2.medianBlur(_u8(image), kernel_size)


def rotate(
    image: NDArray[np.generic], angle_degrees: float = 5.0
) -> NDArray[np.uint8]:
    """Rotate around image center while preserving canvas size."""

    source = _u8(image)
    height, width = source.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, 1.0)
    return cv2.warpAffine(
        source,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def scale(
    image: NDArray[np.generic], factor: float = 0.75, *, restore_size: bool = True
) -> NDArray[np.uint8]:
    """Rescale an image and optionally restore its original dimensions."""

    if factor <= 0:
        raise ValueError("factor must be positive")
    source = _u8(image)
    height, width = source.shape[:2]
    resized = cv2.resize(
        source,
        (max(1, round(width * factor)), max(1, round(height * factor))),
        interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC,
    )
    if not restore_size:
        return resized
    return cv2.resize(resized, (width, height), interpolation=cv2.INTER_CUBIC)


def crop_and_resize(
    image: NDArray[np.generic], keep_fraction: float = 0.85
) -> NDArray[np.uint8]:
    """Center-crop to ``keep_fraction`` area per axis and resize to original."""

    if not 0.1 <= keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in [0.1, 1.0]")
    source = _u8(image)
    height, width = source.shape[:2]
    crop_h = max(1, round(height * keep_fraction))
    crop_w = max(1, round(width * keep_fraction))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = source[top : top + crop_h, left : left + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)


def salt_and_pepper(
    image: NDArray[np.generic], amount: float = 0.01, *, seed: int = 0
) -> NDArray[np.uint8]:
    """Replace an ``amount`` fraction of pixels with black/white impulses."""

    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be in [0, 1]")
    output = _u8(image).copy()
    height, width = output.shape[:2]
    count = int(round(height * width * amount))
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, height, size=count)
    xs = rng.integers(0, width, size=count)
    values = rng.integers(0, 2, size=count, dtype=np.uint8) * 255
    output[ys, xs] = values[:, None] if output.ndim == 3 else values
    return output
