"""Imperceptibility and robustness metrics for image/video watermarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def _as_float_pair(
    reference: NDArray[np.generic], candidate: NDArray[np.generic]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    return a, b


def mse(reference: NDArray[np.generic], candidate: NDArray[np.generic]) -> float:
    """Mean squared error over all pixels and channels."""

    a, b = _as_float_pair(reference, candidate)
    return float(np.mean(np.square(a - b), dtype=np.float64))


def psnr(
    reference: NDArray[np.generic],
    candidate: NDArray[np.generic],
    data_range: float = 255.0,
) -> float:
    """Peak signal-to-noise ratio in decibels."""

    error = mse(reference, candidate)
    if error <= np.finfo(np.float64).eps:
        return float("inf")
    return float(10.0 * np.log10((data_range * data_range) / error))


def _ssim_channel(
    reference: NDArray[np.float64], candidate: NDArray[np.float64], data_range: float
) -> float:
    kernel = (11, 11)
    sigma = 1.5
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = cv2.GaussianBlur(reference, kernel, sigma)
    mu_b = cv2.GaussianBlur(candidate, kernel, sigma)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(reference * reference, kernel, sigma) - mu_a2
    sigma_b2 = cv2.GaussianBlur(candidate * candidate, kernel, sigma) - mu_b2
    sigma_ab = cv2.GaussianBlur(reference * candidate, kernel, sigma) - mu_ab
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    score_map = numerator / np.maximum(denominator, np.finfo(np.float64).eps)
    border = kernel[0] // 2
    if min(score_map.shape) > 2 * border:
        score_map = score_map[border:-border, border:-border]
    return float(np.mean(score_map, dtype=np.float64))


def ssim(
    reference: NDArray[np.generic],
    candidate: NDArray[np.generic],
    data_range: float = 255.0,
) -> float:
    """Windowed structural similarity (SSIM), averaged over channels."""

    a, b = _as_float_pair(reference, candidate)
    if a.ndim == 2:
        return _ssim_channel(a, b, data_range)
    if a.ndim != 3:
        raise ValueError("SSIM expects a 2-D grayscale or 3-D color image")
    return float(np.mean([_ssim_channel(a[..., i], b[..., i], data_range) for i in range(a.shape[2])]))


def bit_error_rate(
    expected: NDArray[np.generic], extracted: NDArray[np.generic]
) -> float:
    """Fraction of unequal bits; missing extracted bits count as errors."""

    a = np.asarray(expected, dtype=np.uint8).reshape(-1)
    b = np.asarray(extracted, dtype=np.uint8).reshape(-1)
    if a.size == 0:
        return 0.0 if b.size == 0 else 1.0
    overlap = min(a.size, b.size)
    errors = int(np.count_nonzero(a[:overlap] != b[:overlap])) + abs(a.size - b.size)
    return float(errors / max(a.size, b.size))


def byte_bit_error_rate(expected: bytes, extracted: bytes) -> float:
    """BER between byte strings, counting missing bytes as eight bit errors."""

    a = np.unpackbits(np.frombuffer(expected, dtype=np.uint8)) if expected else np.empty(0, dtype=np.uint8)
    b = np.unpackbits(np.frombuffer(extracted, dtype=np.uint8)) if extracted else np.empty(0, dtype=np.uint8)
    return bit_error_rate(a, b)


def normalized_correlation(
    expected: NDArray[np.generic], extracted: NDArray[np.generic]
) -> float:
    """Normalized correlation of two zero-mean numeric watermark arrays."""

    a, b = _as_float_pair(expected, extracted)
    a = a.reshape(-1) - float(np.mean(a))
    b = b.reshape(-1) - float(np.mean(b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


@dataclass(frozen=True, slots=True)
class ImageQuality:
    psnr_db: float
    ssim: float
    mse: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def image_quality(
    reference: NDArray[np.generic], candidate: NDArray[np.generic]
) -> ImageQuality:
    """Compute the standard imperceptibility metric bundle."""

    return ImageQuality(
        psnr_db=psnr(reference, candidate),
        ssim=ssim(reference, candidate),
        mse=mse(reference, candidate),
    )
