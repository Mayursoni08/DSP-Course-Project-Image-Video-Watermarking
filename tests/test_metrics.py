from __future__ import annotations

import numpy as np

from core.metrics import bit_error_rate, image_quality, normalized_correlation, psnr, ssim


def test_identical_image_metrics() -> None:
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.isinf(psnr(image, image))
    assert ssim(image, image) == 1.0
    assert image_quality(image, image).mse == 0.0


def test_robustness_metrics() -> None:
    assert bit_error_rate([0, 1, 1, 0], [0, 0, 1, 0]) == 0.25
    assert normalized_correlation(np.arange(10), np.arange(10)) == 1.0
    assert normalized_correlation(np.arange(10), -np.arange(10)) == -1.0
