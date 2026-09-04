from __future__ import annotations

import numpy as np
import pytest

from attacks.compression import jpeg_compress
from attacks.spatial import (
    awgn,
    crop_and_resize,
    gaussian_blur,
    median_filter,
    rotate,
    salt_and_pepper,
    scale,
)


@pytest.mark.parametrize(
    "attack",
    [
        lambda image: awgn(image, 5.0, seed=1),
        lambda image: gaussian_blur(image, 3, 0.8),
        lambda image: median_filter(image, 3),
        lambda image: rotate(image, 3.0),
        lambda image: scale(image, 0.8),
        lambda image: crop_and_resize(image, 0.9),
        lambda image: salt_and_pepper(image, 0.01, seed=1),
        lambda image: jpeg_compress(image, 70),
    ],
)
def test_image_attacks_preserve_shape_and_dtype(cover, attack) -> None:
    result = attack(cover)
    assert result.shape == cover.shape
    assert result.dtype == np.uint8
