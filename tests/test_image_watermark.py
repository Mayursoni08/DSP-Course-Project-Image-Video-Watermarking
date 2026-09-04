from __future__ import annotations

import numpy as np
import pytest

from attacks.compression import jpeg_compress
from attacks.spatial import awgn, median_filter, rotate, scale
from core.config import ImageWatermarkConfig
from core.embedder import ImageWatermarker
from core.exceptions import CapacityError
from core.extractor import ImageWatermarkExtractor
from core.metrics import byte_bit_error_rate, psnr, ssim


def test_default_blind_roundtrip_and_imperceptibility(
    cover, payload, secret_key, image_config, embedded
) -> None:
    watermarked, report = embedded
    result = ImageWatermarkExtractor(image_config).extract_array(
        watermarked,
        secret_key,
        expected_payload_bytes=len(payload),
        expected_shape=cover.shape[:2],
    )
    assert result.valid
    assert result.payload == payload
    assert report.capacity_bytes >= len(payload)
    assert psnr(cover, watermarked) > 42.0
    assert ssim(cover, watermarked) > 0.98


@pytest.mark.parametrize(
    ("attack", "registered"),
    [
        (lambda image: jpeg_compress(image, 50), False),
        (lambda image: median_filter(image, 3), False),
        (lambda image: awgn(image, 10.0, seed=7), False),
        (lambda image: rotate(image, 5.0), True),
        (lambda image: scale(image, 0.75), True),
    ],
)
def test_required_attack_recovery(
    cover, payload, secret_key, image_config, embedded, attack, registered
) -> None:
    watermarked, _ = embedded
    attacked = attack(watermarked)
    result = ImageWatermarkExtractor(image_config).extract_array(
        attacked,
        secret_key,
        reference=cover if registered else None,
        expected_payload_bytes=len(payload),
        expected_shape=cover.shape[:2],
    )
    assert byte_bit_error_rate(payload, result.payload) <= 0.01
    assert result.valid


def test_wrong_key_is_rejected(cover, payload, embedded, image_config) -> None:
    watermarked, _ = embedded
    result = ImageWatermarkExtractor(image_config).extract_array(
        watermarked,
        "wrong-secret-key-2026",
        expected_payload_bytes=len(payload),
        expected_shape=cover.shape[:2],
    )
    assert not result.valid


def test_svd_qim_backend_exact_roundtrip(cover, secret_key) -> None:
    config = ImageWatermarkConfig(
        algorithm="svd-qim",
        dwt_level=2,
        block_size=4,
        delta=14.0,
        ecc_mode="repetition",
        header_repetition=3,
        payload_repetition=3,
        auth_tag_bytes=4,
    )
    payload = b"SVD"
    watermarked, _ = ImageWatermarker(config).embed_array(
        cover, payload, secret_key, collect_debug=False
    )
    result = ImageWatermarkExtractor(config).extract_array(
        watermarked, secret_key, expected_payload_bytes=len(payload)
    )
    assert result.valid
    assert result.payload == payload


def test_capacity_rejection(secret_key) -> None:
    tiny = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(CapacityError):
        ImageWatermarker().embed_array(tiny, b"payload", secret_key)


def test_grayscale_bgra_and_odd_shapes(cover, secret_key) -> None:
    import cv2

    payload = b"MODE"
    alpha = np.tile(np.arange(cover.shape[1], dtype=np.uint8), (cover.shape[0], 1))
    variants = [
        cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY),
        np.dstack((cover, alpha)),
        cover[:501, :507],
    ]
    for source in variants:
        watermarked, _ = ImageWatermarker().embed_array(
            source, payload, secret_key, collect_debug=False
        )
        assert watermarked.shape == source.shape
        result = ImageWatermarkExtractor().extract_array(
            watermarked,
            secret_key,
            expected_payload_bytes=len(payload),
            expected_shape=source.shape[:2],
        )
        assert result.valid and result.payload == payload
    np.testing.assert_array_equal(variants[1][..., 3], ImageWatermarker().embed_array(
        variants[1], payload, secret_key, collect_debug=False
    )[0][..., 3])
