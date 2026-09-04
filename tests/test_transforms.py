from __future__ import annotations

import numpy as np

from core.transforms import (
    batched_svd,
    blocks_view,
    crop_padding,
    dct2_blocks,
    dwt2,
    idct2_blocks,
    idwt2,
    merge_blocks,
    multilevel_dwt2,
    multilevel_idwt2,
    pad_to_multiple,
    reconstruct_svd,
    temporal_dwt,
    temporal_idwt,
)


def test_padding_roundtrip() -> None:
    source = np.arange(35, dtype=np.float64).reshape(5, 7)
    padded, metadata = pad_to_multiple(source, 8)
    assert padded.shape == (8, 8)
    np.testing.assert_array_equal(crop_padding(padded, metadata), source)


def test_dwt_and_multilevel_roundtrip() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(64, 96))
    coefficients = dwt2(source)
    np.testing.assert_allclose(idwt2(coefficients), source, rtol=0, atol=1e-11)
    approximation, details = multilevel_dwt2(source, 3)
    np.testing.assert_allclose(
        multilevel_idwt2(approximation, details), source, rtol=0, atol=1e-10
    )


def test_block_dct_and_svd_roundtrip() -> None:
    rng = np.random.default_rng(9)
    source = rng.normal(size=(32, 48))
    blocks = blocks_view(source, 8)
    coefficients = dct2_blocks(blocks)
    np.testing.assert_allclose(idct2_blocks(coefficients), blocks, atol=1e-11)
    u, singular, vh = batched_svd(coefficients)
    np.testing.assert_allclose(reconstruct_svd(u, singular, vh), coefficients, atol=1e-11)
    np.testing.assert_allclose(merge_blocks(blocks), source, atol=0)


def test_temporal_haar_roundtrip() -> None:
    rng = np.random.default_rng(11)
    frames = rng.normal(size=(8, 32, 40))
    low, high = temporal_dwt(frames)
    assert low.shape == high.shape == (4, 32, 40)
    np.testing.assert_allclose(temporal_idwt(low, high), frames, atol=1e-11)
