"""Vectorized transform wrappers used by the watermarking core.

PyWavelets is used for production DWT operations. A numerically exact Haar
fallback is included so the package can still run in constrained environments
where optional binary wheels are unavailable. SciPy's orthonormal DCT-II and
IDCT-II are used block-wise, and NumPy supplies batched SVD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn, idctn

try:  # pragma: no cover - branch depends on deployment environment
    import pywt  # type: ignore
except ImportError:  # pragma: no cover
    pywt = None

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Padding:
    """Padding metadata required to crop an inverse-transform result."""

    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    pad_bottom: int
    pad_right: int


def pad_to_multiple(
    image: NDArray[np.generic], multiple: int, mode: str = "reflect"
) -> tuple[FloatArray, Padding]:
    """Pad a 2-D plane so each dimension is divisible by ``multiple``."""

    if image.ndim != 2:
        raise ValueError("pad_to_multiple expects a 2-D array")
    if multiple < 1:
        raise ValueError("multiple must be positive")
    height, width = image.shape
    pad_bottom = (-height) % multiple
    pad_right = (-width) % multiple
    pad_mode = mode
    if (height == 1 and pad_bottom) or (width == 1 and pad_right):
        pad_mode = "edge"
    padded = np.pad(
        np.asarray(image, dtype=np.float64),
        ((0, pad_bottom), (0, pad_right)),
        mode=pad_mode,
    )
    return padded, Padding(
        original_shape=(height, width),
        padded_shape=tuple(padded.shape),
        pad_bottom=pad_bottom,
        pad_right=pad_right,
    )


def crop_padding(image: NDArray[np.generic], padding: Padding) -> FloatArray:
    """Remove padding introduced by :func:`pad_to_multiple`."""

    height, width = padding.original_shape
    return np.asarray(image[:height, :width], dtype=np.float64)


def _haar_dwt2(data: FloatArray) -> tuple[FloatArray, tuple[FloatArray, FloatArray, FloatArray]]:
    if data.shape[0] % 2 or data.shape[1] % 2:
        raise ValueError("Haar fallback requires even spatial dimensions")
    root2 = np.sqrt(2.0)
    low_x = (data[:, 0::2] + data[:, 1::2]) / root2
    high_x = (data[:, 0::2] - data[:, 1::2]) / root2
    c_a = (low_x[0::2, :] + low_x[1::2, :]) / root2
    c_h = (low_x[0::2, :] - low_x[1::2, :]) / root2
    c_v = (high_x[0::2, :] + high_x[1::2, :]) / root2
    c_d = (high_x[0::2, :] - high_x[1::2, :]) / root2
    return c_a, (c_h, c_v, c_d)


def _haar_idwt2(
    coeffs: tuple[FloatArray, tuple[FloatArray, FloatArray, FloatArray]]
) -> FloatArray:
    c_a, (c_h, c_v, c_d) = coeffs
    root2 = np.sqrt(2.0)
    low_x = np.empty((c_a.shape[0] * 2, c_a.shape[1]), dtype=np.float64)
    high_x = np.empty_like(low_x)
    low_x[0::2, :] = (c_a + c_h) / root2
    low_x[1::2, :] = (c_a - c_h) / root2
    high_x[0::2, :] = (c_v + c_d) / root2
    high_x[1::2, :] = (c_v - c_d) / root2
    output = np.empty((low_x.shape[0], low_x.shape[1] * 2), dtype=np.float64)
    output[:, 0::2] = (low_x + high_x) / root2
    output[:, 1::2] = (low_x - high_x) / root2
    return output


def dwt2(
    data: NDArray[np.generic], wavelet: str = "haar", mode: str = "periodization"
) -> tuple[FloatArray, tuple[FloatArray, FloatArray, FloatArray]]:
    """Compute one-level 2-D DWT as ``(LL, (LH, HL, HH))``.

    PyWavelets calls the detail arrays ``cH``, ``cV``, and ``cD``. This project
    maps them to the conventional ``LH``, ``HL``, and ``HH`` labels.
    """

    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("dwt2 expects a 2-D array")
    if pywt is not None:
        c_a, (c_h, c_v, c_d) = pywt.dwt2(array, wavelet, mode=mode)
        return (
            np.asarray(c_a, dtype=np.float64),
            (
                np.asarray(c_h, dtype=np.float64),
                np.asarray(c_v, dtype=np.float64),
                np.asarray(c_d, dtype=np.float64),
            ),
        )
    if wavelet not in {"haar", "db1"}:
        raise ImportError(
            "PyWavelets is required for non-Haar wavelets. Install PyWavelets."
        )
    return _haar_dwt2(array)


def idwt2(
    coeffs: tuple[FloatArray, tuple[FloatArray, FloatArray, FloatArray]],
    wavelet: str = "haar",
    mode: str = "periodization",
) -> FloatArray:
    """Reconstruct a 2-D signal from one-level DWT coefficients."""

    if pywt is not None:
        return np.asarray(pywt.idwt2(coeffs, wavelet, mode=mode), dtype=np.float64)
    if wavelet not in {"haar", "db1"}:
        raise ImportError(
            "PyWavelets is required for non-Haar wavelets. Install PyWavelets."
        )
    return _haar_idwt2(coeffs)


def blocks_view(data: NDArray[np.generic], block_size: int) -> FloatArray:
    """Return a copy arranged as ``(block_row, block_col, y, x)``."""

    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("blocks_view expects a 2-D array")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    height, width = array.shape
    if height % block_size or width % block_size:
        raise ValueError("array dimensions must be divisible by block_size")
    return (
        array.reshape(height // block_size, block_size, width // block_size, block_size)
        .transpose(0, 2, 1, 3)
        .copy()
    )


def merge_blocks(blocks: NDArray[np.generic]) -> FloatArray:
    """Inverse of :func:`blocks_view`."""

    array = np.asarray(blocks, dtype=np.float64)
    if array.ndim != 4 or array.shape[2] != array.shape[3]:
        raise ValueError("blocks must have shape (rows, cols, size, size)")
    rows, cols, size, _ = array.shape
    return array.transpose(0, 2, 1, 3).reshape(rows * size, cols * size)


def dct2_blocks(blocks: NDArray[np.generic]) -> FloatArray:
    """Apply orthonormal DCT-II over the last two axes of a block stack."""

    return np.asarray(
        dctn(np.asarray(blocks, dtype=np.float64), axes=(-2, -1), type=2, norm="ortho"),
        dtype=np.float64,
    )


def idct2_blocks(coefficients: NDArray[np.generic]) -> FloatArray:
    """Apply the exact orthonormal inverse of :func:`dct2_blocks`."""

    return np.asarray(
        idctn(
            np.asarray(coefficients, dtype=np.float64),
            axes=(-2, -1),
            type=2,
            norm="ortho",
        ),
        dtype=np.float64,
    )


def batched_svd(
    matrices: NDArray[np.generic],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute reduced SVD for every matrix in a stack."""

    u, singular, vh = np.linalg.svd(
        np.asarray(matrices, dtype=np.float64), full_matrices=False
    )
    return (
        np.asarray(u, dtype=np.float64),
        np.asarray(singular, dtype=np.float64),
        np.asarray(vh, dtype=np.float64),
    )


def reconstruct_svd(u: FloatArray, singular: FloatArray, vh: FloatArray) -> FloatArray:
    """Reconstruct a matrix stack without explicitly forming diagonal arrays."""

    return np.matmul(u * singular[..., np.newaxis, :], vh)


def temporal_dwt(
    frames: NDArray[np.generic], wavelet: str = "haar", mode: str = "periodization"
) -> tuple[FloatArray, FloatArray]:
    """One-level DWT along the temporal axis of ``(time, height, width)`` data."""

    array = np.asarray(frames, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] % 2:
        raise ValueError("temporal_dwt expects an even-length (T,H,W) array")
    if pywt is not None:
        low, high = pywt.dwt(array, wavelet, mode=mode, axis=0)
        return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    if wavelet not in {"haar", "db1"}:
        raise ImportError("PyWavelets is required for this temporal wavelet")
    root2 = np.sqrt(2.0)
    return (array[0::2] + array[1::2]) / root2, (array[0::2] - array[1::2]) / root2


def temporal_idwt(
    low: NDArray[np.generic],
    high: NDArray[np.generic],
    wavelet: str = "haar",
    mode: str = "periodization",
) -> FloatArray:
    """Inverse of :func:`temporal_dwt`."""

    low_array = np.asarray(low, dtype=np.float64)
    high_array = np.asarray(high, dtype=np.float64)
    if low_array.shape != high_array.shape:
        raise ValueError("temporal low/high arrays must have identical shapes")
    if pywt is not None:
        return np.asarray(
            pywt.idwt(low_array, high_array, wavelet, mode=mode, axis=0),
            dtype=np.float64,
        )
    if wavelet not in {"haar", "db1"}:
        raise ImportError("PyWavelets is required for this temporal wavelet")
    root2 = np.sqrt(2.0)
    output = np.empty((low_array.shape[0] * 2, *low_array.shape[1:]), dtype=np.float64)
    output[0::2] = (low_array + high_array) / root2
    output[1::2] = (low_array - high_array) / root2
    return output


def normalize_for_display(array: NDArray[np.generic], percentile: float = 99.0) -> NDArray[np.uint8]:
    """Symmetrically normalize signed transform coefficients for visualization."""

    values = np.asarray(array, dtype=np.float64)
    if values.size == 0:
        return np.empty(values.shape, dtype=np.uint8)
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    scale = float(np.percentile(np.abs(values), percentile))
    if scale <= np.finfo(np.float64).eps:
        return np.full(values.shape, 127, dtype=np.uint8)
    normalized = np.clip(127.5 + 127.5 * values / scale, 0, 255)
    return normalized.astype(np.uint8)


def available_wavelets() -> Iterable[str]:
    """Return available discrete wavelets (Haar-only without PyWavelets)."""

    if pywt is None:
        return ("haar", "db1")
    return tuple(pywt.wavelist(kind="discrete"))


def multilevel_dwt2(
    data: NDArray[np.generic],
    level: int,
    wavelet: str = "haar",
    mode: str = "periodization",
) -> tuple[FloatArray, list[tuple[FloatArray, FloatArray, FloatArray]]]:
    """Recursively decompose an image; details are ordered level 1..N."""

    if level < 1:
        raise ValueError("level must be >= 1")
    approximation = np.asarray(data, dtype=np.float64)
    details: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for _ in range(level):
        approximation, level_details = dwt2(approximation, wavelet, mode)
        details.append(level_details)
    return approximation, details


def multilevel_idwt2(
    approximation: NDArray[np.generic],
    details: list[tuple[FloatArray, FloatArray, FloatArray]],
    wavelet: str = "haar",
    mode: str = "periodization",
) -> FloatArray:
    """Reconstruct details produced by :func:`multilevel_dwt2`."""

    current = np.asarray(approximation, dtype=np.float64)
    for level_details in reversed(details):
        current = idwt2((current, level_details), wavelet, mode)
    return current
