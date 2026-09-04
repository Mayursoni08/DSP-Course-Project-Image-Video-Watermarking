"""Keyed dithered quantization-index modulation (QIM) primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BitArray = NDArray[np.uint8]


def embed_parity_qim(
    values: NDArray[np.generic],
    bits: NDArray[np.generic],
    dithers: NDArray[np.generic],
    delta: float,
    *,
    lower_bounds: NDArray[np.generic] | float = 0.0,
    upper_bounds: NDArray[np.generic] | float = np.inf,
) -> FloatArray:
    """Quantize values to keyed parity lattices.

    Bit 0 occupies even lattice indices and bit 1 occupies odd indices. A
    subtractive key-derived dither shifts every carrier's lattice, preventing
    meaningful bit decisions without the secret key.
    """

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    d = np.asarray(dithers, dtype=np.float64).reshape(-1)
    if x.size != b.size or x.size != d.size:
        raise ValueError("values, bits, and dithers must have equal lengths")
    if delta <= 0:
        raise ValueError("delta must be positive")

    lower = np.broadcast_to(np.asarray(lower_bounds, dtype=np.float64), x.shape)
    upper = np.broadcast_to(np.asarray(upper_bounds, dtype=np.float64), x.shape)
    if np.any(lower > upper):
        raise ValueError("lower_bounds cannot exceed upper_bounds")

    scaled = (x - d) / delta
    q_floor = np.floor(scaled).astype(np.int64)
    q_lower = q_floor - np.mod(q_floor - b.astype(np.int64), 2)
    q_upper = q_lower + 2

    finite_lower = np.isfinite(lower)
    min_q = np.full(x.shape, np.iinfo(np.int64).min // 4, dtype=np.int64)
    raw_min = np.ceil((lower[finite_lower] - d[finite_lower]) / delta).astype(np.int64)
    raw_min += np.mod(b[finite_lower].astype(np.int64) - raw_min, 2)
    min_q[finite_lower] = raw_min
    finite_upper = np.isfinite(upper)
    max_q = np.full(x.shape, np.iinfo(np.int64).max // 4, dtype=np.int64)
    raw_max = np.floor((upper[finite_upper] - d[finite_upper]) / delta).astype(np.int64)
    raw_max -= np.mod(raw_max - b[finite_upper].astype(np.int64), 2)
    max_q[finite_upper] = raw_max

    q_lower = np.maximum(q_lower, min_q)
    q_upper = np.maximum(q_upper, min_q)
    q_lower = np.minimum(q_lower, max_q)
    q_upper = np.minimum(q_upper, max_q)

    dist_lower = np.abs(scaled - q_lower)
    dist_upper = np.abs(scaled - q_upper)
    q = np.where(dist_lower <= dist_upper, q_lower, q_upper)

    infeasible = min_q > max_q
    if np.any(infeasible):
        # This can happen only for a non-leading singular value with a very
        # narrow ordering interval. Keep it inside the interval; callers using
        # the default leading singular value do not encounter this branch.
        projected = np.clip(x[infeasible], lower[infeasible], upper[infeasible])
        result = q.astype(np.float64) * delta + d
        result[infeasible] = projected
        return result
    return q.astype(np.float64) * delta + d


def parity_qim_scores(
    values: NDArray[np.generic], dithers: NDArray[np.generic], delta: float
) -> FloatArray:
    """Return soft bit scores in ``[-1, 1]`` (negative=0, positive=1)."""

    if delta <= 0:
        raise ValueError("delta must be positive")
    x = (
        np.asarray(values, dtype=np.float64).reshape(-1)
        - np.asarray(dithers, dtype=np.float64).reshape(-1)
    ) / delta
    nearest_even = 2.0 * np.rint(x / 2.0)
    nearest_odd = 2.0 * np.rint((x - 1.0) / 2.0) + 1.0
    distance_even = np.abs(x - nearest_even)
    distance_odd = np.abs(x - nearest_odd)
    return np.clip(distance_even - distance_odd, -1.0, 1.0)
