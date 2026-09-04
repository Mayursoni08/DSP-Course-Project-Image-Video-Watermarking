"""Transform-band carrier layout utilities shared by embedding and extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BandLayout:
    name: str
    offset: int
    count: int
    block_rows: int
    block_cols: int
    shape: tuple[int, int]


def detail_bands(
    details: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
) -> dict[str, NDArray[np.float64]]:
    """Map PyWavelets-style detail arrays to LH/HL/HH names."""

    c_h, c_v, c_d = details
    return {"LH": c_h, "HL": c_v, "HH": c_d}


def build_layouts(
    bands: dict[str, NDArray[np.float64]],
    selected_names: tuple[str, ...],
    block_size: int,
) -> tuple[BandLayout, ...]:
    layouts: list[BandLayout] = []
    offset = 0
    for name in selected_names:
        band = bands[name]
        rows = band.shape[0] // block_size
        cols = band.shape[1] // block_size
        if rows * block_size != band.shape[0] or cols * block_size != band.shape[1]:
            raise ValueError(f"subband {name} is not divisible by block_size")
        count = rows * cols
        layouts.append(BandLayout(name, offset, count, rows, cols, tuple(band.shape)))
        offset += count
    return tuple(layouts)


def total_carriers(layouts: tuple[BandLayout, ...]) -> int:
    return sum(layout.count for layout in layouts)


def locate_carrier(
    carrier_id: int, layouts: tuple[BandLayout, ...]
) -> tuple[BandLayout, int]:
    for layout in layouts:
        if layout.offset <= carrier_id < layout.offset + layout.count:
            return layout, carrier_id - layout.offset
    raise IndexError(f"carrier id {carrier_id} is outside the layout")
