"""DWT-DCT spread-transform dither modulation (STDM) helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .payload import keyed_dither, keyed_permutation, keyed_pn_signs
from .qim import embed_parity_qim, parity_qim_scores
from .transforms import blocks_view, dct2_blocks, idct2_blocks, merge_blocks


@dataclass(slots=True)
class BandCoefficientState:
    name: str
    block_rows: int
    block_cols: int
    coefficients: NDArray[np.float64]
    positions: tuple[tuple[int, int], ...]
    offset: int
    count: int


def zigzag_positions(block_size: int) -> tuple[tuple[int, int], ...]:
    """Return JPEG-style low-to-high-frequency zigzag coordinates."""

    positions: list[tuple[int, int]] = []
    for diagonal in range(2 * block_size - 1):
        line: list[tuple[int, int]] = []
        row_start = max(0, diagonal - block_size + 1)
        row_end = min(block_size - 1, diagonal)
        for row in range(row_start, row_end + 1):
            column = diagonal - row
            line.append((row, column))
        if diagonal % 2 == 0:
            line.reverse()
        positions.extend(line)
    return tuple(positions)


def analyze_bands(
    bands: dict[str, NDArray[np.float64]],
    selected_names: tuple[str, ...],
    block_size: int,
    coefficients_per_block: int,
) -> tuple[list[BandCoefficientState], NDArray[np.float64]]:
    """DCT-transform selected bands and pack mid-frequency scalar carriers."""

    all_positions = [position for position in zigzag_positions(block_size) if position != (0, 0)]
    if coefficients_per_block > len(all_positions):
        raise ValueError("coefficients_per_block exceeds non-DC DCT positions")
    positions = tuple(all_positions[:coefficients_per_block])
    rows_index = np.array([row for row, _ in positions], dtype=np.intp)
    cols_index = np.array([col for _, col in positions], dtype=np.intp)

    states: list[BandCoefficientState] = []
    chunks: list[NDArray[np.float64]] = []
    offset = 0
    for name in selected_names:
        block_grid = blocks_view(bands[name], block_size)
        block_rows, block_cols = block_grid.shape[:2]
        flat_blocks = block_grid.reshape(-1, block_size, block_size)
        coefficients = dct2_blocks(flat_blocks)
        selected = coefficients[:, rows_index, cols_index].reshape(-1).copy()
        count = selected.size
        states.append(
            BandCoefficientState(
                name=name,
                block_rows=block_rows,
                block_cols=block_cols,
                coefficients=coefficients,
                positions=positions,
                offset=offset,
                count=count,
            )
        )
        chunks.append(selected)
        offset += count
    packed = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    return states, packed


def synthesize_bands(
    states: list[BandCoefficientState],
    packed: NDArray[np.generic],
    bands: dict[str, NDArray[np.float64]],
    block_size: int,
) -> None:
    """Write packed carriers back into DCT blocks and inverse-transform bands."""

    vector = np.asarray(packed, dtype=np.float64).reshape(-1)
    for state in states:
        rows_index = np.array([row for row, _ in state.positions], dtype=np.intp)
        cols_index = np.array([col for _, col in state.positions], dtype=np.intp)
        count_per_block = len(state.positions)
        selected = vector[state.offset : state.offset + state.count].reshape(
            -1, count_per_block
        )
        state.coefficients[:, rows_index, cols_index] = selected
        blocks = idct2_blocks(state.coefficients).reshape(
            state.block_rows,
            state.block_cols,
            block_size,
            block_size,
        )
        bands[state.name] = merge_blocks(blocks)


def physical_bit_capacity(scalar_carriers: int, chip_length: int) -> int:
    return scalar_carriers // chip_length


def embed_stdm(
    packed: NDArray[np.generic],
    physical_bits: NDArray[np.generic],
    secret_key: str | bytes,
    *,
    delta: float,
    chip_length: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Embed physical bits by QIM-quantizing PN correlation projections."""

    vector = np.asarray(packed, dtype=np.float64).copy().reshape(-1)
    bits = np.asarray(physical_bits, dtype=np.uint8).reshape(-1)
    required = bits.size * chip_length
    if required > vector.size:
        raise ValueError("insufficient scalar carriers for requested STDM chips")
    order = keyed_permutation(vector.size, secret_key, b"stdm-coefficient-order/v1")
    selected = order[:required]
    chips = vector[selected].reshape(bits.size, chip_length)
    pn = keyed_pn_signs(bits.size, chip_length, secret_key, b"stdm-pn/v1")
    projections = np.mean(chips * pn, axis=1)
    dithers = keyed_dither(bits.size, delta, secret_key, b"stdm-dither/v1")
    targets = embed_parity_qim(
        projections,
        bits,
        dithers,
        delta,
        lower_bounds=-np.inf,
        upper_bounds=np.inf,
    )
    chips += (targets - projections)[:, np.newaxis] * pn
    vector[selected] = chips.reshape(-1)
    return vector, selected


def extract_stdm_scores(
    packed: NDArray[np.generic],
    secret_key: str | bytes,
    *,
    delta: float,
    chip_length: int,
) -> NDArray[np.float64]:
    """Extract soft physical-bit scores from every complete keyed chip."""

    vector = np.asarray(packed, dtype=np.float64).reshape(-1)
    bit_capacity = physical_bit_capacity(vector.size, chip_length)
    if bit_capacity == 0:
        return np.empty(0, dtype=np.float64)
    used = bit_capacity * chip_length
    order = keyed_permutation(vector.size, secret_key, b"stdm-coefficient-order/v1")
    chips = vector[order[:used]].reshape(bit_capacity, chip_length)
    pn = keyed_pn_signs(bit_capacity, chip_length, secret_key, b"stdm-pn/v1")
    projections = np.mean(chips * pn, axis=1)
    dithers = keyed_dither(bit_capacity, delta, secret_key, b"stdm-dither/v1")
    return parity_qim_scores(projections, dithers, delta)


def selected_blocks_from_scalar_ids(
    scalar_ids: NDArray[np.generic], states: list[BandCoefficientState]
) -> list[tuple[str, int, int]]:
    """Map scalar carrier IDs to unique DWT-band block coordinates."""

    identifiers = np.asarray(scalar_ids, dtype=np.int64).reshape(-1)
    output: set[tuple[str, int, int]] = set()
    for state in states:
        mask = (identifiers >= state.offset) & (identifiers < state.offset + state.count)
        local = identifiers[mask] - state.offset
        if local.size == 0:
            continue
        blocks = np.unique(local // len(state.positions))
        for block_id in blocks.tolist():
            row, column = divmod(int(block_id), state.block_cols)
            output.add((state.name, row, column))
    return sorted(output)
