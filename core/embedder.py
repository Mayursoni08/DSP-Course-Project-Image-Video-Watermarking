"""Image and video embedding pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .carriers import build_layouts, detail_bands, total_carriers
from .config import ImageWatermarkConfig, VideoWatermarkConfig
from .media import merge_luminance, read_image, split_luminance, write_image
from .payload import PayloadCodec, keyed_dither, keyed_permutation
from .qim import embed_parity_qim
from .spread_spectrum import (
    analyze_bands,
    embed_stdm,
    physical_bit_capacity,
    selected_blocks_from_scalar_ids,
    synthesize_bands,
)
from .transforms import (
    batched_svd,
    blocks_view,
    crop_padding,
    dct2_blocks,
    idct2_blocks,
    merge_blocks,
    multilevel_dwt2,
    multilevel_idwt2,
    pad_to_multiple,
    reconstruct_svd,
    temporal_dwt,
    temporal_idwt,
)
from .video_io import create_lossless_writer, encode_and_remux, probe_video, temporary_video_directory


@dataclass(frozen=True, slots=True)
class ModifiedBlock:
    band: str
    row: int
    column: int
    bit: int


@dataclass(frozen=True, slots=True)
class ImageEmbeddingReport:
    algorithm: str
    payload_bytes: int
    capacity_bytes: int
    carriers_used: int
    total_carriers: int
    transform_coefficients_used: int
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    modified_blocks: tuple[ModifiedBlock, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["modified_blocks"] = [asdict(item) for item in self.modified_blocks]
        return data


@dataclass(frozen=True, slots=True)
class VideoEmbeddingReport:
    algorithm: str
    payload_bytes: int
    capacity_bytes_per_temporal_plane: int
    frames_processed: int
    temporal_groups: int
    embedded_groups: int
    embedded_temporal_planes: int
    width: int
    height: int
    fps: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ImageWatermarker:
    """Blind keyed hybrid DWT-DCT watermark embedder.

    The default STDM backend spreads every coded bit across a balanced PN chip
    of mid-frequency DCT coefficients. The alternative SVD-QIM backend modifies
    a selected singular value in each keyed DCT block.
    """

    _SVD_ORDER_CONTEXT = b"image-carrier-order/v1"
    _SVD_DITHER_CONTEXT = b"image-qim-dither/v1"

    def __init__(self, config: ImageWatermarkConfig | None = None) -> None:
        self.config = config or ImageWatermarkConfig()

    def _capacity_from_bands(
        self, bands: dict[str, NDArray[np.float64]]
    ) -> tuple[int, int]:
        """Return (physical-bit carriers, scalar transform carriers)."""

        config = self.config
        if config.algorithm == "stdm":
            _, packed = analyze_bands(
                bands,
                config.subbands,
                config.block_size,
                config.spread_coefficients_per_block,
            )
            return (
                physical_bit_capacity(packed.size, config.spread_chip_length),
                packed.size,
            )
        layouts = build_layouts(bands, config.subbands, config.block_size)
        count = total_carriers(layouts)
        return count, count

    def capacity_for_shape(self, shape: tuple[int, int]) -> int:
        dummy = np.zeros(shape, dtype=np.float64)
        padded, _ = pad_to_multiple(dummy, self.config.spatial_multiple)
        _, detail_levels = multilevel_dwt2(
            padded, self.config.dwt_level, self.config.wavelet, self.config.wavelet_mode
        )
        bit_carriers, _ = self._capacity_from_bands(detail_bands(detail_levels[-1]))
        return PayloadCodec.capacity_bytes(bit_carriers, self.config)

    def _embed_stdm_bands(
        self,
        bands: dict[str, NDArray[np.float64]],
        payload: bytes,
        secret_key: str | bytes,
        *,
        utf8: bool,
        collect_debug: bool,
    ) -> tuple[int, int, int, tuple[ModifiedBlock, ...], int]:
        config = self.config
        states, packed = analyze_bands(
            bands,
            config.subbands,
            config.block_size,
            config.spread_coefficients_per_block,
        )
        bit_capacity = physical_bit_capacity(packed.size, config.spread_chip_length)
        encoded = PayloadCodec.encode(
            payload, bit_capacity, secret_key, config, utf8=utf8
        )
        modified, selected_scalar_ids = embed_stdm(
            packed,
            encoded.physical_bits,
            secret_key,
            delta=config.delta,
            chip_length=config.spread_chip_length,
        )
        synthesize_bands(states, modified, bands, config.block_size)
        debug: tuple[ModifiedBlock, ...] = ()
        if collect_debug:
            debug = tuple(
                ModifiedBlock(name, row, column, -1)
                for name, row, column in selected_blocks_from_scalar_ids(
                    selected_scalar_ids, states
                )
            )
        return (
            encoded.capacity_bytes,
            encoded.physical_bits.size,
            bit_capacity,
            debug,
            selected_scalar_ids.size,
        )

    def _embed_svd_bands(
        self,
        bands: dict[str, NDArray[np.float64]],
        payload: bytes,
        secret_key: str | bytes,
        *,
        utf8: bool,
        collect_debug: bool,
    ) -> tuple[int, int, int, tuple[ModifiedBlock, ...], int]:
        config = self.config
        layouts = build_layouts(bands, config.subbands, config.block_size)
        carrier_count = total_carriers(layouts)
        encoded = PayloadCodec.encode(
            payload, carrier_count, secret_key, config, utf8=utf8
        )
        order = keyed_permutation(
            carrier_count, secret_key, self._SVD_ORDER_CONTEXT
        )
        used_order = order[: encoded.physical_bits.size]
        dithers = keyed_dither(
            encoded.physical_bits.size,
            config.delta,
            secret_key,
            self._SVD_DITHER_CONTEXT,
        )

        debug_blocks: list[ModifiedBlock] = []
        for layout in layouts:
            in_band = (used_order >= layout.offset) & (
                used_order < layout.offset + layout.count
            )
            if not np.any(in_band):
                continue
            physical_positions = np.flatnonzero(in_band)
            local_ids = used_order[physical_positions] - layout.offset
            bits = encoded.physical_bits[physical_positions]
            local_dithers = dithers[physical_positions]

            block_grid = blocks_view(bands[layout.name], config.block_size)
            flat_blocks = block_grid.reshape(-1, config.block_size, config.block_size)
            selected = flat_blocks[local_ids]
            coefficients = dct2_blocks(selected)
            u, singular, vh = batched_svd(coefficients)
            index = config.singular_index
            lower = (
                singular[:, index + 1] + 1e-9
                if index + 1 < singular.shape[1]
                else np.zeros(singular.shape[0], dtype=np.float64)
            )
            upper = (
                singular[:, index - 1] - 1e-9
                if index > 0
                else np.full(singular.shape[0], np.inf, dtype=np.float64)
            )
            singular[:, index] = embed_parity_qim(
                singular[:, index],
                bits,
                local_dithers,
                config.delta,
                lower_bounds=lower,
                upper_bounds=upper,
            )
            flat_blocks[local_ids] = idct2_blocks(
                reconstruct_svd(u, singular, vh)
            )
            bands[layout.name] = merge_blocks(
                flat_blocks.reshape(
                    layout.block_rows,
                    layout.block_cols,
                    config.block_size,
                    config.block_size,
                )
            )
            if collect_debug:
                for local_id, bit in zip(local_ids.tolist(), bits.tolist(), strict=True):
                    row, column = divmod(int(local_id), layout.block_cols)
                    debug_blocks.append(
                        ModifiedBlock(layout.name, row, column, int(bit))
                    )
        return (
            encoded.capacity_bytes,
            encoded.physical_bits.size,
            carrier_count,
            tuple(debug_blocks),
            encoded.physical_bits.size,
        )

    def _embed_plane(
        self,
        plane: NDArray[np.generic],
        payload: bytes,
        secret_key: str | bytes,
        *,
        utf8: bool = False,
        collect_debug: bool = False,
    ) -> tuple[NDArray[np.float64], ImageEmbeddingReport]:
        config = self.config
        padded, padding = pad_to_multiple(plane, config.spatial_multiple)
        approximation, detail_levels = multilevel_dwt2(
            padded, config.dwt_level, config.wavelet, config.wavelet_mode
        )
        bands = detail_bands(detail_levels[-1])

        if config.algorithm == "stdm":
            capacity, used, total, debug, scalar_used = self._embed_stdm_bands(
                bands,
                payload,
                secret_key,
                utf8=utf8,
                collect_debug=collect_debug,
            )
        else:
            capacity, used, total, debug, scalar_used = self._embed_svd_bands(
                bands,
                payload,
                secret_key,
                utf8=utf8,
                collect_debug=collect_debug,
            )

        detail_levels[-1] = (bands["LH"], bands["HL"], bands["HH"])
        reconstructed = multilevel_idwt2(
            approximation, detail_levels, config.wavelet, config.wavelet_mode
        )
        output = crop_padding(reconstructed, padding)
        report = ImageEmbeddingReport(
            algorithm=config.algorithm,
            payload_bytes=len(payload),
            capacity_bytes=capacity,
            carriers_used=used,
            total_carriers=total,
            transform_coefficients_used=scalar_used,
            original_shape=padding.original_shape,
            padded_shape=padding.padded_shape,
            modified_blocks=debug,
        )
        return output, report

    def embed_array(
        self,
        image: NDArray[np.generic],
        payload: bytes,
        secret_key: str | bytes,
        *,
        utf8: bool = False,
        collect_debug: bool = False,
    ) -> tuple[NDArray[np.uint8], ImageEmbeddingReport]:
        """Embed bytes in an OpenCV-style grayscale/BGR/BGRA image array."""

        luminance, context = split_luminance(image)
        watermarked_luminance, report = self._embed_plane(
            luminance,
            payload,
            secret_key,
            utf8=utf8,
            collect_debug=collect_debug,
        )
        return merge_luminance(watermarked_luminance, context), report

    def embed_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        payload: bytes,
        secret_key: str | bytes,
        *,
        utf8: bool = False,
        collect_debug: bool = False,
    ) -> ImageEmbeddingReport:
        image = read_image(input_path)
        watermarked, report = self.embed_array(
            image,
            payload,
            secret_key,
            utf8=utf8,
            collect_debug=collect_debug,
        )
        write_image(output_path, watermarked)
        return report


class VideoWatermarker:
    """Separable 3-D watermarking using temporal Haar + spatial DWT-DCT."""

    def __init__(self, config: VideoWatermarkConfig | None = None) -> None:
        self.config = config or VideoWatermarkConfig()
        self.image_watermarker = ImageWatermarker(self.config.image)

    def _process_group(
        self,
        frames: list[NDArray[np.uint8]],
        payload: bytes,
        secret_key: str | bytes,
        *,
        utf8: bool,
    ) -> tuple[list[NDArray[np.uint8]], ImageEmbeddingReport, int]:
        original_count = len(frames)
        if not frames:
            raise ValueError("video group cannot be empty")
        padded_frames = list(frames)
        while len(padded_frames) < self.config.temporal_window:
            padded_frames.append(padded_frames[-1].copy())

        color_planes: list[NDArray[np.float64]] = []
        luminance_planes: list[NDArray[np.float64]] = []
        for frame in padded_frames:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float64)
            color_planes.append(ycrcb)
            luminance_planes.append(ycrcb[..., 0].copy())
        low, high = temporal_dwt(
            np.stack(luminance_planes, axis=0),
            self.config.temporal_wavelet,
            self.config.image.wavelet_mode,
        )

        first_report: ImageEmbeddingReport | None = None
        for index in range(low.shape[0]):
            low[index], report = self.image_watermarker._embed_plane(
                low[index],
                payload,
                secret_key,
                utf8=utf8,
                collect_debug=False,
            )
            first_report = first_report or report
        reconstructed = temporal_idwt(
            low,
            high,
            self.config.temporal_wavelet,
            self.config.image.wavelet_mode,
        )[: len(padded_frames)]

        output: list[NDArray[np.uint8]] = []
        for index in range(original_count):
            ycrcb = color_planes[index]
            ycrcb[..., 0] = np.clip(reconstructed[index], 0, 255)
            output.append(
                cv2.cvtColor(
                    np.clip(np.rint(ycrcb), 0, 255).astype(np.uint8),
                    cv2.COLOR_YCrCb2BGR,
                )
            )
        if first_report is None:  # pragma: no cover
            raise RuntimeError("temporal transform produced no low-pass planes")
        return output, first_report, low.shape[0]

    def embed_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        payload: bytes,
        secret_key: str | bytes,
        *,
        utf8: bool = False,
    ) -> VideoEmbeddingReport:
        info = probe_video(input_path)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {input_path}")

        frames_processed = 0
        group_index = 0
        embedded_groups = 0
        embedded_planes = 0
        capacity = 0
        with temporary_video_directory() as directory:
            writer, intermediate = create_lossless_writer(
                directory, info.width, info.height, info.fps
            )
            try:
                while True:
                    group: list[NDArray[np.uint8]] = []
                    for _ in range(self.config.temporal_window):
                        ok, frame = capture.read()
                        if not ok:
                            break
                        group.append(frame)
                    if not group:
                        break
                    if group_index % self.config.embed_group_interval == 0:
                        processed, plane_report, plane_count = self._process_group(
                            group, payload, secret_key, utf8=utf8
                        )
                        capacity = plane_report.capacity_bytes
                        embedded_groups += 1
                        embedded_planes += plane_count
                    else:
                        processed = group
                    for frame in processed:
                        writer.write(frame)
                    frames_processed += len(group)
                    group_index += 1
                    if len(group) < self.config.temporal_window:
                        break
            finally:
                capture.release()
                writer.release()

            encode_and_remux(
                intermediate,
                input_path,
                output_path,
                codec=self.config.codec,
                crf=self.config.crf,
                preset=self.config.preset,
                pixel_format=self.config.pixel_format,
                preserve_audio=self.config.preserve_audio,
                keyframe_interval_seconds=(
                    self.config.temporal_window
                    * self.config.embed_group_interval
                    / info.fps
                    if self.config.align_output_keyframes
                    else None
                ),
                gop_size=(
                    self.config.temporal_window * self.config.embed_group_interval
                    if self.config.align_output_keyframes
                    else None
                ),
            )

        return VideoEmbeddingReport(
            algorithm=self.config.image.algorithm,
            payload_bytes=len(payload),
            capacity_bytes_per_temporal_plane=capacity,
            frames_processed=frames_processed,
            temporal_groups=group_index,
            embedded_groups=embedded_groups,
            embedded_temporal_planes=embedded_planes,
            width=info.width,
            height=info.height,
            fps=info.fps,
        )
