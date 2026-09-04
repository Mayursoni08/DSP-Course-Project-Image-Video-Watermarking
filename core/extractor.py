"""Blind/semi-blind image extraction and temporally aggregated video extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .carriers import build_layouts, detail_bands, total_carriers
from .config import ImageWatermarkConfig, VideoWatermarkConfig
from .media import read_image, split_luminance
from .payload import DecodedPayload, PayloadCodec, keyed_dither, keyed_permutation
from .qim import parity_qim_scores
from .registration import align_to_reference
from .spread_spectrum import analyze_bands, extract_stdm_scores
from .transforms import batched_svd, blocks_view, dct2_blocks, multilevel_dwt2, pad_to_multiple, temporal_dwt
from .video_io import probe_video


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    payload: bytes
    valid: bool
    header_ok: bool
    crc_ok: bool
    auth_ok: bool
    confidence: float
    declared_length: int | None
    message: str
    is_utf8: bool
    registration_method: str | None = None
    temporal_phase: int | None = None
    temporal_samples: int | None = None

    @property
    def text(self) -> str | None:
        if not self.is_utf8:
            return None
        try:
            return self.payload.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def to_dict(self, *, include_payload_hex: bool = True) -> dict[str, object]:
        data = asdict(self)
        data["text"] = self.text
        if include_payload_hex:
            data["payload_hex"] = self.payload.hex()
        data.pop("payload", None)
        return data


class ImageWatermarkExtractor:
    """Extract a keyed payload without the original cover image."""

    _SVD_ORDER_CONTEXT = b"image-carrier-order/v1"
    _SVD_DITHER_CONTEXT = b"image-qim-dither/v1"

    def __init__(self, config: ImageWatermarkConfig | None = None) -> None:
        self.config = config or ImageWatermarkConfig()

    def _svd_scores(
        self,
        bands: dict[str, NDArray[np.float64]],
        secret_key: str | bytes,
    ) -> NDArray[np.float64]:
        config = self.config
        layouts = build_layouts(bands, config.subbands, config.block_size)
        count = total_carriers(layouts)
        singular_by_carrier = np.empty(count, dtype=np.float64)
        for layout in layouts:
            blocks = blocks_view(bands[layout.name], config.block_size)
            flat = blocks.reshape(-1, config.block_size, config.block_size)
            _, singular, _ = batched_svd(dct2_blocks(flat))
            singular_by_carrier[
                layout.offset : layout.offset + layout.count
            ] = singular[:, config.singular_index]
        order = keyed_permutation(count, secret_key, self._SVD_ORDER_CONTEXT)
        dithers = keyed_dither(
            count, config.delta, secret_key, self._SVD_DITHER_CONTEXT
        )
        return parity_qim_scores(singular_by_carrier[order], dithers, config.delta)

    def _ordered_scores_from_plane(
        self, plane: NDArray[np.generic], secret_key: str | bytes
    ) -> NDArray[np.float64]:
        config = self.config
        padded, _ = pad_to_multiple(plane, config.spatial_multiple)
        _, detail_levels = multilevel_dwt2(
            padded, config.dwt_level, config.wavelet, config.wavelet_mode
        )
        bands = detail_bands(detail_levels[-1])
        if config.algorithm == "stdm":
            _, packed = analyze_bands(
                bands,
                config.subbands,
                config.block_size,
                config.spread_coefficients_per_block,
            )
            return extract_stdm_scores(
                packed,
                secret_key,
                delta=config.delta,
                chip_length=config.spread_chip_length,
            )
        return self._svd_scores(bands, secret_key)

    @staticmethod
    def _result_from_decoded(
        decoded: DecodedPayload,
        *,
        registration_method: str | None = None,
        temporal_phase: int | None = None,
        temporal_samples: int | None = None,
    ) -> ExtractionResult:
        return ExtractionResult(
            payload=decoded.payload,
            valid=decoded.valid,
            header_ok=decoded.header_ok,
            crc_ok=decoded.crc_ok,
            auth_ok=decoded.auth_ok,
            confidence=decoded.confidence,
            declared_length=decoded.declared_length,
            message=decoded.message,
            is_utf8=decoded.is_utf8,
            registration_method=registration_method,
            temporal_phase=temporal_phase,
            temporal_samples=temporal_samples,
        )

    def extract_array(
        self,
        image: NDArray[np.generic],
        secret_key: str | bytes,
        *,
        reference: NDArray[np.generic] | None = None,
        expected_payload_bytes: int | None = None,
        expected_shape: tuple[int, int] | None = None,
    ) -> ExtractionResult:
        working = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        registration_method: str | None = None
        if reference is not None:
            reference_u8 = np.clip(np.rint(reference), 0, 255).astype(np.uint8)
            registration = align_to_reference(working, reference_u8)
            working = registration.image
            registration_method = registration.method
        elif expected_shape is not None and tuple(working.shape[:2]) != tuple(expected_shape):
            working = cv2.resize(
                working,
                (expected_shape[1], expected_shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            registration_method = "manifest-resize"
        luminance, _ = split_luminance(working)
        decoded = PayloadCodec.decode(
            self._ordered_scores_from_plane(luminance, secret_key),
            secret_key,
            self.config,
            expected_payload_bytes=expected_payload_bytes,
        )
        return self._result_from_decoded(
            decoded, registration_method=registration_method
        )

    def extract_file(
        self,
        input_path: str | Path,
        secret_key: str | bytes,
        *,
        reference_path: str | Path | None = None,
        expected_payload_bytes: int | None = None,
        expected_shape: tuple[int, int] | None = None,
    ) -> ExtractionResult:
        image = read_image(input_path)
        reference = read_image(reference_path) if reference_path is not None else None
        return self.extract_array(
            image,
            secret_key,
            reference=reference,
            expected_payload_bytes=expected_payload_bytes,
            expected_shape=expected_shape,
        )


class VideoWatermarkExtractor:
    """Extract video watermarks through temporal phase search and soft voting."""

    def __init__(self, config: VideoWatermarkConfig | None = None) -> None:
        self.config = config or VideoWatermarkConfig()
        self.image_extractor = ImageWatermarkExtractor(self.config.image)

    def _collect_phase_scores(
        self,
        input_path: str | Path,
        secret_key: str | bytes,
        phase: int,
        *,
        expected_shape: tuple[int, int] | None,
    ) -> tuple[NDArray[np.float64] | None, int]:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {input_path}")
        for _ in range(phase):
            if not capture.grab():
                capture.release()
                return None, 0

        score_sum: NDArray[np.float64] | None = None
        samples = 0
        group_index = 0
        try:
            while samples < self.config.extraction_max_samples:
                frames: list[NDArray[np.uint8]] = []
                for _ in range(self.config.temporal_window):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if expected_shape is not None and tuple(frame.shape[:2]) != tuple(expected_shape):
                        frame = cv2.resize(
                            frame,
                            (expected_shape[1], expected_shape[0]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    frames.append(frame)
                if not frames:
                    break
                partial_group = len(frames) < self.config.temporal_window
                while len(frames) < self.config.temporal_window:
                    frames.append(frames[-1].copy())
                if group_index % self.config.embed_group_interval == 0:
                    low, _ = temporal_dwt(
                        np.stack(
                            [
                                cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
                                for frame in frames
                            ],
                            axis=0,
                        ),
                        self.config.temporal_wavelet,
                        self.config.image.wavelet_mode,
                    )
                    for plane in low:
                        scores = self.image_extractor._ordered_scores_from_plane(
                            plane, secret_key
                        )
                        if score_sum is None:
                            score_sum = np.zeros_like(scores)
                        score_sum += np.clip(scores, -1.0, 1.0)
                        samples += 1
                        if samples >= self.config.extraction_max_samples:
                            break
                group_index += 1
                if partial_group:
                    break
        finally:
            capture.release()
        if score_sum is None or samples == 0:
            return None, 0
        return score_sum / samples, samples

    def extract_file(
        self,
        input_path: str | Path,
        secret_key: str | bytes,
        *,
        expected_payload_bytes: int | None = None,
        expected_shape: tuple[int, int] | None = None,
    ) -> ExtractionResult:
        info = probe_video(input_path)
        shape = expected_shape or (info.height, info.width)
        phases = range(self.config.temporal_window) if self.config.search_temporal_phase else range(1)
        candidates: list[ExtractionResult] = []
        for phase in phases:
            averaged_scores, samples = self._collect_phase_scores(
                input_path, secret_key, phase, expected_shape=shape
            )
            if averaged_scores is None:
                continue
            decoded = PayloadCodec.decode(
                averaged_scores,
                secret_key,
                self.config.image,
                expected_payload_bytes=expected_payload_bytes,
            )
            candidates.append(
                self.image_extractor._result_from_decoded(
                    decoded, temporal_phase=phase, temporal_samples=samples
                )
            )
            if decoded.valid and phase == 0:
                break
        if not candidates:
            return ExtractionResult(
                b"", False, False, False, False, 0.0, None,
                "no complete temporal windows were available", False
            )
        candidates.sort(
            key=lambda result: (
                result.valid,
                result.header_ok,
                result.crc_ok,
                result.auth_ok,
                result.confidence,
            ),
            reverse=True,
        )
        best = candidates[0]
        data = asdict(best)
        data["message"] = (
            f"{best.message}; temporal phase={best.temporal_phase}, "
            f"samples={best.temporal_samples}"
        )
        return ExtractionResult(**data)
