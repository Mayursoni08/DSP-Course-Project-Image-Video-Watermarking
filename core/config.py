"""Validated configuration objects for image and video watermarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .exceptions import ConfigurationError


_ALLOWED_SUBBANDS = frozenset({"LH", "HL", "HH"})
_ALLOWED_ALGORITHMS = frozenset({"stdm", "svd-qim"})


@dataclass(frozen=True, slots=True)
class ImageWatermarkConfig:
    """Configuration for hybrid DWT-DCT watermarking.

    ``algorithm="stdm"`` is the robust default. It applies keyed balanced PN
    chips and spread-transform dither modulation to mid-frequency DCT
    coefficients inside the DWT LH/HL bands. ``algorithm="svd-qim"`` selects
    the alternative SVD singular-value QIM backend.
    """

    algorithm: str = "stdm"
    wavelet: str = "haar"
    wavelet_mode: str = "periodization"
    block_size: int = 4
    dwt_level: int = 3
    delta: float = 12.0
    subbands: tuple[str, ...] = ("LH", "HL")
    spread_chip_length: int = 8
    spread_coefficients_per_block: int = 15
    singular_index: int = 0
    ecc_mode: str = "convolutional"
    header_repetition: int = 3
    payload_repetition: int = 3
    auth_tag_bytes: int = 8
    luminance_only: bool = True

    def __post_init__(self) -> None:
        algorithm = self.algorithm.lower()
        if algorithm not in _ALLOWED_ALGORITHMS:
            raise ConfigurationError(f"algorithm must be one of {sorted(_ALLOWED_ALGORITHMS)}")
        object.__setattr__(self, "algorithm", algorithm)
        if not self.wavelet:
            raise ConfigurationError("wavelet must be a non-empty name")
        if self.block_size < 2 or self.block_size > 32:
            raise ConfigurationError("block_size must be in [2, 32]")
        if self.dwt_level < 1 or self.dwt_level > 4:
            raise ConfigurationError("dwt_level must be in [1, 4]")
        if self.delta <= 0:
            raise ConfigurationError("delta must be positive")
        normalized = tuple(name.upper() for name in self.subbands)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ConfigurationError("subbands must be unique and non-empty")
        if not set(normalized).issubset(_ALLOWED_SUBBANDS):
            raise ConfigurationError("supported subbands are LH, HL, and HH")
        object.__setattr__(self, "subbands", normalized)
        available_dct_positions = self.block_size * self.block_size - 1
        if not 1 <= self.spread_coefficients_per_block <= available_dct_positions:
            raise ConfigurationError(
                "spread_coefficients_per_block must fit non-DC DCT positions"
            )
        if self.spread_chip_length < 2 or self.spread_chip_length % 2:
            raise ConfigurationError("spread_chip_length must be an even integer >= 2")
        if not 0 <= self.singular_index < self.block_size:
            raise ConfigurationError("singular_index must be inside the DCT block")
        ecc_mode = self.ecc_mode.lower()
        if ecc_mode not in {"convolutional", "repetition"}:
            raise ConfigurationError("ecc_mode must be convolutional or repetition")
        object.__setattr__(self, "ecc_mode", ecc_mode)
        for name, repetition in (
            ("header_repetition", self.header_repetition),
            ("payload_repetition", self.payload_repetition),
        ):
            if repetition < 1 or repetition % 2 == 0:
                raise ConfigurationError(f"{name} must be a positive odd integer")
        if not 4 <= self.auth_tag_bytes <= 32:
            raise ConfigurationError("auth_tag_bytes must be in [4, 32]")
        if not self.luminance_only:
            raise ConfigurationError(
                "only luminance_only=True is currently supported"
            )

    @property
    def spatial_multiple(self) -> int:
        return (2 ** self.dwt_level) * self.block_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageWatermarkConfig":
        values = dict(data)
        if "subbands" in values:
            values["subbands"] = tuple(values["subbands"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class VideoWatermarkConfig:
    """Configuration for separable spatio-temporal watermarking."""

    image: ImageWatermarkConfig = field(
        default_factory=lambda: ImageWatermarkConfig(delta=16.0)
    )
    temporal_wavelet: str = "haar"
    temporal_window: int = 8
    embed_group_interval: int = 1
    codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    preserve_audio: bool = True
    align_output_keyframes: bool = True
    extraction_max_samples: int = 64
    search_temporal_phase: bool = True

    def __post_init__(self) -> None:
        if self.temporal_wavelet not in {"haar", "db1"}:
            raise ConfigurationError("the temporal path currently supports haar/db1")
        if self.temporal_window < 2 or self.temporal_window % 2:
            raise ConfigurationError("temporal_window must be an even integer >= 2")
        if self.embed_group_interval < 1:
            raise ConfigurationError("embed_group_interval must be >= 1")
        if not 0 <= self.crf <= 51:
            raise ConfigurationError("crf must be in [0, 51]")
        if self.extraction_max_samples < 1:
            raise ConfigurationError("extraction_max_samples must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["image"] = self.image.to_dict()
        return values

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VideoWatermarkConfig":
        values = dict(data)
        image = values.get("image", {})
        values["image"] = image if isinstance(image, ImageWatermarkConfig) else ImageWatermarkConfig.from_dict(image)
        return cls(**values)
