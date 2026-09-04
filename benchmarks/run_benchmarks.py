"""Reproducible image robustness benchmark suite."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from attacks.compression import jpeg_compress
from attacks.spatial import awgn, crop_and_resize, gaussian_blur, median_filter, rotate, scale
from core.config import ImageWatermarkConfig
from core.embedder import ImageWatermarker
from core.extractor import ImageWatermarkExtractor
from core.media import read_image
from core.metrics import byte_bit_error_rate, psnr, ssim


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    attack: str
    ber: float
    psnr_db: float
    ssim: float
    valid: bool
    header_ok: bool
    crc_ok: bool
    auth_ok: bool
    confidence: float
    registration: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Attack = tuple[
    str,
    Callable[[NDArray[np.uint8]], NDArray[np.uint8]],
    bool,
]


def default_attacks() -> tuple[Attack, ...]:
    return (
        ("No attack", lambda image: image.copy(), False),
        ("JPEG Q=50", lambda image: jpeg_compress(image, 50), False),
        ("Rotation 5°", lambda image: rotate(image, 5.0), True),
        ("Median filter 3×3", lambda image: median_filter(image, 3), False),
        ("AWGN σ=10", lambda image: awgn(image, 10.0, seed=7), False),
        ("Gaussian blur 3×3", lambda image: gaussian_blur(image, 3, 0.8), False),
        ("Scale 0.75×", lambda image: scale(image, 0.75), True),
        ("Center crop 90%", lambda image: crop_and_resize(image, 0.90), True),
    )


def run_image_attack_suite(
    cover: NDArray[np.uint8],
    watermarked: NDArray[np.uint8],
    payload: bytes,
    secret_key: str | bytes,
    config: ImageWatermarkConfig,
    *,
    attacks: tuple[Attack, ...] | None = None,
) -> list[BenchmarkRow]:
    """Apply attacks, extract with soft ECC, and report quality/BER."""

    extractor = ImageWatermarkExtractor(config)
    rows: list[BenchmarkRow] = []
    for name, attack, geometric in attacks or default_attacks():
        attacked = attack(watermarked)
        result = extractor.extract_array(
            attacked,
            secret_key,
            reference=cover if geometric else None,
            expected_payload_bytes=len(payload),
            expected_shape=cover.shape[:2],
        )
        # For geometric attacks, compute quality after the same registration
        # used by extraction so the metric represents recovered coordinates.
        quality_image = attacked
        if geometric:
            from core.registration import align_to_reference

            quality_image = align_to_reference(attacked, cover).image
        rows.append(
            BenchmarkRow(
                attack=name,
                ber=byte_bit_error_rate(payload, result.payload),
                psnr_db=psnr(cover, quality_image),
                ssim=ssim(cover, quality_image),
                valid=result.valid,
                header_ok=result.header_ok,
                crc_ok=result.crc_ok,
                auth_ok=result.auth_ok,
                confidence=result.confidence,
                registration=result.registration_method,
            )
        )
    return rows


def rows_to_markdown(rows: list[BenchmarkRow]) -> str:
    lines = [
        "| Attack | BER | PSNR (dB) | SSIM | Authenticated |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        psnr_text = "∞" if np.isinf(row.psnr_db) else f"{row.psnr_db:.2f}"
        lines.append(
            f"| {row.attack} | {row.ber:.4f} | {psnr_text} | {row.ssim:.4f} | {'✅' if row.valid else '❌'} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--key", required=True)
    parser.add_argument("--message", default="DSP-WATERMARK")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    cover = read_image(args.input)
    config = ImageWatermarkConfig()
    payload = args.message.encode("utf-8")
    watermarked, _ = ImageWatermarker(config).embed_array(
        cover, payload, args.key, utf8=True, collect_debug=False
    )
    rows = run_image_attack_suite(cover, watermarked, payload, args.key, config)
    print(rows_to_markdown(rows))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([row.to_dict() for row in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
