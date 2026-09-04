"""Generate reproducible README assets and benchmark reports.

Run from the repository root::

    python generate_assets.py

No external image download is required; the script creates a deterministic,
license-free synthetic cover with gradients, texture, edges, and geometric
features. All generated files are written below ``assets/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from attacks.compression import jpeg_compress
from benchmarks.run_benchmarks import rows_to_markdown, run_image_attack_suite
from benchmarks.run_video_benchmarks import (
    rows_to_markdown as video_rows_to_markdown,
    run_video_attack_suite,
)
from benchmarks.sample_data import synthetic_cover, synthetic_video_frames
from core.carriers import detail_bands
from core.config import ImageWatermarkConfig, VideoWatermarkConfig
from core.embedder import ImageWatermarker
from core.extractor import ImageWatermarkExtractor
from core.media import split_luminance, write_image
from core.metrics import byte_bit_error_rate, psnr, ssim
from core.transforms import multilevel_dwt2, normalize_for_display, pad_to_multiple

DEFAULT_KEY = "docs-only-demo-key-2026"
DEFAULT_MESSAGE = "DSP-WATERMARK-2026"


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_comparison(
    cover: np.ndarray,
    watermarked: np.ndarray,
    output: Path,
    *,
    psnr_db: float,
    ssim_value: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(_rgb(cover))
    axes[0].set_title("Original cover")
    axes[1].imshow(_rgb(watermarked))
    axes[1].set_title(f"Watermarked\nPSNR {psnr_db:.2f} dB · SSIM {ssim_value:.4f}")
    for axis in axes:
        axis.axis("off")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_residual(cover: np.ndarray, watermarked: np.ndarray, output: Path) -> None:
    residual = watermarked.astype(np.float64) - cover.astype(np.float64)
    magnified = np.clip(127.5 + 10.0 * residual, 0.0, 255.0).astype(np.uint8)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.imshow(_rgb(magnified))
    maximum = float(np.max(np.abs(residual)))
    rms = float(np.sqrt(np.mean(residual * residual)))
    axis.set_title(f"Signed residual ×10 (mid-gray = zero)\nmax |Δ|={maximum:.0f}, RMS={rms:.3f}")
    axis.axis("off")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_dwt_subbands(
    cover: np.ndarray,
    watermarked: np.ndarray,
    config: ImageWatermarkConfig,
    output: Path,
) -> None:
    cover_y, _ = split_luminance(cover)
    marked_y, _ = split_luminance(watermarked)
    cover_pad, _ = pad_to_multiple(cover_y, config.spatial_multiple)
    marked_pad, _ = pad_to_multiple(marked_y, config.spatial_multiple)
    ll, details = multilevel_dwt2(
        cover_pad, config.dwt_level, config.wavelet, config.wavelet_mode
    )
    _, marked_details = multilevel_dwt2(
        marked_pad, config.dwt_level, config.wavelet, config.wavelet_mode
    )
    bands = detail_bands(details[-1])
    marked_bands = detail_bands(marked_details[-1])

    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes[0, 0].imshow(normalize_for_display(ll), cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"LL{config.dwt_level} approximation")
    axes[1, 0].axis("off")
    for column, name in enumerate(("LH", "HL", "HH"), start=1):
        axes[0, column].imshow(normalize_for_display(bands[name]), cmap="gray", vmin=0, vmax=255)
        suffix = " · embedded" if name in config.subbands else " · untouched"
        axes[0, column].set_title(f"{name}{config.dwt_level}{suffix}")
        difference = np.abs(marked_bands[name] - bands[name])
        upper = max(float(np.percentile(difference, 99.5)), 1e-9)
        image = axes[1, column].imshow(difference, cmap="magma", vmin=0.0, vmax=upper)
        axes[1, column].set_title(f"|Δ {name}{config.dwt_level}| coefficients")
        figure.colorbar(image, ax=axes[1, column], fraction=0.045, pad=0.03)
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle("Level-3 2D-DWT sub-bands and keyed embedding support", fontsize=15)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_compression_curve(
    cover: np.ndarray,
    watermarked: np.ndarray,
    payload: bytes,
    key: str,
    config: ImageWatermarkConfig,
    output: Path,
) -> list[dict[str, float | int | bool]]:
    qualities = [100, 95, 90, 80, 70, 60, 50, 40, 30]
    extractor = ImageWatermarkExtractor(config)
    rows: list[dict[str, float | int | bool]] = []
    for quality in qualities:
        attacked = jpeg_compress(watermarked, quality)
        result = extractor.extract_array(
            attacked,
            key,
            expected_payload_bytes=len(payload),
            expected_shape=cover.shape[:2],
        )
        rows.append(
            {
                "quality": quality,
                "psnr_db": psnr(cover, attacked),
                "ssim": ssim(cover, attacked),
                "ber": byte_bit_error_rate(payload, result.payload),
                "authenticated": result.valid,
            }
        )

    figure, axis_quality = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    x = [int(row["quality"]) for row in rows]
    axis_quality.plot(x, [float(row["psnr_db"]) for row in rows], marker="o", label="PSNR (dB)")
    axis_quality.plot(x, [100.0 * float(row["ssim"]) for row in rows], marker="s", label="SSIM ×100")
    axis_quality.set_xlabel("JPEG quality factor")
    axis_quality.set_ylabel("Quality metric")
    axis_quality.set_xlim(max(x), min(x))
    axis_quality.grid(True, alpha=0.25)

    axis_ber = axis_quality.twinx()
    axis_ber.plot(x, [float(row["ber"]) for row in rows], marker="^", linestyle="--", label="Payload BER")
    axis_ber.set_ylabel("Bit error rate")
    axis_ber.set_ylim(-0.01, 1.0)
    handles_a, labels_a = axis_quality.get_legend_handles_labels()
    handles_b, labels_b = axis_ber.get_legend_handles_labels()
    axis_quality.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    axis_quality.set_title("Image quality and authenticated payload recovery after JPEG")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return rows


def generate_video_benchmark_assets(
    output_dir: Path, *, size: int = 512
) -> dict[str, object]:
    """Generate machine-readable video robustness results using real FFmpeg attacks."""

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required for --include-video")
    key = "docs-only-video-key-2026"
    payload = b"VIDEO"
    config = VideoWatermarkConfig(preset="veryfast")
    with tempfile.TemporaryDirectory(prefix="dsp-watermark-assets-") as directory:
        root = Path(directory)
        source = root / "source.avi"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (size, size)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the synthetic video fixture")
        for frame in synthetic_video_frames(24, size):
            writer.write(frame)
        writer.release()
        rows, _ = run_video_attack_suite(
            source, payload, key, config, output_dir=root / "results"
        )
    data: dict[str, object] = {
        "fixture": {
            "description": "deterministic 24-frame synthetic sequence",
            "width": size,
            "height": size,
            "fps": 12.0,
            "payload": payload.decode("ascii"),
        },
        "config": config.to_dict(),
        "rows": [row.to_dict() for row in rows],
    }
    (output_dir / "video_robustness.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "video_robustness.md").write_text(
        video_rows_to_markdown(rows) + "\n", encoding="utf-8"
    )
    return data


def generate_assets(
    output_dir: Path, *, size: int = 512, include_video: bool = False
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ImageWatermarkConfig()
    cover = synthetic_cover(size)
    payload = DEFAULT_MESSAGE.encode("utf-8")
    watermarked, report = ImageWatermarker(config).embed_array(
        cover, payload, DEFAULT_KEY, utf8=True, collect_debug=True
    )
    quality_psnr = psnr(cover, watermarked)
    quality_ssim = ssim(cover, watermarked)

    result = ImageWatermarkExtractor(config).extract_array(
        watermarked,
        DEFAULT_KEY,
        expected_payload_bytes=len(payload),
        expected_shape=cover.shape[:2],
    )
    if not result.valid or result.payload != payload:
        raise RuntimeError("asset-generation self-check failed to recover the embedded payload")

    write_image(output_dir / "original.png", cover)
    write_image(output_dir / "watermarked.png", watermarked)
    _save_comparison(
        cover,
        watermarked,
        output_dir / "original_vs_watermarked.png",
        psnr_db=quality_psnr,
        ssim_value=quality_ssim,
    )
    _save_residual(cover, watermarked, output_dir / "residual_x10.png")
    _save_dwt_subbands(cover, watermarked, config, output_dir / "dwt_subbands.png")
    compression_rows = _save_compression_curve(
        cover,
        watermarked,
        payload,
        DEFAULT_KEY,
        config,
        output_dir / "compression_curve.png",
    )

    benchmark_rows = run_image_attack_suite(
        cover, watermarked, payload, DEFAULT_KEY, config
    )
    benchmark_json = [row.to_dict() for row in benchmark_rows]
    (output_dir / "robustness.json").write_text(
        json.dumps(benchmark_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "robustness.md").write_text(
        rows_to_markdown(benchmark_rows) + "\n", encoding="utf-8"
    )
    with (output_dir / "robustness.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(benchmark_json[0]))
        writer.writeheader()
        writer.writerows(benchmark_json)
    (output_dir / "compression_curve.json").write_text(
        json.dumps(compression_rows, indent=2) + "\n", encoding="utf-8"
    )

    summary: dict[str, object] = {
        "message": DEFAULT_MESSAGE,
        "image_size": list(cover.shape),
        "config": config.to_dict(),
        "embedding_report": report.to_dict(),
        "baseline": {
            "psnr_db": quality_psnr,
            "ssim": quality_ssim,
            "authenticated": result.valid,
        },
        "benchmark": benchmark_json,
    }
    if include_video:
        summary["video_benchmark"] = generate_video_benchmark_assets(
            output_dir, size=size
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="also run the real H.264/frame-timing benchmark (requires FFmpeg)",
    )
    args = parser.parse_args()
    summary = generate_assets(
        args.output_dir, size=args.size, include_video=args.include_video
    )
    baseline = summary["baseline"]
    print(
        f"Generated assets in {args.output_dir} | "
        f"PSNR={baseline['psnr_db']:.2f} dB | SSIM={baseline['ssim']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
