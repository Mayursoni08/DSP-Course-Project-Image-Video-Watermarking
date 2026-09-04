"""Reproducible H.264 and temporal-attack benchmark suite for video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from attacks.compression import compress_video
from attacks.temporal import change_frame_rate, drop_frames
from core.config import VideoWatermarkConfig
from core.embedder import VideoWatermarker
from core.extractor import VideoWatermarkExtractor
from core.metrics import byte_bit_error_rate


@dataclass(frozen=True, slots=True)
class VideoBenchmarkRow:
    attack: str
    ber: float
    valid: bool
    header_ok: bool
    crc_ok: bool
    auth_ok: bool
    confidence: float
    temporal_phase: int | None
    temporal_samples: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


VideoAttack = tuple[str, Callable[[Path, Path], None]]


def default_video_attacks(source_fps: float) -> tuple[VideoAttack, ...]:
    target_fps = max(1.0, source_fps * 0.75)
    return (
        ("H.264 CRF 28", lambda source, output: compress_video(source, output, codec="libx264", crf=28)),
        ("H.265 CRF 30", lambda source, output: compress_video(source, output, codec="libx265", crf=30, preset="fast")),
        ("Drop every 8th frame", lambda source, output: drop_frames(source, output, drop_every=8)),
        (f"Frame-rate conversion {target_fps:.2f} fps", lambda source, output: change_frame_rate(source, output, fps=target_fps)),
    )


def run_video_attack_suite(
    source_video: str | Path,
    payload: bytes,
    secret_key: str | bytes,
    config: VideoWatermarkConfig,
    *,
    output_dir: str | Path | None = None,
) -> tuple[list[VideoBenchmarkRow], Path]:
    """Embed once, apply codec/timing attacks, and extract by temporal voting."""

    source = Path(source_video)
    from core.video_io import probe_video

    info = probe_video(source)
    managed: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        managed = tempfile.TemporaryDirectory(prefix="dsp-watermark-benchmark-")
        root = Path(managed.name)
    else:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

    marked = root / "watermarked.mp4"
    VideoWatermarker(config).embed_file(source, marked, payload, secret_key)
    extractor = VideoWatermarkExtractor(config)
    candidates: list[tuple[str, Path]] = [("No additional attack", marked)]
    for index, (name, attack) in enumerate(default_video_attacks(info.fps), start=1):
        destination = root / f"attack-{index:02d}.mp4"
        attack(marked, destination)
        candidates.append((name, destination))

    rows: list[VideoBenchmarkRow] = []
    for name, path in candidates:
        result = extractor.extract_file(
            path,
            secret_key,
            expected_payload_bytes=len(payload),
            expected_shape=(info.height, info.width),
        )
        rows.append(
            VideoBenchmarkRow(
                attack=name,
                ber=byte_bit_error_rate(payload, result.payload),
                valid=result.valid,
                header_ok=result.header_ok,
                crc_ok=result.crc_ok,
                auth_ok=result.auth_ok,
                confidence=result.confidence,
                temporal_phase=result.temporal_phase,
                temporal_samples=result.temporal_samples,
            )
        )

    # Keep results when the caller supplied a directory. With a managed temp
    # directory, copy only the watermarked stream to a stable NamedTemporaryFile
    # so the return value remains useful until process exit.
    if managed is not None:
        descriptor, stable_name = tempfile.mkstemp(
            prefix="dsp-watermarked-", suffix=".mp4"
        )
        os.close(descriptor)
        stable = Path(stable_name)
        shutil.copy2(marked, stable)
        managed.cleanup()
        return rows, stable
    return rows, marked


def rows_to_markdown(rows: list[VideoBenchmarkRow]) -> str:
    lines = [
        "| Video attack | BER | Authenticated | Phase | Temporal samples |",
        "|---|---:|:---:|---:|---:|",
    ]
    for row in rows:
        phase = "—" if row.temporal_phase is None else str(row.temporal_phase)
        samples = "—" if row.temporal_samples is None else str(row.temporal_samples)
        lines.append(
            f"| {row.attack} | {row.ber:.4f} | {'✅' if row.valid else '❌'} | {phase} | {samples} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--key", required=True)
    parser.add_argument("--message", default="VIDEO-WATERMARK")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    config = VideoWatermarkConfig()
    rows, marked = run_video_attack_suite(
        args.input,
        args.message.encode("utf-8"),
        args.key,
        config,
        output_dir=args.output_dir,
    )
    print(rows_to_markdown(rows))
    print(f"Watermarked stream: {marked}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([row.to_dict() for row in rows], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
