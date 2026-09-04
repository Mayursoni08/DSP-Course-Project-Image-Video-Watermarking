"""Command-line interface for image and video invisible watermarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from attacks.compression import compress_video
from benchmarks.run_benchmarks import rows_to_markdown, run_image_attack_suite
from benchmarks.run_video_benchmarks import (
    rows_to_markdown as video_rows_to_markdown,
    run_video_attack_suite,
)
from core.config import ImageWatermarkConfig, VideoWatermarkConfig
from core.embedder import ImageWatermarker, VideoWatermarker
from core.exceptions import WatermarkError
from core.extractor import ImageWatermarkExtractor, VideoWatermarkExtractor
from core.media import read_image
from core.metrics import image_quality
from generate_assets import generate_assets

MANIFEST_SCHEMA = "dsp-invisible-watermark/manifest/v1"
DEFAULT_KEY_ENV = "DSP_WATERMARK_KEY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return value


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_key(args: argparse.Namespace) -> str | bytes:
    supplied = [
        getattr(args, "key", None) is not None,
        getattr(args, "key_file", None) is not None,
    ]
    if sum(supplied) > 1:
        raise ValueError("use only one of --key or --key-file")
    if getattr(args, "key", None) is not None:
        key: str | bytes = args.key
    elif getattr(args, "key_file", None) is not None:
        try:
            key = args.key_file.read_bytes().rstrip(b"\r\n")
        except OSError as error:
            raise ValueError(f"unable to read key file: {args.key_file}") from error
    else:
        environment_name = getattr(args, "key_env", DEFAULT_KEY_ENV)
        value = os.environ.get(environment_name)
        if value is None:
            raise ValueError(
                f"no key supplied; set {environment_name}, use --key-file, or use --key"
            )
        key = value
    if len(key.encode("utf-8") if isinstance(key, str) else key) < 8:
        raise ValueError("the secret key must contain at least 8 bytes")
    return key


def _payload_from_args(args: argparse.Namespace) -> tuple[bytes, bool]:
    if getattr(args, "message", None) is not None:
        return args.message.encode("utf-8"), True
    if getattr(args, "payload_file", None) is not None:
        try:
            return args.payload_file.read_bytes(), False
        except OSError as error:
            raise ValueError(f"unable to read payload: {args.payload_file}") from error
    raise ValueError("one of --message or --payload-file is required")


def _add_key_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--key", help="secret key (prefer the environment or a protected key file)")
    parser.add_argument("--key-file", type=Path, help="file containing the secret key")
    parser.add_argument(
        "--key-env",
        default=DEFAULT_KEY_ENV,
        help=f"environment variable used when no key flag is supplied (default: {DEFAULT_KEY_ENV})",
    )


def _add_payload_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message", help="UTF-8 text payload")
    group.add_argument("--payload-file", type=Path, help="arbitrary binary payload")


def _add_image_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--algorithm", choices=("stdm", "svd-qim"))
    parser.add_argument("--wavelet")
    parser.add_argument("--wavelet-mode")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--dwt-level", type=int)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--subbands", help="comma-separated subset of LH,HL,HH")
    parser.add_argument("--chip-length", type=int, dest="spread_chip_length")
    parser.add_argument(
        "--coefficients-per-block",
        type=int,
        dest="spread_coefficients_per_block",
    )
    parser.add_argument("--singular-index", type=int)
    parser.add_argument("--ecc", choices=("convolutional", "repetition"), dest="ecc_mode")
    parser.add_argument("--header-repetition", type=int)
    parser.add_argument("--payload-repetition", type=int)
    parser.add_argument("--auth-tag-bytes", type=int)


def _image_config_from_args(
    args: argparse.Namespace,
    base: ImageWatermarkConfig | None = None,
) -> ImageWatermarkConfig:
    values = (base or ImageWatermarkConfig()).to_dict()
    valid_names = {field.name for field in fields(ImageWatermarkConfig)}
    for name in valid_names:
        if name == "luminance_only":
            continue
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    if isinstance(values.get("subbands"), str):
        values["subbands"] = tuple(
            part.strip().upper() for part in values["subbands"].split(",") if part.strip()
        )
    return ImageWatermarkConfig.from_dict(values)


def _add_video_config_arguments(parser: argparse.ArgumentParser) -> None:
    _add_image_config_arguments(parser)
    parser.add_argument("--temporal-window", type=int)
    parser.add_argument("--group-interval", type=int, dest="embed_group_interval")
    parser.add_argument("--codec")
    parser.add_argument("--crf", type=int)
    parser.add_argument("--preset")
    parser.add_argument("--pixel-format")
    parser.add_argument("--max-samples", type=int, dest="extraction_max_samples")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-keyframe-alignment", action="store_true")
    parser.add_argument("--no-phase-search", action="store_true")


def _video_config_from_args(
    args: argparse.Namespace,
    base: VideoWatermarkConfig | None = None,
) -> VideoWatermarkConfig:
    existing = base or VideoWatermarkConfig()
    image_config = _image_config_from_args(args, existing.image)
    values = existing.to_dict()
    values["image"] = image_config.to_dict()
    for name in (
        "temporal_window",
        "embed_group_interval",
        "codec",
        "crf",
        "preset",
        "pixel_format",
        "extraction_max_samples",
    ):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    if getattr(args, "no_audio", False):
        values["preserve_audio"] = False
    if getattr(args, "no_keyframe_alignment", False):
        values["align_output_keyframes"] = False
    if getattr(args, "no_phase_search", False):
        values["search_temporal_phase"] = False
    return VideoWatermarkConfig.from_dict(values)


def _manifest_config(
    path: Path | None,
    media_type: str,
) -> tuple[dict[str, Any] | None, ImageWatermarkConfig | VideoWatermarkConfig | None]:
    if path is None:
        return None, None
    manifest = _read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema in {path}")
    if manifest.get("media_type") != media_type:
        raise ValueError(f"manifest media_type is not {media_type!r}")
    raw_config = manifest.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("manifest is missing a config object")
    if media_type == "image":
        return manifest, ImageWatermarkConfig.from_dict(raw_config)
    return manifest, VideoWatermarkConfig.from_dict(raw_config)


def _image_embed(args: argparse.Namespace) -> int:
    key = _resolve_key(args)
    payload, utf8 = _payload_from_args(args)
    config = _image_config_from_args(args)
    report = ImageWatermarker(config).embed_file(
        args.input, args.output, payload, key, utf8=utf8
    )
    cover = read_image(args.input)
    marked = read_image(args.output)
    quality = image_quality(cover, marked)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "media_type": "image",
        "config": config.to_dict(),
        "payload_bytes": len(payload),
        "payload_encoding": "utf-8" if utf8 else "binary",
        "expected_shape": list(cover.shape[:2]),
        "cover_sha256": _sha256(args.input),
        "watermarked_sha256": _sha256(args.output),
        "embedding_report": report.to_dict(),
        "quality": quality.to_dict(),
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".wm.json")
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), **manifest}, indent=2))
    return 0


def _image_extract(args: argparse.Namespace) -> int:
    key = _resolve_key(args)
    manifest, base_config = _manifest_config(args.manifest, "image")
    config = _image_config_from_args(
        args, base_config if isinstance(base_config, ImageWatermarkConfig) else None
    )
    expected_length = args.expected_bytes
    expected_shape: tuple[int, int] | None = None
    if manifest is not None:
        if expected_length is None:
            raw_length = manifest.get("payload_bytes")
            expected_length = int(raw_length) if raw_length is not None else None
        raw_shape = manifest.get("expected_shape")
        if isinstance(raw_shape, list) and len(raw_shape) == 2:
            expected_shape = (int(raw_shape[0]), int(raw_shape[1]))
    result = ImageWatermarkExtractor(config).extract_file(
        args.input,
        key,
        reference_path=args.reference,
        expected_payload_bytes=expected_length,
        expected_shape=expected_shape,
    )
    if args.output_payload is not None:
        args.output_payload.parent.mkdir(parents=True, exist_ok=True)
        args.output_payload.write_bytes(result.payload)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.valid else 2


def _image_capacity(args: argparse.Namespace) -> int:
    config = _image_config_from_args(args)
    capacity = ImageWatermarker(config).capacity_for_shape((args.height, args.width))
    print(json.dumps({"width": args.width, "height": args.height, "capacity_bytes": capacity, "config": config.to_dict()}, indent=2))
    return 0


def _image_benchmark(args: argparse.Namespace) -> int:
    key = _resolve_key(args)
    payload, utf8 = _payload_from_args(args)
    config = _image_config_from_args(args)
    cover = read_image(args.input)
    watermarked, report = ImageWatermarker(config).embed_array(
        cover, payload, key, utf8=utf8, collect_debug=False
    )
    rows = run_image_attack_suite(cover, watermarked, payload, key, config)
    markdown = rows_to_markdown(rows)
    print(markdown)
    if args.json_output:
        _write_json(
            args.json_output,
            {
                "config": config.to_dict(),
                "embedding_report": report.to_dict(),
                "rows": [row.to_dict() for row in rows],
            },
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown + "\n", encoding="utf-8")
    return 0


def _video_embed(args: argparse.Namespace) -> int:
    key = _resolve_key(args)
    payload, utf8 = _payload_from_args(args)
    config = _video_config_from_args(args)
    report = VideoWatermarker(config).embed_file(
        args.input, args.output, payload, key, utf8=utf8
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "media_type": "video",
        "config": config.to_dict(),
        "payload_bytes": len(payload),
        "payload_encoding": "utf-8" if utf8 else "binary",
        "expected_shape": [report.height, report.width],
        "source_sha256": _sha256(args.input),
        "watermarked_sha256": _sha256(args.output),
        "embedding_report": report.to_dict(),
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".wm.json")
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), **manifest}, indent=2))
    return 0


def _video_extract(args: argparse.Namespace) -> int:
    key = _resolve_key(args)
    manifest, base_config = _manifest_config(args.manifest, "video")
    config = _video_config_from_args(
        args, base_config if isinstance(base_config, VideoWatermarkConfig) else None
    )
    expected_length = args.expected_bytes
    expected_shape: tuple[int, int] | None = None
    if manifest is not None:
        if expected_length is None:
            raw_length = manifest.get("payload_bytes")
            expected_length = int(raw_length) if raw_length is not None else None
        raw_shape = manifest.get("expected_shape")
        if isinstance(raw_shape, list) and len(raw_shape) == 2:
            expected_shape = (int(raw_shape[0]), int(raw_shape[1]))
    result = VideoWatermarkExtractor(config).extract_file(
        args.input,
        key,
        expected_payload_bytes=expected_length,
        expected_shape=expected_shape,
    )
    if args.output_payload is not None:
        args.output_payload.parent.mkdir(parents=True, exist_ok=True)
        args.output_payload.write_bytes(result.payload)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.valid else 2


def _video_benchmark(args: argparse.Namespace) -> int:
    key = _resolve_key(args)
    payload, _ = _payload_from_args(args)
    config = _video_config_from_args(args)
    rows, marked = run_video_attack_suite(
        args.input, payload, key, config, output_dir=args.output_dir
    )
    markdown = video_rows_to_markdown(rows)
    print(markdown)
    print(f"Watermarked stream: {marked}")
    if args.json_output:
        _write_json(
            args.json_output,
            {
                "config": config.to_dict(),
                "watermarked_video": str(marked),
                "rows": [row.to_dict() for row in rows],
            },
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown + "\n", encoding="utf-8")
    return 0


def _video_compress(args: argparse.Namespace) -> int:
    compress_video(
        args.input,
        args.output,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        pixel_format=args.pixel_format,
        copy_audio=not args.no_audio,
    )
    print(json.dumps({"input": str(args.input), "output": str(args.output), "codec": args.codec, "crf": args.crf}, indent=2))
    return 0


def _assets(args: argparse.Namespace) -> int:
    summary = generate_assets(
        args.output_dir, size=args.size, include_video=args.include_video
    )
    print(json.dumps(summary["baseline"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsp-watermark",
        description="Keyed DWT-DCT-STDM/SVD invisible watermarking for images and video.",
    )
    parser.add_argument("--version", action="version", version="dsp-invisible-watermark 1.0.0")
    domains = parser.add_subparsers(dest="domain", required=True)

    image = domains.add_parser("image", help="image watermark operations")
    image_commands = image.add_subparsers(dest="operation", required=True)

    image_embed = image_commands.add_parser("embed", help="embed a payload in an image")
    image_embed.add_argument("--input", required=True, type=Path)
    image_embed.add_argument("--output", required=True, type=Path)
    image_embed.add_argument("--manifest", type=Path)
    _add_payload_arguments(image_embed)
    _add_key_arguments(image_embed)
    _add_image_config_arguments(image_embed)
    image_embed.set_defaults(handler=_image_embed)

    image_extract = image_commands.add_parser("extract", help="blind or registered image extraction")
    image_extract.add_argument("--input", required=True, type=Path)
    image_extract.add_argument("--manifest", type=Path)
    image_extract.add_argument("--reference", type=Path, help="optional original cover for geometric registration")
    image_extract.add_argument("--expected-bytes", type=int)
    image_extract.add_argument("--output-payload", type=Path)
    _add_key_arguments(image_extract)
    _add_image_config_arguments(image_extract)
    image_extract.set_defaults(handler=_image_extract)

    image_capacity = image_commands.add_parser("capacity", help="calculate payload capacity")
    image_capacity.add_argument("--width", required=True, type=int)
    image_capacity.add_argument("--height", required=True, type=int)
    _add_image_config_arguments(image_capacity)
    image_capacity.set_defaults(handler=_image_capacity)

    image_benchmark = image_commands.add_parser("benchmark", help="run the spatial/compression attack suite")
    image_benchmark.add_argument("--input", required=True, type=Path)
    image_benchmark.add_argument("--json-output", type=Path)
    image_benchmark.add_argument("--markdown-output", type=Path)
    _add_payload_arguments(image_benchmark)
    _add_key_arguments(image_benchmark)
    _add_image_config_arguments(image_benchmark)
    image_benchmark.set_defaults(handler=_image_benchmark)

    video = domains.add_parser("video", help="video watermark operations")
    video_commands = video.add_subparsers(dest="operation", required=True)

    video_embed = video_commands.add_parser("embed", help="embed a payload using temporal Haar groups")
    video_embed.add_argument("--input", required=True, type=Path)
    video_embed.add_argument("--output", required=True, type=Path)
    video_embed.add_argument("--manifest", type=Path)
    _add_payload_arguments(video_embed)
    _add_key_arguments(video_embed)
    _add_video_config_arguments(video_embed)
    video_embed.set_defaults(handler=_video_embed)

    video_extract = video_commands.add_parser("extract", help="extract using temporal voting and phase search")
    video_extract.add_argument("--input", required=True, type=Path)
    video_extract.add_argument("--manifest", type=Path)
    video_extract.add_argument("--expected-bytes", type=int)
    video_extract.add_argument("--output-payload", type=Path)
    _add_key_arguments(video_extract)
    _add_video_config_arguments(video_extract)
    video_extract.set_defaults(handler=_video_extract)

    video_benchmark = video_commands.add_parser("benchmark", help="run H.264, frame-drop, and frame-rate attacks")
    video_benchmark.add_argument("--input", required=True, type=Path)
    video_benchmark.add_argument("--output-dir", type=Path)
    video_benchmark.add_argument("--json-output", type=Path)
    video_benchmark.add_argument("--markdown-output", type=Path)
    _add_payload_arguments(video_benchmark)
    _add_key_arguments(video_benchmark)
    _add_video_config_arguments(video_benchmark)
    video_benchmark.set_defaults(handler=_video_benchmark)

    video_compress = video_commands.add_parser("compress", help="apply an FFmpeg H.264/H.265 compression attack")
    video_compress.add_argument("--input", required=True, type=Path)
    video_compress.add_argument("--output", required=True, type=Path)
    video_compress.add_argument("--codec", choices=("libx264", "libx265"), default="libx264")
    video_compress.add_argument("--crf", type=int, default=28)
    video_compress.add_argument("--preset", default="medium")
    video_compress.add_argument("--pixel-format", default="yuv420p")
    video_compress.add_argument("--no-audio", action="store_true")
    video_compress.set_defaults(handler=_video_compress)

    assets = domains.add_parser("assets", help="generate documentation figures and benchmark files")
    assets.add_argument("--output-dir", type=Path, default=Path("assets"))
    assets.add_argument("--size", type=int, default=512)
    assets.add_argument("--include-video", action="store_true")
    assets.set_defaults(handler=_assets)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (WatermarkError, ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
