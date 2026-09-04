"""Minimal API example using the generated documentation cover."""

from __future__ import annotations

from pathlib import Path

from core import ImageWatermarkExtractor, ImageWatermarker, image_quality
from core.media import read_image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "assets" / "original.png"
    output = root / "output" / "quickstart-watermarked.png"
    key = "replace-this-demo-key"
    payload = b"asset-id:42"

    report = ImageWatermarker().embed_file(source, output, payload, key)
    result = ImageWatermarkExtractor().extract_file(
        output, key, expected_payload_bytes=len(payload)
    )
    quality = image_quality(read_image(source), read_image(output))

    print(
        "Embedding:",
        {
            "algorithm": report.algorithm,
            "payload_bytes": report.payload_bytes,
            "capacity_bytes": report.capacity_bytes,
            "carriers_used": report.carriers_used,
        },
    )
    print("Extraction:", result.to_dict())
    print("Quality:", quality.to_dict())
    if not result.valid or result.payload != payload:
        raise RuntimeError("quickstart round-trip failed")


if __name__ == "__main__":
    main()
