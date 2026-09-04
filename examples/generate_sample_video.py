"""Generate a deterministic 24-frame AVI for video CLI experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from benchmarks.sample_data import synthetic_video_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/sample-input.avi"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (args.size, args.size),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create an MJPEG AVI")
    for frame in synthetic_video_frames(args.frames, args.size):
        writer.write(frame)
    writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
