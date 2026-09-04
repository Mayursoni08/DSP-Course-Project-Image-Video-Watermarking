"""Deterministic synthetic covers used by documentation and regression tests.

The generator deliberately mixes smooth gradients, textured regions, edges,
text, and fine detail. This produces a license-free benchmark cover while
exercising the frequency-selective watermarking path more realistically than a
flat test chart.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def synthetic_cover(size: int = 512, *, seed: int = 2026) -> NDArray[np.uint8]:
    """Return a deterministic BGR cover image of shape ``(size, size, 3)``."""

    if size < 128:
        raise ValueError("size must be at least 128 pixels")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    xn = x / max(size - 1, 1)
    yn = y / max(size - 1, 1)

    # Broad, photographic-looking illumination and chroma fields.
    b = 45.0 + 95.0 * xn + 32.0 * np.sin(2.0 * np.pi * (1.7 * yn + 0.25 * xn))
    g = 55.0 + 100.0 * yn + 25.0 * np.cos(2.0 * np.pi * (1.3 * xn - 0.2 * yn))
    r = 72.0 + 80.0 * (1.0 - xn) + 28.0 * np.sin(2.0 * np.pi * (xn + yn))
    image = np.dstack((b, g, r))

    # Correlated fine texture rather than independent salt noise.
    texture = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=max(0.7, size / 900.0))
    texture /= float(np.std(texture) + 1e-9)
    image += texture[..., None] * 8.0

    output = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    scale = size / 512.0

    def point(px: int, py: int) -> tuple[int, int]:
        return int(round(px * scale)), int(round(py * scale))

    cv2.rectangle(output, point(42, 58), point(220, 234), (210, 92, 48), -1)
    cv2.rectangle(output, point(60, 76), point(202, 216), (235, 168, 92), 3)
    cv2.circle(output, point(365, 145), int(round(92 * scale)), (56, 178, 224), -1)
    cv2.circle(output, point(365, 145), int(round(61 * scale)), (215, 235, 245), 4)
    polygon = np.array(
        [point(262, 290), point(457, 270), point(435, 438), point(292, 462)],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(output, polygon, (72, 121, 210))
    cv2.polylines(output, [polygon], True, (235, 235, 235), 3)

    # Repeated lines provide stable geometric features for ORB registration.
    for index in range(7):
        y0 = int(round((280 + 22 * index) * scale))
        cv2.line(
            output,
            (int(round(35 * scale)), y0),
            (int(round((220 - 7 * index) * scale)), y0),
            (232, 230, 220),
            max(1, int(round(2 * scale))),
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        "DSP",
        point(72, 170),
        cv2.FONT_HERSHEY_DUPLEX,
        2.0 * scale,
        (24, 34, 42),
        max(1, int(round(3 * scale))),
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "INVISIBLE WATERMARK",
        point(40, 492),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82 * scale,
        (245, 245, 245),
        max(1, int(round(2 * scale))),
        cv2.LINE_AA,
    )
    return output


def synthetic_video_frames(
    count: int = 24,
    size: int = 512,
    *,
    seed: int = 2026,
) -> list[NDArray[np.uint8]]:
    """Create a deterministic moving sequence for video integration tests."""

    if count < 1:
        raise ValueError("count must be positive")
    base = synthetic_cover(size, seed=seed)
    frames: list[NDArray[np.uint8]] = []
    scale = size / 512.0
    for index in range(count):
        frame = base.copy()
        phase = 2.0 * np.pi * index / max(count, 1)
        center = (
            int(round((255 + 72 * np.cos(phase)) * scale)),
            int(round((245 + 46 * np.sin(phase)) * scale)),
        )
        cv2.circle(frame, center, max(3, int(round(13 * scale))), (18, 236, 94), -1)
        cv2.putText(
            frame,
            f"t={index:02d}",
            (int(round(395 * scale)), int(round(492 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale,
            (20, 20, 20),
            max(1, int(round(scale))),
            cv2.LINE_AA,
        )
        frames.append(frame)
    return frames
