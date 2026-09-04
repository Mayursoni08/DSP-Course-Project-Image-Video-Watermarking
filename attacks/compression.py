"""JPEG and FFmpeg-backed video compression attacks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from core.exceptions import MediaIOError

try:  # pragma: no cover - environment dependent
    import ffmpeg  # type: ignore
except ImportError:  # pragma: no cover
    ffmpeg = None


def jpeg_compress(
    image: NDArray[np.generic], quality: int = 50
) -> NDArray[np.uint8]:
    """Round-trip an image through baseline JPEG at a given quality factor."""

    if not 1 <= quality <= 100:
        raise ValueError("quality must be in [1, 100]")
    source = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    success, encoded = cv2.imencode(
        ".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not success:
        raise MediaIOError("OpenCV failed to encode JPEG")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise MediaIOError("OpenCV failed to decode JPEG attack result")
    return decoded


def compress_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    codec: str = "libx264",
    crf: int = 28,
    preset: str = "medium",
    pixel_format: str = "yuv420p",
    copy_audio: bool = True,
) -> None:
    """Transcode a video through H.264/H.265 or another FFmpeg encoder."""

    if not 0 <= crf <= 51:
        raise ValueError("crf must be in [0, 51]")
    if ffmpeg is None or shutil.which("ffmpeg") is None:
        raise MediaIOError("ffmpeg-python and the FFmpeg executable are required")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = ffmpeg.input(str(input_path))
    kwargs: dict[str, Any] = {
        "vcodec": codec,
        "crf": crf,
        "preset": preset,
        "pix_fmt": pixel_format,
        "movflags": "+faststart",
    }
    streams: list[Any] = [source.video]
    if copy_audio:
        try:
            probe = ffmpeg.probe(str(input_path))
            if any(stream.get("codec_type") == "audio" for stream in probe.get("streams", [])):
                streams.append(source.audio)
                kwargs["acodec"] = "aac"
                kwargs["audio_bitrate"] = "192k"
        except Exception:
            pass
    try:
        (
            ffmpeg.output(*streams, str(destination), **kwargs)
            .global_args("-hide_banner", "-loglevel", "error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as error:  # type: ignore[attr-defined]
        stderr = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
        raise MediaIOError(f"FFmpeg compression failed: {stderr}") from error


def h264_compress(
    input_path: str | Path, output_path: str | Path, *, crf: int = 28
) -> None:
    compress_video(input_path, output_path, codec="libx264", crf=crf)


def h265_compress(
    input_path: str | Path, output_path: str | Path, *, crf: int = 30
) -> None:
    compress_video(input_path, output_path, codec="libx265", crf=crf)
