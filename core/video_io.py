"""Video probing, lossless intermediate writing, encoding, and audio remuxing."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .exceptions import MediaIOError

try:  # pragma: no cover - deployment dependent
    import ffmpeg  # type: ignore
except ImportError:  # pragma: no cover
    ffmpeg = None


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool
    duration_seconds: float | None


def probe_video(path: str | Path) -> VideoInfo:
    """Probe essential video metadata with OpenCV and optional ffprobe."""

    source = str(path)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise MediaIOError(f"unable to open video: {source}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise MediaIOError(f"invalid video metadata for: {source}")

    has_audio = False
    duration: float | None = frame_count / fps if frame_count > 0 else None
    if ffmpeg is not None and shutil.which("ffprobe"):
        try:
            metadata: dict[str, Any] = ffmpeg.probe(source)
            has_audio = any(stream.get("codec_type") == "audio" for stream in metadata.get("streams", []))
            raw_duration = metadata.get("format", {}).get("duration")
            if raw_duration is not None:
                duration = float(raw_duration)
        except Exception:
            pass
    return VideoInfo(width, height, fps, frame_count, has_audio, duration)


def create_lossless_writer(
    directory: str | Path, width: int, height: int, fps: float
) -> tuple[cv2.VideoWriter, Path]:
    """Create a lossless intermediate writer, with a portable MJPEG fallback."""

    root = Path(directory)
    candidates = (("FFV1", root / "watermarked_intermediate.mkv"), ("MJPG", root / "watermarked_intermediate.avi"))
    for code, path in candidates:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*code), fps, (width, height)
        )
        if writer.isOpened():
            return writer, path
        writer.release()
    raise MediaIOError("OpenCV could not create an FFV1 or MJPEG intermediate")


def encode_and_remux(
    intermediate: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    codec: str,
    crf: int,
    preset: str,
    pixel_format: str,
    preserve_audio: bool,
    keyframe_interval_seconds: float | None = None,
    gop_size: int | None = None,
) -> None:
    """Encode an intermediate video and remux source audio through FFmpeg."""

    intermediate_path = Path(intermediate)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg is None or shutil.which("ffmpeg") is None:
        # Retain a playable output rather than silently failing. The extension
        # is adjusted because copying MKV bytes into an MP4 path is misleading.
        fallback = output_path.with_suffix(intermediate_path.suffix)
        shutil.copy2(intermediate_path, fallback)
        raise MediaIOError(
            f"FFmpeg is unavailable; lossless intermediate preserved at {fallback}"
        )

    info = probe_video(source)
    processed = ffmpeg.input(str(intermediate_path))
    output_kwargs: dict[str, Any] = {
        "vcodec": codec,
        "crf": crf,
        "preset": preset,
        "pix_fmt": pixel_format,
        "movflags": "+faststart",
    }
    if keyframe_interval_seconds is not None:
        if keyframe_interval_seconds <= 0:
            raise ValueError("keyframe_interval_seconds must be positive")
        output_kwargs["force_key_frames"] = (
            f"expr:gte(t,n_forced*{keyframe_interval_seconds:.9f})"
        )
    if gop_size is not None:
        if gop_size < 1:
            raise ValueError("gop_size must be positive")
        output_kwargs["g"] = int(gop_size)

    streams: list[Any] = [processed.video]
    if preserve_audio and info.has_audio:
        original = ffmpeg.input(str(source))
        streams.append(original.audio)
        output_kwargs.update({"acodec": "aac", "audio_bitrate": "192k"})
    try:
        (
            ffmpeg.output(*streams, str(output_path), **output_kwargs)
            .global_args("-hide_banner", "-loglevel", "error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as error:  # type: ignore[attr-defined]
        stderr = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
        raise MediaIOError(f"FFmpeg encoding failed: {stderr}") from error


def temporary_video_directory() -> tempfile.TemporaryDirectory[str]:
    """Return a managed temporary directory for intermediate streams."""

    return tempfile.TemporaryDirectory(prefix="dsp-watermark-")
