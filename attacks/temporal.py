"""FFmpeg-backed temporal attacks: frame dropping and frame-rate conversion."""

from __future__ import annotations

import shutil
from pathlib import Path

from core.exceptions import MediaIOError

try:  # pragma: no cover
    import ffmpeg  # type: ignore
except ImportError:  # pragma: no cover
    ffmpeg = None


def _require_ffmpeg() -> None:
    if ffmpeg is None or shutil.which("ffmpeg") is None:
        raise MediaIOError("ffmpeg-python and the FFmpeg executable are required")


def drop_frames(
    input_path: str | Path,
    output_path: str | Path,
    *,
    drop_every: int = 10,
    codec: str = "libx264",
    crf: int = 20,
) -> None:
    """Drop every Nth frame and regenerate monotonic presentation timestamps."""

    _require_ffmpeg()
    if drop_every < 2:
        raise ValueError("drop_every must be >= 2")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = ffmpeg.input(str(input_path))
    video = source.video.filter("select", f"not(eq(mod(n,{drop_every}),0))").filter("setpts", "N/FRAME_RATE/TB")
    try:
        (
            ffmpeg.output(video, str(destination), vcodec=codec, crf=crf, preset="medium", pix_fmt="yuv420p")
            .global_args("-hide_banner", "-loglevel", "error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as error:  # type: ignore[attr-defined]
        stderr = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
        raise MediaIOError(f"frame-drop attack failed: {stderr}") from error


def change_frame_rate(
    input_path: str | Path,
    output_path: str | Path,
    *,
    fps: float,
    codec: str = "libx264",
    crf: int = 20,
) -> None:
    """Resample a stream to a target frame rate."""

    _require_ffmpeg()
    if fps <= 0:
        raise ValueError("fps must be positive")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = ffmpeg.input(str(input_path))
    video = source.video.filter("fps", fps=fps)
    try:
        (
            ffmpeg.output(video, str(destination), vcodec=codec, crf=crf, preset="medium", pix_fmt="yuv420p")
            .global_args("-hide_banner", "-loglevel", "error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as error:  # type: ignore[attr-defined]
        stderr = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
        raise MediaIOError(f"frame-rate attack failed: {stderr}") from error
