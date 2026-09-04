from __future__ import annotations

import shutil

import cv2
import pytest

from attacks.compression import compress_video
from attacks.temporal import change_frame_rate, drop_frames
from benchmarks.sample_data import synthetic_video_frames
from core.config import VideoWatermarkConfig
from core.embedder import VideoWatermarker
from core.extractor import VideoWatermarkExtractor


def _write_test_video(path, frame_count: int = 24) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (512, 512)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPEG writer unavailable")
    for frame in synthetic_video_frames(frame_count):
        writer.write(frame)
    writer.release()


@pytest.mark.integration
def test_video_h264_and_temporal_attacks(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg unavailable")
    source = tmp_path / "source.avi"
    marked = tmp_path / "marked.mp4"
    recompressed = tmp_path / "recompressed.mp4"
    recompressed_h265 = tmp_path / "recompressed-h265.mp4"
    dropped = tmp_path / "dropped.mp4"
    rate_changed = tmp_path / "rate-changed.mp4"
    _write_test_video(source)
    key = "video-secret-key-2026"
    payload = b"VIDEO"
    config = VideoWatermarkConfig(crf=18, preset="veryfast", extraction_max_samples=32)
    report = VideoWatermarker(config).embed_file(source, marked, payload, key)
    assert report.frames_processed == 24

    compress_video(marked, recompressed, codec="libx264", crf=28, preset="veryfast")
    compress_video(marked, recompressed_h265, codec="libx265", crf=30, preset="fast")
    drop_frames(marked, dropped, drop_every=8, crf=18)
    change_frame_rate(marked, rate_changed, fps=9.0, crf=18)
    extractor = VideoWatermarkExtractor(config)
    for attacked in (marked, recompressed, recompressed_h265, dropped, rate_changed):
        result = extractor.extract_file(
            attacked,
            key,
            expected_payload_bytes=len(payload),
            expected_shape=(512, 512),
        )
        assert result.valid, (attacked.name, result.message)
        assert result.payload == payload
