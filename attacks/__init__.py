"""Reusable image and video attack simulations."""

from .compression import compress_video, h264_compress, h265_compress, jpeg_compress
from .spatial import awgn, crop_and_resize, gaussian_blur, median_filter, rotate, salt_and_pepper, scale
from .temporal import change_frame_rate, drop_frames

__all__ = [
    "awgn",
    "change_frame_rate",
    "compress_video",
    "crop_and_resize",
    "drop_frames",
    "gaussian_blur",
    "h264_compress",
    "h265_compress",
    "jpeg_compress",
    "median_filter",
    "rotate",
    "salt_and_pepper",
    "scale",
]
