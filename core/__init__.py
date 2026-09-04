"""Public API for the DSP invisible watermarking package."""

from .config import ImageWatermarkConfig, VideoWatermarkConfig
from .embedder import ImageEmbeddingReport, ImageWatermarker, VideoEmbeddingReport, VideoWatermarker
from .extractor import ExtractionResult, ImageWatermarkExtractor, VideoWatermarkExtractor
from .metrics import bit_error_rate, byte_bit_error_rate, image_quality, normalized_correlation, psnr, ssim

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "ExtractionResult",
    "ImageEmbeddingReport",
    "ImageWatermarkConfig",
    "ImageWatermarkExtractor",
    "ImageWatermarker",
    "VideoEmbeddingReport",
    "VideoWatermarkConfig",
    "VideoWatermarkExtractor",
    "VideoWatermarker",
    "bit_error_rate",
    "byte_bit_error_rate",
    "image_quality",
    "normalized_correlation",
    "psnr",
    "ssim",
]
