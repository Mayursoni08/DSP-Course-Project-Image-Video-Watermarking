"""Domain-specific exceptions for the watermarking pipeline."""


class WatermarkError(RuntimeError):
    """Base exception for watermarking failures."""


class CapacityError(WatermarkError):
    """Raised when a payload cannot fit in the available transform carriers."""


class ExtractionError(WatermarkError):
    """Raised when a watermark cannot be decoded or authenticated."""


class MediaIOError(WatermarkError):
    """Raised when an image/video cannot be read, written, or transcoded."""


class ConfigurationError(WatermarkError, ValueError):
    """Raised when a watermark configuration is internally inconsistent."""
