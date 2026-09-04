from __future__ import annotations

import pytest

from benchmarks.sample_data import synthetic_cover
from core.config import ImageWatermarkConfig
from core.embedder import ImageWatermarker


@pytest.fixture(scope="session")
def cover():
    return synthetic_cover(512)


@pytest.fixture(scope="session")
def secret_key() -> str:
    return "pytest-secret-key-2026"


@pytest.fixture(scope="session")
def payload() -> bytes:
    return b"DSP-TEST"


@pytest.fixture(scope="session")
def image_config() -> ImageWatermarkConfig:
    return ImageWatermarkConfig()


@pytest.fixture(scope="session")
def embedded(cover, payload, secret_key, image_config):
    return ImageWatermarker(image_config).embed_array(
        cover, payload, secret_key, utf8=True, collect_debug=False
    )
