from __future__ import annotations

import numpy as np
import pytest

from core.config import ImageWatermarkConfig
from core.exceptions import CapacityError
from core.payload import (
    PayloadCodec,
    convolutional_decode_soft,
    convolutional_encode,
    keyed_permutation,
    keyed_pn_signs,
)


def test_keyed_generators_are_deterministic_and_balanced() -> None:
    first = keyed_permutation(100, "deterministic-key", b"test")
    second = keyed_permutation(100, "deterministic-key", b"test")
    np.testing.assert_array_equal(first, second)
    signs = keyed_pn_signs(13, 8, "deterministic-key", b"pn")
    np.testing.assert_array_equal(signs.sum(axis=1), np.zeros(13))


def test_convolutional_code_exact_and_soft_decoding() -> None:
    rng = np.random.default_rng(13)
    bits = rng.integers(0, 2, 160, dtype=np.uint8)
    encoded = convolutional_encode(bits)
    scores = 2.0 * encoded.astype(np.float64) - 1.0
    # Corrupt isolated coded symbols; soft Viterbi should still recover the message.
    scores[[7, 51, 109, 203]] *= -0.25
    np.testing.assert_array_equal(convolutional_decode_soft(scores, bits.size), bits)


def test_payload_frame_roundtrip_and_wrong_key() -> None:
    config = ImageWatermarkConfig()
    payload = b"authenticated bytes"
    encoded = PayloadCodec.encode(payload, 1000, "correct-key-2026", config, utf8=True)
    scores = 2.0 * encoded.physical_bits.astype(np.float64) - 1.0
    decoded = PayloadCodec.decode(scores, "correct-key-2026", config)
    assert decoded.valid
    assert decoded.payload == payload
    assert decoded.is_utf8
    wrong = PayloadCodec.decode(
        scores, "different-key-2026", config, expected_payload_bytes=len(payload)
    )
    assert not wrong.valid


def test_payload_capacity_error() -> None:
    config = ImageWatermarkConfig()
    with pytest.raises(CapacityError):
        PayloadCodec.encode(b"too large", 10, "capacity-key-2026", config)
