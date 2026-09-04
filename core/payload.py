"""Key derivation, framing, whitening, authentication, and error-correcting codes."""

from __future__ import annotations

import hashlib
import hmac
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ImageWatermarkConfig
from .exceptions import CapacityError

BitArray = NDArray[np.uint8]
ScoreArray = NDArray[np.float64]

_MAGIC = b"DW"
_VERSION = 1
_HEADER = struct.Struct(">2sBBHI")
_HEADER_BITS = _HEADER.size * 8
_FLAG_UTF8 = 0x01
_CONSTRAINT_LENGTH = 7
_TAIL_BITS = _CONSTRAINT_LENGTH - 1
_GENERATORS = (0o171, 0o133)
_STATE_COUNT = 1 << _TAIL_BITS


def _key_bytes(secret_key: str | bytes) -> bytes:
    secret = secret_key.encode("utf-8") if isinstance(secret_key, str) else bytes(secret_key)
    if len(secret) < 8:
        raise ValueError("secret key must contain at least 8 bytes")
    return hashlib.sha256(b"dsp-invisible-watermark/v1\x00" + secret).digest()


def _hmac_stream(secret_key: str | bytes, context: bytes, n_bytes: int) -> bytes:
    key = _key_bytes(secret_key)
    output = bytearray()
    counter = 0
    while len(output) < n_bytes:
        output.extend(hmac.new(key, context + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(output[:n_bytes])


def keystream_bits(secret_key: str | bytes, context: bytes, n_bits: int) -> BitArray:
    raw = _hmac_stream(secret_key, context, (n_bits + 7) // 8)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n_bits].astype(np.uint8)


def keyed_permutation(total: int, secret_key: str | bytes, context: bytes) -> NDArray[np.int64]:
    if total < 0:
        raise ValueError("total must be non-negative")
    seed = int.from_bytes(_hmac_stream(secret_key, b"perm/" + context, 16), "little")
    return np.random.Generator(np.random.PCG64(seed)).permutation(total)


def keyed_dither(total: int, delta: float, secret_key: str | bytes, context: bytes) -> ScoreArray:
    seed = int.from_bytes(_hmac_stream(secret_key, b"dither/" + context, 16), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.uniform(0.0, delta, size=total).astype(np.float64)


def keyed_pn_signs(rows: int, length: int, secret_key: str | bytes, context: bytes) -> ScoreArray:
    """Generate balanced pseudo-noise chips containing equal +/-1 signs."""

    if rows < 0 or length < 2 or length % 2:
        raise ValueError("PN length must be a positive even integer")
    seed = int.from_bytes(_hmac_stream(secret_key, b"pn/" + context, 16), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    order = np.argsort(rng.random((rows, length)), axis=1)
    signs = -np.ones((rows, length), dtype=np.float64)
    row_indices = np.arange(rows)[:, np.newaxis]
    signs[row_indices, order[:, : length // 2]] = 1.0
    return signs


def bytes_to_bits(data: bytes) -> BitArray:
    if not data:
        return np.empty(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: NDArray[np.generic]) -> bytes:
    array = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if array.size % 8:
        raise ValueError("bit count must be divisible by 8")
    return np.packbits(array).tobytes()


def repeat_bits(bits: BitArray, repetition: int) -> BitArray:
    return np.repeat(bits, repetition).astype(np.uint8)


def decode_repetition(scores: ScoreArray, logical_bits: int, repetition: int) -> ScoreArray:
    needed = logical_bits * repetition
    if scores.size < needed:
        raise CapacityError(f"need {needed} carrier scores, received {scores.size}")
    return np.asarray(scores[:needed], dtype=np.float64).reshape(logical_bits, repetition).mean(axis=1)


def convolutional_encoded_length(logical_bits: int) -> int:
    return 2 * (logical_bits + _TAIL_BITS)


def _parity(value: int) -> int:
    return value.bit_count() & 1


def convolutional_encode(bits: NDArray[np.generic]) -> BitArray:
    """Rate-1/2, K=7 convolutional code using generators (171, 133)_octal."""

    source = np.asarray(bits, dtype=np.uint8).reshape(-1)
    terminated = np.concatenate((source, np.zeros(_TAIL_BITS, dtype=np.uint8)))
    output = np.empty(terminated.size * 2, dtype=np.uint8)
    state = 0
    for index, bit in enumerate(terminated.tolist()):
        register = (state << 1) | int(bit)
        output[2 * index] = _parity(register & _GENERATORS[0])
        output[2 * index + 1] = _parity(register & _GENERATORS[1])
        state = register & (_STATE_COUNT - 1)
    return output


def _trellis() -> tuple[NDArray[np.int16], NDArray[np.uint8]]:
    next_state = np.empty((_STATE_COUNT, 2), dtype=np.int16)
    outputs = np.empty((_STATE_COUNT, 2, 2), dtype=np.uint8)
    for state in range(_STATE_COUNT):
        for bit in (0, 1):
            register = (state << 1) | bit
            next_state[state, bit] = register & (_STATE_COUNT - 1)
            outputs[state, bit, 0] = _parity(register & _GENERATORS[0])
            outputs[state, bit, 1] = _parity(register & _GENERATORS[1])
    return next_state, outputs


_NEXT_STATE, _OUTPUT_BITS = _trellis()


def convolutional_decode_soft(scores: NDArray[np.generic], logical_bits: int) -> BitArray:
    """Soft-decision Viterbi decoder for :func:`convolutional_encode`."""

    encoded_length = convolutional_encoded_length(logical_bits)
    observations = np.asarray(scores, dtype=np.float64).reshape(-1)
    if observations.size < encoded_length:
        raise CapacityError(f"need {encoded_length} convolutional scores")
    observations = observations[:encoded_length].reshape(-1, 2)
    observations = np.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=-1.0)
    steps = observations.shape[0]
    infinity = np.finfo(np.float64).max / 8.0
    metrics = np.full(_STATE_COUNT, infinity, dtype=np.float64)
    metrics[0] = 0.0
    previous_states = np.full((steps, _STATE_COUNT), -1, dtype=np.int16)
    previous_bits = np.zeros((steps, _STATE_COUNT), dtype=np.uint8)

    for step, observed in enumerate(observations):
        new_metrics = np.full(_STATE_COUNT, infinity, dtype=np.float64)
        for state in range(_STATE_COUNT):
            base_metric = metrics[state]
            if base_metric >= infinity:
                continue
            for bit in (0, 1):
                destination = int(_NEXT_STATE[state, bit])
                expected = 2.0 * _OUTPUT_BITS[state, bit].astype(np.float64) - 1.0
                branch_metric = -float(np.dot(observed, expected))
                candidate = base_metric + branch_metric
                if candidate < new_metrics[destination]:
                    new_metrics[destination] = candidate
                    previous_states[step, destination] = state
                    previous_bits[step, destination] = bit
        metrics = new_metrics

    state = 0 if metrics[0] < infinity else int(np.argmin(metrics))
    decoded = np.empty(steps, dtype=np.uint8)
    for step in range(steps - 1, -1, -1):
        predecessor = int(previous_states[step, state])
        if predecessor < 0:
            raise CapacityError("Viterbi traceback did not reach a valid state")
        decoded[step] = previous_bits[step, state]
        state = predecessor
    return decoded[:logical_bits]


def _unwhiten_bits(bits: BitArray, secret_key: str | bytes, context: bytes) -> BitArray:
    return bits ^ keystream_bits(secret_key, context, bits.size)


def _unwhiten_scores(scores: ScoreArray, secret_key: str | bytes, context: bytes) -> ScoreArray:
    stream = keystream_bits(secret_key, context, scores.size)
    return scores * (1.0 - 2.0 * stream.astype(np.float64))


def _auth_tag(secret_key: str | bytes, payload: bytes, length: int) -> bytes:
    return hmac.new(_key_bytes(secret_key), b"auth/v1\x00" + payload, hashlib.sha256).digest()[:length]


@dataclass(frozen=True, slots=True)
class EncodedPayload:
    physical_bits: BitArray
    header_carriers: int
    body_carriers: int
    capacity_bytes: int
    payload_bytes: int
    flags: int


@dataclass(frozen=True, slots=True)
class DecodedPayload:
    payload: bytes
    header_ok: bool
    crc_ok: bool
    auth_ok: bool
    flags: int
    declared_length: int | None
    confidence: float
    message: str

    @property
    def valid(self) -> bool:
        return self.header_ok and self.crc_ok and self.auth_ok

    @property
    def is_utf8(self) -> bool:
        return bool(self.flags & _FLAG_UTF8)


class PayloadCodec:
    """Frame arbitrary bytes and map them to key-whitened ECC bits."""

    header_bits = _HEADER_BITS

    @staticmethod
    def _encoded_length(logical_bits: int, repetition: int, config: ImageWatermarkConfig) -> int:
        if config.ecc_mode == "convolutional":
            return convolutional_encoded_length(logical_bits)
        return logical_bits * repetition

    @classmethod
    def capacity_bytes(cls, total_carriers: int, config: ImageWatermarkConfig) -> int:
        header_carriers = cls._encoded_length(_HEADER_BITS, config.header_repetition, config)
        remaining = max(0, total_carriers - header_carriers)
        if config.ecc_mode == "convolutional":
            body_bits = max(0, remaining // 2 - _TAIL_BITS)
        else:
            body_bits = remaining // config.payload_repetition
        return max(0, body_bits // 8 - config.auth_tag_bytes)

    @staticmethod
    def _encode_bits(bits: BitArray, repetition: int, config: ImageWatermarkConfig) -> BitArray:
        if config.ecc_mode == "convolutional":
            return convolutional_encode(bits)
        return repeat_bits(bits, repetition)

    @staticmethod
    def _decode_bits(
        scores: ScoreArray,
        logical_bits: int,
        repetition: int,
        config: ImageWatermarkConfig,
    ) -> tuple[BitArray, int]:
        if config.ecc_mode == "convolutional":
            used = convolutional_encoded_length(logical_bits)
            return convolutional_decode_soft(scores[:used], logical_bits), used
        used = logical_bits * repetition
        return (decode_repetition(scores[:used], logical_bits, repetition) > 0.0).astype(np.uint8), used

    @classmethod
    def encode(
        cls,
        payload: bytes,
        total_carriers: int,
        secret_key: str | bytes,
        config: ImageWatermarkConfig,
        *,
        utf8: bool = False,
    ) -> EncodedPayload:
        payload = bytes(payload)
        if len(payload) > 0xFFFF:
            raise CapacityError("payload exceeds the 65535-byte framing limit")
        capacity = cls.capacity_bytes(total_carriers, config)
        if len(payload) > capacity:
            raise CapacityError(f"payload is {len(payload)} bytes; transform capacity is {capacity} bytes")
        flags = _FLAG_UTF8 if utf8 else 0
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = _HEADER.pack(_MAGIC, _VERSION, flags, len(payload), crc)
        body = payload + _auth_tag(secret_key, payload, config.auth_tag_bytes)

        header_bits = bytes_to_bits(header)
        header_bits ^= keystream_bits(secret_key, b"header-whitening/v1", header_bits.size)
        body_bits = bytes_to_bits(body)
        body_bits ^= keystream_bits(secret_key, b"body-whitening/v1", body_bits.size)
        encoded_header = cls._encode_bits(header_bits, config.header_repetition, config)
        encoded_body = cls._encode_bits(body_bits, config.payload_repetition, config)
        physical = np.concatenate((encoded_header, encoded_body)).astype(np.uint8)
        return EncodedPayload(
            physical, encoded_header.size, encoded_body.size, capacity, len(payload), flags
        )

    @classmethod
    def decode(
        cls,
        ordered_scores: ScoreArray,
        secret_key: str | bytes,
        config: ImageWatermarkConfig,
        *,
        expected_payload_bytes: int | None = None,
    ) -> DecodedPayload:
        scores = np.asarray(ordered_scores, dtype=np.float64).reshape(-1)
        header_carriers = cls._encoded_length(_HEADER_BITS, config.header_repetition, config)
        if scores.size < header_carriers:
            return DecodedPayload(b"", False, False, False, 0, None, 0.0, "insufficient carriers for header")
        try:
            encoded_header_bits, _ = cls._decode_bits(
                scores[:header_carriers], _HEADER_BITS, config.header_repetition, config
            )
            header_bits = _unwhiten_bits(encoded_header_bits, secret_key, b"header-whitening/v1")
            magic, version, flags, declared_length, declared_crc = _HEADER.unpack(bits_to_bytes(header_bits))
        except (CapacityError, struct.error, ValueError):
            magic, version, flags, declared_length, declared_crc = b"", 0, 0, 0, 0

        header_ok = magic == _MAGIC and version == _VERSION
        payload_length = declared_length if header_ok else expected_payload_bytes
        confidence = float(np.mean(np.abs(scores[:header_carriers])))
        if payload_length is None:
            return DecodedPayload(b"", False, False, False, int(flags), None, confidence, "header magic/version check failed")
        if payload_length < 0 or payload_length > 0xFFFF:
            return DecodedPayload(b"", header_ok, False, False, int(flags), None, confidence, "invalid payload length")

        logical_body_bits = (payload_length + config.auth_tag_bytes) * 8
        body_carriers = cls._encoded_length(logical_body_bits, config.payload_repetition, config)
        start, end = header_carriers, header_carriers + body_carriers
        if scores.size < end:
            return DecodedPayload(b"", header_ok, False, False, int(flags), int(payload_length), confidence, "insufficient carriers for declared payload")
        try:
            encoded_body_bits, _ = cls._decode_bits(
                scores[start:end], logical_body_bits, config.payload_repetition, config
            )
        except CapacityError:
            return DecodedPayload(b"", header_ok, False, False, int(flags), int(payload_length), confidence, "ECC decoding failed")
        body_bits = _unwhiten_bits(encoded_body_bits, secret_key, b"body-whitening/v1")
        body = bits_to_bytes(body_bits)
        payload = body[:payload_length]
        extracted_tag = body[payload_length:]
        auth_ok = hmac.compare_digest(
            extracted_tag, _auth_tag(secret_key, payload, config.auth_tag_bytes)
        )
        crc_ok = header_ok and (zlib.crc32(payload) & 0xFFFFFFFF) == declared_crc
        confidence = float(np.mean(np.abs(scores[:end])))
        if header_ok and crc_ok and auth_ok:
            message = "payload decoded and authenticated"
        elif not header_ok:
            message = "header failed; payload decoded using expected length"
        else:
            message = "payload decoded but integrity/authentication failed"
        return DecodedPayload(payload, header_ok, crc_ok, auth_ok, int(flags), int(payload_length), confidence, message)
