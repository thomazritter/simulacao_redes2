
import numpy as np
from src.encoding import text_to_bits, bits_to_text, manchester_encode, manchester_decode


def test_text_bits_roundtrip():
    msg = "Ola Mundo!"
    bits = text_to_bits(msg)
    recovered = bits_to_text(bits)
    assert recovered == msg


def test_manchester_roundtrip():
    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=128, dtype=np.uint8)
    encoded = manchester_encode(bits)
    decoded = manchester_decode(encoded)
    assert np.array_equal(bits, decoded)
