
import numpy as np
from src.modulation import (
    bpsk_modulate,
    bpsk_demodulate,
    qpsk_modulate,
    qpsk_demodulate,
)


def test_bpsk_no_noise():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=256, dtype=np.uint8)
    symbols = bpsk_modulate(bits)
    # sem ruído: demod deve recuperar exatamente
    recovered = bpsk_demodulate(symbols)
    assert np.array_equal(bits, recovered)


def test_qpsk_no_noise():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=255, dtype=np.uint8)  # tamanho ímpar para testar padding
    symbols, padding = qpsk_modulate(bits)
    recovered = qpsk_demodulate(symbols, padding)
    assert np.array_equal(bits, recovered)
