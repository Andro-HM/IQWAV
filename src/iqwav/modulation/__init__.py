"""Modulation-related functionality and synthetic signal generation."""

from .digital import bpsk_modulate, qpsk_modulate
from .tones import generate_iq_tone, generate_real_tone
from .waveform import bpsk_waveform, qpsk_waveform, symbols_to_samples

__all__ = [
    "bpsk_modulate",
    "bpsk_waveform",
    "generate_iq_tone",
    "generate_real_tone",
    "qpsk_modulate",
    "qpsk_waveform",
    "symbols_to_samples",
]
