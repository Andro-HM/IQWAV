"""Receiver-side synchronization for IQWAV."""

from .frequency import correct_frequency_offset
from .phase import correct_phase_offset

__all__ = ["correct_frequency_offset", "correct_phase_offset"]
