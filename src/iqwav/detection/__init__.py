"""Capture-level detection hints; activity is not a signal-presence gate."""

from .activity import ActivityDetection, ActivityRegion, detect_activity

__all__ = ["ActivityDetection", "ActivityRegion", "detect_activity"]
