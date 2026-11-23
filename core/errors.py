"""Exceptions shared by the engine and the processor."""
from __future__ import annotations


class Cancelled(Exception):
    """Raised to unwind processing when cancellation was requested (not an error)."""
