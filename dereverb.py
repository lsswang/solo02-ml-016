"""Current single-recording WPE entry point.

The pinned upstream kernel supports one continuous recording per call.
Frequency bins are processed independently, including their power floors.
"""
import numpy as np

from vendor.nara_wpe_subset import wpe_v6


def single_wpe(spectrum, taps=2, delay=2, iterations=3):
    """Return an F x C x T residual for a continuous complex spectrum."""
    y = np.asarray(spectrum, dtype=np.complex128)
    if y.ndim != 3 or min(y.shape) < 1 or not np.isfinite(y).all():
        raise ValueError("spectrum must be a finite F x C x T array")
    for name, value in (("taps", taps), ("delay", delay), ("iterations", iterations)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if y.shape[-1] < delay + taps:
        raise ValueError("recording has no complete prediction history")
    return np.stack([
        wpe_v6(yf, taps=taps, delay=delay, iterations=iterations,
               psd_context=0, statistics_mode="valid")
        for yf in y
    ])
