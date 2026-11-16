"""Current positive-definite, one-response spatial filter and array helpers."""
import numpy as np


def mvdr(noise_psd, steering):
    """One frequency: minimize w^H R w subject to v^H w = 1; R is PD."""
    r = np.asarray(noise_psd, dtype=np.complex128)
    v = np.asarray(steering, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or v.shape != (r.shape[0],):
        raise ValueError("incompatible covariance/steering shapes")
    if not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("inputs must be finite")
    if not np.allclose(r, r.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("noise covariance must be Hermitian")
    if np.linalg.norm(v) == 0:
        raise ValueError("steering cannot be zero")
    try:
        np.linalg.cholesky(r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ordinary MVDR requires positive-definite covariance") from exc
    numerator = np.linalg.solve(r, v)
    return numerator / np.vdot(v, numerator).real


def weighted_psd(segments, frame_weights):
    """PSD across all frames of all supplied segments; zero weight gives zero R."""
    if not segments or len(segments) != len(frame_weights):
        raise ValueError("one weight array is required per segment")
    first = np.asarray(segments[0])
    if first.ndim != 3:
        raise ValueError("spectra must have shape F x C x T")
    f, c = first.shape[:2]
    numerator = np.zeros((f, c, c), dtype=np.complex128)
    denominator = np.zeros(f)
    for spectrum, weights in zip(segments, frame_weights):
        y = np.asarray(spectrum, dtype=np.complex128)
        w = np.asarray(weights)
        if y.ndim != 3 or y.shape[:2] != (f, c) or w.shape != (f, y.shape[-1]):
            raise ValueError("spectrum/weight dimensions differ")
        if np.iscomplexobj(w) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("frame weights must be finite, real and nonnegative")
        if not np.isfinite(y).all():
            raise ValueError("spectra must be finite")
        numerator += np.einsum("fct,fdt,ft->fcd", y, y.conj(), w, optimize=False)
        denominator += w.sum(axis=1)
    result = np.zeros_like(numerator)
    mask = denominator > 0
    result[mask] = numerator[mask] / denominator[mask, None, None]
    return result


def apply_weights(weights, spectrum):
    """Apply one spatial weight vector per frequency to one F x C x T segment."""
    w = np.asarray(weights, dtype=np.complex128)
    y = np.asarray(spectrum, dtype=np.complex128)
    if y.ndim != 3 or w.shape != y.shape[:2]:
        raise ValueError("weights must have shape F x C")
    if not np.isfinite(w).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    return np.einsum("fc,fct->ft", w.conj(), y, optimize=False)
