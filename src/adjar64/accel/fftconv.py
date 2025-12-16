from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from adjar64 import get_logger, require_optional

log = get_logger("accel.fftconv")

# SciPy provides robust next_fast_len and FFT backends; fall back if unavailable.
try:
    require_optional("scipy", "FFT utilities for fast convolution")
    from scipy.fft import rfft, irfft, fft, ifft, next_fast_len  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    rfft = irfft = fft = ifft = None  # type: ignore
    next_fast_len = None  # type: ignore


def _next_fast_len(n: int) -> int:
    if _HAVE_SCIPY and next_fast_len is not None:
        return int(next_fast_len(int(n)))
    # Fallback: power of two
    p = 1
    while p < n:
        p <<= 1
    return p


def _validate_1d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    if x.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/inf")
    return x


def _validate_2d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/inf")
    return x


def fft_convolve_1d(
    a: np.ndarray,
    b: np.ndarray,
    mode: str = "full",
    use_real_fft: Optional[bool] = None,
) -> np.ndarray:
    """
    FFT-based 1D convolution.

    Args:
        a, b: 1D arrays
        mode: 'full', 'same', or 'valid' (NumPy-like)
        use_real_fft: If True, use rfft/irfft (faster) when arrays are real.
                      If None, choose automatically.

    Returns:
        Convolved array according to mode.

    Notes:
        - Always returns float64 if inputs are real; complex if needed.
        - Uses SciPy FFT when available; else falls back to NumPy FFT.
    """
    a = _validate_1d(a, "a")
    b = _validate_1d(b, "b")

    if mode not in ("full", "same", "valid"):
        raise ValueError("mode must be one of: 'full', 'same', 'valid'")

    na = a.size
    nb = b.size
    n_full = na + nb - 1
    n_fft = _next_fast_len(n_full)

    a_is_real = np.isrealobj(a)
    b_is_real = np.isrealobj(b)
    if use_real_fft is None:
        use_real_fft = bool(a_is_real and b_is_real)

    if use_real_fft:
        if _HAVE_SCIPY and rfft is not None and irfft is not None:
            A = rfft(a, n_fft)
            B = rfft(b, n_fft)
            y = irfft(A * B, n_fft)
        else:
            # NumPy fallback
            A = np.fft.rfft(a, n_fft)
            B = np.fft.rfft(b, n_fft)
            y = np.fft.irfft(A * B, n_fft)
        y = y[:n_full].astype(np.float64, copy=False)
    else:
        if _HAVE_SCIPY and fft is not None and ifft is not None:
            A = fft(a, n_fft)
            B = fft(b, n_fft)
            y = ifft(A * B, n_fft)
        else:
            A = np.fft.fft(a, n_fft)
            B = np.fft.fft(b, n_fft)
            y = np.fft.ifft(A * B, n_fft)
        y = y[:n_full]

    return _apply_mode(y, na, nb, mode)


def fft_convolve_batch(
    X: np.ndarray,
    h: np.ndarray,
    axis: int = -1,
    mode: str = "full",
    use_real_fft: Optional[bool] = None,
) -> np.ndarray:
    """
    Convolve a batch/matrix of signals with a single kernel using FFT.

    Typical use:
      - X: (channels, times)
      - h: (times,) distribution or impulse response
      - output: (channels, times_out)

    Args:
        X: 2D array (batch, n) or similar; convolution happens along `axis`.
        h: 1D kernel.
        axis: axis in X to convolve along (default last axis).
        mode: 'full', 'same', 'valid'
        use_real_fft: auto if None

    Returns:
        Convolved array with same batch dims, time dims adjusted by mode.

    Notes:
        - Internally reshapes so convolution is done along last axis.
        - Uses shared FFT of h for efficiency.
    """
    X = np.asarray(X)
    if X.ndim < 1:
        raise ValueError("X must have at least 1 dimension")
    h = _validate_1d(h, "h")

    if mode not in ("full", "same", "valid"):
        raise ValueError("mode must be one of: 'full', 'same', 'valid'")

    # Move convolution axis to last
    X_moved = np.moveaxis(X, axis, -1)
    n = X_moved.shape[-1]
    nh = h.size
    n_full = n + nh - 1
    n_fft = _next_fast_len(n_full)

    x_is_real = np.isrealobj(X_moved)
    h_is_real = np.isrealobj(h)
    if use_real_fft is None:
        use_real_fft = bool(x_is_real and h_is_real)

    if use_real_fft:
        if _HAVE_SCIPY and rfft is not None and irfft is not None:
            H = rfft(h, n_fft)
            Xf = rfft(X_moved, n_fft, axis=-1)
            Y = irfft(Xf * H, n_fft, axis=-1)
        else:
            H = np.fft.rfft(h, n_fft)
            Xf = np.fft.rfft(X_moved, n_fft, axis=-1)
            Y = np.fft.irfft(Xf * H, n_fft, axis=-1)
        Y = Y[..., :n_full].astype(np.float64, copy=False)
    else:
        if _HAVE_SCIPY and fft is not None and ifft is not None:
            H = fft(h, n_fft)
            Xf = fft(X_moved, n_fft, axis=-1)
            Y = ifft(Xf * H, n_fft, axis=-1)
        else:
            H = np.fft.fft(h, n_fft)
            Xf = np.fft.fft(X_moved, n_fft, axis=-1)
            Y = np.fft.ifft(Xf * H, n_fft, axis=-1)
        Y = Y[..., :n_full]

    # Apply mode along last axis
    Y = _apply_mode_batch(Y, n, nh, mode)

    # Move axis back
    return np.moveaxis(Y, -1, axis)


def _apply_mode(y: np.ndarray, na: int, nb: int, mode: str) -> np.ndarray:
    if mode == "full":
        return y
    if mode == "same":
        # output length = na, centered
        start = (nb - 1) // 2
        end = start + na
        return y[start:end]
    # valid
    if na < nb:
        # following NumPy: valid with na<nb yields empty
        return y[:0]
    start = nb - 1
    end = start + (na - nb + 1)
    return y[start:end]


def _apply_mode_batch(Y: np.ndarray, n: int, nh: int, mode: str) -> np.ndarray:
    if mode == "full":
        return Y
    if mode == "same":
        start = (nh - 1) // 2
        end = start + n
        return Y[..., start:end]
    # valid
    if n < nh:
        return Y[..., :0]
    start = nh - 1
    end = start + (n - nh + 1)
    return Y[..., start:end]
