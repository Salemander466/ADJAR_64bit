from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from adjar64 import get_logger

log = get_logger("core.metrics")


def rms(x: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """
    Root-mean-square value.

    Args:
        x: numeric array
        axis: axis for RMS reduction; None returns scalar RMS over all elements

    Returns:
        RMS scalar or array depending on axis.
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("rms: input is empty")
    if not np.isfinite(x).all():
        raise ValueError("rms: input contains NaN/inf")

    return np.sqrt(np.mean(np.square(x), axis=axis))


def rms_diff(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """
    RMS of the difference between two arrays.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"rms_diff: shape mismatch {a.shape} vs {b.shape}")
    return rms(a - b, axis=axis)


def _time_index_window(
    fs_hz: float,
    tmin_ms: float,
    n_times: int,
    window_ms: Optional[Tuple[float, float]],
) -> slice:
    """
    Convert a time window in ms to a slice over sample indices.

    If window_ms is None, return full slice.
    """
    if window_ms is None:
        return slice(0, n_times)

    w0, w1 = window_ms
    if w1 <= w0:
        raise ValueError("window_ms must satisfy (start < end)")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0 to compute time indices")

    dt_ms = 1000.0 / fs_hz
    # time axis: t = tmin_ms + i*dt_ms
    i0 = int(np.floor((w0 - tmin_ms) / dt_ms))
    i1 = int(np.ceil((w1 - tmin_ms) / dt_ms))

    i0 = max(0, min(n_times, i0))
    i1 = max(0, min(n_times, i1))
    if i1 <= i0:
        # empty slice; allow but warn
        log.warning("Computed empty time window slice for window_ms=%s", window_ms)
        return slice(0, 0)
    return slice(i0, i1)


def convergence_rms(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    fs_hz: float,
    tmin_ms: float,
    window_ms: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Convergence metric for iterative ADJAR (Level 2):
    RMS(prev - curr) over (channels, time_window).

    Args:
        prev, curr: arrays (channels, times)
        fs_hz, tmin_ms: sampling rate and epoch start for time-window mapping
        window_ms: optional time window (start_ms, end_ms) to restrict convergence

    Returns:
        Scalar RMS.
    """
    prev = np.asarray(prev)
    curr = np.asarray(curr)
    if prev.shape != curr.shape:
        raise ValueError(f"convergence_rms: shape mismatch {prev.shape} vs {curr.shape}")
    if prev.ndim != 2:
        raise ValueError("convergence_rms expects 2D arrays (channels, times)")

    if not np.isfinite(prev).all() or not np.isfinite(curr).all():
        raise ValueError("convergence_rms: inputs contain NaN/inf")

    n_times = prev.shape[1]
    sl = _time_index_window(fs_hz, tmin_ms, n_times, window_ms)
    diff = prev[:, sl] - curr[:, sl]
    if diff.size == 0:
        return 0.0
    return float(rms(diff))


@dataclass(frozen=True)
class QAStats:
    """
    Simple quality assurance stats for debugging/plot annotations.
    """
    n_trials: int
    n_channels: int
    n_times: int
    fs_hz: float
    tmin_ms: float
    tmax_ms: float
    isi_min_ms: float
    isi_max_ms: float
    isi_mean_ms: float
    isi_std_ms: float


def qa_stats(
    data: np.ndarray,
    isi_pre_ms: np.ndarray,
    *,
    fs_hz: float,
    tmin_ms: float,
    tmax_ms: float,
) -> QAStats:
    """
    Basic QA summary for epoched data and ISI distribution.

    Args:
        data: (trials, channels, times)
        isi_pre_ms: (trials,)
    """
    data = np.asarray(data)
    isi = np.asarray(isi_pre_ms, dtype=np.float64)

    if data.ndim != 3:
        raise ValueError("qa_stats: data must be 3D (trials, channels, times)")
    if isi.ndim != 1 or isi.shape[0] != data.shape[0]:
        raise ValueError("qa_stats: isi_pre_ms must be 1D and match n_trials")
    if not np.isfinite(data).all():
        raise ValueError("qa_stats: data contains NaN/inf")
    if not np.isfinite(isi).all():
        raise ValueError("qa_stats: isi_pre_ms contains NaN/inf")

    n_trials, n_ch, n_times = data.shape
    return QAStats(
        n_trials=int(n_trials),
        n_channels=int(n_ch),
        n_times=int(n_times),
        fs_hz=float(fs_hz),
        tmin_ms=float(tmin_ms),
        tmax_ms=float(tmax_ms),
        isi_min_ms=float(np.min(isi)),
        isi_max_ms=float(np.max(isi)),
        isi_mean_ms=float(np.mean(isi)),
        isi_std_ms=float(np.std(isi, ddof=1)) if isi.size > 1 else 0.0,
    )
