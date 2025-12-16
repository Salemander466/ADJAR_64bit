from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np

from adjar64 import get_logger

log = get_logger("core.distributions")


@dataclass(frozen=True)
class DistributionResult:
    """
    Output container for a preceding-event distribution kernel.

    Attributes:
        d_pre: 1D kernel aligned to the epoch time axis (length n_times).
               Intended for convolution with R_prev(t) to predict overlap in the
               conventional ERP aligned at time 0 (current event onset).
        time_ms: 1D time axis (ms) corresponding to d_pre indices.
        hist_counts: raw histogram counts per bin (before normalization), same length as d_pre.
        hist_prob: probability distribution per bin (sum ~ 1), same length as d_pre.
        dt_ms: sampling period in ms.
        notes: short string describing construction choices.
    """
    d_pre: np.ndarray
    time_ms: np.ndarray
    hist_counts: np.ndarray
    hist_prob: np.ndarray
    dt_ms: float
    notes: str


KernelKind = Literal["probability", "rate_per_trial", "impulse"]


def build_preceding_distribution(
    isi_pre_ms: np.ndarray,
    *,
    fs_hz: float,
    tmin_ms: float,
    n_times: int,
    normalize: KernelKind = "probability",
    smoothing_ms: float = 0.0,
    clamp_to_epoch: bool = True,
) -> DistributionResult:
    """
    Build the preceding-event time-lag distribution kernel D_pre(t) aligned to the epoch.

    Interpretation:
      - For each trial, the preceding event occurred at lag = -ISI_pre relative to the
        current event at time 0.
      - We represent these lags on the epoch time grid and form a histogram (or impulses),
        which becomes D_pre(t) used in overlap prediction:
            overlap_pred(t) = (D_pre * R_prev)(t)

    Args:
        isi_pre_ms:
            1D array of preceding intervals (ms), length n_trials.
        fs_hz:
            Sampling rate in Hz.
        tmin_ms:
            Epoch start in ms (e.g., -200).
        n_times:
            Number of samples in the epoch.
        normalize:
            - "probability": D_pre sums to 1 across the epoch bins that are included.
            - "rate_per_trial": D_pre sums to 1 as well, but labeled for interpretation as
              expected overlap per trial (kept identical numerically here; useful as semantic flag).
            - "impulse": returns an impulse-train kernel (counts, optionally smoothed), not
              probability-normalized (sum equals number of contributing trials).
        smoothing_ms:
            Optional Gaussian smoothing (in ms) applied to the histogram on the time axis.
            Use 0.0 for no smoothing.
        clamp_to_epoch:
            If True, intervals whose lag falls outside [tmin_ms, tmax_ms] are dropped (ignored).
            If False, they are clipped into the closest bin (not recommended for ADJAR).

    Returns:
        DistributionResult containing kernel and diagnostics.

    Raises:
        ValueError for invalid inputs.
    """
    isi = np.asarray(isi_pre_ms, dtype=np.float64)
    if isi.ndim != 1:
        raise ValueError(f"isi_pre_ms must be 1D, got shape {isi.shape}")
    if isi.size < 1:
        raise ValueError("isi_pre_ms must be non-empty")
    if not np.isfinite(isi).all():
        raise ValueError("isi_pre_ms contains NaN/inf")
    if np.any(isi < 0):
        raise ValueError("isi_pre_ms contains negative values; must be >= 0 ms")

    if fs_hz <= 0 or not np.isfinite(fs_hz):
        raise ValueError(f"fs_hz must be finite and > 0, got {fs_hz}")
    if n_times < 2:
        raise ValueError("n_times must be >= 2")
    if not np.isfinite(tmin_ms):
        raise ValueError("tmin_ms must be finite")

    dt_ms = 1000.0 / float(fs_hz)
    time_ms = tmin_ms + np.arange(n_times, dtype=np.float64) * dt_ms
    tmax_ms = float(time_ms[-1])

    # Preceding event lag relative to current event is negative:
    # lag_ms = -isi_pre_ms
    lag_ms = -isi

    # Map lag_ms to nearest time-bin index.
    # index i corresponds to time_ms[i] = tmin_ms + i*dt_ms
    # i = round((lag_ms - tmin_ms)/dt_ms)
    idx = np.rint((lag_ms - tmin_ms) / dt_ms).astype(np.int64)

    if clamp_to_epoch:
        keep = (idx >= 0) & (idx < n_times)
        idx_kept = idx[keep]
        n_dropped = int(np.size(idx) - np.size(idx_kept))
        if idx_kept.size == 0:
            raise ValueError(
                "All preceding-event lags fall outside the epoch window. "
                f"Epoch=[{tmin_ms:.3f},{tmax_ms:.3f}] ms; "
                f"lag range=[{lag_ms.min():.3f},{lag_ms.max():.3f}] ms. "
                "Increase epoch length or verify isi_pre_ms."
            )
        if n_dropped > 0:
            log.info("Dropped %d trials from D_pre because lag fell outside epoch.", n_dropped)
        idx = idx_kept
    else:
        # Clip indices into range (not recommended, but available).
        idx = np.clip(idx, 0, n_times - 1)

    # Histogram counts per epoch bin.
    hist_counts = np.bincount(idx, minlength=n_times).astype(np.float64)

    # Optional smoothing (Gaussian on discrete grid).
    if smoothing_ms > 0.0:
        hist_counts = _gaussian_smooth_1d(hist_counts, sigma_ms=float(smoothing_ms), dt_ms=dt_ms)

    # Convert to a distribution kernel according to normalize mode.
    if normalize in ("probability", "rate_per_trial"):
        s = float(np.sum(hist_counts))
        if s <= 0:
            raise ValueError("Histogram sum is zero; cannot normalize distribution.")
        d_pre = (hist_counts / s).astype(np.float64, copy=False)
        hist_prob = d_pre.copy()
        notes = (
            f"Histogram of lag=-ISI_pre mapped to epoch grid; normalized={normalize}; "
            f"smoothing_ms={smoothing_ms}; clamp_to_epoch={clamp_to_epoch}"
        )
    elif normalize == "impulse":
        d_pre = hist_counts.astype(np.float64, copy=False)
        s = float(np.sum(hist_counts))
        hist_prob = (hist_counts / s).astype(np.float64, copy=False) if s > 0 else np.zeros_like(hist_counts)
        notes = (
            f"Impulse-train kernel (counts) of lag=-ISI_pre; smoothing_ms={smoothing_ms}; "
            f"clamp_to_epoch={clamp_to_epoch}"
        )
    else:
        raise ValueError("normalize must be one of: 'probability', 'rate_per_trial', 'impulse'")

    return DistributionResult(
        d_pre=d_pre,
        time_ms=time_ms,
        hist_counts=hist_counts,
        hist_prob=hist_prob,
        dt_ms=dt_ms,
        notes=notes,
    )


def distribution_from_single_lag(
    lag_ms: float,
    *,
    fs_hz: float,
    tmin_ms: float,
    n_times: int,
    kind: KernelKind = "probability",
) -> DistributionResult:
    """
    Convenience helper: build D_pre(t) for a single lag (e.g., bin center).

    This is useful when constructing per-bin distributions for subaverage regression.

    Args:
        lag_ms: time of preceding event relative to current event (should be negative for preceding)
        kind: normalization mode, same meaning as build_preceding_distribution

    Returns:
        DistributionResult with a single impulse at the nearest time bin.
    """
    dt_ms = 1000.0 / float(fs_hz)
    time_ms = tmin_ms + np.arange(n_times, dtype=np.float64) * dt_ms
    idx = int(np.rint((lag_ms - tmin_ms) / dt_ms))
    idx = int(np.clip(idx, 0, n_times - 1))

    hist_counts = np.zeros((n_times,), dtype=np.float64)
    hist_counts[idx] = 1.0

    if kind in ("probability", "rate_per_trial"):
        d_pre = hist_counts.copy()
        hist_prob = hist_counts.copy()
    elif kind == "impulse":
        d_pre = hist_counts.copy()
        hist_prob = hist_counts.copy()
    else:
        raise ValueError("kind must be one of: 'probability', 'rate_per_trial', 'impulse'")

    return DistributionResult(
        d_pre=d_pre,
        time_ms=time_ms,
        hist_counts=hist_counts,
        hist_prob=hist_prob,
        dt_ms=dt_ms,
        notes=f"Single-lag impulse at {lag_ms} ms mapped to nearest bin index {idx}",
    )


def _gaussian_smooth_1d(x: np.ndarray, *, sigma_ms: float, dt_ms: float) -> np.ndarray:
    """
    Gaussian smoothing on a 1D array using an explicit kernel.

    Args:
        x: 1D array
        sigma_ms: Gaussian sigma in ms
        dt_ms: sampling interval in ms

    Returns:
        Smoothed array, same shape.

    Notes:
        - Kernel is truncated at +/- 4 sigma (rounded up).
        - Uses FFT convolution if SciPy is available; otherwise direct convolution.
    """
    x = np.asarray(x, dtype=np.float64)
    if sigma_ms <= 0:
        return x

    sigma_samp = sigma_ms / dt_ms
    if sigma_samp <= 0:
        return x

    radius = int(np.ceil(4.0 * sigma_samp))
    if radius < 1:
        return x

    k_idx = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (k_idx / sigma_samp) ** 2)
    k /= np.sum(k)

    # Prefer FFT-based convolution from our accel module for speed.
    try:
        from adjar64.accel.fftconv import fft_convolve_1d

        y_full = fft_convolve_1d(x, k, mode="full", use_real_fft=True)
        # 'same' alignment (centered)
        start = (k.size - 1) // 2
        end = start + x.size
        y = y_full[start:end]
        return y.astype(np.float64, copy=False)
    except Exception:
        # Fallback direct convolution
        y = np.convolve(x, k, mode="same")
        return y.astype(np.float64, copy=False)
