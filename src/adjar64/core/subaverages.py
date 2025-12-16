from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from adjar64 import get_logger

log = get_logger("core.subaverages")


@dataclass(frozen=True)
class SubaverageBin:
    """
    Represents one ISI bin and its computed subaverage.

    Attributes:
        bin_index: index of this bin in the returned bin list (0..K-1)
        left_ms, right_ms: inclusive/exclusive edges of the bin
        center_ms: center of the bin
        trial_indices: indices of trials that fall into this bin
        n_trials: number of trials in bin
        erp: mean waveform over trials in this bin, shape (channels, times)
        isi_values_ms: ISI values for the trials in this bin, shape (n_trials,)
    """
    bin_index: int
    left_ms: float
    right_ms: float
    center_ms: float
    trial_indices: np.ndarray
    n_trials: int
    erp: np.ndarray
    isi_values_ms: np.ndarray


@dataclass(frozen=True)
class SubaverageResult:
    """
    Output of binning + subaverage computation.

    Attributes:
        bins: list of SubaverageBin
        edges_ms: bin edges used (length K+1)
        bin_width_ms: bin width
        n_trials_total: number of input trials
        n_trials_used: number of trials assigned to bins (after drop rules)
        dropped_trial_indices: trials dropped (e.g., isi out of range)
        conventional_erp: mean over all (used) trials, shape (channels, times)
    """
    bins: List[SubaverageBin]
    edges_ms: np.ndarray
    bin_width_ms: float
    n_trials_total: int
    n_trials_used: int
    dropped_trial_indices: np.ndarray
    conventional_erp: np.ndarray


def bin_trials_by_isi(
    isi_pre_ms: np.ndarray,
    *,
    bin_width_ms: float,
    isi_range_ms: Optional[Tuple[float, float]] = None,
    min_trials_per_bin: int = 10,
    drop_empty_bins: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each trial to an ISI bin.

    Args:
        isi_pre_ms: 1D array length n_trials
        bin_width_ms: bin width in ms
        isi_range_ms: optional explicit (min, max) range. If None, inferred from data.
        min_trials_per_bin: bins with fewer trials can be dropped later.
        drop_empty_bins: if True, later steps typically ignore bins with 0 trials.

    Returns:
        bin_indices: length n_trials, -1 for dropped/out-of-range
        edges_ms: bin edges used, shape (K+1,)
        dropped_trials: indices where bin_indices == -1

    Notes:
        - This function only assigns bins; it does not compute subaverages.
        - Bins are half-open [left, right), except the last which is [left, right].
    """
    isi = np.asarray(isi_pre_ms, dtype=np.float64)
    if isi.ndim != 1:
        raise ValueError(f"isi_pre_ms must be 1D, got {isi.shape}")
    if isi.size < 1:
        raise ValueError("isi_pre_ms must be non-empty")
    if not np.isfinite(isi).all():
        raise ValueError("isi_pre_ms contains NaN/inf")
    if np.any(isi < 0):
        raise ValueError("isi_pre_ms contains negative values; must be >= 0")

    if bin_width_ms <= 0:
        raise ValueError("bin_width_ms must be > 0")

    if isi_range_ms is None:
        lo = float(np.min(isi))
        hi = float(np.max(isi))
    else:
        lo, hi = float(isi_range_ms[0]), float(isi_range_ms[1])
        if hi <= lo:
            raise ValueError("isi_range_ms must satisfy (min < max)")

    # Expand hi slightly so the max value falls inside last bin.
    hi_eps = hi + 1e-9

    # Build edges from lo to hi inclusive
    n_bins = int(np.ceil((hi_eps - lo) / bin_width_ms))
    if n_bins < 1:
        n_bins = 1
    edges = lo + np.arange(n_bins + 1, dtype=np.float64) * float(bin_width_ms)
    # Ensure last edge >= hi_eps
    if edges[-1] < hi_eps:
        edges = np.append(edges, edges[-1] + float(bin_width_ms))

    # Digitize: returns bin in 1..K, where 0 is below range
    # Right=False gives bins: edges[i-1] <= x < edges[i]
    b = np.digitize(isi, edges, right=False) - 1  # to 0..K-1
    # Out of range -> -1
    out = (isi < edges[0]) | (isi > edges[-1])
    b[out] = -1

    dropped = np.where(b == -1)[0].astype(np.int64)
    return b.astype(np.int64), edges, dropped


def compute_subaverages(
    data: np.ndarray,
    isi_pre_ms: np.ndarray,
    *,
    bin_width_ms: float,
    isi_range_ms: Optional[Tuple[float, float]] = None,
    min_trials_per_bin: int = 10,
    drop_empty_bins: bool = True,
    drop_small_bins: bool = True,
    use_only_in_range_trials_for_conventional: bool = True,
) -> SubaverageResult:
    """
    Compute ISI-binned subaverages (ERPs) for ADJAR estimation.

    Args:
        data: (trials, channels, times)
        isi_pre_ms: (trials,)
        bin_width_ms: bin width for preceding ISI
        isi_range_ms: optional explicit range
        min_trials_per_bin: minimum trials required to keep a bin if drop_small_bins is True
        drop_empty_bins: drop bins with zero trials
        drop_small_bins: drop bins with n_trials < min_trials_per_bin
        use_only_in_range_trials_for_conventional: if True, conventional ERP is computed
            only from trials that are not out-of-range. This matches typical ADJAR practice
            where you only correct what you model.

    Returns:
        SubaverageResult
    """
    X = np.asarray(data)
    if X.ndim != 3:
        raise ValueError(f"data must be 3D (trials, channels, times), got shape {X.shape}")
    if X.shape[0] < 1:
        raise ValueError("data must have at least 1 trial")
    if not np.isfinite(X).all():
        raise ValueError("data contains NaN/inf")

    isi = np.asarray(isi_pre_ms, dtype=np.float64)
    if isi.ndim != 1 or isi.shape[0] != X.shape[0]:
        raise ValueError("isi_pre_ms must be 1D and match n_trials")
    if not np.isfinite(isi).all():
        raise ValueError("isi_pre_ms contains NaN/inf")
    if np.any(isi < 0):
        raise ValueError("isi_pre_ms contains negative values; must be >= 0")

    bin_idx, edges, dropped_trials = bin_trials_by_isi(
        isi, bin_width_ms=bin_width_ms, isi_range_ms=isi_range_ms,
        min_trials_per_bin=min_trials_per_bin, drop_empty_bins=drop_empty_bins
    )

    n_bins = edges.size - 1
    bins: List[SubaverageBin] = []

    # Determine which trials are in range
    in_range_mask = bin_idx >= 0
    in_range_trials = np.where(in_range_mask)[0].astype(np.int64)

    if use_only_in_range_trials_for_conventional:
        conv = np.mean(X[in_range_trials, :, :], axis=0) if in_range_trials.size > 0 else np.mean(X, axis=0)
    else:
        conv = np.mean(X, axis=0)

    n_used = 0
    dropped_due_to_small: List[int] = []

    for k in range(n_bins):
        trials_k = np.where(bin_idx == k)[0].astype(np.int64)
        n_k = int(trials_k.size)

        if n_k == 0 and drop_empty_bins:
            continue
        if drop_small_bins and n_k > 0 and n_k < int(min_trials_per_bin):
            dropped_due_to_small.extend(trials_k.tolist())
            continue

        if n_k > 0:
            erp_k = np.mean(X[trials_k, :, :], axis=0)
            isi_vals = isi[trials_k].astype(np.float64, copy=False)
            n_used += n_k
        else:
            # Keep empty bin only if drop_empty_bins is False
            erp_k = np.zeros((X.shape[1], X.shape[2]), dtype=np.float64)
            isi_vals = np.zeros((0,), dtype=np.float64)

        left = float(edges[k])
        right = float(edges[k + 1])
        center = 0.5 * (left + right)

        bins.append(
            SubaverageBin(
                bin_index=len(bins),
                left_ms=left,
                right_ms=right,
                center_ms=center,
                trial_indices=trials_k,
                n_trials=n_k,
                erp=erp_k.astype(np.float64, copy=False),
                isi_values_ms=isi_vals,
            )
        )

    dropped_all = np.unique(np.concatenate([dropped_trials, np.asarray(dropped_due_to_small, dtype=np.int64)]))
    dropped_all.sort()

    return SubaverageResult(
        bins=bins,
        edges_ms=edges.astype(np.float64, copy=False),
        bin_width_ms=float(bin_width_ms),
        n_trials_total=int(X.shape[0]),
        n_trials_used=int(n_used),
        dropped_trial_indices=dropped_all.astype(np.int64, copy=False),
        conventional_erp=conv.astype(np.float64, copy=False),
    )
