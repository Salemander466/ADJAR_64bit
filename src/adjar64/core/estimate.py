from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from adjar64 import get_logger

from .distributions import distribution_from_single_lag
from .subaverages import SubaverageResult

log = get_logger("core.estimate")


@dataclass(frozen=True)
class EstimateResult:
    """
    Result of estimating the previous-response waveform R_prev(t).

    Attributes:
        r_prev: estimated previous-response waveform, shape (channels, times)
        erp_true: estimated underlying (overlap-free) ERP, shape (channels, times)
        used_bin_centers_ms: bin centers used in estimation
        used_bin_sizes: trials per bin used
        notes: description of the method and constraints
    """
    r_prev: np.ndarray
    erp_true: np.ndarray
    used_bin_centers_ms: np.ndarray
    used_bin_sizes: np.ndarray
    notes: str


def estimate_previous_response(
    subavg: SubaverageResult,
    *,
    fs_hz: float,
    tmin_ms: float,
    ridge_lambda: float = 0.0,
    lag_mode: str = "bin_center_impulse",
    min_bins: int = 3,
) -> EstimateResult:
    """
    Estimate R_prev(t) from ISI-binned subaverages using a regression formulation.

    Core model (per channel, per timepoint t):
        ERP_sub_k(t) = ERP_true(t) + Overlap_k(t)
    where Overlap_k(t) is the preceding-response contribution determined by the
    lag distribution for bin k.

    Practical approach implemented here:
    - For each ISI bin k, approximate its preceding-event distribution as a single
      impulse at lag = -center_ms (preceding event occurs before time 0).
    - The overlap waveform becomes a shifted version of R_prev:
        Overlap_k(t) = R_prev(t + center_ms)  (sign consistent with lag = -center_ms)
      In discrete time, this corresponds to indexing shift.

    Then we solve, per (channel, timepoint), a linear regression across bins:
        y_k = ERP_sub_k(t) = ERP_true(t) + x_k
    where x_k is the shifted R_prev contribution at that timepoint.

    Implementation notes:
    - We estimate ERP_true as the average across bins after removing best-fit overlap.
    - We estimate R_prev with ridge regularization (optional) to improve stability.
    - This estimator is designed to be a good first-pass (Level 1) estimator.
      You can refine it later to match exact Woldorff derivations if needed.

    Args:
        subavg: SubaverageResult containing bins and ERPs.
        fs_hz, tmin_ms: epoch grid metadata (needed for lag mapping).
        ridge_lambda: ridge penalty for stability (0 disables).
        lag_mode: currently only "bin_center_impulse" supported.
        min_bins: minimum number of bins required for estimation.

    Returns:
        EstimateResult

    Raises:
        ValueError if insufficient bins/trials or invalid shapes.
    """
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be >= 0")

    bins = [b for b in subavg.bins if b.n_trials > 0]
    if len(bins) < min_bins:
        raise ValueError(
            f"Need at least {min_bins} non-empty bins for estimation; got {len(bins)}. "
            "Decrease min_trials_per_bin or increase dataset size."
        )

    # Assume all bins have ERP shape (channels, times)
    n_ch, n_times = bins[0].erp.shape
    for b in bins:
        if b.erp.shape != (n_ch, n_times):
            raise ValueError("Inconsistent ERP shapes across bins")

    # Stack subaverages: (K, channels, times)
    K = len(bins)
    Y = np.stack([b.erp for b in bins], axis=0).astype(np.float64, copy=False)

    centers_ms = np.asarray([b.center_ms for b in bins], dtype=np.float64)
    sizes = np.asarray([b.n_trials for b in bins], dtype=np.int64)

    dt_ms = 1000.0 / float(fs_hz)

    # Convert lag = -center_ms to sample shift.
    # If preceding event is at time -center_ms, then its waveform appears in current epoch
    # shifted right by center_ms. For overlap at time t (index i), contribution is R_prev at (i + shift).
    shift_samp = np.rint(centers_ms / dt_ms).astype(np.int64)

    # Build a design matrix that relates R_prev to observed Y across bins.
    #
    # We solve for each channel separately:
    # For each timepoint i, we have K observations y_k(i) = ERP_true(i) + R_prev(i + shift_k).
    #
    # This is not a standard linear regression in one step because it references shifted indices of R_prev.
    # We reformulate as a global least squares problem over all timepoints:
    # Unknowns: ERP_true[time] and R_prev[time]
    #
    # For each bin k and time i:
    #   y = ERP_true[i] + R_prev[j]   where j = i + shift_k
    # Only include constraints where j is within [0, n_times-1].
    #
    # This yields a sparse linear system:
    #   y = A * theta
    # theta = [ERP_true(0..T-1), R_prev(0..T-1)]
    #
    # We solve independently per channel using dense normal equations with careful construction.
    # Complexity: O(K*T) constraints; unknowns ~ 2T. For typical ERP lengths, this is fine.
    #
    # Ridge regularization applies only to R_prev part by default.
    #
    # This is fast enough for practical use, and deterministic.
    #
    T = n_times
    n_theta = 2 * T

    # Precompute constraint indices
    # For each bin k and time i, map to row r, and theta indices for ERP_true and R_prev.
    rows_true: List[np.ndarray] = []
    rows_prev: List[np.ndarray] = []
    obs_index: List[Tuple[int, np.ndarray]] = []

    # We'll build lists of (i_valid, j_valid) per bin to avoid huge temporary arrays.
    for k in range(K):
        s = int(shift_samp[k])
        i = np.arange(T, dtype=np.int64)
        j = i + s
        valid = (j >= 0) & (j < T)
        i_v = i[valid]
        j_v = j[valid]
        if i_v.size == 0:
            # This bin contributes no constraints; skip it
            continue
        rows_true.append(i_v)         # ERP_true index is i
        rows_prev.append(j_v)         # R_prev index is j
        obs_index.append((k, i_v))    # Y[k, :, i_v] are observations

    if len(obs_index) < min_bins:
        raise ValueError(
            "Too few usable bins after lag-to-epoch mapping. "
            "Your epoch may be too short relative to ISIs."
        )

    # Total number of constraints (rows)
    M = int(sum(idx[1].size for idx in obs_index))

    # For each channel, solve:
    # minimize ||A*theta - y||^2 + ridge_lambda * ||R_prev||^2
    #
    # We form normal equations: (A^T A + R) theta = A^T y
    # where R applies ridge to the R_prev block.
    #
    # Build A^T A analytically without forming full A (sparse structure):
    #
    # Each row has two ones: one at ERP_true[i], one at R_prev[j].
    # Therefore:
    # - ERP_true block: counts of how many times each i appears
    # - R_prev block: counts of how many times each j appears
    # - cross block: counts for pairs (i, j) occurrences
    #
    # A^T A is (2T x 2T). We construct it as dense float64 for simplicity.
    #
    ATA = np.zeros((n_theta, n_theta), dtype=np.float64)

    # Fill diagonals and cross terms
    # ERP_true diagonal counts
    true_counts = np.zeros((T,), dtype=np.float64)
    prev_counts = np.zeros((T,), dtype=np.float64)

    # Cross counts between ERP_true[i] and R_prev[j]
    # We accumulate into a dense (T x T) cross block.
    cross = np.zeros((T, T), dtype=np.float64)

    for (k, i_v), j_v in zip(obs_index, rows_prev):
        # Each valid constraint contributes:
        # true_counts[i] += 1
        # prev_counts[j] += 1
        # cross[i, j] += 1
        true_counts[i_v] += 1.0
        prev_counts[j_v] += 1.0
        cross[i_v, j_v] += 1.0

    # Assemble ATA:
    # [diag(true_counts)        cross
    #  cross.T                 diag(prev_counts) + ridge]
    ATA[:T, :T] = np.diag(true_counts)
    ATA[T:, T:] = np.diag(prev_counts)
    ATA[:T, T:] = cross
    ATA[T:, :T] = cross.T

    if ridge_lambda > 0:
        # Ridge on R_prev block only
        ATA[T:, T:] += ridge_lambda * np.eye(T, dtype=np.float64)

    # Now compute ATy for each channel.
    # ATy = [sum over rows of y for each true index i,
    #        sum over rows of y for each prev index j]
    #
    # We'll accumulate from Y.
    #
    r_prev = np.zeros((n_ch, T), dtype=np.float64)
    erp_true = np.zeros((n_ch, T), dtype=np.float64)

    # Precompute mapping arrays for fast accumulation
    # Create flattened lists of (k, i_v, j_v) for constraints.
    ks_list: List[np.ndarray] = []
    i_list: List[np.ndarray] = []
    j_list: List[np.ndarray] = []
    for (k, i_v), j_v in zip(obs_index, rows_prev):
        ks_list.append(np.full(i_v.shape, k, dtype=np.int64))
        i_list.append(i_v.astype(np.int64, copy=False))
        j_list.append(j_v.astype(np.int64, copy=False))

    ks_flat = np.concatenate(ks_list) if ks_list else np.zeros((0,), dtype=np.int64)
    i_flat = np.concatenate(i_list) if i_list else np.zeros((0,), dtype=np.int64)
    j_flat = np.concatenate(j_list) if j_list else np.zeros((0,), dtype=np.int64)

    if ks_flat.size != M:
        # Sanity check; not critical but helps debugging.
        log.warning("Constraint count mismatch: expected %d, got %d", M, ks_flat.size)
        M = ks_flat.size

    for ch in range(n_ch):
        # Observations y for this channel across constraints:
        # y_m = Y[k_m, ch, i_m]
        y = Y[ks_flat, ch, i_flat].astype(np.float64, copy=False)

        ATy = np.zeros((n_theta,), dtype=np.float64)
        # Accumulate into ERP_true indices (0..T-1) and R_prev indices (T..2T-1)
        np.add.at(ATy, i_flat, y)
        np.add.at(ATy, T + j_flat, y)

        # Solve linear system
        try:
            theta = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            # Fall back to least squares if singular
            theta, *_ = np.linalg.lstsq(ATA, ATy, rcond=None)

        erp_true[ch, :] = theta[:T]
        r_prev[ch, :] = theta[T:]

    notes = (
        "Estimated R_prev and ERP_true from ISI-binned subaverages using a sparse-structure "
        "global least squares system: y[k,i]=ERP_true[i]+R_prev[i+shift_k]. "
        f"Bins used={K}, ridge_lambda={ridge_lambda}, dt_ms={dt_ms:.6g}."
    )

    return EstimateResult(
        r_prev=r_prev,
        erp_true=erp_true,
        used_bin_centers_ms=centers_ms,
        used_bin_sizes=sizes,
        notes=notes,
    )
