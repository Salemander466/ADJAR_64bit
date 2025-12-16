from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adjar64 import get_logger
from adjar64.accel.fftconv import fft_convolve_batch

from .distributions import build_preceding_distribution, DistributionResult
from .estimate import estimate_previous_response, EstimateResult
from .filter import GainCurve, compute_gain_curve, apply_gain_filter, apply_manual_lowpass
from .metrics import convergence_rms, qa_stats
from .subaverages import compute_subaverages, SubaverageResult
from . import AdjarConfig

log = get_logger("core.correct")


# =============================================================================
# Public API
# =============================================================================

def run_level1(
    data: np.ndarray,
    isi_pre_ms: np.ndarray,
    *,
    config: AdjarConfig,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run ADJAR Level 1 overlap correction.

    Inputs:
        data: (trials, channels, times) float-like
        isi_pre_ms: (trials,) preceding intervals in ms
        config: AdjarConfig

    Returns dict:
        erp_conv: (channels, times)
        erp_l1: (channels, times)
        r_prev: (channels, times) estimated previous-response waveform (filtered if applicable)
        r_prev_raw: (channels, times) raw estimated previous-response before filtering
        erp_true_est: (channels, times) estimated underlying ERP from estimator
        d_pre: (times,) distribution kernel aligned to epoch
        gain_curve: (F,) gain values (rFFT bins) if used
        gain_freqs_hz: (F,) freqs for gain curve
        meta: dict
    """
    X, isi = _validate_inputs(data, isi_pre_ms)
    n_trials, n_ch, n_times = X.shape

    _validate_config_basic(config, n_times)

    erp_conv = np.mean(X, axis=0)

    # Build D_pre(t) aligned to epoch
    dist = build_preceding_distribution(
        isi,
        fs_hz=config.fs_hz,
        tmin_ms=config.tmin_ms,
        n_times=n_times,
        normalize="probability",
        smoothing_ms=0.0,
        clamp_to_epoch=True,
    )

    # Subaverages for estimation
    subavg = compute_subaverages(
        X,
        isi,
        bin_width_ms=config.isi_bin_width_ms,
        isi_range_ms=None,
        min_trials_per_bin=10,
        drop_empty_bins=True,
        drop_small_bins=True,
        use_only_in_range_trials_for_conventional=True,
    )

    # Estimate R_prev (raw)
    est = estimate_previous_response(
        subavg,
        fs_hz=config.fs_hz,
        tmin_ms=config.tmin_ms,
        ridge_lambda=config.ridge_lambda,
        lag_mode="bin_center_impulse",
        min_bins=3,
    )

    r_prev_raw = est.r_prev
    r_prev = r_prev_raw

    gain = None
    if config.manual_lowpass_hz is not None and config.manual_lowpass_hz > 0:
        # Manual low-pass option: prefer FFT gain path for reproducibility
        gain = compute_gain_curve(
            fs_hz=config.fs_hz,
            n_times=n_times,
            kind="manual_lowpass",
            manual_lowpass_hz=float(config.manual_lowpass_hz),
            transition_width_hz=None,
        )
        r_prev = apply_gain_filter(r_prev, gain, axis=-1)
    elif config.use_woldorff_gain_filter:
        gain = compute_gain_curve(
            fs_hz=config.fs_hz,
            n_times=n_times,
            isi_pre_ms=isi,
            kind="woldorff_placeholder",
            manual_lowpass_hz=None,
            transition_width_hz=None,
        )
        r_prev = apply_gain_filter(r_prev, gain, axis=-1)

    # Predict overlap in ERP via convolution with D_pre
    overlap_pred = _predict_overlap_erp(r_prev, dist.d_pre)

    erp_l1 = erp_conv - overlap_pred

    meta = _build_meta(
        config=config,
        n_trials=n_trials,
        n_channels=n_ch,
        n_times=n_times,
        channel_names=channel_names,
        level="L1",
        extra={
            "distribution_notes": dist.notes,
            "estimate_notes": est.notes,
            "gain_kind": gain.kind if gain is not None else "none",
            "gain_params": gain.params if gain is not None else {},
        },
        isi=isi,
    )

    out: Dict[str, Any] = {
        "erp_conv": erp_conv,
        "erp_l1": erp_l1,
        "r_prev_raw": r_prev_raw,
        "r_prev": r_prev,
        "erp_true_est": est.erp_true,
        "d_pre": dist.d_pre,
        "d_pre_counts": dist.hist_counts,
        "d_pre_prob": dist.hist_prob,
        "meta": meta,
    }
    if gain is not None:
        out["gain_curve"] = gain.gain
        out["gain_freqs_hz"] = gain.freqs_hz
    return out


def run_level2(
    data: np.ndarray,
    isi_pre_ms: np.ndarray,
    *,
    config: AdjarConfig,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run ADJAR Level 2 iterative overlap correction.

    Approach implemented:
      - Iteratively estimate R_prev from subaverages of the current corrected trial data.
      - Predict per-trial overlap using trial-specific ISI shifts (impulse model).
      - Subtract predicted overlap from the original trial data to obtain updated corrected trials.
      - Converge based on RMS change in corrected ERP across iterations.

    Returns dict (superset of Level 1 style keys):
        erp_conv, erp_l2, r_prev, r_prev_raw, d_pre, gain_curve, meta, plus:
        level2_history: list of per-iter diagnostics (rms_delta, effective_cutoff, etc.)
    """
    X, isi = _validate_inputs(data, isi_pre_ms)
    n_trials, n_ch, n_times = X.shape

    _validate_config_basic(config, n_times)

    erp_conv = np.mean(X, axis=0)

    # Distribution for ERP-level diagnostics and (optional) ERP-level overlap prediction
    dist = build_preceding_distribution(
        isi,
        fs_hz=config.fs_hz,
        tmin_ms=config.tmin_ms,
        n_times=n_times,
        normalize="probability",
        smoothing_ms=0.0,
        clamp_to_epoch=True,
    )

    # Iteration state:
    # Start with no correction: corrected_trials = original trials
    corrected_trials = X.copy()
    prev_erp = np.mean(corrected_trials, axis=0)

    history: List[Dict[str, Any]] = []

    # Fixed binning settings
    bin_width = config.isi_bin_width_ms

    final_gain: Optional[GainCurve] = None
    final_est: Optional[EstimateResult] = None
    final_r_prev_raw: Optional[np.ndarray] = None
    final_r_prev: Optional[np.ndarray] = None

    for it in range(int(config.max_iter)):
        # Recompute subaverages from current corrected trials
        subavg = compute_subaverages(
            corrected_trials,
            isi,
            bin_width_ms=bin_width,
            isi_range_ms=None,
            min_trials_per_bin=10,
            drop_empty_bins=True,
            drop_small_bins=True,
            use_only_in_range_trials_for_conventional=True,
        )

        # Estimate R_prev from these subaverages
        est = estimate_previous_response(
            subavg,
            fs_hz=config.fs_hz,
            tmin_ms=config.tmin_ms,
            ridge_lambda=config.ridge_lambda,
            lag_mode="bin_center_impulse",
            min_bins=3,
        )

        r_prev_raw = est.r_prev
        r_prev = r_prev_raw

        gain = None
        if config.manual_lowpass_hz is not None and config.manual_lowpass_hz > 0:
            gain = compute_gain_curve(
                fs_hz=config.fs_hz,
                n_times=n_times,
                kind="manual_lowpass",
                manual_lowpass_hz=float(config.manual_lowpass_hz),
                transition_width_hz=None,
            )
            r_prev = apply_gain_filter(r_prev, gain, axis=-1)
        elif config.use_woldorff_gain_filter:
            gain = compute_gain_curve(
                fs_hz=config.fs_hz,
                n_times=n_times,
                isi_pre_ms=isi,
                kind="woldorff_placeholder",
                manual_lowpass_hz=None,
                transition_width_hz=None,
            )
            r_prev = apply_gain_filter(r_prev, gain, axis=-1)

        # Predict per-trial overlap using ISI shift (fast slicing)
        overlap_trials = _predict_overlap_trials(r_prev, isi, fs_hz=config.fs_hz)

        # Update corrected trials by subtracting predicted overlap from ORIGINAL trials.
        # This avoids compounding errors by repeatedly subtracting from an already corrected signal.
        corrected_trials = X - overlap_trials

        # Compute corrected ERP for convergence
        curr_erp = np.mean(corrected_trials, axis=0)
        delta = convergence_rms(
            prev_erp,
            curr_erp,
            fs_hz=config.fs_hz,
            tmin_ms=config.tmin_ms,
            window_ms=config.convergence_window_ms,
        )

        hist_item = {
            "iter": int(it + 1),
            "rms_delta": float(delta),
            "gain_kind": gain.kind if gain is not None else "none",
            "gain_params": gain.params if gain is not None else {},
        }
        history.append(hist_item)

        # Store final iteration results
        final_gain = gain
        final_est = est
        final_r_prev_raw = r_prev_raw
        final_r_prev = r_prev

        prev_erp = curr_erp

        log.info("Level2 iter %d: rms_delta=%.6g", it + 1, delta)

        if delta <= float(config.tol_rms):
            break

    if final_est is None or final_r_prev is None or final_r_prev_raw is None:
        raise RuntimeError("Level 2 did not produce an estimate; check input data and binning settings.")

    # For output, also provide ERP-level overlap prediction via distribution convolution (diagnostic)
    overlap_pred_erp = _predict_overlap_erp(final_r_prev, dist.d_pre)
    erp_l2 = erp_conv - overlap_pred_erp

    meta = _build_meta(
        config=config,
        n_trials=n_trials,
        n_channels=n_ch,
        n_times=n_times,
        channel_names=channel_names,
        level="L2",
        extra={
            "distribution_notes": dist.notes,
            "estimate_notes": final_est.notes,
            "gain_kind": final_gain.kind if final_gain is not None else "none",
            "gain_params": final_gain.params if final_gain is not None else {},
            "level2_history": history,
            "level2_iters": int(len(history)),
        },
        isi=isi,
    )

    out: Dict[str, Any] = {
        "erp_conv": erp_conv,
        "erp_l2": erp_l2,
        "r_prev_raw": final_r_prev_raw,
        "r_prev": final_r_prev,
        "erp_true_est": final_est.erp_true,
        "d_pre": dist.d_pre,
        "d_pre_counts": dist.hist_counts,
        "d_pre_prob": dist.hist_prob,
        "level2_history": history,
        "meta": meta,
    }
    if final_gain is not None:
        out["gain_curve"] = final_gain.gain
        out["gain_freqs_hz"] = final_gain.freqs_hz
    return out


def run_all(
    data: np.ndarray,
    isi_pre_ms: np.ndarray,
    *,
    config: AdjarConfig,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper that runs Level 1 and (optionally) Level 2 and merges results.

    If config.level2_enabled is False, this returns only Level 1 outputs.
    """
    out1 = run_level1(data, isi_pre_ms, config=config, channel_names=channel_names)
    if not config.level2_enabled:
        return out1

    out2 = run_level2(data, isi_pre_ms, config=config, channel_names=channel_names)

    merged = dict(out1)
    # Prefer Level 2 keys where overlapping
    for k, v in out2.items():
        if k in ("erp_conv",):
            continue
        merged[k] = v

    # Provide a unified "best" ERP key for convenience
    if "erp_l2" in merged:
        merged["erp_best"] = merged["erp_l2"]
    else:
        merged["erp_best"] = merged.get("erp_l1", merged["erp_conv"])

    return merged


# =============================================================================
# Internal helpers
# =============================================================================

def _validate_inputs(data: np.ndarray, isi_pre_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(data, dtype=np.float64, order="C")
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
    return X, isi


def _validate_config_basic(config: AdjarConfig, n_times: int) -> None:
    if config.fs_hz <= 0:
        raise ValueError("config.fs_hz must be > 0")
    if config.tmax_ms <= config.tmin_ms:
        raise ValueError("config.tmax_ms must be > config.tmin_ms")
    if config.isi_bin_width_ms <= 0:
        raise ValueError("config.isi_bin_width_ms must be > 0")
    if config.max_iter < 1:
        raise ValueError("config.max_iter must be >= 1")
    if config.tol_rms < 0:
        raise ValueError("config.tol_rms must be >= 0")
    if n_times < 2:
        raise ValueError("epoch length too short")


def _predict_overlap_erp(r_prev: np.ndarray, d_pre: np.ndarray) -> np.ndarray:
    """
    Predict overlap in ERP using convolution:
        overlap(t) = (d_pre * r_prev)(t)
    computed per channel in FFT domain.

    Returns:
        overlap_pred: (channels, times) in "same" mode w.r.t. r_prev length
    """
    r_prev = np.asarray(r_prev, dtype=np.float64)
    d_pre = np.asarray(d_pre, dtype=np.float64)
    if r_prev.ndim != 2:
        raise ValueError("r_prev must be 2D (channels, times)")
    if d_pre.ndim != 1:
        raise ValueError("d_pre must be 1D (times,)")

    # Convolve along time axis
    overlap = fft_convolve_batch(r_prev, d_pre, axis=-1, mode="same", use_real_fft=True)
    overlap = np.asarray(overlap, dtype=np.float64)
    return overlap


def _predict_overlap_trials(r_prev: np.ndarray, isi_pre_ms: np.ndarray, *, fs_hz: float) -> np.ndarray:
    """
    Predict overlap contribution for each trial using an impulse-at-lag model.

    For trial with ISI = s ms (preceding event at -s ms), the preceding waveform appears
    shifted right by shift samples in the current epoch:
        overlap_trial[:, i] = r_prev[:, i + shift] (for i+shift within bounds)

    Returns:
        overlap_trials: (trials, channels, times)
    """
    r_prev = np.asarray(r_prev, dtype=np.float64)
    isi = np.asarray(isi_pre_ms, dtype=np.float64)

    if r_prev.ndim != 2:
        raise ValueError("r_prev must be (channels, times)")
    if isi.ndim != 1:
        raise ValueError("isi_pre_ms must be 1D")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")

    n_ch, T = r_prev.shape
    n_trials = isi.shape[0]
    dt_ms = 1000.0 / float(fs_hz)

    shifts = np.rint(isi / dt_ms).astype(np.int64)
    shifts = np.clip(shifts, 0, T)  # shift==T -> zero overlap

    out = np.zeros((n_trials, n_ch, T), dtype=np.float64)

    # Efficient per-trial slicing (loop over trials; each iteration is a single slice assignment)
    # This is typically fast enough in NumPy for thousands of trials.
    for tr in range(n_trials):
        s = int(shifts[tr])
        if s <= 0:
            # overlap_trial[:, i] = r_prev[:, i]
            out[tr, :, :] = r_prev
        elif s >= T:
            continue
        else:
            # for i in [0, T-s): overlap[:, i] = r_prev[:, i+s]
            out[tr, :, : (T - s)] = r_prev[:, s:]
    return out


def _build_meta(
    *,
    config: AdjarConfig,
    n_trials: int,
    n_channels: int,
    n_times: int,
    channel_names: Optional[List[str]],
    level: str,
    extra: Dict[str, Any],
    isi: np.ndarray,
) -> Dict[str, Any]:
    # Basic QA stats (helps debugging and reporting)
    stats = qa_stats(
        data=np.zeros((n_trials, n_channels, n_times), dtype=np.float64),  # dummy for shape
        isi_pre_ms=isi,
        fs_hz=config.fs_hz,
        tmin_ms=config.tmin_ms,
        tmax_ms=config.tmax_ms,
    )

    meta: Dict[str, Any] = {
        "level": level,
        "fs_hz": float(config.fs_hz),
        "tmin_ms": float(config.tmin_ms),
        "tmax_ms": float(config.tmax_ms),
        "n_trials": int(n_trials),
        "n_channels": int(n_channels),
        "n_times": int(n_times),
        "channel_names": channel_names,
        "config": asdict(config),
        "isi_summary": {
            "min_ms": float(np.min(isi)),
            "max_ms": float(np.max(isi)),
            "mean_ms": float(np.mean(isi)),
            "std_ms": float(np.std(isi, ddof=1)) if isi.size > 1 else 0.0,
        },
    }
    meta.update(extra)
    return meta
