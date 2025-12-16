from __future__ import annotations

import numpy as np

from adjar64.core import AdjarConfig
from adjar64.core.correct import run_level1
from adjar64.core.distributions import build_preceding_distribution
from adjar64.accel.fftconv import fft_convolve_batch


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def _gauss(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _make_convolution_consistent_synthetic(seed: int = 123):
    rng = np.random.default_rng(seed)

    fs_hz = 250.0
    tmin_ms = -200.0
    tmax_ms = 800.0
    dt_ms = 1000.0 / fs_hz
    n_times = int(np.round((tmax_ms - tmin_ms) / dt_ms)) + 1
    t_ms = tmin_ms + np.arange(n_times, dtype=np.float64) * dt_ms

    n_trials = 4000  # more trials -> subaverages are stable
    n_channels = 8

    # True ERP and true preceding response
    base_erp = 1.5 * _gauss(t_ms, 120.0, 25.0) - 1.0 * _gauss(t_ms, 180.0, 35.0) + 0.7 * _gauss(t_ms, 320.0, 60.0)
    base_erp *= (t_ms >= 0).astype(np.float64)

    base_prev = 1.2 * _gauss(t_ms, 90.0, 22.0) - 0.8 * _gauss(t_ms, 150.0, 30.0) + 0.4 * _gauss(t_ms, 260.0, 55.0)
    base_prev *= (t_ms >= 0).astype(np.float64)

    ch_scale_erp = 1.0 + 0.10 * rng.standard_normal(n_channels)
    ch_scale_prev = 1.0 + 0.10 * rng.standard_normal(n_channels)

    erp_true = (ch_scale_erp[:, None] * base_erp[None, :]).astype(np.float64)
    r_prev_true = (ch_scale_prev[:, None] * base_prev[None, :]).astype(np.float64)

    # Draw ISIs that produce meaningful overlap inside epoch
    isi_pre_ms = rng.normal(loc=220.0, scale=70.0, size=n_trials).astype(np.float64)
    isi_pre_ms = np.clip(isi_pre_ms, 50.0, 450.0)

    # Build D_pre exactly as Level 1 does
    dist = build_preceding_distribution(
        isi_pre_ms,
        fs_hz=fs_hz,
        tmin_ms=tmin_ms,
        n_times=n_times,
        normalize="probability",
        smoothing_ms=0.0,
        clamp_to_epoch=True,
    )

    # Expected-value overlap (ERP-level) as Level 1 predicts
    overlap_expected = fft_convolve_batch(r_prev_true, dist.d_pre, axis=-1, mode="same", use_real_fft=True)
    overlap_expected = np.asarray(overlap_expected, dtype=np.float64)

    # Generate trials around the expected ERP with small noise
    noise_std = 0.15
    data = np.zeros((n_trials, n_channels, n_times), dtype=np.float64)
    for tr in range(n_trials):
        data[tr] = erp_true + overlap_expected + noise_std * rng.standard_normal((n_channels, n_times))

    return data, isi_pre_ms, erp_true, fs_hz, tmin_ms, tmax_ms


def test_level1_reduces_error_vs_conventional() -> None:
    data, isi_pre_ms, erp_true, fs_hz, tmin_ms, tmax_ms = _make_convolution_consistent_synthetic(seed=123)

    cfg = AdjarConfig(
        fs_hz=fs_hz,
        tmin_ms=tmin_ms,
        tmax_ms=tmax_ms,
        isi_bin_width_ms=50.0,
        use_woldorff_gain_filter=False,
        manual_lowpass_hz=None,
        level2_enabled=True,
        max_iter=8,
        tol_rms=1e-6,
        ridge_lambda=1e-3,
        convergence_window_ms=None,
    )

    out = run_level1(data, isi_pre_ms, config=cfg)

    erp_conv = np.asarray(out["erp_conv"], dtype=np.float64)
    erp_l1 = np.asarray(out["erp_l1"], dtype=np.float64)

    err_conv = _rms(erp_conv - erp_true)
    err_l1 = _rms(erp_l1 - erp_true)

    # With convolution-consistent synthetic data, L1 should improve.
    assert err_l1 < err_conv, f"Expected L1 error < conv error, got {err_l1} vs {err_conv}"
