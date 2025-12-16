from __future__ import annotations

import numpy as np

from adjar64.core import AdjarConfig
from adjar64.core.correct import run_level1, run_level2


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def _make_synthetic(seed: int = 321) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    rng = np.random.default_rng(seed)

    fs_hz = 250.0
    tmin_ms = -200.0
    tmax_ms = 800.0
    dt_ms = 1000.0 / fs_hz
    n_times = int(np.round((tmax_ms - tmin_ms) / dt_ms)) + 1
    t_ms = tmin_ms + np.arange(n_times) * dt_ms

    def gauss(mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((t_ms - mu) / sigma) ** 2)

    n_trials = 1600
    n_channels = 8

    base_erp = 1.4 * gauss(110.0, 24.0) - 0.9 * gauss(175.0, 34.0) + 0.6 * gauss(310.0, 55.0)
    base_erp *= (t_ms >= 0).astype(np.float64)

    base_prev = 1.3 * gauss(85.0, 21.0) - 0.7 * gauss(145.0, 29.0) + 0.45 * gauss(250.0, 50.0)
    base_prev *= (t_ms >= 0).astype(np.float64)

    ch_scale_erp = 1.0 + 0.12 * rng.standard_normal(n_channels)
    ch_scale_prev = 1.0 + 0.12 * rng.standard_normal(n_channels)

    erp_true = (ch_scale_erp[:, None] * base_erp[None, :]).astype(np.float64)
    r_prev_true = (ch_scale_prev[:, None] * base_prev[None, :]).astype(np.float64)

    # Make overlap moderately severe and variable
    isi_pre_ms = rng.normal(loc=200.0, scale=60.0, size=n_trials).astype(np.float64)
    isi_pre_ms = np.clip(isi_pre_ms, 60.0, 420.0)

    shifts = np.rint(isi_pre_ms / dt_ms).astype(np.int64)
    shifts = np.clip(shifts, 0, n_times)

    noise_std = 0.22
    data = np.zeros((n_trials, n_channels, n_times), dtype=np.float64)

    for tr in range(n_trials):
        s = int(shifts[tr])
        overlap = np.zeros((n_channels, n_times), dtype=np.float64)
        if s <= 0:
            overlap[:, :] = r_prev_true
        elif s < n_times:
            overlap[:, : (n_times - s)] = r_prev_true[:, s:]

        data[tr, :, :] = erp_true + overlap + noise_std * rng.standard_normal((n_channels, n_times))

    return data, isi_pre_ms, erp_true, fs_hz, tmin_ms, tmax_ms


def test_level2_improves_over_level1_and_converges() -> None:
    data, isi_pre_ms, erp_true, fs_hz, tmin_ms, tmax_ms = _make_synthetic(seed=321)

    cfg = AdjarConfig(
        fs_hz=fs_hz,
        tmin_ms=tmin_ms,
        tmax_ms=tmax_ms,
        isi_bin_width_ms=50.0,
        use_woldorff_gain_filter=False,
        manual_lowpass_hz=None,
        level2_enabled=True,
        max_iter=10,
        tol_rms=1e-6,
        ridge_lambda=1e-3,
        convergence_window_ms=None,
    )

    out1 = run_level1(data, isi_pre_ms, config=cfg)
    out2 = run_level2(data, isi_pre_ms, config=cfg)

    erp_l1 = np.asarray(out1["erp_l1"])
    erp_l2 = np.asarray(out2["erp_l2"])

    err_l1 = _rms(erp_l1 - erp_true)
    err_l2 = _rms(erp_l2 - erp_true)

    # Level 2 should be at least slightly better than Level 1 on this synthetic construction.
    # Conservative margin to avoid flakiness.
    assert err_l2 < 0.99 * err_l1, f"Expected L2 error < 0.99*L1 error, got {err_l2} vs {err_l1}"

    hist = out2.get("level2_history", [])
    assert isinstance(hist, list) and len(hist) >= 1

    # Convergence behavior: the final rms_delta should be <= the first rms_delta in most cases.
    first_delta = float(hist[0]["rms_delta"])
    last_delta = float(hist[-1]["rms_delta"])
    assert last_delta <= first_delta + 1e-12, f"Expected convergence (delta not increasing), got {last_delta} vs {first_delta}"
