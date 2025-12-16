from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np

from adjar64.core import AdjarConfig
from adjar64.core.correct import run_level1, run_level2
from adjar64.io import save_npz, export_erp_npz


def _gaussian(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def generate_synthetic(
    *,
    seed: int,
    n_trials: int,
    n_channels: int,
    fs_hz: float,
    tmin_ms: float,
    tmax_ms: float,
    isi_mean_ms: float,
    isi_std_ms: float,
    noise_std: float,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    dt_ms = 1000.0 / fs_hz
    n_times = int(np.round((tmax_ms - tmin_ms) / dt_ms)) + 1
    t_ms = tmin_ms + np.arange(n_times, dtype=np.float64) * dt_ms

    base_erp = (
        1.5 * _gaussian(t_ms, 120.0, 25.0)
        - 1.0 * _gaussian(t_ms, 180.0, 35.0)
        + 0.7 * _gaussian(t_ms, 320.0, 60.0)
    )
    base_erp *= (t_ms >= 0).astype(np.float64)

    base_prev = (
        1.2 * _gaussian(t_ms, 90.0, 22.0)
        - 0.8 * _gaussian(t_ms, 150.0, 30.0)
        + 0.4 * _gaussian(t_ms, 260.0, 55.0)
    )
    base_prev *= (t_ms >= 0).astype(np.float64)

    ch_scale_erp = 1.0 + 0.15 * rng.standard_normal(n_channels)
    ch_scale_prev = 1.0 + 0.15 * rng.standard_normal(n_channels)

    erp_true = (ch_scale_erp[:, None] * base_erp[None, :]).astype(np.float64)
    r_prev_true = (ch_scale_prev[:, None] * base_prev[None, :]).astype(np.float64)

    isi_pre_ms = rng.normal(loc=isi_mean_ms, scale=isi_std_ms, size=n_trials).astype(np.float64)
    isi_pre_ms = np.clip(isi_pre_ms, 50.0, 450.0)

    shifts = np.rint(isi_pre_ms / dt_ms).astype(np.int64)
    shifts = np.clip(shifts, 0, n_times)

    data = np.zeros((n_trials, n_channels, n_times), dtype=np.float64)
    for tr in range(n_trials):
        s = int(shifts[tr])
        overlap = np.zeros((n_channels, n_times), dtype=np.float64)
        if s <= 0:
            overlap[:, :] = r_prev_true
        elif s < n_times:
            overlap[:, : (n_times - s)] = r_prev_true[:, s:]
        data[tr] = erp_true + overlap + noise_std * rng.standard_normal((n_channels, n_times))

    return {
        "data": data,
        "isi_pre_ms": isi_pre_ms,
        "erp_true": erp_true,
        "r_prev_true": r_prev_true,
        "t_ms": t_ms,
        "fs_hz": np.array(fs_hz, dtype=np.float64),
        "tmin_ms": np.array(tmin_ms, dtype=np.float64),
        "tmax_ms": np.array(tmax_ms, dtype=np.float64),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ADJAR Level 1/2 on synthetic data and write NPZ outputs.")
    ap.add_argument("--outdir", type=str, default="demo_out")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--trials", type=int, default=1400)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--tmin", type=float, default=-200.0)
    ap.add_argument("--tmax", type=float, default=800.0)
    ap.add_argument("--isi_mean", type=float, default=220.0)
    ap.add_argument("--isi_std", type=float, default=70.0)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--bin_width", type=float, default=50.0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max_iter", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    payload = generate_synthetic(
        seed=args.seed,
        n_trials=args.trials,
        n_channels=args.channels,
        fs_hz=args.fs,
        tmin_ms=args.tmin,
        tmax_ms=args.tmax,
        isi_mean_ms=args.isi_mean,
        isi_std_ms=args.isi_std,
        noise_std=args.noise,
    )

    data = payload["data"]
    isi = payload["isi_pre_ms"]
    erp_true = payload["erp_true"]

    cfg = AdjarConfig(
        fs_hz=float(args.fs),
        tmin_ms=float(args.tmin),
        tmax_ms=float(args.tmax),
        isi_bin_width_ms=float(args.bin_width),
        use_woldorff_gain_filter=False,  # keep demo focused on mechanics
        manual_lowpass_hz=None,
        level2_enabled=True,
        max_iter=int(args.max_iter),
        tol_rms=1e-6,
        ridge_lambda=float(args.ridge),
        convergence_window_ms=None,
    )

    # Save GUI-loadable input
    input_path = os.path.join(args.outdir, "synthetic_input.npz")
    save_npz(
        input_path,
        data=data,
        fs_hz=cfg.fs_hz,
        tmin_ms=cfg.tmin_ms,
        tmax_ms=cfg.tmax_ms,
        isi_pre_ms=isi,
        channel_names=[f"Ch{i}" for i in range(data.shape[1])],
        trial_labels=["A"] * data.shape[0],
        extra={"synthetic": True, "seed": int(args.seed)},
    )

    # Save ground truth
    gt_path = os.path.join(args.outdir, "synthetic_ground_truth.npz")
    np.savez_compressed(
        gt_path,
        erp_true=payload["erp_true"],
        r_prev_true=payload["r_prev_true"],
        time_ms=payload["t_ms"],
    )

    out1 = run_level1(data, isi, config=cfg)
    out2 = run_level2(data, isi, config=cfg)

    erp_conv = np.asarray(out1["erp_conv"])
    erp_l1 = np.asarray(out1["erp_l1"])
    erp_l2 = np.asarray(out2["erp_l2"])

    err_conv = _rms(erp_conv - erp_true)
    err_l1 = _rms(erp_l1 - erp_true)
    err_l2 = _rms(erp_l2 - erp_true)

    print("Errors vs ground truth ERP_true (RMS over channels,time):")
    print(f"  Conventional: {err_conv:.6g}")
    print(f"  Level 1:      {err_l1:.6g}  (improvement: {(err_conv - err_l1)/max(err_conv,1e-12):.2%})")
    print(f"  Level 2:      {err_l2:.6g}  (improvement vs L1: {(err_l1 - err_l2)/max(err_l1,1e-12):.2%})")

    hist = out2.get("level2_history", [])
    if isinstance(hist, list) and hist:
        print("Level 2 convergence (rms_delta per iter):")
        for h in hist:
            print(f"  iter={h['iter']:2d}  rms_delta={float(h['rms_delta']):.6g}")

    # Export results for GUI inspection
    res1_path = os.path.join(args.outdir, "results_level1.npz")
    res2_path = os.path.join(args.outdir, "results_level2.npz")
    export_erp_npz(res1_path, out1)
    export_erp_npz(res2_path, out2)

    print(f"Wrote input NPZ:   {input_path}")
    print(f"Wrote truth NPZ:   {gt_path}")
    print(f"Wrote L1 NPZ:      {res1_path}")
    print(f"Wrote L2 NPZ:      {res2_path}")
    print("You can load synthetic_input.npz in the GUI and run L1/L2 for visual comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
