from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from adjar64.io import save_npz


def _gaussian(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def generate_synthetic_dataset(
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

    # Ground-truth current-event ERP waveform (ERP_true), per channel scaled slightly.
    base_erp = (
        1.5 * _gaussian(t_ms, 120.0, 25.0)
        - 1.0 * _gaussian(t_ms, 180.0, 35.0)
        + 0.7 * _gaussian(t_ms, 320.0, 60.0)
    )
    base_erp *= (t_ms >= 0).astype(np.float64)

    # Ground-truth preceding-event response waveform (R_prev), distinct shape.
    base_prev = (
        1.2 * _gaussian(t_ms, 90.0, 22.0)
        - 0.8 * _gaussian(t_ms, 150.0, 30.0)
        + 0.4 * _gaussian(t_ms, 260.0, 55.0)
    )
    base_prev *= (t_ms >= 0).astype(np.float64)

    # Channel scaling (kept modest to maintain consistency)
    ch_scale_erp = 1.0 + 0.15 * rng.standard_normal(n_channels)
    ch_scale_prev = 1.0 + 0.15 * rng.standard_normal(n_channels)

    erp_true = (ch_scale_erp[:, None] * base_erp[None, :]).astype(np.float64)
    r_prev_true = (ch_scale_prev[:, None] * base_prev[None, :]).astype(np.float64)

    # Draw ISIs (ms), clip to ensure they remain within a realistic range.
    # We want a range that creates overlap within the epoch.
    isi_pre_ms = rng.normal(loc=isi_mean_ms, scale=isi_std_ms, size=n_trials).astype(np.float64)
    isi_pre_ms = np.clip(isi_pre_ms, 50.0, 450.0)

    # Convert to integer sample shifts consistent with your correct.py overlap model:
    # overlap_trial[:, i] = r_prev[:, i + shift]
    shifts = np.rint(isi_pre_ms / dt_ms).astype(np.int64)
    shifts = np.clip(shifts, 0, n_times)

    # Build trials: data[trial, channel, time]
    data = np.zeros((n_trials, n_channels, n_times), dtype=np.float64)

    for tr in range(n_trials):
        s = int(shifts[tr])

        # overlap contribution for this trial
        overlap = np.zeros((n_channels, n_times), dtype=np.float64)
        if s <= 0:
            overlap[:, :] = r_prev_true
        elif s < n_times:
            overlap[:, : (n_times - s)] = r_prev_true[:, s:]
        else:
            # s == n_times -> no overlap
            pass

        noise = noise_std * rng.standard_normal((n_channels, n_times))
        data[tr, :, :] = erp_true + overlap + noise

    channel_names = [f"Ch{c}" for c in range(n_channels)]
    trial_labels = ["A"] * n_trials  # single condition for now

    return {
        "data": data,
        "fs_hz": np.array(fs_hz, dtype=np.float64),
        "tmin_ms": np.array(tmin_ms, dtype=np.float64),
        "tmax_ms": np.array(tmax_ms, dtype=np.float64),
        "isi_pre_ms": isi_pre_ms,
        "channel_names": np.array(channel_names, dtype=object),
        "trial_labels": np.array(trial_labels, dtype=object),
        "erp_true": erp_true,
        "r_prev_true": r_prev_true,
        "time_ms": t_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic ADJAR dataset NPZ with ground truth.")
    ap.add_argument("--out", type=str, default="synthetic_adjar_input.npz")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--trials", type=int, default=1200)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--tmin", type=float, default=-200.0)
    ap.add_argument("--tmax", type=float, default=800.0)
    ap.add_argument("--isi_mean", type=float, default=220.0)
    ap.add_argument("--isi_std", type=float, default=70.0)
    ap.add_argument("--noise", type=float, default=0.25)
    args = ap.parse_args()

    payload = generate_synthetic_dataset(
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

    out_path = args.out
    # Save the canonical input NPZ (what the GUI expects)
    save_npz(
        out_path,
        data=payload["data"],
        fs_hz=float(payload["fs_hz"]),
        tmin_ms=float(payload["tmin_ms"]),
        tmax_ms=float(payload["tmax_ms"]),
        isi_pre_ms=payload["isi_pre_ms"],
        channel_names=[str(x) for x in payload["channel_names"].tolist()],
        trial_labels=[str(x) for x in payload["trial_labels"].tolist()],
        extra={
            "synthetic": True,
            "seed": int(args.seed),
            "notes": "Includes overlap from known r_prev_true and known erp_true (ground truth saved separately).",
        },
    )

    # Also save ground truth next to it for analysis convenience
    gt_path = os.path.splitext(out_path)[0] + "_ground_truth.npz"
    np.savez_compressed(
        gt_path,
        erp_true=payload["erp_true"],
        r_prev_true=payload["r_prev_true"],
        time_ms=payload["time_ms"],
    )

    print(f"Wrote input NPZ: {out_path}")
    print(f"Wrote ground truth NPZ: {gt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
