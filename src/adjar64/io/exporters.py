from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from adjar64 import get_logger

log = get_logger("io.exporters")


def save_npz(
    path: str,
    *,
    data: np.ndarray,
    fs_hz: float,
    tmin_ms: float,
    tmax_ms: float,
    isi_pre_ms: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
    trial_labels: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save epoched data to the canonical ADJAR_64BIT NPZ schema.

    This is intended for saving preprocessed input datasets in a stable way.
    """
    _ensure_parent_dir(path)
    if not path.lower().endswith(".npz"):
        path = path + ".npz"

    data = np.asarray(data, dtype=np.float64, order="C")
    isi_pre_ms = np.asarray(isi_pre_ms, dtype=np.float64)

    payload: Dict[str, Any] = {
        "data": data,
        "fs_hz": float(fs_hz),
        "tmin_ms": float(tmin_ms),
        "tmax_ms": float(tmax_ms),
        "isi_pre_ms": isi_pre_ms,
    }

    if channel_names is not None:
        payload["channel_names"] = np.asarray([str(x) for x in channel_names], dtype=object)
    if trial_labels is not None:
        payload["trial_labels"] = np.asarray([str(x) for x in trial_labels], dtype=object)

    if extra is None:
        extra = {}
    extra = dict(extra)
    extra.setdefault("saved_utc", datetime.utcnow().isoformat() + "Z")
    payload["extra_json"] = json.dumps(extra, ensure_ascii=False)

    np.savez_compressed(path, **payload)


# =============================================================================
# RESULTS EXPORTS (from GUI pipeline output dict)
# =============================================================================

def export_erp_csv(path: str, results: Dict[str, Any]) -> None:
    """
    Export ERP waveforms to CSV in a simple wide format suitable for Excel/R.

    Output columns:
      time_ms,
      conv_ch0, conv_ch1, ...
      l1_ch0, l1_ch1, ... (if present)
      l2_ch0, l2_ch1, ... (if present)

    Metadata is not embedded in CSV; it should be exported via NPZ for full fidelity.
    """
    _ensure_parent_dir(path)
    if not path.lower().endswith(".csv"):
        path = path + ".csv"

    meta = results.get("meta", {})
    fs_hz = float(meta.get("fs_hz", 0.0))
    tmin_ms = float(meta.get("tmin_ms", 0.0))

    erp_conv = _as_2d(results.get("erp_conv"), "erp_conv")
    erp_l1 = results.get("erp_l1", None)
    erp_l2 = results.get("erp_l2", None)

    erp_l1_2d = _as_2d(erp_l1, "erp_l1") if erp_l1 is not None else None
    erp_l2_2d = _as_2d(erp_l2, "erp_l2") if erp_l2 is not None else None

    n_ch, n_times = erp_conv.shape

    if fs_hz <= 0:
        # Infer dt from t axis if present
        # Otherwise assume 1 ms step (fallback).
        dt_ms = float(meta.get("dt_ms", 1.0))
    else:
        dt_ms = 1000.0 / fs_hz

    t_ms = tmin_ms + np.arange(n_times) * dt_ms

    # Build header
    cols = ["time_ms"]
    cols += [f"conv_ch{c}" for c in range(n_ch)]
    if erp_l1_2d is not None:
        cols += [f"l1_ch{c}" for c in range(n_ch)]
    if erp_l2_2d is not None:
        cols += [f"l2_ch{c}" for c in range(n_ch)]

    # Build matrix
    blocks = [t_ms.reshape(-1, 1), erp_conv.T]
    if erp_l1_2d is not None:
        blocks.append(erp_l1_2d.T)
    if erp_l2_2d is not None:
        blocks.append(erp_l2_2d.T)

    mat = np.concatenate(blocks, axis=1)

    # Save CSV
    header = ",".join(cols)
    np.savetxt(path, mat, delimiter=",", header=header, comments="", fmt="%.10g")


def export_erp_npz(path: str, results: Dict[str, Any]) -> None:
    """
    Export the full pipeline results to NPZ.

    This is the recommended export for complete reproducibility, including:
    - erp_conv, erp_l1, erp_l2 (if present)
    - r_prev (if present)
    - d_pre (if present)
    - gain_curve (if present)
    - meta_json (full metadata)
    """
    _ensure_parent_dir(path)
    if not path.lower().endswith(".npz"):
        path = path + ".npz"

    payload: Dict[str, Any] = {}

    # Standard outputs
    payload["erp_conv"] = _as_2d(results.get("erp_conv"), "erp_conv").astype(np.float64, copy=False)
    if "erp_l1" in results and results["erp_l1"] is not None:
        payload["erp_l1"] = _as_2d(results["erp_l1"], "erp_l1").astype(np.float64, copy=False)
    if "erp_l2" in results and results["erp_l2"] is not None:
        payload["erp_l2"] = _as_2d(results["erp_l2"], "erp_l2").astype(np.float64, copy=False)

    # Optional diagnostics
    if "r_prev" in results and results["r_prev"] is not None:
        payload["r_prev"] = _as_2d(results["r_prev"], "r_prev").astype(np.float64, copy=False)
    if "d_pre" in results and results["d_pre"] is not None:
        payload["d_pre"] = np.asarray(results["d_pre"], dtype=np.float64)
    if "gain_curve" in results and results["gain_curve"] is not None:
        payload["gain_curve"] = np.asarray(results["gain_curve"], dtype=np.float64)

    # Meta as JSON
    meta = results.get("meta", {})
    meta = dict(meta)
    meta.setdefault("exported_utc", datetime.utcnow().isoformat() + "Z")
    payload["meta_json"] = json.dumps(meta, ensure_ascii=False)

    np.savez_compressed(path, **payload)


# =============================================================================
# Helpers
# =============================================================================

def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _as_2d(x: Any, name: str) -> np.ndarray:
    if x is None:
        raise ValueError(f"{name} is missing")
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (channels, times). Got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/inf")
    return arr
