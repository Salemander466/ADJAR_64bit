from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from adjar64 import get_logger

from . import EpochedData

log = get_logger("io.loaders")


# =============================================================================
# NPZ SCHEMA (recommended canonical storage)
# =============================================================================
#
# Required keys:
#   data        : float array, shape (n_trials, n_channels, n_times)
#   fs_hz       : scalar float
#   tmin_ms     : scalar float
#   tmax_ms     : scalar float
#   isi_pre_ms  : float array, shape (n_trials,)
#
# Optional keys:
#   channel_names : array-like of strings, length n_channels
#   trial_labels  : array-like of strings, length n_trials
#   extra_json    : JSON string or dict-like serializable payload (metadata)
#
# Notes:
# - All numeric arrays are loaded into float64 for computational stability.
# - Strings are loaded as Python strings.
# - If you want to store "extra" as a dict in NPZ, store it as JSON under extra_json.
# =============================================================================


def load_npz(path: str) -> EpochedData:
    """
    Load epoched EEG/ERP data from a canonical ADJAR_64BIT NPZ file.

    Args:
        path: Path to .npz file

    Returns:
        EpochedData object with required fields.

    Raises:
        ValueError on schema mismatch.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not path.lower().endswith(".npz"):
        raise ValueError("load_npz expects a .npz file")

    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)

        required = {"data", "fs_hz", "tmin_ms", "tmax_ms", "isi_pre_ms"}
        missing = required - keys
        if missing:
            raise ValueError(f"NPZ missing required keys: {sorted(missing)}. Found keys: {sorted(keys)}")

        data = np.asarray(z["data"], dtype=np.float64, order="C")
        fs_hz = float(np.asarray(z["fs_hz"]).item())
        tmin_ms = float(np.asarray(z["tmin_ms"]).item())
        tmax_ms = float(np.asarray(z["tmax_ms"]).item())
        isi_pre_ms = np.asarray(z["isi_pre_ms"], dtype=np.float64)

        channel_names = _load_str_array(z, "channel_names", expected_len=data.shape[1])
        trial_labels = _load_str_array(z, "trial_labels", expected_len=data.shape[0])

        extra = None
        if "extra_json" in keys:
            extra = _load_extra_json(z["extra_json"])
        elif "extra" in keys:
            # Backward/alternate support: "extra" can be JSON str or pickled dict
            extra = _load_extra_json(z["extra"])

    _validate_epoched_data_arrays(data, fs_hz, tmin_ms, tmax_ms, isi_pre_ms, channel_names, trial_labels)

    return EpochedData(
        data=data,
        fs_hz=fs_hz,
        tmin_ms=tmin_ms,
        tmax_ms=tmax_ms,
        channel_names=channel_names,
        trial_labels=trial_labels,
        isi_pre_ms=isi_pre_ms,
        extra=extra,
    )


def _load_extra_json(obj: Any) -> Dict[str, Any]:
    """
    Accept either:
      - JSON string
      - numpy scalar with JSON string
      - dict-like already
      - bytes containing JSON
    """
    if obj is None:
        return {}
    # numpy scalar -> Python
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8", errors="replace")
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            # If JSON fails, store raw
            return {"_raw_extra": s}
    if isinstance(obj, dict):
        return obj
    # Pickled python objects might arrive as numpy object arrays
    try:
        if hasattr(obj, "item"):
            maybe = obj.item()
            if isinstance(maybe, dict):
                return maybe
    except Exception:
        pass
    return {"_raw_extra": repr(obj)}


def _load_str_array(
    z: Any, key: str, expected_len: Optional[int] = None
) -> Optional[List[str]]:
    if key not in z.files:
        return None
    arr = z[key]
    # allow_pickle may yield object dtype
    if isinstance(arr, np.ndarray) and arr.shape == ():
        arr = arr.item()
    if arr is None:
        return None
    if isinstance(arr, (list, tuple)):
        out = [str(x) for x in arr]
    else:
        a = np.asarray(arr)
        if a.ndim == 0:
            # single string, not an array
            out = [str(a.item())]
        else:
            out = [str(x) for x in a.tolist()]

    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f"{key} length mismatch: expected {expected_len}, got {len(out)}")
    return out


def _validate_epoched_data_arrays(
    data: np.ndarray,
    fs_hz: float,
    tmin_ms: float,
    tmax_ms: float,
    isi_pre_ms: np.ndarray,
    channel_names: Optional[Sequence[str]],
    trial_labels: Optional[Sequence[str]],
) -> None:
    if data.ndim != 3:
        raise ValueError(f"data must be 3D (trials, channels, times), got shape {data.shape}")
    n_trials, n_channels, n_times = data.shape
    if n_trials < 1 or n_channels < 1 or n_times < 2:
        raise ValueError(f"Invalid data shape {data.shape}")

    if not np.isfinite(data).all():
        raise ValueError("data contains NaN or inf")

    if fs_hz <= 0 or not np.isfinite(fs_hz):
        raise ValueError(f"fs_hz must be finite and > 0, got {fs_hz}")

    if not np.isfinite(tmin_ms) or not np.isfinite(tmax_ms):
        raise ValueError("tmin_ms/tmax_ms must be finite")
    if tmax_ms <= tmin_ms:
        raise ValueError(f"tmax_ms must be > tmin_ms, got tmin={tmin_ms}, tmax={tmax_ms}")

    if isi_pre_ms.ndim != 1:
        raise ValueError(f"isi_pre_ms must be 1D, got shape {isi_pre_ms.shape}")
    if isi_pre_ms.shape[0] != n_trials:
        raise ValueError(f"isi_pre_ms length {isi_pre_ms.shape[0]} != n_trials {n_trials}")
    if not np.isfinite(isi_pre_ms).all():
        raise ValueError("isi_pre_ms contains NaN or inf")
    if np.any(isi_pre_ms < 0):
        raise ValueError("isi_pre_ms contains negative values; preceding intervals must be >= 0 ms")

    if channel_names is not None and len(channel_names) != n_channels:
        raise ValueError("channel_names length must match n_channels")
    if trial_labels is not None and len(trial_labels) != n_trials:
        raise ValueError("trial_labels length must match n_trials")


# =============================================================================
# CSV SCHEMA (simple, explicit; not as efficient as NPZ)
# =============================================================================
#
# Format: LONG (recommended for human inspection and easy generation)
#
# Required columns:
#   trial, channel, time_ms, value, isi_pre_ms
#
# Optional columns:
#   label (condition/bin label for the trial)
#
# Rules:
# - trial: integer [0..n_trials-1]
# - channel: integer [0..n_channels-1]
# - time_ms: float (must repeat identically for all trials/channels)
# - value: float (EEG/ERP amplitude)
# - isi_pre_ms: float (must be constant within each trial)
# - label: str (must be constant within each trial if present)
#
# Loader will reconstruct 3D array data[trial, channel, time_index].
#
# For large datasets, CSV is slow. Use NPZ for production.
# =============================================================================


def load_csv_epochs(path: str) -> EpochedData:
    """
    Load epoched EEG/ERP data from CSV in the defined LONG schema.

    Args:
        path: path to .csv file

    Returns:
        EpochedData
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not path.lower().endswith(".csv"):
        raise ValueError("load_csv_epochs expects a .csv file")

    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr is None or arr.size == 0:
        raise ValueError("CSV appears empty or unreadable")

    colnames = set(arr.dtype.names or [])
    required_cols = {"trial", "channel", "time_ms", "value", "isi_pre_ms"}
    missing = required_cols - colnames
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}. Found: {sorted(colnames)}")

    trial = np.asarray(arr["trial"], dtype=np.int64)
    channel = np.asarray(arr["channel"], dtype=np.int64)
    time_ms = np.asarray(arr["time_ms"], dtype=np.float64)
    value = np.asarray(arr["value"], dtype=np.float64)
    isi_pre_ms_col = np.asarray(arr["isi_pre_ms"], dtype=np.float64)

    label_col = None
    if "label" in colnames:
        label_col = np.asarray(arr["label"], dtype=str)

    if np.any(trial < 0) or np.any(channel < 0):
        raise ValueError("trial/channel indices must be >= 0")

    n_trials = int(trial.max()) + 1
    n_channels = int(channel.max()) + 1

    # Identify unique time axis and enforce consistency
    uniq_times = np.unique(time_ms)
    uniq_times.sort()
    n_times = uniq_times.size
    if n_times < 2:
        raise ValueError("time_ms must have at least 2 unique points")

    # Map each row's time_ms to an index in uniq_times
    # Use searchsorted; requires exact match.
    t_idx = np.searchsorted(uniq_times, time_ms)
    if np.any(t_idx < 0) or np.any(t_idx >= n_times):
        raise ValueError("Invalid time_ms mapping")
    if not np.allclose(uniq_times[t_idx], time_ms, rtol=0, atol=1e-12):
        raise ValueError("time_ms values must match a consistent shared time grid exactly")

    # Allocate and fill data array
    data = np.empty((n_trials, n_channels, n_times), dtype=np.float64)
    data.fill(np.nan)

    data[trial, channel, t_idx] = value

    if np.isnan(data).any():
        # Identify missing entries for clarity
        missing_count = int(np.isnan(data).sum())
        raise ValueError(
            f"CSV does not fully cover the trial/channel/time grid. Missing cells: {missing_count}. "
            f"Ensure every (trial, channel, time_ms) combination exists."
        )

    # Build isi_pre_ms per trial: must be constant within each trial
    isi_pre_ms = np.empty((n_trials,), dtype=np.float64)
    isi_pre_ms.fill(np.nan)
    for tr in range(n_trials):
        vals = isi_pre_ms_col[trial == tr]
        if vals.size == 0:
            raise ValueError(f"No isi_pre_ms rows for trial {tr}")
        v0 = float(vals[0])
        if not np.allclose(vals, v0, rtol=0, atol=1e-9):
            raise ValueError(f"isi_pre_ms is not constant within trial {tr}")
        isi_pre_ms[tr] = v0

    if np.any(isi_pre_ms < 0) or not np.isfinite(isi_pre_ms).all():
        raise ValueError("isi_pre_ms must be finite and >= 0")

    trial_labels = None
    if label_col is not None:
        labels = []
        for tr in range(n_trials):
            vals = label_col[trial == tr]
            if vals.size == 0:
                raise ValueError(f"No label rows for trial {tr}")
            v0 = str(vals[0])
            if not np.all(vals == v0):
                raise ValueError(f"label is not constant within trial {tr}")
            labels.append(v0)
        trial_labels = labels

    # Infer fs and tmin/tmax from time grid
    # Use median dt for robustness.
    dt_ms = np.diff(uniq_times)
    if np.any(dt_ms <= 0):
        raise ValueError("time_ms grid must be strictly increasing")
    dt_med = float(np.median(dt_ms))
    if dt_med <= 0:
        raise ValueError("Invalid dt inferred from time_ms")

    fs_hz = 1000.0 / dt_med
    tmin_ms = float(uniq_times[0])
    tmax_ms = float(uniq_times[-1])

    # CSV has no channel names by default; could be inferred later.
    channel_names = [f"Ch{c}" for c in range(n_channels)]

    _validate_epoched_data_arrays(data, fs_hz, tmin_ms, tmax_ms, isi_pre_ms, channel_names, trial_labels)

    return EpochedData(
        data=data,
        fs_hz=fs_hz,
        tmin_ms=tmin_ms,
        tmax_ms=tmax_ms,
        channel_names=channel_names,
        trial_labels=trial_labels,
        isi_pre_ms=isi_pre_ms,
        extra={"source_csv": os.path.abspath(path), "schema": "long_v1"},
    )
