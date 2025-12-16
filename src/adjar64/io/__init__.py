"""
adjar64.io

Loading and exporting utilities.

The IO layer should:
- keep file-format concerns out of the math engine
- normalize incoming data into a shared in-memory representation:
  data: (n_trials, n_channels, n_times) float64
  meta: fs_hz, tmin_ms, channel_names, condition labels, isi_pre, etc.

This __init__.py:
- provides a stable surface for the GUI to call
- supports optional dependencies (MNE) gracefully
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from adjar64 import get_logger, require_optional

log = get_logger("io")


__all__ = [
    "EpochedData",
    "load_npz",
    "save_npz",
    "load_csv_epochs",
    "export_erp_csv",
    "export_erp_npz",
]


@dataclass(frozen=True)
class EpochedData:
    """
    Standard in-memory representation for epoched EEG/ERP data.
    """
    data: np.ndarray  # shape (trials, channels, times)
    fs_hz: float
    tmin_ms: float
    tmax_ms: float
    channel_names: Optional[Sequence[str]] = None
    trial_labels: Optional[Sequence[str]] = None
    isi_pre_ms: Optional[np.ndarray] = None  # shape (trials,)
    extra: Optional[Dict[str, Any]] = None


def _lazy_import_error(name: str, module: str) -> Any:
    raise ImportError(
        f"adjar64.io: '{name}' is not available because '{module}' could not be imported. "
        f"Implement the module or verify your installation."
    )


try:
    from .loaders import load_npz, load_csv_epochs  # type: ignore
except Exception as e:
    log.debug("Failed importing loaders.py: %s", e)
    load_npz = lambda *a, **k: _lazy_import_error("load_npz", "loaders")  # type: ignore
    load_csv_epochs = lambda *a, **k: _lazy_import_error("load_csv_epochs", "loaders")  # type: ignore

try:
    from .exporters import save_npz, export_erp_csv, export_erp_npz  # type: ignore
except Exception as e:
    log.debug("Failed importing exporters.py: %s", e)
    save_npz = lambda *a, **k: _lazy_import_error("save_npz", "exporters")  # type: ignore
    export_erp_csv = lambda *a, **k: _lazy_import_error("export_erp_csv", "exporters")  # type: ignore
    export_erp_npz = lambda *a, **k: _lazy_import_error("export_erp_npz", "exporters")  # type: ignore
