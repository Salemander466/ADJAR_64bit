"""
adjar64.core

Core ADJAR computational engine.

This package should have no GUI dependencies.

Public API (intended):
- run_level1 / run_level2 / run_all
- build_preceding_distribution
- bin_by_isi, compute_subaverages
- estimate_previous_response
- apply_gain_filter
- convergence metrics and QA helpers

This __init__.py provides:
- a stable public API surface via re-exports (once implemented)
- lazy-import stubs so the package can be imported even before all modules exist
- a consistent logging entry point
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import logging

from adjar64 import get_logger

log = get_logger("core")


__all__ = [
    # High-level pipelines
    "run_level1",
    "run_level2",
    "run_all",
    # Core building blocks
    "build_preceding_distribution",
    "bin_trials_by_isi",
    "compute_subaverages",
    "estimate_previous_response",
    "compute_gain_curve",
    "apply_gain_filter",
    # Metrics
    "rms",
    "convergence_rms",
    # Data structures
    "AdjarConfig",
]


@dataclass(frozen=True)
class AdjarConfig:
    """
    Central configuration container for the ADJAR pipeline.

    Keep this conservative and explicit, because it will be used by the GUI.
    """
    fs_hz: float
    tmin_ms: float
    tmax_ms: float

    isi_bin_width_ms: float = 50.0

    use_woldorff_gain_filter: bool = True
    manual_lowpass_hz: Optional[float] = None

    level2_enabled: bool = True
    max_iter: int = 8
    tol_rms: float = 1e-6

    ridge_lambda: float = 0.0

    # When comparing iterations, restrict to a stable window if desired
    convergence_window_ms: Optional[Tuple[float, float]] = None


# ---------------------------------------------------------------------
# Lazy imports / public re-exports
# ---------------------------------------------------------------------

def _lazy_import_error(name: str, module: str) -> Any:
    raise ImportError(
        f"adjar64.core: '{name}' is not available because '{module}' could not be imported. "
        f"Implement the module or verify your installation."
    )


try:
    from .correct import run_level1, run_level2, run_all  # type: ignore
except Exception as e:
    log.debug("Failed importing correct.py: %s", e)
    run_level1 = lambda *a, **k: _lazy_import_error("run_level1", "correct")  # type: ignore
    run_level2 = lambda *a, **k: _lazy_import_error("run_level2", "correct")  # type: ignore
    run_all = lambda *a, **k: _lazy_import_error("run_all", "correct")  # type: ignore

try:
    from .distributions import build_preceding_distribution  # type: ignore
except Exception as e:
    log.debug("Failed importing distributions.py: %s", e)
    build_preceding_distribution = lambda *a, **k: _lazy_import_error(  # type: ignore
        "build_preceding_distribution", "distributions"
    )

try:
    from .subaverages import bin_trials_by_isi, compute_subaverages  # type: ignore
except Exception as e:
    log.debug("Failed importing subaverages.py: %s", e)
    bin_trials_by_isi = lambda *a, **k: _lazy_import_error("bin_trials_by_isi", "subaverages")  # type: ignore
    compute_subaverages = lambda *a, **k: _lazy_import_error("compute_subaverages", "subaverages")  # type: ignore

try:
    from .estimate import estimate_previous_response  # type: ignore
except Exception as e:
    log.debug("Failed importing estimate.py: %s", e)
    estimate_previous_response = lambda *a, **k: _lazy_import_error(  # type: ignore
        "estimate_previous_response", "estimate"
    )

try:
    from .filter import compute_gain_curve, apply_gain_filter  # type: ignore
except Exception as e:
    log.debug("Failed importing filter.py: %s", e)
    compute_gain_curve = lambda *a, **k: _lazy_import_error("compute_gain_curve", "filter")  # type: ignore
    apply_gain_filter = lambda *a, **k: _lazy_import_error("apply_gain_filter", "filter")  # type: ignore

try:
    from .metrics import rms, convergence_rms  # type: ignore
except Exception as e:
    log.debug("Failed importing metrics.py: %s", e)
    rms = lambda *a, **k: _lazy_import_error("rms", "metrics")  # type: ignore
    convergence_rms = lambda *a, **k: _lazy_import_error("convergence_rms", "metrics")  # type: ignore
