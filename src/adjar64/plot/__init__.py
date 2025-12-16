"""
adjar64.plots

Matplotlib figure builders for:
- ERP comparisons (conventional vs L1 vs L2)
- ISI distributions
- gain curve visualizations
- subaverage alignment diagnostics

This package is intentionally "pure plotting" and should not contain ADJAR math.
"""

from __future__ import annotations

from typing import Any
from adjar64 import get_logger, require_optional

log = get_logger("plots")

__all__ = [
    "make_erp_comparison_figure",
    "make_isi_hist_figure",
    "make_gain_curve_figure",
    "make_subaverage_alignment_figure",
]


def _lazy_import_error(name: str, module: str) -> Any:
    raise ImportError(
        f"adjar64.plots: '{name}' is not available because '{module}' could not be imported. "
        f"Implement the module or verify your installation."
    )


# Ensure matplotlib is available before plotting.
try:
    require_optional("matplotlib", "plotting figures")
except Exception:
    # Do not raise at import time; raise only when user calls figure builders.
    pass


try:
    from .figures import (
        make_erp_comparison_figure,
        make_isi_hist_figure,
        make_gain_curve_figure,
        make_subaverage_alignment_figure,
    )  # type: ignore
except Exception as e:
    log.debug("Failed importing figures.py: %s", e)
    make_erp_comparison_figure = lambda *a, **k: _lazy_import_error("make_erp_comparison_figure", "figures")  # type: ignore
    make_isi_hist_figure = lambda *a, **k: _lazy_import_error("make_isi_hist_figure", "figures")  # type: ignore
    make_gain_curve_figure = lambda *a, **k: _lazy_import_error("make_gain_curve_figure", "figures")  # type: ignore
    make_subaverage_alignment_figure = lambda *a, **k: _lazy_import_error("make_subaverage_alignment_figure", "figures")  # type: ignore
