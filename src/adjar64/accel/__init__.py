"""
adjar64.accel

Performance helpers and optional acceleration.

This package is safe to import without optional dependencies.
If numba is present, accelerated kernels may be used by core modules.
"""

from __future__ import annotations

from typing import Any
from adjar64 import get_logger

log = get_logger("accel")

__all__ = [
    "fft_convolve_1d",
    "fft_convolve_batch",
    "has_numba",
]


def has_numba() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def _lazy_import_error(name: str, module: str) -> Any:
    raise ImportError(
        f"adjar64.accel: '{name}' is not available because '{module}' could not be imported. "
        f"Implement the module or verify your installation."
    )


try:
    from .fftconv import fft_convolve_1d, fft_convolve_batch  # type: ignore
except Exception as e:
    log.debug("Failed importing fftconv.py: %s", e)
    fft_convolve_1d = lambda *a, **k: _lazy_import_error("fft_convolve_1d", "fftconv")  # type: ignore
    fft_convolve_batch = lambda *a, **k: _lazy_import_error("fft_convolve_batch", "fftconv")  # type: ignore
