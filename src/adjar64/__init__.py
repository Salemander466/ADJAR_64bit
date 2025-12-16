"""
adjar64

64-bit capable ADJAR (Woldorff-style) overlap correction engine + GUI.

This package follows a src-layout:
- src/adjar64/core   : math engine (Level 1, Level 2)
- src/adjar64/io     : loaders/exporters
- src/adjar64/plots  : matplotlib figure builders
- src/adjar64/accel  : performance helpers (FFT conv, optional numba)
- src/adjar64/gui    : PySide6 desktop application

Design principles:
- Core code does not depend on GUI.
- GUI is a thin wrapper around core + io + plots.
- Optional dependencies are detected gracefully with clear error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as _metadata
from importlib.util import find_spec as _find_spec
import logging
import os
import platform
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Versioning and package metadata
# ---------------------------------------------------------------------

def _read_version() -> str:
    try:
        return _metadata.version("adjar64")
    except Exception:
        # Fallback for editable installs or when metadata is not yet available.
        return "0.1.0"


__version__ = _read_version()
__all__ = [
    "__version__",
    "PackageInfo",
    "package_info",
    "get_logger",
    "optional_dependency_status",
    "require_optional",
]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

_LOGGER_NAME = "adjar64"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger for the package.

    If the consuming app does not configure logging, this will attach a
    NullHandler to avoid "No handler could be found" warnings.
    """
    logger = logging.getLogger(_LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


log = get_logger()


# ---------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------

def _has_module(module_name: str) -> bool:
    try:
        return _find_spec(module_name) is not None
    except Exception:
        return False


def optional_dependency_status() -> Dict[str, bool]:
    """
    Returns a dict of optional dependency availability.

    This is useful for:
    - GUI enabling/disabling
    - choosing optimized paths (numba)
    - validating environment before running
    """
    return {
        "scipy": _has_module("scipy"),
        "mne": _has_module("mne"),
        "pyside6": _has_module("PySide6"),
        "matplotlib": _has_module("matplotlib"),
        "numba": _has_module("numba"),
        "pyinstaller": _has_module("PyInstaller"),
    }


def require_optional(module_name: str, purpose: str) -> None:
    """
    Raise a clear error if an optional dependency is not installed.
    """
    if not _has_module(module_name):
        raise RuntimeError(
            f"Missing optional dependency '{module_name}'. It is required for: {purpose}. "
            f"Install an appropriate extra or install it directly."
        )


# ---------------------------------------------------------------------
# Runtime/package information
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PackageInfo:
    name: str
    version: str
    python_version: str
    platform: str
    platform_release: str
    machine: str
    is_64bit_python: bool
    env: Dict[str, str]
    optional_deps: Dict[str, bool]


def package_info() -> PackageInfo:
    """
    Collect runtime environment info helpful for debugging user installs.
    """
    py_ver = platform.python_version()
    plat = platform.system()
    rel = platform.release()
    mach = platform.machine()
    is_64 = (platform.architecture()[0] == "64bit")

    env_keys = [
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "QT_PLUGIN_PATH",
        "MPLBACKEND",
    ]
    env = {k: os.environ.get(k, "") for k in env_keys if os.environ.get(k) is not None}

    return PackageInfo(
        name="adjar64",
        version=__version__,
        python_version=py_ver,
        platform=plat,
        platform_release=rel,
        machine=mach,
        is_64bit_python=is_64,
        env=env,
        optional_deps=optional_dependency_status(),
    )
