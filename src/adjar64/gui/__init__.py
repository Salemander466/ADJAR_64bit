"""
adjar64.gui

PySide6 desktop GUI package.

This package should:
- only import PySide6 when needed
- keep long-running computations out of the UI thread
- call into adjar64.core for math and adjar64.io for loading/exporting

If PySide6 is not installed, importing adjar64.gui should not crash unless
you try to launch the GUI.
"""

from __future__ import annotations

from typing import Any, Optional
from adjar64 import get_logger, require_optional

log = get_logger("gui")

__all__ = [
    "launch",
]


def launch(argv: Optional[list[str]] = None) -> int:
    """
    Launch the desktop application.

    Returns:
        Process exit code (0 on normal exit).
    """
    require_optional("PySide6", "running the desktop GUI")
    from .main import main  # local import to avoid hard dependency at import time
    return main(argv)
