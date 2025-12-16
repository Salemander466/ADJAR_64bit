# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# Ensure PyInstaller runs from the project root (ADJAR_64BIT)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

# Entry script (your GUI)
ENTRY_SCRIPT = os.path.join(SRC_ROOT, "adjar64", "gui", "main.py")

# Collect Qt (PySide6) + Matplotlib assets
# Qt plugins (platforms, styles, imageformats) are critical for a working Qt EXE.
pyside6 = collect_all("PySide6")

# Matplotlib backends and data (fonts, mpl-data) are common failure points.
mpl = collect_all("matplotlib")

# NumPy/SciPy sometimes need hidden imports; these collections help.
numpy_all = collect_all("numpy")
scipy_all = collect_all("scipy")

# Optional: if you later use MNE or other packages, add similarly:
# mne_all = collect_all("mne")

datas = []
binaries = []
hiddenimports = []

datas += pyside6.datas + mpl.datas + numpy_all.datas + scipy_all.datas
binaries += pyside6.binaries + mpl.binaries + numpy_all.binaries + scipy_all.binaries

hiddenimports += pyside6.hiddenimports + mpl.hiddenimports + numpy_all.hiddenimports + scipy_all.hiddenimports

# Matplotlib backend explicit hidden imports (defensive)
hiddenimports += [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qt5",
    "matplotlib.backends.backend_agg",
]

# If you use QtAgg, Qt tries to load platform plugins dynamically.
# collect_all(PySide6) usually includes them, but adding plugin dirs as datas helps.
# If you hit "Could not load the Qt platform plugin 'windows'", this is the area to adjust.

a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[PROJECT_ROOT, SRC_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ADJAR_64BIT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
)

# One-folder build is more reliable for Qt. You can switch to one-file later if desired.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ADJAR_64BIT",
)
