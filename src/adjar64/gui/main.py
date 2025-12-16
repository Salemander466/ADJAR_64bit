from __future__ import annotations

import sys
import traceback
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from adjar64 import get_logger, package_info, require_optional
from adjar64.core import AdjarConfig

log = get_logger("gui.main")


# GUI imports are kept inside a guarded block for clearer error messages.
def _import_qt() -> Any:
    require_optional("PySide6", "running the desktop GUI")
    from PySide6.QtCore import (
        QObject,
        QRunnable,
        QThreadPool,
        Qt,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSpinBox,
        QDoubleSpinBox,
        QCheckBox,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QComboBox,
    )

    return {
        "QObject": QObject,
        "QRunnable": QRunnable,
        "QThreadPool": QThreadPool,
        "Qt": Qt,
        "Signal": Signal,
        "Slot": Slot,
        "QAction": QAction,
        "QApplication": QApplication,
        "QFileDialog": QFileDialog,
        "QFormLayout": QFormLayout,
        "QGroupBox": QGroupBox,
        "QHBoxLayout": QHBoxLayout,
        "QLabel": QLabel,
        "QLineEdit": QLineEdit,
        "QMainWindow": QMainWindow,
        "QMessageBox": QMessageBox,
        "QPushButton": QPushButton,
        "QProgressBar": QProgressBar,
        "QSpinBox": QSpinBox,
        "QDoubleSpinBox": QDoubleSpinBox,
        "QCheckBox": QCheckBox,
        "QTabWidget": QTabWidget,
        "QTextEdit": QTextEdit,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
        "QComboBox": QComboBox,
    }


def _import_mpl_qt() -> Any:
    require_optional("matplotlib", "rendering plots in the GUI")
    # Force a Qt-compatible backend. This is safe if Qt bindings exist.
    import matplotlib

    try:
        matplotlib.use("QtAgg")
    except Exception:
        # If backend selection fails, matplotlib may still work; keep going.
        pass

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    return {"FigureCanvas": FigureCanvas, "Figure": Figure}


QT = _import_qt()
MPL = _import_mpl_qt()


# -------------------------
# Utility: robust parsing
# -------------------------
def _safe_float(text: str, default: float) -> float:
    try:
        return float(text.strip())
    except Exception:
        return default


def _safe_int(text: str, default: int) -> int:
    try:
        return int(float(text.strip()))
    except Exception:
        return default


# -------------------------
# Background worker
# -------------------------
class WorkerSignals(QT["QObject"]):
    started = QT["Signal"]()
    progress = QT["Signal"](int, str)  # percent, message
    finished = QT["Signal"](object)  # result payload
    error = QT["Signal"](str)  # formatted error string


class PipelineWorker(QT["QRunnable"]):
    """
    Runs the ADJAR pipeline off the UI thread.

    Inputs:
      - data payload (EpochedData or dict-like)
      - config (AdjarConfig)
      - run_l1, run_l2 flags
      - selected condition/bin (optional)
    """

    def __init__(
        self,
        payload: Any,
        config: AdjarConfig,
        run_l1: bool,
        run_l2: bool,
        condition: Optional[str],
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.payload = payload
        self.config = config
        self.run_l1 = run_l1
        self.run_l2 = run_l2
        self.condition = condition

    @QT["Slot"]()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            self.signals.progress.emit(2, "Preparing inputs")

            # Payload normalization
            # Expect EpochedData (from adjar64.io) or dict with keys.
            data = getattr(self.payload, "data", None)
            fs_hz = getattr(self.payload, "fs_hz", None)
            tmin_ms = getattr(self.payload, "tmin_ms", None)
            tmax_ms = getattr(self.payload, "tmax_ms", None)
            isi_pre_ms = getattr(self.payload, "isi_pre_ms", None)
            trial_labels = getattr(self.payload, "trial_labels", None)
            channel_names = getattr(self.payload, "channel_names", None)

            if data is None and isinstance(self.payload, dict):
                data = self.payload.get("data")
                fs_hz = self.payload.get("fs_hz")
                tmin_ms = self.payload.get("tmin_ms")
                tmax_ms = self.payload.get("tmax_ms")
                isi_pre_ms = self.payload.get("isi_pre_ms")
                trial_labels = self.payload.get("trial_labels")
                channel_names = self.payload.get("channel_names")

            if data is None:
                raise ValueError("No epoched data found. Expected payload.data or payload['data'].")

            data = np.asarray(data, dtype=np.float64, order="C")

            if fs_hz is None:
                fs_hz = self.config.fs_hz
            if tmin_ms is None:
                tmin_ms = self.config.tmin_ms
            if tmax_ms is None:
                tmax_ms = self.config.tmax_ms

            if isi_pre_ms is not None:
                isi_pre_ms = np.asarray(isi_pre_ms, dtype=np.float64)

            self.signals.progress.emit(8, "Selecting trials (condition/bin)")

            # Condition selection (optional)
            trial_mask = None
            if self.condition and trial_labels is not None:
                labels = np.asarray(trial_labels)
                trial_mask = (labels == self.condition)
                if not np.any(trial_mask):
                    raise ValueError(f"No trials match selected condition: {self.condition!r}")

            if trial_mask is not None:
                data_sel = data[trial_mask, :, :]
                isi_sel = isi_pre_ms[trial_mask] if isi_pre_ms is not None else None
            else:
                data_sel = data
                isi_sel = isi_pre_ms

            # Basic sanity checks
            if data_sel.ndim != 3:
                raise ValueError(f"Expected data shape (trials, channels, times). Got {data_sel.shape}")
            if data_sel.shape[0] < 5:
                raise ValueError("Too few trials selected. Need at least ~5 for stable averages.")

            if isi_sel is None:
                raise ValueError(
                    "Missing isi_pre_ms for ADJAR correction. Provide preceding intervals per trial."
                )

            self.signals.progress.emit(15, "Running ADJAR pipeline")

            # Import pipeline functions lazily to avoid import-time failures.
            from adjar64.core import run_level1, run_level2

            # Expected return payload contract (you can refine later):
            # {
            #   'erp_conv': (channels, times),
            #   'erp_l1': (channels, times) optional,
            #   'erp_l2': (channels, times) optional,
            #   'r_prev': (channels, times) optional,
            #   'd_pre': (times,) optional,
            #   'meta': {...}
            # }
            result: Dict[str, Any] = {}
            # Conventional average for reference
            result["erp_conv"] = np.mean(data_sel, axis=0)

            if self.run_l1:
                self.signals.progress.emit(35, "Level 1 correction")
                out_l1 = run_level1(
                    data_sel,
                    isi_sel,
                    config=self.config,
                    channel_names=channel_names,
                )
                # Permit either dict return or direct array.
                if isinstance(out_l1, dict):
                    result.update(out_l1)
                else:
                    result["erp_l1"] = out_l1

            if self.run_l2:
                self.signals.progress.emit(70, "Level 2 correction (iterative)")
                out_l2 = run_level2(
                    data_sel,
                    isi_sel,
                    config=self.config,
                    channel_names=channel_names,
                )
                if isinstance(out_l2, dict):
                    result.update(out_l2)
                else:
                    result["erp_l2"] = out_l2

            # Attach minimal meta for the GUI
            result.setdefault("meta", {})
            result["meta"].update(
                {
                    "fs_hz": float(fs_hz),
                    "tmin_ms": float(tmin_ms),
                    "tmax_ms": float(tmax_ms),
                    "n_trials": int(data_sel.shape[0]),
                    "n_channels": int(data_sel.shape[1]),
                    "n_times": int(data_sel.shape[2]),
                    "condition": self.condition,
                    "config": asdict(self.config),
                }
            )

            self.signals.progress.emit(100, "Done")
            self.signals.finished.emit(result)

        except Exception:
            err = traceback.format_exc()
            self.signals.error.emit(err)


# -------------------------
# Matplotlib plotting widget
# -------------------------
class MplPanel(QT["QWidget"]):
    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.fig = MPL["Figure"](dpi=100, constrained_layout=True)
        self.canvas = MPL["FigureCanvas"](self.fig)

        layout = QT["QVBoxLayout"]()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self._last_title = ""

    def clear(self) -> None:
        self.fig.clear()
        self.canvas.draw_idle()

    def plot_erp_compare(
        self,
        t_ms: np.ndarray,
        conv: np.ndarray,
        l1: Optional[np.ndarray],
        l2: Optional[np.ndarray],
        channel_idx: int = 0,
        title: str = "ERP comparison",
    ) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        ax.plot(t_ms, conv[channel_idx, :], label="Conventional")
        if l1 is not None:
            ax.plot(t_ms, l1[channel_idx, :], label="ADJAR L1")
        if l2 is not None:
            ax.plot(t_ms, l2[channel_idx, :], label="ADJAR L2")

        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.legend(loc="best")
        ax.axvline(0.0, linewidth=1.0)

        self.canvas.draw_idle()


# -------------------------
# Main window
# -------------------------
class MainWindow(QT["QMainWindow"]):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ADJAR_64BIT")
        self.resize(1100, 720)

        self.threadpool = QT["QThreadPool"]()

        # State
        self.payload: Optional[Any] = None
        self.last_result: Optional[Dict[str, Any]] = None

        # Central widget with tabs
        self.tabs = QT["QTabWidget"]()
        self.setCentralWidget(self.tabs)

        self._build_menu()
        self._build_tab_data()
        self._build_tab_settings()
        self._build_tab_run()
        self._build_tab_results()

        self._append_log("Application started.")
        info = package_info()
        self._append_log(f"Python 64-bit: {info.is_64bit_python}; Version: {info.version}")

    # -------------------------
    # Menu
    # -------------------------
    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        help_menu = menubar.addMenu("Help")

        act_load = QT["QAction"]("Load data...", self)
        act_load.triggered.connect(self._on_load_clicked)
        file_menu.addAction(act_load)

        act_quit = QT["QAction"]("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        act_about = QT["QAction"]("About", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    def _on_about(self) -> None:
        info = package_info()
        deps = info.optional_deps
        text = (
            f"ADJAR_64BIT (adjar64)\n"
            f"Version: {info.version}\n"
            f"Python: {info.python_version} ({'64-bit' if info.is_64bit_python else '32-bit'})\n"
            f"Platform: {info.platform} {info.platform_release} ({info.machine})\n\n"
            f"Optional deps:\n"
            f"  PySide6: {deps.get('pyside6')}\n"
            f"  Matplotlib: {deps.get('matplotlib')}\n"
            f"  SciPy: {deps.get('scipy')}\n"
            f"  MNE: {deps.get('mne')}\n"
            f"  Numba: {deps.get('numba')}\n"
        )
        QT["QMessageBox"].information(self, "About", text)

    # -------------------------
    # Tabs
    # -------------------------
    def _build_tab_data(self) -> None:
        tab = QT["QWidget"]()
        layout = QT["QVBoxLayout"]()

        group = QT["QGroupBox"]("Input data")
        form = QT["QFormLayout"]()

        self.path_edit = QT["QLineEdit"]()
        self.path_edit.setPlaceholderText("Select an input file (NPZ or CSV)")

        btn_row = QT["QHBoxLayout"]()
        self.btn_load = QT["QPushButton"]("Load...")
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_validate = QT["QPushButton"]("Validate")
        self.btn_validate.clicked.connect(self._on_validate_clicked)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_validate)
        btn_row.addStretch(1)

        self.fs_edit = QT["QLineEdit"]()
        self.fs_edit.setPlaceholderText("e.g., 250 or 512")
        self.tmin_edit = QT["QLineEdit"]()
        self.tmin_edit.setPlaceholderText("e.g., -200")
        self.tmax_edit = QT["QLineEdit"]()
        self.tmax_edit.setPlaceholderText("e.g., 800")

        self.condition_combo = QT["QComboBox"]()
        self.condition_combo.addItem("(all trials)")
        self.condition_combo.setEnabled(False)

        form.addRow("File path:", self.path_edit)
        form.addRow("", self._wrap_layout(btn_row))
        form.addRow("Sampling rate (Hz):", self.fs_edit)
        form.addRow("Epoch start tmin (ms):", self.tmin_edit)
        form.addRow("Epoch end tmax (ms):", self.tmax_edit)
        form.addRow("Condition/bin:", self.condition_combo)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch(1)
        tab.setLayout(layout)

        self.tabs.addTab(tab, "Data")

    def _build_tab_settings(self) -> None:
        tab = QT["QWidget"]()
        layout = QT["QVBoxLayout"]()

        group = QT["QGroupBox"]("ADJAR settings")
        form = QT["QFormLayout"]()

        self.cb_woldorff_gain = QT["QCheckBox"]("Use Woldorff gain filter (recommended)")
        self.cb_woldorff_gain.setChecked(True)

        self.manual_lowpass = QT["QDoubleSpinBox"]()
        self.manual_lowpass.setRange(0.0, 200.0)
        self.manual_lowpass.setDecimals(2)
        self.manual_lowpass.setValue(0.0)
        self.manual_lowpass.setToolTip("0.0 means disabled (use Woldorff gain or no manual lowpass).")

        self.isi_bin_width = QT["QDoubleSpinBox"]()
        self.isi_bin_width.setRange(1.0, 1000.0)
        self.isi_bin_width.setDecimals(1)
        self.isi_bin_width.setValue(50.0)

        self.ridge_lambda = QT["QDoubleSpinBox"]()
        self.ridge_lambda.setRange(0.0, 1e6)
        self.ridge_lambda.setDecimals(6)
        self.ridge_lambda.setValue(0.0)
        self.ridge_lambda.setToolTip("Ridge regularization strength for estimation stability (0 = none).")

        self.max_iter = QT["QSpinBox"]()
        self.max_iter.setRange(1, 50)
        self.max_iter.setValue(8)

        self.tol_rms = QT["QDoubleSpinBox"]()
        self.tol_rms.setRange(0.0, 1.0)
        self.tol_rms.setDecimals(12)
        self.tol_rms.setValue(1e-6)

        form.addRow(self.cb_woldorff_gain)
        form.addRow("Manual lowpass (Hz):", self.manual_lowpass)
        form.addRow("ISI bin width (ms):", self.isi_bin_width)
        form.addRow("Ridge lambda:", self.ridge_lambda)
        form.addRow("Level 2 max iterations:", self.max_iter)
        form.addRow("Level 2 tolerance (RMS):", self.tol_rms)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch(1)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Settings")

    def _build_tab_run(self) -> None:
        tab = QT["QWidget"]()
        layout = QT["QVBoxLayout"]()

        group = QT["QGroupBox"]("Run ADJAR")
        v = QT["QVBoxLayout"]()

        run_row = QT["QHBoxLayout"]()
        self.cb_run_l1 = QT["QCheckBox"]("Run Level 1")
        self.cb_run_l1.setChecked(True)
        self.cb_run_l2 = QT["QCheckBox"]("Run Level 2")
        self.cb_run_l2.setChecked(True)

        self.btn_run = QT["QPushButton"]("Run")
        self.btn_run.clicked.connect(self._on_run_clicked)

        run_row.addWidget(self.cb_run_l1)
        run_row.addWidget(self.cb_run_l2)
        run_row.addStretch(1)
        run_row.addWidget(self.btn_run)

        self.progress = QT["QProgressBar"]()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.log_box = QT["QTextEdit"]()
        self.log_box.setReadOnly(True)

        v.addLayout(run_row)
        v.addWidget(self.progress)
        v.addWidget(QT["QLabel"]("Log:"))
        v.addWidget(self.log_box)

        group.setLayout(v)
        layout.addWidget(group)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Run")

    def _build_tab_results(self) -> None:
        tab = QT["QWidget"]()
        layout = QT["QVBoxLayout"]()

        top = QT["QHBoxLayout"]()
        self.channel_spin = QT["QSpinBox"]()
        self.channel_spin.setRange(0, 0)
        self.channel_spin.valueChanged.connect(self._refresh_plot)

        self.btn_export_npz = QT["QPushButton"]("Export results (NPZ)")
        self.btn_export_npz.clicked.connect(self._on_export_npz)
        self.btn_export_npz.setEnabled(False)

        self.btn_export_csv = QT["QPushButton"]("Export ERP (CSV)")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_csv.setEnabled(False)

        top.addWidget(QT["QLabel"]("Channel index:"))
        top.addWidget(self.channel_spin)
        top.addStretch(1)
        top.addWidget(self.btn_export_csv)
        top.addWidget(self.btn_export_npz)

        self.plot_panel = MplPanel()

        layout.addLayout(top)
        layout.addWidget(self.plot_panel)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Results")

    # -------------------------
    # Helpers
    # -------------------------
    def _wrap_layout(self, layout_obj: Any) -> Any:
        w = QT["QWidget"]()
        w.setLayout(layout_obj)
        return w

    def _append_log(self, msg: str) -> None:
        self.log_box.append(msg)
        self.log_box.ensureCursorVisible()

    def _show_error(self, title: str, details: str) -> None:
        log.error("%s\n%s", title, details)
        QT["QMessageBox"].critical(self, title, details)

    def _get_selected_condition(self) -> Optional[str]:
        if not self.condition_combo.isEnabled():
            return None
        txt = self.condition_combo.currentText()
        if txt == "(all trials)":
            return None
        return txt

    def _build_config(self) -> AdjarConfig:
        fs_hz = _safe_float(self.fs_edit.text(), 0.0)
        tmin_ms = _safe_float(self.tmin_edit.text(), 0.0)
        tmax_ms = _safe_float(self.tmax_edit.text(), 0.0)

        manual_lp = float(self.manual_lowpass.value())
        manual_lp_val = None if manual_lp <= 0.0 else manual_lp

        cfg = AdjarConfig(
            fs_hz=fs_hz,
            tmin_ms=tmin_ms,
            tmax_ms=tmax_ms,
            isi_bin_width_ms=float(self.isi_bin_width.value()),
            use_woldorff_gain_filter=bool(self.cb_woldorff_gain.isChecked()),
            manual_lowpass_hz=manual_lp_val,
            level2_enabled=bool(self.cb_run_l2.isChecked()),
            max_iter=int(self.max_iter.value()),
            tol_rms=float(self.tol_rms.value()),
            ridge_lambda=float(self.ridge_lambda.value()),
            convergence_window_ms=None,
        )
        return cfg

    def _time_axis_ms(self, meta: Dict[str, Any]) -> np.ndarray:
        fs = float(meta["fs_hz"])
        tmin = float(meta["tmin_ms"])
        n_times = int(meta["n_times"])
        dt_ms = 1000.0 / fs
        return tmin + np.arange(n_times) * dt_ms

    # -------------------------
    # Data loading / validation
    # -------------------------
    def _on_load_clicked(self) -> None:
        dlg = QT["QFileDialog"](self, "Select input file")
        dlg.setFileMode(QT["QFileDialog"].ExistingFile)
        dlg.setNameFilters(["NPZ files (*.npz)", "CSV files (*.csv)", "All files (*.*)"])

        if dlg.exec() != QT["QFileDialog"].Accepted:
            return

        path = dlg.selectedFiles()[0]
        self.path_edit.setText(path)

        try:
            self._append_log(f"Loading: {path}")

            from adjar64.io import load_npz, load_csv_epochs

            if path.lower().endswith(".npz"):
                payload = load_npz(path)
            elif path.lower().endswith(".csv"):
                payload = load_csv_epochs(path)
            else:
                raise ValueError("Unsupported file type. Use .npz or .csv.")

            self.payload = payload

            # Populate suggested fs/tmin/tmax
            self.fs_edit.setText(str(getattr(payload, "fs_hz", "")))
            self.tmin_edit.setText(str(getattr(payload, "tmin_ms", "")))
            self.tmax_edit.setText(str(getattr(payload, "tmax_ms", "")))

            # Populate conditions if available
            trial_labels = getattr(payload, "trial_labels", None)
            self.condition_combo.clear()
            self.condition_combo.addItem("(all trials)")
            if trial_labels is not None:
                labels = np.asarray(trial_labels)
                uniq = sorted(set([str(x) for x in labels.tolist()]))
                for u in uniq:
                    self.condition_combo.addItem(u)
                self.condition_combo.setEnabled(True)
            else:
                self.condition_combo.setEnabled(False)

            self._append_log("Load completed.")
            self._on_validate_clicked()

        except Exception:
            self._show_error("Load failed", traceback.format_exc())

    def _on_validate_clicked(self) -> None:
        try:
            if self.payload is None:
                raise ValueError("No data loaded.")

            data = getattr(self.payload, "data", None)
            isi = getattr(self.payload, "isi_pre_ms", None)

            if data is None:
                raise ValueError("Payload missing .data.")
            data = np.asarray(data)
            if data.ndim != 3:
                raise ValueError(f"Expected 3D epoched data (trials, channels, times). Got shape {data.shape}")

            if isi is None:
                raise ValueError("Payload missing isi_pre_ms (preceding interval per trial). Required for ADJAR.")

            isi = np.asarray(isi)
            if isi.shape[0] != data.shape[0]:
                raise ValueError(
                    f"isi_pre_ms length {isi.shape[0]} does not match n_trials {data.shape[0]}"
                )

            fs_hz = _safe_float(self.fs_edit.text(), 0.0)
            tmin_ms = _safe_float(self.tmin_edit.text(), 0.0)
            tmax_ms = _safe_float(self.tmax_edit.text(), 0.0)

            if fs_hz <= 0:
                raise ValueError("Sampling rate (fs_hz) must be > 0.")
            if tmax_ms <= tmin_ms:
                raise ValueError("tmax_ms must be greater than tmin_ms.")

            self._append_log(
                f"Validated: trials={data.shape[0]}, channels={data.shape[1]}, times={data.shape[2]}"
            )

        except Exception:
            self._show_error("Validation failed", traceback.format_exc())

    # -------------------------
    # Run pipeline
    # -------------------------
    def _on_run_clicked(self) -> None:
        try:
            if self.payload is None:
                raise ValueError("No data loaded.")
            self._on_validate_clicked()

            cfg = self._build_config()
            if cfg.fs_hz <= 0:
                raise ValueError("Sampling rate (Hz) is required.")
            if cfg.tmax_ms <= cfg.tmin_ms:
                raise ValueError("Epoch window (tmin/tmax) is required.")

            run_l1 = bool(self.cb_run_l1.isChecked())
            run_l2 = bool(self.cb_run_l2.isChecked())
            if not run_l1 and not run_l2:
                raise ValueError("Select at least one of Level 1 or Level 2.")

            cond = self._get_selected_condition()

            self.progress.setValue(0)
            self.btn_run.setEnabled(False)
            self.btn_export_csv.setEnabled(False)
            self.btn_export_npz.setEnabled(False)
            self.last_result = None
            self.plot_panel.clear()

            self._append_log("Starting pipeline...")

            worker = PipelineWorker(
                payload=self.payload,
                config=cfg,
                run_l1=run_l1,
                run_l2=run_l2,
                condition=cond,
            )
            worker.signals.started.connect(lambda: self._append_log("Worker started."))
            worker.signals.progress.connect(self._on_worker_progress)
            worker.signals.finished.connect(self._on_worker_finished)
            worker.signals.error.connect(self._on_worker_error)

            self.threadpool.start(worker)

            # Switch to Run tab
            self.tabs.setCurrentIndex(2)

        except Exception:
            self.btn_run.setEnabled(True)
            self._show_error("Run failed", traceback.format_exc())

    def _on_worker_progress(self, pct: int, msg: str) -> None:
        self.progress.setValue(int(pct))
        self._append_log(f"[{pct:3d}%] {msg}")

    def _on_worker_finished(self, result: object) -> None:
        self.btn_run.setEnabled(True)
        self.progress.setValue(100)

        if not isinstance(result, dict):
            self._show_error("Pipeline error", "Worker returned an unexpected result type.")
            return

        self.last_result = result
        meta = result.get("meta", {})
        self._append_log("Pipeline completed.")
        self._append_log(f"Meta: trials={meta.get('n_trials')}, channels={meta.get('n_channels')}")

        # Update channel selector range
        n_ch = int(meta.get("n_channels", 1))
        self.channel_spin.setRange(0, max(0, n_ch - 1))
        self.channel_spin.setValue(0)

        self.btn_export_csv.setEnabled(True)
        self.btn_export_npz.setEnabled(True)

        self._refresh_plot()
        self.tabs.setCurrentIndex(3)

    def _on_worker_error(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.progress.setValue(0)
        self._show_error("Pipeline failed", err)

    def _refresh_plot(self) -> None:
        if self.last_result is None:
            return

        meta = self.last_result.get("meta", {})
        t_ms = self._time_axis_ms(meta)

        erp_conv = self.last_result.get("erp_conv", None)
        erp_l1 = self.last_result.get("erp_l1", None)
        erp_l2 = self.last_result.get("erp_l2", None)

        if erp_conv is None:
            return

        conv = np.asarray(erp_conv)
        l1 = np.asarray(erp_l1) if erp_l1 is not None else None
        l2 = np.asarray(erp_l2) if erp_l2 is not None else None

        ch = int(self.channel_spin.value())

        title = "ERP comparison"
        cond = meta.get("condition", None)
        if cond:
            title += f" (condition={cond})"

        self.plot_panel.plot_erp_compare(
            t_ms=t_ms,
            conv=conv,
            l1=l1,
            l2=l2,
            channel_idx=ch,
            title=title,
        )

    # -------------------------
    # Export
    # -------------------------
    def _on_export_npz(self) -> None:
        try:
            if self.last_result is None:
                raise ValueError("No results to export.")
            path, _ = QT["QFileDialog"].getSaveFileName(
                self,
                "Save results (NPZ)",
                "adjar_results.npz",
                "NPZ files (*.npz)",
            )
            if not path:
                return

            from adjar64.io import export_erp_npz

            export_erp_npz(path, self.last_result)
            self._append_log(f"Exported NPZ: {path}")

        except Exception:
            self._show_error("Export failed", traceback.format_exc())

    def _on_export_csv(self) -> None:
        try:
            if self.last_result is None:
                raise ValueError("No results to export.")
            path, _ = QT["QFileDialog"].getSaveFileName(
                self,
                "Save ERP (CSV)",
                "adjar_erp.csv",
                "CSV files (*.csv)",
            )
            if not path:
                return

            from adjar64.io import export_erp_csv

            export_erp_csv(path, self.last_result)
            self._append_log(f"Exported CSV: {path}")

        except Exception:
            self._show_error("Export failed", traceback.format_exc())


# -------------------------
# Entrypoint
# -------------------------
def main(argv: Optional[list[str]] = None) -> int:
    """
    GUI entrypoint used by:
      - `python -m adjar64.gui.main`
      - console script `adjar64-gui`
      - PyInstaller EXE

    Returns process exit code.
    """
    argv = sys.argv if argv is None else argv

    app = QT["QApplication"](argv)
    app.setApplicationName("ADJAR_64BIT")
    app.setOrganizationName("ADJAR_64BIT")

    win = MainWindow()
    win.show()

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
