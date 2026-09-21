"""
app_window.py - main window module v13.0

Core updates in this version:
  Drift estimation has been fully switched from whole-frame cross-correlation
  to fiducial tracking. Single molecules detected in TIRF videos are used as
  natural fiducial markers, tracked frame by frame with subpixel centroids,
  then aggregated with a weighted median for zero-accumulation, high-precision
  drift correction (~0.05 px per molecule).

New in v13.0:
  - Replaced print() with the logging module
  - Added a bottom log viewer panel
  - Added project/session save and load support
  - Unified long-running work under cancellable process tasks
  - Added null guards in update_video_display()
  - Added interactive intensity plot zoom/pan/channel hiding/right-click export

Unchanged capabilities:
  - Subpixel intensity extraction via cv2.getRectSubPix
  - Automatic coordinate conversion for non-532 detection channels
  - Intensity scale correction
  - Affine registration
  - Drift trajectory visualization
  - Batch detection mode
  - Split CSV export
"""

import sys
import os
import glob
import logging
import numpy as np
import multiprocessing as mp
from functools import partial

from gpu_backend import gpu_status
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
    QListWidget, QFileDialog, QMessageBox, QProgressBar,
    QSplitter, QGroupBox, QGridLayout, QDialog,
    QMenu, QScrollArea, QDoubleSpinBox,
    QLineEdit, QCheckBox, QSpinBox, QTabWidget,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QSettings, QTimer, pyqtSlot

from file_io import ND2Reader
from channel_profiles import ALEX_LOGICAL_CHANNELS, physical_channel_name
from processing import (
    DriftCalculator,
)
from controllers import MainWindowController
from runtime_services import (
    DEFAULT_CACHE_BUDGET_MB,
    ProcessTaskService,
    compute_file_fingerprint,
    fingerprint_matches,
    load_project_bundle,
    max_projection_first_n_frames,
    save_project_bundle,
    _format_drop_summary,
    _strict_filter_complete_molecules,
    write_intensities_csv_split,
)

from ui_widgets import (
    VideoWidget,
    SyncZoomManager,
    IntensityPlotWidget,
    MultiLevelCache,
    DetectionPreviewDialog,
    LUTControlPanel,
    QtLogHandler,
    LogPanel,
)

from registration import (
    RegistrationIO,
    RegistrationModel,
    RegistrationValidationError,
    REFERENCE_CHANNEL,
    validate_quantitative_registration,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

logger = logging.getLogger(__name__)

SETTINGS_ORGANIZATION = "SingleMoleculeND2"
SETTINGS_APPLICATION = "IntensityExtractionTool"

FIXED_CHANNEL_ORDER = list(ALEX_LOGICAL_CHANNELS)


# ==================== Logging Setup ====================

_qt_log_handler = QtLogHandler()


def _qt_log_handler_is_alive() -> bool:
    try:
        _ = _qt_log_handler.emitter.log_signal
        return True
    except RuntimeError:
        return False


def _ensure_qt_log_handler() -> QtLogHandler:
    global _qt_log_handler
    if _qt_log_handler_is_alive():
        return _qt_log_handler

    root = logging.getLogger()
    try:
        root.removeHandler(_qt_log_handler)
    except Exception:
        pass
    _qt_log_handler = QtLogHandler()
    _qt_log_handler.setLevel(logging.DEBUG)
    for handler in root.handlers:
        formatter = getattr(handler, "formatter", None)
        if formatter is not None:
            _qt_log_handler.setFormatter(formatter)
            break
    root.addHandler(_qt_log_handler)
    return _qt_log_handler


def setup_logging():
    if getattr(setup_logging, "_configured", False):
        return

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    handler = _ensure_qt_log_handler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(fmt)
    if handler not in root.handlers:
        root.addHandler(handler)
    setup_logging._configured = True


_ORIGINAL_EXCEPTHOOK = sys.excepthook


def _log_unhandled_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
        return
    logger.critical(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )
    _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)


# ==================== Helper Functions ====================


def _app_settings() -> QSettings:
    return QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


def _dialog_dir(key: str = "last_dir") -> str:
    settings = _app_settings()
    value = settings.value(f"paths/{key}", "", type=str) or ""
    if not value:
        value = settings.value("paths/last_dir", "", type=str) or ""
    return value if value and os.path.isdir(value) else ""


def _remember_dialog_path(path: str, key: str = "last_dir") -> None:
    if not path:
        return
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if not folder:
        return
    settings = _app_settings()
    settings.setValue(f"paths/{key}", folder)
    settings.setValue("paths/last_dir", folder)


def _default_dialog_path(key: str, filename: str) -> str:
    folder = _dialog_dir(key)
    return os.path.join(folder, filename) if folder else filename


def max_projection_frame_range_reader(
    reader: ND2Reader,
    channel_names,
    start_frame: int = 0,
    n: int = 10,
):
    num_frames = int(reader.num_frames)
    start = max(0, min(int(start_frame), num_frames - 1))
    n_use = max(1, min(int(n), num_frames - start))
    num_ch = int(reader.num_channels)

    proj = [None] * num_ch
    for fi in range(start, start + n_use):
        frame = reader.get_frame_data(fi)
        for ci in range(num_ch):
            img = frame[ci]
            if proj[ci] is None:
                proj[ci] = img.copy()
            else:
                proj[ci] = np.maximum(proj[ci], img)
    for ci in range(num_ch):
        if proj[ci] is None:
            proj[ci] = np.zeros(
                (reader.height, reader.width), dtype=np.uint16
            )
        else:
            proj[ci] = proj[ci].astype(np.uint16)
    return proj


def max_projection_first_n_frames_reader(
    reader: ND2Reader,
    channel_names,
    n: int = 10,
):
    return max_projection_frame_range_reader(
        reader,
        channel_names,
        start_frame=0,
        n=n,
    )


# ==================== Drift Plot Dialog ====================

class DriftPlotDialog(QDialog):
    """Display the drift trajectory figure produced by DriftCalculator."""

    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drift Trajectory - Fiducial Tracking")
        self.resize(1100, 780)
        self._figure = figure

        layout = QVBoxLayout()

        self._canvas = FigureCanvas(figure)
        layout.addWidget(self._canvas, stretch=1)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Image...")
        save_btn.clicked.connect(self._save_figure)
        btn_layout.addWidget(save_btn)
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _save_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Drift Trajectory",
            _default_dialog_path("export", "drift_trajectory.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            try:
                self._figure.savefig(path, dpi=200, bbox_inches='tight')
                _remember_dialog_path(path, "export")
                QMessageBox.information(
                    self, "Success", f"Drift trajectory saved to:\n{path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")


# ==================== Exposure Override Dialog ====================

class ExposureOverrideDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        metadata_exposure_ms: float | None = None,
        metadata_exposure_source: str = "",
        override_enabled: bool = False,
        override_ms: float | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Exposure Time Override")
        self.resize(480, 210)
        self.metadata_exposure_ms = metadata_exposure_ms
        self.metadata_exposure_source = str(metadata_exposure_source or "ND2 metadata")

        layout = QVBoxLayout()
        if metadata_exposure_ms is None:
            metadata_text = "ND2 metadata exposure: unavailable"
            default_ms = 100.0
        else:
            metadata_text = (
                f"ND2 metadata exposure: {float(metadata_exposure_ms):.4f} ms "
                f"({self.metadata_exposure_source})"
            )
            default_ms = float(metadata_exposure_ms)
        layout.addWidget(QLabel(metadata_text))

        self.override_check = QCheckBox("Override ND2 metadata exposure for exported Time_sec")
        self.override_check.setChecked(bool(override_enabled))
        layout.addWidget(self.override_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Manual exposure:"))
        self.override_spin = QDoubleSpinBox()
        self.override_spin.setRange(0.001, 1_000_000.0)
        self.override_spin.setDecimals(4)
        self.override_spin.setSingleStep(1.0)
        self.override_spin.setSuffix(" ms")
        self.override_spin.setValue(float(override_ms if override_ms is not None else default_ms))
        self.override_spin.setEnabled(bool(override_enabled))
        row.addWidget(self.override_spin)
        layout.addLayout(row)

        note = QLabel(
            "This changes the time axis used by plots, project state, and CSV export. "
            "It does not modify the ND2 file or intensity values."
        )
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { color: #666; }")
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch()
        ok = QPushButton("Apply")
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.override_check.stateChanged.connect(
            lambda _state: self.override_spin.setEnabled(self.override_check.isChecked())
        )

    def get_override(self) -> tuple[bool, float | None]:
        enabled = bool(self.override_check.isChecked())
        return enabled, (float(self.override_spin.value()) if enabled else None)


# ==================== Batch Detection Dialog ====================

class BatchDetectionDialog(QDialog):
    def __init__(self, parent, initial_intensity_scales: dict):
        super().__init__(parent)
        self.setWindowTitle("Batch Detection Mode (ND2 Folder)")
        self.resize(900, 520)

        self.folder = ""
        self.sample_file = ""
        self.sample_channel_names = []
        self.sample_exposure_ms = None
        self.sample_exposure_source = ""
        self.sample_num_frames = 0
        self.initial_intensity_scales = dict(initial_intensity_scales or {})
        self._preview_frame = None
        self._sample_proj_by_channel = []

        self._params = {
            "sigma": 3.5,
            "min_snr": 4.0,
            "min_r2": 0.0,
            "min_distance": None,
            "auto_bg": True,
            "bg_radius": None,
        }

        self._scale_spins = {}

        self._build_ui()
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)

    def _build_ui(self):
        layout = QVBoxLayout()

        folder_group = QGroupBox("1) Select the folder containing ND2 files")
        fg = QGridLayout()

        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        browse_btn = QPushButton("Choose Folder...")
        browse_btn.clicked.connect(self._choose_folder)

        fg.addWidget(QLabel("Folder:"), 0, 0)
        fg.addWidget(self.folder_edit, 0, 1)
        fg.addWidget(browse_btn, 0, 2)

        self.sample_info = QLabel("No folder selected yet")
        self.sample_info.setWordWrap(True)
        self.sample_info.setStyleSheet("QLabel { color: #666; }")
        fg.addWidget(self.sample_info, 1, 0, 1, 3)

        folder_group.setLayout(fg)
        layout.addWidget(folder_group)

        time_group = QGroupBox("2) Exposure Time")
        tg = QGridLayout()
        self.exposure_info = QLabel("Exposure time is read from ND2 metadata.")
        self.exposure_info.setWordWrap(True)
        tg.addWidget(self.exposure_info, 0, 0, 1, 3)
        self.exposure_override_check = QCheckBox("Override metadata exposure for CSV Time_sec")
        self.exposure_override_check.stateChanged.connect(self._on_exposure_override_toggled)
        tg.addWidget(self.exposure_override_check, 1, 0, 1, 3)
        tg.addWidget(QLabel("Manual exposure:"), 2, 0)
        self.exposure_override_spin = QDoubleSpinBox()
        self.exposure_override_spin.setRange(0.001, 1_000_000.0)
        self.exposure_override_spin.setDecimals(4)
        self.exposure_override_spin.setSingleStep(1.0)
        self.exposure_override_spin.setSuffix(" ms")
        self.exposure_override_spin.setValue(100.0)
        self.exposure_override_spin.setEnabled(False)
        self.exposure_override_spin.valueChanged.connect(lambda _value: self._update_batch_exposure_info())
        tg.addWidget(self.exposure_override_spin, 2, 1)
        hint = QLabel("Default uses ND2 metadata. Override only changes exported time, not intensity values.")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; }")
        tg.addWidget(hint, 3, 0, 1, 3)
        time_group.setLayout(tg)
        layout.addWidget(time_group)

        projection_group = QGroupBox("3) Detection Projection Frames")
        prg = QGridLayout()
        self.proj_start_spin = QSpinBox()
        self.proj_start_spin.setRange(0, 0)
        self.proj_start_spin.setValue(0)
        self.proj_start_spin.setSuffix(" frame")
        self.proj_start_spin.setEnabled(False)
        self.proj_count_spin = QSpinBox()
        self.proj_count_spin.setRange(10, 10)
        self.proj_count_spin.setValue(10)
        self.proj_count_spin.setSuffix(" frames")
        self.proj_count_spin.setEnabled(False)
        prg.addWidget(QLabel("Start frame:"), 0, 0)
        prg.addWidget(self.proj_start_spin, 0, 1)
        prg.addWidget(QLabel("Window size:"), 1, 0)
        prg.addWidget(self.proj_count_spin, 1, 1)
        prg.addWidget(QLabel("Batch detection is fixed to frames 0:10."), 2, 0, 1, 2)
        projection_group.setLayout(prg)
        layout.addWidget(projection_group)

        channel_group = QGroupBox(
            "4) Detection Channel (matched by channel name; files are skipped if absent)"
        )
        cg = QHBoxLayout()
        cg.addWidget(QLabel("Detection channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["(Choose a folder first)"])
        cg.addWidget(self.channel_combo)
        cg.addStretch()
        channel_group.setLayout(cg)
        layout.addWidget(channel_group)

        param_group = QGroupBox("5) Detection Parameters")
        pg = QGridLayout()
        r = 0

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.5, 10.0)
        self.sigma_spin.setValue(self._params["sigma"])
        self.sigma_spin.setSingleStep(0.1)
        pg.addWidget(QLabel("Sigma:"), r, 0)
        pg.addWidget(self.sigma_spin, r, 1)
        r += 1

        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(1.0, 10.0)
        self.snr_spin.setValue(self._params["min_snr"])
        self.snr_spin.setSingleStep(0.5)
        pg.addWidget(QLabel("Min SNR:"), r, 0)
        pg.addWidget(self.snr_spin, r, 1)
        r += 1

        self.r2_spin = QDoubleSpinBox()
        self.r2_spin.setRange(0.0, 0.99)
        self.r2_spin.setValue(self._params["min_r2"])
        self.r2_spin.setSingleStep(0.05)
        pg.addWidget(QLabel("Min R^2:"), r, 0)
        pg.addWidget(self.r2_spin, r, 1)
        r += 1

        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(0, 100)
        self.min_dist_spin.setSpecialValueText("Auto")
        self.min_dist_spin.setValue(0)
        pg.addWidget(QLabel("Min distance (px) 0=Auto:"), r, 0)
        pg.addWidget(self.min_dist_spin, r, 1)
        r += 1

        self.bg_check = QCheckBox("Enable rolling-ball background subtraction")
        self.bg_check.setChecked(True)
        pg.addWidget(self.bg_check, r, 0, 1, 2)
        r += 1

        self.bg_radius_spin = QSpinBox()
        self.bg_radius_spin.setRange(0, 200)
        self.bg_radius_spin.setSpecialValueText("Auto")
        self.bg_radius_spin.setValue(0)
        pg.addWidget(QLabel("Rolling-ball radius (px) 0=Auto:"), r, 0)
        pg.addWidget(self.bg_radius_spin, r, 1)
        r += 1

        self.preview_btn = QPushButton("Preview max projection of the selected sample frames...")
        self.preview_btn.clicked.connect(self._preview)
        pg.addWidget(self.preview_btn, r, 0, 1, 2)
        r += 1

        param_group.setLayout(pg)
        layout.addWidget(param_group)

        reg_group = QGroupBox("6) Registration (required for multi-channel quantitative output)")
        rg = QGridLayout()
        self.reg_check = QCheckBox("Use registration JSON")
        self.reg_check.stateChanged.connect(self._on_reg_toggled)
        rg.addWidget(self.reg_check, 0, 0, 1, 2)

        self.reg_path_edit = QLineEdit()
        self.reg_path_edit.setReadOnly(True)
        self.reg_path_edit.setEnabled(False)
        reg_btn = QPushButton("Choose JSON...")
        reg_btn.setEnabled(False)
        reg_btn.clicked.connect(self._choose_reg_json)
        self._reg_btn = reg_btn

        rg.addWidget(QLabel("JSON:"), 1, 0)
        rg.addWidget(self.reg_path_edit, 1, 1)
        rg.addWidget(reg_btn, 1, 2)

        reg_group.setLayout(rg)
        layout.addWidget(reg_group)

        scale_group = QGroupBox("7) Intensity Scale")
        self.scale_layout = QGridLayout()
        scale_group.setLayout(self.scale_layout)
        layout.addWidget(scale_group)

        self._rebuild_scale_editor([])

        perf_group = QGroupBox("8) Performance")
        pf = QGridLayout()
        self.parallel_workers_spin = QSpinBox()
        self.parallel_workers_spin.setRange(1, 4)
        self.parallel_workers_spin.setValue(2)
        self.parallel_workers_spin.setSuffix(" worker(s)")
        pf.addWidget(QLabel("Parallel ND2 workers:"), 0, 0)
        pf.addWidget(self.parallel_workers_spin, 0, 1)
        self.fast_drift_check = QCheckBox("Fast drift tracking (1 s sampling)")
        self.fast_drift_check.setChecked(False)
        pf.addWidget(self.fast_drift_check, 1, 0, 1, 2)
        note = QLabel("Use 1 for strict serial processing or low-memory disks. Fast drift is optional and falls back to full tracking if sampled frames are unreliable.")
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { color: #666; }")
        pf.addWidget(note, 2, 0, 1, 2)
        perf_group.setLayout(pf)
        layout.addWidget(perf_group)

        btns = QHBoxLayout()
        btns.addStretch()
        ok = QPushButton("Confirm (Start Batch)")
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

        self.setLayout(layout)

    def _on_exposure_override_toggled(self):
        enabled = self.exposure_override_check.isChecked()
        self.exposure_override_spin.setEnabled(enabled)
        if enabled and self.sample_exposure_ms is not None and self.exposure_override_spin.value() == 100.0:
            self.exposure_override_spin.setValue(float(self.sample_exposure_ms))
        self._update_batch_exposure_info()

    def _update_batch_exposure_info(self):
        if self.sample_exposure_ms is None:
            self.exposure_info.setText("Exposure time is read from ND2 metadata.")
            return
        if self.exposure_override_check.isChecked():
            self.exposure_info.setText(
                f"Effective export exposure: {self.exposure_override_spin.value():.4f} ms (manual_override)\n"
                f"ND2 metadata exposure: {self.sample_exposure_ms:.4f} ms ({self.sample_exposure_source})"
            )
        else:
            self.exposure_info.setText(
                f"Exposure: {self.sample_exposure_ms:.4f} ms ({self.sample_exposure_source})"
            )

    def _on_reg_toggled(self):
        enabled = self.reg_check.isChecked()
        self.reg_path_edit.setEnabled(enabled)
        self._reg_btn.setEnabled(enabled)

    def _choose_reg_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Registration JSON",
            _dialog_dir("registration"),
            "JSON Files (*.json)",
        )
        if path:
            _remember_dialog_path(path, "registration")
            self.reg_path_edit.setText(path)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose the Folder Containing ND2 Files",
            _dialog_dir("batch_folder"),
        )
        if not folder:
            return
        _remember_dialog_path(folder, "batch_folder")
        self.folder = folder
        self.folder_edit.setText(folder)

        nd2_files = sorted(glob.glob(os.path.join(folder, "*.nd2")))
        if not nd2_files:
            self.sample_file = ""
            self.sample_channel_names = []
            self.sample_num_frames = 0
            self.channel_combo.clear()
            self.channel_combo.addItems(["(No .nd2 files in this folder)"])
            self.sample_info.setText("No .nd2 files in this folder")
            self._preview_frame = None
            self._sample_proj_by_channel = []
            self._rebuild_scale_editor([])
            self._update_batch_exposure_info()
            return

        self.sample_file = nd2_files[0]

        reader = None
        try:
            reader = ND2Reader(self.sample_file)
            self.sample_channel_names = list(reader.channel_names)
            self.sample_num_frames = int(reader.num_frames)
            metadata = reader.get_metadata()
            self.sample_exposure_ms = float(metadata["exposure_ms"])
            self.sample_exposure_source = str(metadata.get("exposure_source", "ND2 metadata"))
            self._sample_proj_by_channel = max_projection_frame_range_reader(
                reader,
                self.sample_channel_names,
                start_frame=0,
                n=10,
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read sample ND2:\n{e}")
            self.sample_channel_names = []
            self.sample_num_frames = 0
            self._preview_frame = None
            self._sample_proj_by_channel = []
            self.sample_exposure_ms = None
            self.sample_exposure_source = ""
            return
        finally:
            if reader is not None:
                reader.close()

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(self.sample_channel_names)
        if REFERENCE_CHANNEL in self.sample_channel_names:
            self.channel_combo.setCurrentText(REFERENCE_CHANNEL)
        self.channel_combo.blockSignals(False)

        self.sample_info.setText(
            f"Sample file: {os.path.basename(self.sample_file)}\n"
            f"Channels: {', '.join(self.sample_channel_names)}\n"
            f"Frames: {self.sample_num_frames}\n"
            f"Exposure: {self.sample_exposure_ms:.4f} ms ({self.sample_exposure_source})\n"
            "Preview/detection window: 0:10"
        )
        self.exposure_info.setText(
            f"Exposure: {self.sample_exposure_ms:.4f} ms ({self.sample_exposure_source})"
        )
        if not self.exposure_override_check.isChecked():
            self.exposure_override_spin.setValue(float(self.sample_exposure_ms))
        self._update_batch_exposure_info()

        self._update_preview_frame_from_current_channel()
        self._rebuild_scale_editor(self.sample_channel_names)

    def _on_channel_changed(self, _):
        self._update_preview_frame_from_current_channel()

    def _refresh_sample_projection(self) -> bool:
        if not self.sample_file:
            return False
        reader = None
        try:
            reader = ND2Reader(self.sample_file)
            metadata = reader.get_metadata()
            self.sample_exposure_ms = float(metadata["exposure_ms"])
            self.sample_exposure_source = str(metadata.get("exposure_source", "ND2 metadata"))
            self._sample_proj_by_channel = max_projection_frame_range_reader(
                reader,
                self.sample_channel_names,
                start_frame=0,
                n=10,
            )
            self.sample_info.setText(
                f"Sample file: {os.path.basename(self.sample_file)}\n"
                f"Channels: {', '.join(self.sample_channel_names)}\n"
                f"Frames: {self.sample_num_frames}\n"
                f"Exposure: {self.sample_exposure_ms:.4f} ms ({self.sample_exposure_source})\n"
                "Preview/detection window: 0:10"
            )
            self.exposure_info.setText(
                f"Exposure: {self.sample_exposure_ms:.4f} ms ({self.sample_exposure_source})"
            )
            if not self.exposure_override_check.isChecked():
                self.exposure_override_spin.setValue(float(self.sample_exposure_ms))
            self._update_batch_exposure_info()
            self._update_preview_frame_from_current_channel()
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to refresh sample projection:\n{exc}")
            self._preview_frame = None
            self._sample_proj_by_channel = []
            return False
        finally:
            if reader is not None:
                reader.close()

    def _update_preview_frame_from_current_channel(self):
        if not self.sample_channel_names or not self._sample_proj_by_channel:
            self._preview_frame = None
            return

        ch_name = self.channel_combo.currentText()
        idx = (self.sample_channel_names.index(ch_name)
               if ch_name in self.sample_channel_names else 0)
        self._preview_frame = (
            self._sample_proj_by_channel[idx]
            if 0 <= idx < len(self._sample_proj_by_channel)
            else None
        )

    def _collect_params(self):
        md = int(self.min_dist_spin.value())
        br = int(self.bg_radius_spin.value())
        return {
            "sigma": float(self.sigma_spin.value()),
            "min_snr": float(self.snr_spin.value()),
            "min_r2": float(self.r2_spin.value()),
            "min_distance": md if md > 0 else None,
            "auto_bg": bool(self.bg_check.isChecked()),
            "bg_radius": br if br > 0 else None,
        }

    def _preview(self):
        self._refresh_sample_projection()
        if self._preview_frame is None:
            QMessageBox.warning(
                self, "Notice", "Choose a folder first and make sure the sample projection can be read"
            )
            return

        params = self._collect_params()
        dlg = DetectionPreviewDialog(
            self._preview_frame, self, initial_params=params
        )
        if dlg.exec_() == QDialog.Accepted:
            p2 = dlg.get_params()
            self.sigma_spin.setValue(p2["sigma"])
            self.snr_spin.setValue(p2["min_snr"])
            self.r2_spin.setValue(p2["min_r2"])
            self.min_dist_spin.setValue(int(p2["min_distance"] or 0))
            self.bg_check.setChecked(bool(p2["auto_bg"]))
            self.bg_radius_spin.setValue(int(p2["bg_radius"] or 0))

    def _rebuild_scale_editor(self, channel_names):
        while self.scale_layout.count():
            item = self.scale_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._scale_spins = {}

        if not channel_names:
            self.scale_layout.addWidget(
                QLabel("Choose a folder to display Intensity Scale settings"), 0, 0
            )
            return

        for r, ch in enumerate(channel_names):
            self.scale_layout.addWidget(QLabel(f"{ch}:"), r, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0.01, 10.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setValue(
                float(self.initial_intensity_scales.get(ch, 1.0))
            )
            self.scale_layout.addWidget(spin, r, 1)
            self.scale_layout.addWidget(
                QLabel("Intensity will be multiplied by this factor"), r, 2
            )
            self._scale_spins[ch] = spin

    def get_config(self):
        if not self.folder:
            raise ValueError("No folder selected")
        if not self.sample_channel_names:
            raise ValueError("Failed to parse sample channels")

        detect_channel_name = self.channel_combo.currentText()

        reg = None
        if self.reg_check.isChecked():
            path = self.reg_path_edit.text().strip()
            if not path or not os.path.exists(path):
                raise ValueError("Registration was enabled but no valid JSON file was selected")
            reg = RegistrationIO.load_json(path).to_dict()
        validate_quantitative_registration(
            reg,
            self.sample_channel_names,
            detect_channel=detect_channel_name,
        )

        intensity_scales = {
            ch: float(spin.value())
            for ch, spin in self._scale_spins.items()
        }
        exposure_override_enabled = bool(self.exposure_override_check.isChecked())
        exposure_override_ms = float(self.exposure_override_spin.value()) if exposure_override_enabled else None
        if exposure_override_enabled and (not np.isfinite(exposure_override_ms) or exposure_override_ms <= 0):
            raise ValueError("Manual exposure override must be finite and > 0 ms")

        return {
            "folder": self.folder,
            "detect_channel_name": detect_channel_name,
            "params": self._collect_params(),
            "registration_params": reg,
            "intensity_scales": intensity_scales,
            "detect_projection_start_frame": 0,
            "detect_projection_frames": 10,
            "auto_search_detection_window": False,
            "compute_backend": "auto",
            "parallel_workers": int(self.parallel_workers_spin.value()),
            "drift_tracking_mode": "sparse" if self.fast_drift_check.isChecked() else "full",
            "sparse_drift_interval_sec": 1.0,
            "exposure_override_enabled": exposure_override_enabled,
            "exposure_override_ms": exposure_override_ms,
        }

# ==================== Main Window ====================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Single-Molecule Fluorescence Intensity Extraction Tool v13.0 [ND2 Edition]"
        )
        self.resize(1800, 1000)

        self.registration_params = None
        self.nd2_file = None
        self.nd2_fingerprint = {}
        self.file_reader = None
        self.multi_cache = None
        self.num_channels = 0
        self.num_frames = 0
        self.exposure_s = 0.1
        self.exposure_source = "ND2 metadata"
        self.metadata_exposure_ms = None
        self.metadata_exposure_source = ""
        self.exposure_override_enabled = False
        self.exposure_override_ms = None
        self.channel_names = []
        self.molecules = []
        self.base_molecules = []
        self.molecule_features = {}
        self.base_molecule_features = {}
        self.deleted_molecules = set()
        self.molecule_intensities = {}
        self.current_frame = 0
        self.selected_molecule = None
        self.detect_channel = 0
        self.analysis_recipe = {}
        self.operation_log = []
        self.uncertainty_filter = {"enabled": False, "max_scalar": 0.5}
        self.active_edit_tool = "pan"
        self.view_transform = {}
        self.results_mode = "normal"
        self.drift_result = {}

        self.lut_panels = []
        self.intensity_scale_spinboxes = {}
        self.intensity_scales = {}

        self.is_playing = False
        self.play_speed = 1.0
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

        self.window_controller = MainWindowController()
        self.task_poll_timer = QTimer()
        self.task_poll_timer.setInterval(50)
        self.task_poll_timer.timeout.connect(self._poll_task_events)
        self.task_service = None
        self._active_task_ids = {}
        self._task_contexts = {}
        self._init_process_task_runtime()
        self.sync_manager = SyncZoomManager()
        self.video_widgets = []
        self._splitter_sizes_initialized = False

        self.scale_reprocess_timer = QTimer()
        self.scale_reprocess_timer.setSingleShot(True)
        self.scale_reprocess_timer.setInterval(800)
        self.scale_reprocess_timer.timeout.connect(
            self._reprocess_after_scale_change
        )
        self._scale_reprocess_pending = False

        self.setup_ui()
        self.setup_menu()

        # Connect the log panel
        _ensure_qt_log_handler().emitter.log_signal.connect(self.log_panel.append_log)

    def showEvent(self, event):
        super().showEvent(event)
        if self._splitter_sizes_initialized:
            return
        if (hasattr(self, "video_splitter")
                and self.video_splitter is not None):
            h = max(1, self.video_splitter.size().height())
            self.video_splitter.setSizes([int(h * 0.7), int(h * 0.3)])
        if (hasattr(self, "display_hsplitter")
                and self.display_hsplitter is not None):
            w = max(1, self.display_hsplitter.size().width())
            self.display_hsplitter.setSizes([int(w * 0.78), int(w * 0.22)])
        self._splitter_sizes_initialized = True

    def setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        load_reg_action = file_menu.addAction("Load Registration")
        load_reg_action.triggered.connect(self.load_registration)

        estimate_reg_action = file_menu.addAction(
            "Estimate Affine Registration (Bead Calibration)"
        )
        estimate_reg_action.triggered.connect(
            self.estimate_affine_registration_from_beads
        )

        file_menu.addSeparator()

        load_data_action = file_menu.addAction("Load ND2 Data")
        load_data_action.triggered.connect(self.load_data)

        export_action = file_menu.addAction("Export CSV")
        export_action.triggered.connect(self.export_csv)

        file_menu.addSeparator()

        save_proj_action = file_menu.addAction("Save Project...")
        save_proj_action.triggered.connect(self.save_project_dialog)

        load_proj_action = file_menu.addAction("Load Project...")
        load_proj_action.triggered.connect(self.load_project_dialog)

        file_menu.addSeparator()

        batch_action = file_menu.addAction("Batch Detection Mode (Folder)")
        batch_action.triggered.connect(self.batch_detection_mode)

        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        analysis_menu = menubar.addMenu("Analysis")
        self.exposure_override_action = analysis_menu.addAction("Set Exposure Time Override...")
        self.exposure_override_action.triggered.connect(self.edit_exposure_override)
        self.exposure_override_action.setEnabled(False)
        analysis_menu.addSeparator()
        drift_plot_action = analysis_menu.addAction("View Drift Trajectory")
        drift_plot_action.triggered.connect(self.show_drift_plot)

        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        display_area = self.create_display_area()
        main_layout.addWidget(display_area, stretch=9)

        central_widget.setLayout(main_layout)

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        reg_group = QGroupBox("Registration")
        reg_layout = QVBoxLayout()
        self.reg_info_label = QLabel("No registration loaded")
        self.reg_info_label.setWordWrap(True)
        reg_layout.addWidget(self.reg_info_label)
        reg_group.setLayout(reg_layout)
        layout.addWidget(reg_group)

        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout()
        self.data_info_label = QLabel("No ND2 data loaded")
        self.data_info_label.setWordWrap(True)
        data_layout.addWidget(self.data_info_label)
        self.exposure_override_btn = QPushButton("Exposure Override...")
        self.exposure_override_btn.clicked.connect(self.edit_exposure_override)
        self.exposure_override_btn.setEnabled(False)
        data_layout.addWidget(self.exposure_override_btn)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        detect_group = QGroupBox("Molecule Detection")
        detect_layout = QVBoxLayout()

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Detection channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(FIXED_CHANNEL_ORDER)
        channel_layout.addWidget(self.channel_combo)
        detect_layout.addLayout(channel_layout)

        self.detect_btn = QPushButton("Interactive Detection (Optimized)")
        self.detect_btn.clicked.connect(self.detect_molecules_interactive)
        self.detect_btn.setEnabled(False)
        detect_layout.addWidget(self.detect_btn)

        self.detect_result_label = QLabel("")
        detect_layout.addWidget(self.detect_result_label)

        hint = QLabel(
            "Tip: detection uses the max projection of the first 10 frames by default; "
            "channel 532 is recommended as the detection channel."
        )
        hint.setStyleSheet("QLabel { color: #666; }")
        hint.setWordWrap(True)
        detect_layout.addWidget(hint)

        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)

        self.process_btn = QPushButton("Start Processing")
        self.process_btn.clicked.connect(self.process_all_frames)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        mol_group = QGroupBox("Molecule List")
        mol_layout = QVBoxLayout()
        self.molecule_list = QListWidget()
        self.molecule_list.setFocusPolicy(Qt.StrongFocus)
        self.molecule_list.currentRowChanged.connect(
            self.on_molecule_row_changed
        )
        self.molecule_list.itemClicked.connect(self.on_molecule_selected)
        self.molecule_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.molecule_list.customContextMenuRequested.connect(
            self.show_molecule_context_menu
        )
        mol_layout.addWidget(self.molecule_list)
        mol_group.setLayout(mol_layout)
        layout.addWidget(mol_group, stretch=1)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_display_area(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        edit_row = QHBoxLayout()
        edit_row.addWidget(QLabel("Image Tools:"))
        self._edit_buttons = {}
        for tool_name, label in [
            ("pan", "Pan"),
            ("zoom", "Zoom"),
            ("add", "Add"),
            ("move", "Move"),
            ("delete", "Delete"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(partial(self.set_edit_tool, tool_name))
            edit_row.addWidget(btn)
            self._edit_buttons[tool_name] = btn
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.fit_all_video_widgets)
        edit_row.addWidget(fit_btn)
        one_to_one_btn = QPushButton("100%")
        one_to_one_btn.clicked.connect(self.reset_all_video_widgets)
        edit_row.addWidget(one_to_one_btn)
        edit_row.addStretch()
        layout.addLayout(edit_row)

        self.display_hsplitter = QSplitter(Qt.Horizontal)
        self.video_splitter = QSplitter(Qt.Vertical)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.video_container = QWidget()
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_layout = QHBoxLayout()
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(4)
        self.video_container.setLayout(self.video_layout)

        scroll_area.setWidget(self.video_container)
        self.video_splitter.addWidget(scroll_area)

        # Bottom tabs: intensity plot + log
        self.bottom_tabs = QTabWidget()
        self.plot_widget = IntensityPlotWidget()
        self.plot_widget.frame_selected.connect(self.on_plot_frame_selected)
        self.bottom_tabs.addTab(self.plot_widget, "Intensity Plot")
        self.log_panel = LogPanel()
        self.bottom_tabs.addTab(self.log_panel, "Log")
        self.video_splitter.addWidget(self.bottom_tabs)

        self.video_splitter.setStretchFactor(0, 70)
        self.video_splitter.setStretchFactor(1, 30)

        self.right_panel_scroll = QScrollArea()
        self.right_panel_scroll.setWidgetResizable(True)

        self.right_panel_container = QWidget()
        self.right_panel_layout = QVBoxLayout()
        self.right_panel_layout.setContentsMargins(6, 6, 6, 6)
        self.right_panel_container.setLayout(self.right_panel_layout)

        self.right_panel_scroll.setWidget(self.right_panel_container)
        self._build_right_panel_placeholder()

        self.display_hsplitter.addWidget(self.video_splitter)
        self.display_hsplitter.addWidget(self.right_panel_scroll)
        self.display_hsplitter.setStretchFactor(0, 8)
        self.display_hsplitter.setStretchFactor(1, 2)

        layout.addWidget(self.display_hsplitter)
        widget.setLayout(layout)
        self.set_edit_tool("pan")
        return widget

    def _clear_layout(self, layout: QVBoxLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _build_right_panel_placeholder(self):
        self._clear_layout(self.right_panel_layout)
        info = QLabel(
            "Right-side control panel\n\nAfter data is loaded, this area will show:\n"
            "- Playback controls\n- LUT control panels for each channel\n"
            "- Intensity Scale controls for each channel"
        )
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { color: #666; }")
        self.right_panel_layout.addWidget(info)
        self.right_panel_layout.addStretch()

    def _default_intensity_scales_for_channels(self, channel_names):
        defaults = {"488": 1.0, "532": 0.7, "638": 1.3}
        return {ch: float(defaults.get(physical_channel_name(ch), 1.0)) for ch in channel_names}

    @staticmethod
    def _empty_task_id_map():
        return {
            "projection": None,
            "affine_registration": None,
            "process_intensities": None,
            "batch": None,
        }

    def _init_process_task_runtime(self):
        if self.task_service is None:
            self.task_service = ProcessTaskService()
        self._active_task_ids = self._empty_task_id_map()
        self._task_contexts = {}

    def _reset_progress_ui(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("")

    def _sync_drift_runtime_state(self):
        drift_result = dict(self.drift_result or {})
        required_keys = {
            "drift_offsets",
            "drift_raw",
            "n_valid_per_frame",
            "mad_values",
            "num_frames",
            "num_fiducials",
        }
        if drift_result and required_keys.issubset(drift_result):
            DriftCalculator.set_last_drift_result(drift_result)
        else:
            DriftCalculator.set_last_drift_result(None)

    def _shutdown_process_task_runtime(self, *, recreate: bool):
        self.task_poll_timer.stop()
        if self.task_service is not None:
            try:
                self.task_service.shutdown()
            except Exception:
                logger.exception("Failed to shut down process task service")
        self.task_service = ProcessTaskService() if recreate else None
        self._active_task_ids = self._empty_task_id_map()
        self._task_contexts.clear()

    def _cancel_all_active_tasks(self, *, recreate_runtime: bool = False):
        self.task_poll_timer.stop()
        if self.task_service is not None:
            try:
                self.task_service.shutdown()
            except Exception:
                logger.exception("Failed to cancel all process tasks")
        self.task_service = ProcessTaskService() if recreate_runtime else self.task_service
        self._active_task_ids = self._empty_task_id_map()
        self._task_contexts.clear()
        self._reset_progress_ui()
        self.detect_btn.setEnabled(self.multi_cache is not None and self.results_mode == "normal")
        self.process_btn.setEnabled(bool(self.multi_cache is not None and self.molecules and self.results_mode == "normal"))

    def _cancel_active_task(self, kind):
        task_id = self._active_task_ids.get(kind)
        if not task_id or self.task_service is None:
            return
        self.task_service.cancel_task(task_id)
        self._active_task_ids[kind] = None
        self._task_contexts.pop(task_id, None)
        if (
            self.task_service is not None
            and not any(self._active_task_ids.values())
            and not self.task_service.active
        ):
            self.task_poll_timer.stop()

    def _is_task_active(self, kind):
        task_id = self._active_task_ids.get(kind)
        return bool(
            task_id
            and self.task_service is not None
            and task_id in self.task_service.active
        )

    def _sync_playback_controls(self):
        if hasattr(self, "speed_combo") and self.speed_combo is not None:
            target_text = f"{self.play_speed:g}x"
            idx = self.speed_combo.findText(target_text)
            if idx >= 0:
                self.speed_combo.blockSignals(True)
                self.speed_combo.setCurrentIndex(idx)
                self.speed_combo.blockSignals(False)

        if hasattr(self, "play_btn") and self.play_btn is not None:
            self.play_btn.setText("Pause" if self.is_playing else "Play")

        if self.is_playing and self.multi_cache is not None:
            if self.video_widgets:
                interval = max(1, int(self.exposure_s * 1000 / max(self.play_speed, 1e-6)))
                self.play_timer.start(interval)
        else:
            self.play_timer.stop()
            self.is_playing = False
            if hasattr(self, "play_btn") and self.play_btn is not None:
                self.play_btn.setText("Play")

    def set_edit_tool(self, tool_name: str):
        self.active_edit_tool = str(tool_name or "pan")
        for name, button in getattr(self, "_edit_buttons", {}).items():
            button.blockSignals(True)
            button.setChecked(name == self.active_edit_tool)
            button.blockSignals(False)
        for widget in list(getattr(self, "video_widgets", []) or []):
            widget.set_edit_tool(self.active_edit_tool)

    def fit_all_video_widgets(self):
        for widget in list(getattr(self, "video_widgets", []) or []):
            widget.fit_to_window()

    def reset_all_video_widgets(self):
        for widget in list(getattr(self, "video_widgets", []) or []):
            widget.reset_view()

    def on_plot_frame_selected(self, frame_idx: int):
        if self.multi_cache is None:
            return
        self.current_frame = self._normalize_frame_index(frame_idx)
        if hasattr(self, "time_slider") and self.time_slider is not None:
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(self.current_frame)
            self.time_slider.blockSignals(False)
        self.update_video_display()

    def _start_process_task(self, kind, *, context=None, **kwargs):
        if self.task_service is None:
            self._shutdown_process_task_runtime(recreate=True)
        current_task_id = self._active_task_ids.get(kind)
        if current_task_id:
            self._cancel_active_task(kind)
        handle = self.task_service.start_task(kind, **kwargs)
        self._active_task_ids[kind] = handle.task_id
        self._task_contexts[handle.task_id] = dict(context or {})
        self.task_poll_timer.start()
        return handle.task_id

    def _clear_process_task(self, kind, task_id):
        if self._active_task_ids.get(kind) == task_id:
            self._active_task_ids[kind] = None
        self._task_contexts.pop(task_id, None)
        if (
            self.task_service is not None
            and not any(self._active_task_ids.values())
            and not self.task_service.active
        ):
            self.task_poll_timer.stop()

    def _poll_task_events(self):
        if self.task_service is None:
            self.task_poll_timer.stop()
            return
        try:
            events = self.task_service.drain_events()
        except Exception:
            logger.exception("Failed to drain process task events")
            return

        for event in events:
            if self._active_task_ids.get(event.kind) != event.task_id:
                continue

            if event.event_type == "progress":
                progress = int(event.payload.get("progress", 0))
                message = str(event.payload.get("message", ""))
                if event.kind == "process_intensities":
                    self.on_processing_progress(progress, message)
                elif event.kind == "batch":
                    self._on_batch_progress(progress, message)
                else:
                    self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(progress)
                    self.progress_label.setText(message)
                continue

            if event.event_type == "error":
                message = str(event.payload.get("message", "Task failed"))
                self._clear_process_task(event.kind, event.task_id)
                if event.kind == "projection":
                    self._on_projection_failed_for_detection(message)
                elif event.kind == "affine_registration":
                    self._on_affine_registration_failed(message)
                elif event.kind == "process_intensities":
                    self.on_processing_error(message)
                elif event.kind == "batch":
                    self._on_batch_failed(message)
                continue

            if event.event_type != "result":
                continue

            payload = event.payload.get("payload")
            self._clear_process_task(event.kind, event.task_id)
            if event.kind == "projection":
                self._on_projection_ready_for_detection(payload)
            elif event.kind == "affine_registration":
                model = RegistrationModel.from_dict(payload) if isinstance(payload, dict) else payload
                self._on_affine_registration_finished(model)
            elif event.kind == "process_intensities":
                self.on_processing_complete(payload)
            elif event.kind == "batch":
                self._on_batch_finished(str(payload))

        if not any(self._active_task_ids.values()) and not self.task_service.active:
            self.task_poll_timer.stop()

    def _normalize_detect_channel(self, detect_channel):
        if not self.channel_names:
            return 0
        try:
            idx = int(detect_channel)
        except (TypeError, ValueError):
            idx = 0
        return max(0, min(idx, len(self.channel_names) - 1))

    def _normalize_frame_index(self, frame_index):
        if self.num_frames <= 0:
            return 0
        try:
            idx = int(frame_index)
        except (TypeError, ValueError):
            idx = 0
        return max(0, min(idx, self.num_frames - 1))

    def _apply_exposure_settings(
        self,
        *,
        override_enabled: bool,
        override_ms: float | None,
        metadata_exposure_ms: float | None = None,
        metadata_exposure_source: str | None = None,
    ) -> None:
        if metadata_exposure_ms is not None:
            self.metadata_exposure_ms = float(metadata_exposure_ms)
        if metadata_exposure_source is not None:
            self.metadata_exposure_source = str(metadata_exposure_source or "")

        if override_enabled:
            if override_ms is None:
                raise ValueError("Manual exposure override is enabled but no exposure value was provided")
            override_ms = float(override_ms)
            if not np.isfinite(override_ms) or override_ms <= 0:
                raise ValueError("Manual exposure override must be finite and > 0 ms")
            self.exposure_override_enabled = True
            self.exposure_override_ms = override_ms
            self.exposure_s = override_ms / 1000.0
            self.exposure_source = "manual_override"
        else:
            if self.metadata_exposure_ms is None:
                raise ValueError("ND2 metadata exposure is unavailable")
            self.exposure_override_enabled = False
            self.exposure_override_ms = None
            self.exposure_s = float(self.metadata_exposure_ms) / 1000.0
            self.exposure_source = self.metadata_exposure_source or "ND2 metadata"

    def _refresh_exposure_ui(self) -> None:
        self.window_controller.apply_basic_labels(self)
        if self.selected_molecule in self.molecule_intensities:
            self.select_molecule(self.selected_molecule)

    def edit_exposure_override(self):
        if self.num_frames <= 0 or self.nd2_file is None:
            QMessageBox.information(self, "Exposure Time", "Load an ND2 file before setting an exposure override.")
            return
        dlg = ExposureOverrideDialog(
            self,
            metadata_exposure_ms=self.metadata_exposure_ms,
            metadata_exposure_source=self.metadata_exposure_source,
            override_enabled=self.exposure_override_enabled,
            override_ms=self.exposure_override_ms,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            enabled, override_ms = dlg.get_override()
            self._apply_exposure_settings(
                override_enabled=enabled,
                override_ms=override_ms,
            )
            self.operation_log.append(
                {
                    "event": "exposure_override_changed",
                    "enabled": bool(self.exposure_override_enabled),
                    "effective_exposure_ms": float(self.exposure_s * 1000.0),
                    "metadata_exposure_ms": self.metadata_exposure_ms,
                    "metadata_exposure_source": self.metadata_exposure_source,
                }
            )
            self._refresh_exposure_ui()
            source = "manual override" if self.exposure_override_enabled else (self.metadata_exposure_source or "ND2 metadata")
            QMessageBox.information(
                self,
                "Exposure Time",
                f"Effective exposure is now {self.exposure_s * 1000.0:.4f} ms ({source}).",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Exposure Time", f"Failed to apply exposure override:\n{exc}")

    def _cleanup_video_widgets(self):
        for widget in self.video_widgets:
            try:
                self.sync_manager.unregister(widget)
            except Exception:
                pass
        while self.video_layout.count():
            child = self.video_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                try:
                    self.sync_manager.unregister(w)
                except Exception:
                    pass
                w.deleteLater()
        self.video_widgets = []

    def _cleanup_current_data(self, *, preserve_registration: bool = False):
        preserved_registration = (
            self.registration_params if preserve_registration else None
        )
        try:
            self.is_playing = False
            self.play_timer.stop()
        except Exception:
            logger.exception("Failed to stop playback during cleanup")

        try:
            self.scale_reprocess_timer.stop()
        except Exception:
            pass
        self._scale_reprocess_pending = False

        self._cancel_all_active_tasks(recreate_runtime=True)

        self._cleanup_video_widgets()
        self.sync_manager.clear()

        try:
            if self.multi_cache:
                self.multi_cache.shutdown()
        except Exception:
            logger.exception("Failed to shut down cache during cleanup")
        self.multi_cache = None

        try:
            if self.file_reader:
                self.file_reader.close()
        except Exception:
            logger.exception("Failed to close ND2 reader during cleanup")
        self.file_reader = None

        self.window_controller.state.reset_data()
        self.nd2_file = None
        self.nd2_fingerprint = {}
        self.registration_params = preserved_registration
        self.num_channels = 0
        self.num_frames = 0
        self.exposure_s = 0.1
        self.exposure_source = "ND2 metadata"
        self.metadata_exposure_ms = None
        self.metadata_exposure_source = ""
        self.exposure_override_enabled = False
        self.exposure_override_ms = None
        self.channel_names = []
        self.molecules = []
        self.base_molecules = []
        self.molecule_features = {}
        self.base_molecule_features = {}
        self.deleted_molecules = set()
        self.molecule_intensities = {}
        self.current_frame = 0
        self.selected_molecule = None
        self.detect_channel = 0
        self.intensity_scales = {}
        self.analysis_recipe = {}
        self.operation_log = []
        self.uncertainty_filter = {"enabled": False, "max_scalar": 0.5}
        self.active_edit_tool = "pan"
        self.view_transform = {}
        self.results_mode = "normal"
        self.drift_result = {}
        self._sync_drift_runtime_state()

        self.lut_panels = []
        self.intensity_scale_spinboxes = {}

        self.molecule_list.clear()
        self.plot_widget.clear_plot()
        self._build_right_panel_placeholder()
        if self.registration_params is not None:
            self.window_controller.apply_registration_label(self)
        else:
            self.reg_info_label.setText("No registration loaded")

        self._reset_progress_ui()

        self.detect_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        if hasattr(self, "exposure_override_btn"):
            self.exposure_override_btn.setEnabled(False)
        if hasattr(self, "exposure_override_action"):
            self.exposure_override_action.setEnabled(False)

        self.window_controller.restore_channel_combo(
            self,
            fallback_channels=FIXED_CHANNEL_ORDER,
        )

        if hasattr(self, "frame_label"):
            self.frame_label.setText("0/0")
        if hasattr(self, "time_slider"):
            self.time_slider.setRange(0, 0)
            self.time_slider.setValue(0)
        if hasattr(self, "speed_combo"):
            self.speed_combo.setCurrentText("1x")
        self.set_edit_tool("pan")
        self.window_controller.apply_basic_labels(self)

    def build_right_controls(self):
        self._clear_layout(self.right_panel_layout)
        self.lut_panels = []
        self.intensity_scale_spinboxes = {}

        play_group = QGroupBox("Playback")
        pl = QGridLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(self.multi_cache is not None)
        pl.addWidget(self.play_btn, 0, 0)

        pl.addWidget(QLabel("Speed:"), 0, 1)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x", "5x", "10x"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentTextChanged.connect(self.change_play_speed)
        self.speed_combo.setEnabled(self.multi_cache is not None)
        pl.addWidget(self.speed_combo, 0, 2)

        self.frame_label = QLabel("0/0")
        pl.addWidget(self.frame_label, 0, 3)

        pl.addWidget(QLabel("Time:"), 1, 0)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(self.multi_cache is not None)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        pl.addWidget(self.time_slider, 1, 1, 1, 3)

        play_group.setLayout(pl)
        self.right_panel_layout.addWidget(play_group)

        lut_group = QGroupBox("LUT Controls (Per Channel)")
        lut_layout = QVBoxLayout()
        for ch_name in self.channel_names:
            panel = LUTControlPanel(ch_name)
            self.lut_panels.append(panel)
            lut_layout.addWidget(panel)
        lut_group.setLayout(lut_layout)
        self.right_panel_layout.addWidget(lut_group)

        scale_group = QGroupBox("Intensity Scale")
        scale_layout = QGridLayout()

        default_scales = self._default_intensity_scales_for_channels(
            self.channel_names
        )
        for ch_name, value in dict(self.intensity_scales or {}).items():
            if ch_name in default_scales:
                default_scales[ch_name] = float(value)
        self.intensity_scales = default_scales

        for row, ch_name in enumerate(self.channel_names):
            scale_layout.addWidget(QLabel(f"{ch_name}:"), row, 0)

            spin = QDoubleSpinBox()
            spin.setRange(0.01, 10.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setValue(
                float(self.intensity_scales.get(ch_name, 1.0))
            )
            spin.valueChanged.connect(
                partial(self.on_intensity_scale_changed, ch_name)
            )
            scale_layout.addWidget(spin, row, 1)

            hint = QLabel("Intensity will be multiplied by this factor")
            hint.setStyleSheet("QLabel { color: #666; }")
            scale_layout.addWidget(hint, row, 2)

            self.intensity_scale_spinboxes[ch_name] = spin

        scale_group.setLayout(scale_layout)
        self.right_panel_layout.addWidget(scale_group)

        filter_group = QGroupBox("Localization Uncertainty Filter")
        filter_layout = QGridLayout()
        self.uncertainty_filter_check = QCheckBox("Enable uncertainty filter")
        self.uncertainty_filter_check.setChecked(bool(self.uncertainty_filter.get("enabled", False)))
        self.uncertainty_filter_check.stateChanged.connect(self.on_uncertainty_filter_changed)
        filter_layout.addWidget(self.uncertainty_filter_check, 0, 0, 1, 2)
        filter_layout.addWidget(QLabel("Max scalar uncertainty (px):"), 1, 0)
        self.uncertainty_filter_spin = QDoubleSpinBox()
        self.uncertainty_filter_spin.setRange(0.01, 10.0)
        self.uncertainty_filter_spin.setDecimals(3)
        self.uncertainty_filter_spin.setSingleStep(0.05)
        self.uncertainty_filter_spin.setValue(float(self.uncertainty_filter.get("max_scalar", 0.5) or 0.5))
        self.uncertainty_filter_spin.valueChanged.connect(self.on_uncertainty_filter_changed)
        filter_layout.addWidget(self.uncertainty_filter_spin, 1, 1)
        filter_group.setLayout(filter_layout)
        self.right_panel_layout.addWidget(filter_group)
        self.right_panel_layout.addStretch()
        self._sync_playback_controls()

    def load_registration(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Registration Parameters",
            _dialog_dir("registration"),
            "JSON Files (*.json)",
        )
        if filename:
            try:
                _remember_dialog_path(filename, "registration")
                model = RegistrationIO.load_json(filename)
                self.registration_params = model.to_dict()
                self.window_controller.apply_registration_label(self)

                for vw in self.video_widgets:
                    vw.set_registration_params(self.registration_params)

                self.update_video_display()
                QMessageBox.information(self, "Success", "Registration parameters loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Load failed: {str(e)}")

    def estimate_affine_registration_from_beads(self):
        bead_file, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Bead Calibration ND2 File",
            _dialog_dir("nd2"),
            "ND2 Files (*.nd2)",
        )
        if not bead_file:
            return
        _remember_dialog_path(bead_file, "nd2")

        try:
            if self._active_task_ids.get("affine_registration"):
                QMessageBox.information(
                    self, "Notice", "A registration estimation task is already running"
                )
                return

            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Estimating affine registration in the background...")
            QApplication.processEvents()

            self._start_process_task(
                "affine_registration",
                bead_file=bead_file,
                cache_budget_mb=DEFAULT_CACHE_BUDGET_MB,
            )

        except Exception as e:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Affine registration estimation failed")
            QMessageBox.critical(self, "Error", f"Estimation failed: {e}")

    @pyqtSlot(object)
    def _on_affine_registration_finished(self, model):
        try:
            self.registration_params = model.to_dict()
            self.window_controller.apply_registration_label(self)

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Affine Registration JSON",
                _default_dialog_path("registration", "affine_registration.json"),
                "JSON Files (*.json)",
            )
            if save_path:
                RegistrationIO.save_json(save_path, model)
                _remember_dialog_path(save_path, "registration")

            for vw in self.video_widgets:
                vw.set_registration_params(self.registration_params)

            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_label.setText("Affine registration estimation complete")
            self.update_video_display()
            QMessageBox.information(
                self, "Success", "Affine registration has been estimated and applied."
            )
        except Exception as exc:
            logger.exception("Failed to finalize affine registration")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Affine registration finalization failed")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to finalize affine registration: {exc}",
            )

    @pyqtSlot(str)
    def _on_affine_registration_failed(self, msg):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Affine registration estimation failed")
        QMessageBox.critical(self, "Error", f"Estimation failed: {msg}")

    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load ND2 Data File",
            _dialog_dir("nd2"),
            "ND2 Files (*.nd2)",
        )
        if not filename:
            return
        _remember_dialog_path(filename, "nd2")

        try:
            self._cleanup_current_data(preserve_registration=True)
            self.nd2_file = filename
            self.nd2_fingerprint = compute_file_fingerprint(filename)
            self.results_mode = "normal"

            self.progress_bar.setRange(0, 0)
            self.progress_label.setText("Loading ND2 file...")
            QApplication.processEvents()

            self.file_reader = ND2Reader(filename)
            self.multi_cache = MultiLevelCache(
                self.file_reader,
                cache_budget_mb=DEFAULT_CACHE_BUDGET_MB,
            )

            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_label.setText("Load complete")

            self.num_channels = self.multi_cache.metadata["num_channels"]
            self.num_frames = self.multi_cache.metadata["num_frames"]
            self._apply_exposure_settings(
                override_enabled=False,
                override_ms=None,
                metadata_exposure_ms=self.multi_cache.metadata.get(
                    "exposure_ms",
                    self.multi_cache.metadata.get("time_ms", 100.0),
                ),
                metadata_exposure_source=self.multi_cache.metadata.get("exposure_source", "ND2 metadata"),
            )
            self.channel_names = self.multi_cache.metadata["channel_names"]

            self.window_controller.apply_basic_labels(self)
            self.window_controller.restore_channel_combo(
                self,
                fallback_channels=FIXED_CHANNEL_ORDER,
            )

            self.build_right_controls()

            self.current_frame = 0
            self.time_slider.setRange(0, self.num_frames - 1)
            self.time_slider.setValue(0)

            self.create_video_widgets()

            self.update_video_display()
            self.run_auto_lut_all_channels()

            self.detect_btn.setEnabled(True)
            self.process_btn.setEnabled(False)
            self.exposure_override_btn.setEnabled(True)
            if hasattr(self, "exposure_override_action"):
                self.exposure_override_action.setEnabled(True)

            QMessageBox.information(
                self,
                "Success",
                f"ND2 data loaded successfully.\nExposure time: {self.exposure_s * 1000.0:.4f} ms ({self.exposure_source})\n"
                f"Channels: {', '.join(self.channel_names)}",
            )

        except Exception as e:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Load failed")
            QMessageBox.critical(self, "Error", f"Load failed: {str(e)}")
            logger.exception("Failed to load ND2 data")

    def create_video_widgets(self):
        self._cleanup_video_widgets()

        if len(self.lut_panels) != len(self.channel_names):
            self.lut_panels = [
                LUTControlPanel(ch) for ch in self.channel_names
            ]

        for i, channel_name in enumerate(self.channel_names):
            lut_panel = self.lut_panels[i]

            video_widget = VideoWidget(
                channel_name,
                i,
                self.multi_cache,
                self.sync_manager,
                lut_panel=lut_panel,
                registration_params=self.registration_params,
            )
            video_widget.set_edit_tool(self.active_edit_tool)
            self.video_widgets.append(video_widget)
            self.video_layout.addWidget(video_widget, stretch=1)

            lut_panel.settings_changed.connect(video_widget.on_lut_changed)
            lut_panel.auto_requested.connect(
                partial(self.auto_lut_range, i)
            )
            video_widget.clicked.connect(
                self.on_video_clicked_select_molecule
            )
            video_widget.add_molecule_requested.connect(self.on_video_add_molecule)
            video_widget.delete_molecule_requested.connect(self.on_video_delete_molecule)
            video_widget.move_molecule_requested.connect(self.on_video_move_molecule)

    def on_video_clicked_select_molecule(self, world_x: int, world_y: int):
        if self.active_edit_tool not in {"pan", "zoom"}:
            return
        state = self.window_controller.capture_window_state(self)
        row = self.window_controller.nearest_active_row_for_point(
            state,
            world_x,
            world_y,
            radius=8.0,
        )
        if row is None:
            return
        self.molecule_list.setCurrentRow(row)

    def _detection_reference_image(self) -> np.ndarray | None:
        if self.multi_cache is None:
            return None
        try:
            return max_projection_first_n_frames(
                self.multi_cache,
                int(self.detect_channel),
                n=10,
            ).astype(np.float32)
        except Exception:
            logger.exception("Failed to build detection reference image for manual edit")
            return None

    def on_video_add_molecule(self, world_x: float, world_y: float):
        if self.results_mode != "normal":
            return
        reference = self._detection_reference_image()
        if reference is None:
            return
        from detection import MoleculeDetector

        record = MoleculeDetector().refine_manual_point(reference, world_x, world_y)
        if record is None:
            QMessageBox.warning(self, "Notice", "Unable to place a molecule at this location.")
            return
        mol_id = self.window_controller.add_manual_molecule(self, record)
        self.molecule_intensities = {}
        self.process_btn.setEnabled(True)
        self.update_video_display()
        self.window_controller.restore_molecule_list(self, self.window_controller.capture_window_state(self))
        self.select_molecule(mol_id)

    def on_video_delete_molecule(self, world_x: float, world_y: float):
        state = self.window_controller.capture_window_state(self)
        row = self.window_controller.nearest_active_row_for_point(state, world_x, world_y, radius=10.0)
        if row is None:
            return
        next_row, selection_cleared = self.window_controller.delete_active_molecule_at_row(self, row)
        if selection_cleared:
            self.plot_widget.clear_plot()
        if next_row is not None:
            self.molecule_list.setCurrentRow(next_row)
        self.update_video_display()
        self.window_controller.apply_basic_labels(self)

    def on_video_move_molecule(self, mol_id: int, world_x: float, world_y: float):
        if self.results_mode != "normal":
            return
        reference = self._detection_reference_image()
        if reference is None:
            return
        from detection import MoleculeDetector

        record = MoleculeDetector().refine_manual_point(reference, world_x, world_y)
        if record is None:
            self.update_video_display()
            return
        self.window_controller.move_molecule(self, int(mol_id), record)
        self.molecule_intensities = {}
        self.process_btn.setEnabled(True)
        self.update_video_display()
        self.window_controller.restore_molecule_list(self, self.window_controller.capture_window_state(self), fallback_to_first=False)
        self.select_molecule(int(mol_id))

    def auto_lut_range(self, channel_idx: int):
        if self.multi_cache is None:
            return
        if channel_idx < 0 or channel_idx >= len(self.channel_names):
            return
        if channel_idx >= len(self.lut_panels):
            return

        raw_frame = self.multi_cache.get_raw_frame(self.current_frame)
        if raw_frame is None:
            return
        img = raw_frame[channel_idx]

        min_val = float(np.percentile(img, 0.1))
        max_val = float(np.percentile(img, 99.9))
        if max_val <= min_val:
            max_val = min_val + 1.0

        self.lut_panels[channel_idx].set_auto_range(min_val, max_val)

    def run_auto_lut_all_channels(self):
        for i in range(len(self.channel_names)):
            self.auto_lut_range(i)
        self.update_video_display()

    def detect_molecules_interactive(self):
        if self.multi_cache is None:
            return

        if self._active_task_ids.get("projection"):
            QMessageBox.information(
                self, "Notice", "Projection is already being computed. Please wait..."
            )
            return

        try:
            channel_idx = self.channel_combo.currentIndex()
            self.detect_channel = channel_idx

            current_detect_name = self.channel_combo.currentText()
            if physical_channel_name(current_detect_name) != REFERENCE_CHANNEL:
                QMessageBox.information(
                    self,
                    "Notice",
                    f"The reference channel is fixed to {REFERENCE_CHANNEL}.\n"
                    f"It is recommended to use {REFERENCE_CHANNEL} as the detection channel.\n"
                    f"If {current_detect_name} is used, coordinates will be converted automatically.",
                )

            self.detect_btn.setEnabled(False)
            self.process_btn.setEnabled(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText(
                "Computing the max projection of the first 10 frames in the background..."
            )

            self._start_process_task(
                "projection",
                nd2_path=self.nd2_file,
                channel_idx=channel_idx,
                n=10,
                cache_budget_mb=DEFAULT_CACHE_BUDGET_MB,
            )

        except Exception as e:
            self.detect_btn.setEnabled(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("")
            QMessageBox.critical(
                self, "Error", f"Failed to prepare detection: {str(e)}"
            )

    @pyqtSlot(np.ndarray)
    def _on_projection_ready_for_detection(self, proj_img: np.ndarray):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("")
        self.detect_btn.setEnabled(True)

        try:
            dialog = DetectionPreviewDialog(proj_img, self)
            if dialog.exec_() == QDialog.Accepted:
                detected_results = dialog.get_detected_results()
                if not detected_results:
                    self.process_btn.setEnabled(bool(self.molecules))
                    QMessageBox.warning(
                        self,
                        "Detection Failed",
                        "No molecules were detected in the first 10 frames.\n"
                        "Single-file detection is fixed to the first 10 frames; "
                        "later windows are not searched automatically.",
                    )
                    return
                self.window_controller.apply_detection_results(
                    self,
                    detected_results,
                    params=dialog.get_params(),
                )
                self.update_video_display()
                self.window_controller.apply_basic_labels(self)
                self.process_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            logger.exception("Detection preview failed")

    @pyqtSlot(str)
    def _on_projection_failed_for_detection(self, msg: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("")
        self.detect_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Projection computation failed:\n{msg}")

    def process_all_frames(self):
        if not self.molecules or self.multi_cache is None:
            return

        try:
            validate_quantitative_registration(
                self.registration_params,
                self.channel_names,
                detect_channel=self.detect_channel,
            )
        except RegistrationValidationError as exc:
            QMessageBox.critical(
                self,
                "Registration Required",
                f"Cannot extract quantitative intensities safely:\n{exc}",
            )
            return

        intensity_scales = (
            dict(self.intensity_scales) if self.intensity_scales else {}
        )

        self.process_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting intensity extraction...")

        self._start_process_task(
            "process_intensities",
            nd2_path=self.nd2_file,
            molecules=self.molecules,
            registration_params=self.registration_params,
            detect_channel=self.detect_channel,
            channel_names=self.channel_names,
            intensity_scales=intensity_scales,
            cache_budget_mb=DEFAULT_CACHE_BUDGET_MB,
        )

    @pyqtSlot(int, str)
    def on_processing_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)

    def _apply_strict_single_file_roi_qc(self, intensities: dict) -> list[dict]:
        complete_molecules, complete_intensities, dropped_rois = (
            _strict_filter_complete_molecules(
                self.molecules,
                intensities,
                self.channel_names,
                expected_frames=int(self.num_frames),
            )
        )
        if not dropped_rois:
            self.molecule_intensities = dict(complete_intensities)
            return []

        kept_ids = {int(mol_id) for mol_id, _x, _y in complete_molecules}
        self.molecules = list(complete_molecules)
        self.base_molecules = [
            (int(mol_id), float(x), float(y))
            for mol_id, x, y in list(self.base_molecules or complete_molecules)
            if int(mol_id) in kept_ids
        ]
        if not self.base_molecules:
            self.base_molecules = list(complete_molecules)
        self.molecule_features = {
            int(mol_id): dict(feature or {})
            for mol_id, feature in dict(self.molecule_features or {}).items()
            if int(mol_id) in kept_ids
        }
        self.base_molecule_features = {
            int(mol_id): dict(feature or {})
            for mol_id, feature in dict(self.base_molecule_features or {}).items()
            if int(mol_id) in kept_ids
        }
        self.deleted_molecules = {
            int(mol_id) for mol_id in set(self.deleted_molecules or set())
            if int(mol_id) in kept_ids
        }
        self.molecule_intensities = dict(complete_intensities)
        if self.selected_molecule not in kept_ids:
            self.selected_molecule = int(complete_molecules[0][0]) if complete_molecules else None

        state = self.window_controller.capture_window_state(self)
        self.window_controller.restore_molecule_list(self, state, fallback_to_first=bool(kept_ids))
        self.window_controller.apply_basic_labels(self)
        self.update_video_display()
        return dropped_rois

    @pyqtSlot(dict)
    def on_processing_complete(self, molecule_intensities):
        payload = dict(molecule_intensities or {})
        if "molecule_intensities" in payload:
            raw_intensities = dict(payload.get("molecule_intensities", {}) or {})
            self.drift_result = dict(payload.get("drift_result", {}) or {})
        else:
            raw_intensities = payload
        dropped_rois = self._apply_strict_single_file_roi_qc(raw_intensities)
        self._sync_drift_runtime_state()
        self.progress_bar.setValue(100)

        if not self.molecule_intensities:
            self.progress_label.setText("Processing failed: no complete ROI traces")
            self.process_btn.setEnabled(False)
            self.detect_btn.setEnabled(True)
            self.plot_widget.clear_plot()
            self._scale_reprocess_pending = False
            QMessageBox.critical(
                self,
                "Processing Failed",
                "All detected ROIs were removed by strict NaN/Inf quality control.\n"
                "No CSV can be exported for this ND2 file.",
            )
            return

        self.progress_label.setText("Processing complete")
        self.process_btn.setEnabled(bool(self.molecules))
        self.detect_btn.setEnabled(True)

        if self.selected_molecule is not None:
            self.select_molecule(self.selected_molecule)

        if self._scale_reprocess_pending:
            self.progress_label.setText("Intensity Scale changed. Re-extracting intensities...")
            self.scale_reprocess_timer.start()
            return

        qc_note = ""
        if dropped_rois:
            qc_note = (
                f"\n\nStrict ROI QC removed {len(dropped_rois)} incomplete ROI(s):\n"
                f"{_format_drop_summary(dropped_rois)}"
            )
        QMessageBox.information(
            self,
            "Success",
            "Processing complete.\n"
            "You can review the drift correction result via Analysis -> View Drift Trajectory."
            f"{qc_note}",
        )

    @pyqtSlot(str)
    def on_processing_error(self, error_msg):
        self.progress_bar.setValue(0)
        self.progress_label.setText("Processing failed")
        self.process_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        if self._scale_reprocess_pending:
            self.scale_reprocess_timer.start()
        QMessageBox.critical(self, "Error", f"Processing failed: {error_msg}")

    def update_video_display(self):
        # Null guard
        if self.multi_cache is None:
            self.window_controller.apply_basic_labels(self)
            return
        if not self.video_widgets:
            self.window_controller.apply_basic_labels(self)
            return

        self.multi_cache.trigger_preload(self.current_frame)

        state = self.window_controller.capture_window_state(self)
        molecules_display = self.window_controller.active_molecules(state)

        for video_widget in self.video_widgets:
            try:
                video_widget.set_frame(self.current_frame)
                video_widget.set_molecules(molecules_display)
                video_widget.set_selected_molecule(self.selected_molecule)
                video_widget.set_edit_tool(self.active_edit_tool)
            except RuntimeError:
                pass

        self.plot_widget.update_frame_marker(self.current_frame)
        self.window_controller.apply_basic_labels(self)

    def on_time_slider_changed(self, value):
        if self.is_playing:
            return
        self.current_frame = int(value)
        self.update_video_display()

    def toggle_play(self):
        if self.multi_cache is None:
            return

        if self.is_playing:
            self.is_playing = False
            self.play_btn.setText("Play")
            self.play_timer.stop()
        else:
            self.is_playing = True
            self.play_btn.setText("Pause")
            interval = int(self.exposure_s * 1000 / self.play_speed)
            interval = max(1, interval)
            self.play_timer.start(interval)

    def change_play_speed(self, speed_text):
        self.play_speed = float(speed_text.replace("x", ""))
        if self.is_playing:
            interval = int(self.exposure_s * 1000 / self.play_speed)
            self.play_timer.setInterval(max(1, interval))

    def next_frame(self):
        if self.multi_cache is None:
            self.is_playing = False
            self.play_timer.stop()
            return
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            if (hasattr(self, "time_slider")
                    and self.time_slider is not None):
                self.time_slider.setValue(self.current_frame)
            self.update_video_display()
        else:
            self.is_playing = False
            self.play_btn.setText("Play")
            self.play_timer.stop()

    def on_molecule_selected(self, item):
        row = self.molecule_list.row(item)
        self._select_molecule_by_active_row(row)

    def on_molecule_row_changed(self, row: int):
        self._select_molecule_by_active_row(row)

    def _select_molecule_by_active_row(self, row: int):
        active_molecules = self.window_controller.active_molecules(
            self.window_controller.capture_window_state(self)
        )
        if 0 <= row < len(active_molecules):
            mol_id = active_molecules[row][0]
            self.select_molecule(mol_id)

    def select_molecule(self, mol_id):
        self.selected_molecule = mol_id
        self.update_video_display()

        if mol_id in self.molecule_intensities:
            time_arr = np.arange(self.num_frames) * self.exposure_s
            intensities = self.molecule_intensities[mol_id]
            self.plot_widget.plot_molecule(
                time_arr, intensities, self.channel_names, mol_id
            )
        else:
            self.plot_widget.clear_plot()

    def show_molecule_context_menu(self, position):
        item = self.molecule_list.itemAt(position)
        if item:
            menu = QMenu()
            delete_action = menu.addAction("Delete Molecule")

            action = menu.exec_(
                self.molecule_list.mapToGlobal(position)
            )
            if action == delete_action:
                row = self.molecule_list.row(item)
                next_row, selection_cleared = (
                    self.window_controller.delete_active_molecule_at_row(
                        self,
                        row,
                    )
                )
                if next_row is not None or selection_cleared:
                    if selection_cleared:
                        self.plot_widget.clear_plot()
                    self.update_video_display()
                    self.window_controller.apply_basic_labels(self)

    def on_intensity_scale_changed(self, channel_name: str, value: float):
        self.intensity_scales[channel_name] = float(value)

        if (self.molecule_intensities and self.molecules
                and self.multi_cache is not None):
            if self._is_task_active("process_intensities"):
                self._scale_reprocess_pending = True
                self.progress_label.setText(
                    "Intensity Scale changed. "
                    "Re-extraction will start after the current processing task finishes..."
                )
                return

            self._scale_reprocess_pending = True
            self.progress_label.setText(
                "Intensity Scale changed. Preparing to re-extract intensities..."
            )
            self.scale_reprocess_timer.start()

    def on_uncertainty_filter_changed(self, *_args):
        enabled = bool(getattr(self, "uncertainty_filter_check", None) and self.uncertainty_filter_check.isChecked())
        max_scalar = float(getattr(self, "uncertainty_filter_spin", None).value() if getattr(self, "uncertainty_filter_spin", None) is not None else 0.5)
        self.uncertainty_filter = {"enabled": enabled, "max_scalar": max_scalar}
        state = self.window_controller.capture_window_state(self)
        self.window_controller.restore_molecule_list(self, state, fallback_to_first=False)
        self.update_video_display()

    def _reprocess_after_scale_change(self):
        if not self._scale_reprocess_pending:
            return

        if self._is_task_active("process_intensities"):
            self.scale_reprocess_timer.start()
            return

        self._scale_reprocess_pending = False
        self.process_all_frames()

    # ---- Drift plot visualization ----

    def show_drift_plot(self):
        self._sync_drift_runtime_state()
        fig = DriftCalculator.generate_drift_figure()
        if fig is None:
            QMessageBox.information(
                self,
                "Notice",
                "No drift data is available yet.\nFinish processing first, then view the drift trajectory.",
            )
            return

        dialog = DriftPlotDialog(fig, self)
        dialog.exec_()

    # ---- Project save/load ----

    def save_project_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            _default_dialog_path("project", "project.smproj"),
            "Project Files (*.smproj)"
        )
        if not path:
            return
        try:
            save_project_bundle(
                path,
                **self.window_controller.project_save_kwargs(self),
            )
            _remember_dialog_path(path, "project")
            QMessageBox.information(self, "Success", f"Project saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def load_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            _dialog_dir("project"),
            "Project Files (*.smproj)"
        )
        if not path:
            return
        _remember_dialog_path(path, "project")
        try:
            manifest = load_project_bundle(path)
            self._cleanup_current_data()
            state = self.window_controller.apply_project_manifest(manifest)

            nd2_file = state.nd2_file or ""
            nd2_exists = bool(nd2_file and os.path.exists(nd2_file))
            nd2_matches = bool(nd2_exists and fingerprint_matches(state.nd2_fingerprint, nd2_file))
            if nd2_matches:
                self.file_reader = ND2Reader(nd2_file)
                self.multi_cache = MultiLevelCache(
                    self.file_reader,
                    cache_budget_mb=DEFAULT_CACHE_BUDGET_MB,
                )
                self.window_controller.sync_state_from_runtime_sources(
                    state,
                    file_reader=self.file_reader,
                    multi_cache=self.multi_cache,
                )
                state.results_mode = "normal"
            elif nd2_file:
                state.results_mode = "results-only"
                if nd2_exists and not nd2_matches:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"ND2 fingerprint mismatch:\n{nd2_file}\nThe project results will be loaded in results-only mode.",
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"ND2 file not found:\n{nd2_file}\nThe project results will be loaded without source data.",
                    )

            self.window_controller.apply_project_state_to_window(self, state)
            self._sync_drift_runtime_state()
            self.detect_channel = self._normalize_detect_channel(state.detect_channel)
            self.current_frame = self._normalize_frame_index(state.current_frame)
            self.window_controller.apply_registration_label(self)
            self.window_controller.restore_channel_combo(
                self,
                fallback_channels=FIXED_CHANNEL_ORDER,
            )

            if self.multi_cache is not None:
                self.build_right_controls()
                self.time_slider.setRange(0, max(0, self.num_frames - 1))
                self.time_slider.setValue(self.current_frame)

                restored_lut = self.window_controller.restore_lut_panels(self, state)
                self.window_controller.restore_intensity_scale_controls(self, state)

                self.create_video_widgets()
                if state.view_transform and self.video_widgets:
                    self.video_widgets[0].apply_view_state(state.view_transform)
                    self.sync_manager.sync_transform(self.video_widgets[0])
                if restored_lut:
                    self.update_video_display()
                else:
                    self.run_auto_lut_all_channels()
                self._sync_playback_controls()
                self.detect_btn.setEnabled(self.results_mode == "normal")
                self.process_btn.setEnabled(bool(self.molecules) and self.results_mode == "normal")
            else:
                self.build_right_controls()
                self.time_slider.setRange(0, max(0, self.num_frames - 1))
                self.time_slider.setValue(self.current_frame)
                self.window_controller.restore_lut_panels(self, state)
                self.window_controller.restore_intensity_scale_controls(self, state)
                self.detect_btn.setEnabled(False)
                self.process_btn.setEnabled(False)

            self.set_edit_tool(state.active_edit_tool)
            self.window_controller.restore_molecule_list(self, state)
            self.window_controller.apply_basic_labels(self)
            exposure_controls_enabled = bool(self.num_frames > 0)
            if hasattr(self, "exposure_override_btn"):
                self.exposure_override_btn.setEnabled(exposure_controls_enabled)
            if hasattr(self, "exposure_override_action"):
                self.exposure_override_action.setEnabled(exposure_controls_enabled)
            QMessageBox.information(self, "Success", "Project loaded")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {e}")
            logger.exception("Failed to load project")

    # ---- Export ----

    def export_csv(self):
        if not self.molecule_intensities:
            QMessageBox.warning(self, "Warning", "There is no data to export.")
            return

        dropped_rois = self._apply_strict_single_file_roi_qc(self.molecule_intensities)
        if not self.molecule_intensities:
            QMessageBox.warning(
                self,
                "Warning",
                "All ROIs were removed by strict NaN/Inf quality control; "
                "there is no complete data to export.",
            )
            return

        default_name = (
            os.path.splitext(self.nd2_file)[0] + "_intensities.csv"
            if self.nd2_file else "intensities.csv"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            default_name,
            "CSV Files (*.csv)",
        )
        if not filename:
            return

        try:
            time_arr = np.arange(self.num_frames) * self.exposure_s
            written = write_intensities_csv_split(
                filename,
                molecules=self.molecules,
                deleted_molecules=self.deleted_molecules,
                molecule_intensities=self.molecule_intensities,
                time_arr=time_arr,
                channel_names=self.channel_names,
                max_rows=1_000_000,
            )
            QMessageBox.information(
                self, "Success", "Data exported to:\n" + "\n".join(written)
            )
            if dropped_rois:
                QMessageBox.warning(
                    self,
                    "ROI QC",
                    f"Strict ROI QC removed {len(dropped_rois)} incomplete ROI(s) before export:\n"
                    f"{_format_drop_summary(dropped_rois)}",
                )
            _remember_dialog_path(filename, "export")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
            logger.exception("Failed to export CSV")

    def batch_detection_mode(self):
        dlg = BatchDetectionDialog(
            self, initial_intensity_scales=self.intensity_scales
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        try:
            cfg = dlg.get_config()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid batch configuration:\n{e}")
            return

        if self._active_task_ids.get("batch"):
            QMessageBox.warning(self, "Notice", "A batch task is already running")
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting batch processing...")

        self._start_process_task("batch", config=cfg)

    @pyqtSlot(int, str)
    def _on_batch_progress(self, p, msg):
        self.progress_bar.setValue(int(p))
        self.progress_label.setText(msg)

    @pyqtSlot(str)
    def _on_batch_finished(self, msg):
        self.progress_bar.setValue(100)
        self.progress_label.setText(msg)
        QMessageBox.information(self, "Batch Complete", msg)

    @pyqtSlot(str)
    def _on_batch_failed(self, msg):
        self.progress_bar.setValue(0)
        self.progress_label.setText("Batch failed")
        QMessageBox.critical(self, "Batch Failed", msg)

    def show_about(self):
        about_text = """
Single-Molecule Fluorescence Intensity Extraction Tool

Version: 13.0 - ND2 Edition
Core updates:
- Drift estimation now uses fiducial tracking
  - Selects the brightest, most isolated, most persistent molecules as fiducials
  - Tracks subpixel centroids frame by frame (~0.05 px precision)
  - Uses weighted median aggregation for robust bleaching-resistant estimates
  - Measures drift directly relative to frame 0 (zero accumulation error)
  - Reports automatic quality metrics (valid fiducial count + MAD)
- Savitzky-Golay low-pass smoothing
- Automatic interpolation for low-quality frames
- Drift trajectory visualization (Analysis -> View Drift Trajectory)
- Subpixel intensity extraction (cv2.getRectSubPix)
- NaN marking for out-of-bounds molecules
- Automatic coordinate conversion for non-532 detection channels
- Affine registration (reference channel fixed to 532)
- CSV export preserves NaN values
- Project save/load (.smproj)
- Log viewer panel
- Interactive intensity plots (zoom/pan/channel hiding/right-click export)
- True cancellation via process-based task execution
- print -> logging module

Author: Claude
        """
        QMessageBox.about(self, "About", about_text)

    def closeEvent(self, event):
        if self.is_playing:
            self.play_timer.stop()

        self._cancel_all_active_tasks(recreate_runtime=False)

        self._cleanup_video_widgets()
        self.sync_manager.clear()

        try:
            if self.multi_cache:
                self.multi_cache.shutdown()
        except Exception:
            logger.exception("Failed to shut down cache during close")

        try:
            if self.file_reader:
                self.file_reader.close()
        except Exception:
            logger.exception("Failed to close ND2 reader during close")

        event.accept()


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    setup_logging()
    sys.excepthook = _log_unhandled_exception

    app = QApplication(sys.argv)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Single-Molecule Fluorescence Intensity Extraction Tool v13.0 - ND2 Edition")
    logger.info("=" * 70)
    logger.info(f"CPU cores: {mp.cpu_count()}")
    gpu = gpu_status()
    logger.info(
        "GPU libraries detected: "
        f"CuPy={'yes' if gpu['cupy_detected'] else 'no'} "
        f"OpenCV-CUDA={'yes' if gpu['opencv_cuda_detected'] else 'no'}"
    )
    logger.info(
        f"Active compute backend default: {gpu['default_compute_backend']} "
        "(auto with CPU fallback)"
    )
    logger.info(f"Affine reference channel: {REFERENCE_CHANNEL}")
    logger.info(f"Drift estimation: fiducial tracking v5 "
                f"(max_fiducials={DriftCalculator.MAX_FIDUCIALS})")
    logger.info("=" * 70)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
