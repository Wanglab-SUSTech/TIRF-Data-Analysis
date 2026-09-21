# colocal_classifier.py
import sys
import os
import platform
import logging
import warnings
import pathlib
import argparse
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QCheckBox, QRadioButton,
    QButtonGroup, QLineEdit, QDoubleSpinBox, QGroupBox, QScrollArea, QMenu,
    QShortcut, QDialog, QComboBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QSizePolicy, QStatusBar, QInputDialog,
    QProgressDialog, QSpinBox
)
from PyQt5.QtCore import Qt, QPoint, QTimer, QSettings, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

import pandas as pd
import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

from data_model import (
    ALEX_AF488_INDEX_LANE,
    ALEX_ALPHA_DEFAULT,
    ALEX_BETA_DEFAULT,
    ALEX_CORRECTION_COLUMN_NAMES,
    ALEX_CORRECTION_DISPLAY_LANES,
    ALEX_CY3_ALIVE_LANE,
    ALEX_CY3_TOTAL_LANE,
    ALEX_DERIVED_LANES,
    ALEX_GAMMA_DEFAULT,
    ALEX_LOGICAL_CHANNELS,
    ALEX_S_CORR_LANE,
    DataModel,
    compute_alex_correction_columns,
)
from detection import EventDetector, DetectionParams, EventRecord
from state_manager import AppState


warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None, expanded=True):
        super().__init__(parent)
        self.title = str(title)
        initial_expanded = bool(expanded)

        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(initial_expanded)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setMinimumHeight(32)
        self.toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_button.setToolTip("Click to expand or collapse this section.")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 6px 10px;
                font-weight: 600;
                font-size: 12px;
                color: #14395f;
                background-color: #e7f0fb;
                border: 1px solid #8db6df;
                border-left: 5px solid #2d89ef;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d9eafc;
                border-color: #5f9ed6;
            }
            QPushButton:checked {
                background-color: #dcecff;
            }
            QPushButton:!checked {
                color: #364152;
                background-color: #f1f5f9;
                border-color: #b8c4d1;
                border-left-color: #64748b;
            }
        """)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QWidget()
        self.content_area.setObjectName("collapsibleContent")
        self.content_area.setStyleSheet("""
            QWidget#collapsibleContent {
                background-color: #fcfcfd;
                border-left: 3px solid #d5e3f3;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(14, 8, 8, 10)
        self.content_layout.setSpacing(6)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        self.set_expanded(initial_expanded)

    def on_toggle(self, checked):
        self.content_area.setVisible(checked)
        self._update_toggle_text(checked)

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        previous_state = self.toggle_button.blockSignals(True)
        try:
            self.toggle_button.setChecked(expanded)
        finally:
            self.toggle_button.blockSignals(previous_state)
        self.content_area.setVisible(expanded)
        self._update_toggle_text(expanded)

    def _update_toggle_text(self, expanded):
        marker = "[-]" if expanded else "[+]"
        self.toggle_button.setText(f"{marker} {self.title}")

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


class ExportOptionsDialog(QDialog):
    def __init__(self, classifications, has_unclassified, has_alex_derived=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.setModal(True)
        self.resize(400, 520)

        self.classifications = classifications
        self.has_unclassified = has_unclassified
        self.selected_classes = set()
        self.export_unclassified = False
        self.export_events_summary = True
        self.export_event_signals = True
        self.has_alex_derived = bool(has_alex_derived)
        self.export_alex_derived = bool(has_alex_derived)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        class_group = QGroupBox("Select Classes to Export")
        class_layout = QVBoxLayout()

        self.class_checkboxes = {}
        for class_num in sorted(self.classifications):
            cb = QCheckBox(f"Class {class_num}")
            cb.setChecked(True)
            self.class_checkboxes[class_num] = cb
            class_layout.addWidget(cb)

        if self.has_unclassified:
            self.unclassified_cb = QCheckBox("Unclassified")
            self.unclassified_cb.setChecked(True)
            class_layout.addWidget(self.unclassified_cb)

        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        content_group = QGroupBox("Select Content to Export")
        content_layout = QVBoxLayout()

        self.export_data_cb = QCheckBox("Original data file (*_data.csv)")
        self.export_data_cb.setChecked(True)
        self.export_data_cb.setEnabled(False)
        content_layout.addWidget(self.export_data_cb)

        self.export_alex_derived_cb = QCheckBox("ALEX derived correction columns")
        self.export_alex_derived_cb.setChecked(self.has_alex_derived)
        self.export_alex_derived_cb.setEnabled(self.has_alex_derived)
        content_layout.addWidget(self.export_alex_derived_cb)

        self.export_events_summary_cb = QCheckBox("Event summary file (*_events.csv)")
        self.export_events_summary_cb.setChecked(True)
        content_layout.addWidget(self.export_events_summary_cb)

        self.export_event_signals_cb = QCheckBox("Detailed event signal data (*_event_signal.csv)")
        self.export_event_signals_cb.setChecked(True)
        content_layout.addWidget(self.export_event_signals_cb)

        self.export_params_cb = QCheckBox("Parameter notes file (*_params.txt)")
        self.export_params_cb.setChecked(True)
        content_layout.addWidget(self.export_params_cb)

        content_group.setLayout(content_layout)
        layout.addWidget(content_group)

        quick_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All Classes")
        select_all_btn.clicked.connect(self.select_all_classes)
        quick_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Clear All")
        deselect_all_btn.clicked.connect(self.deselect_all_classes)
        quick_layout.addWidget(deselect_all_btn)
        layout.addLayout(quick_layout)

        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def select_all_classes(self):
        for cb in self.class_checkboxes.values():
            cb.setChecked(True)
        if self.has_unclassified:
            self.unclassified_cb.setChecked(True)

    def deselect_all_classes(self):
        for cb in self.class_checkboxes.values():
            cb.setChecked(False)
        if self.has_unclassified:
            self.unclassified_cb.setChecked(False)

    def get_options(self):
        self.selected_classes = {
            class_num for class_num, cb in self.class_checkboxes.items()
            if cb.isChecked()
        }

        if self.has_unclassified:
            self.export_unclassified = self.unclassified_cb.isChecked()

        self.export_events_summary = self.export_events_summary_cb.isChecked()
        self.export_event_signals = self.export_event_signals_cb.isChecked()
        self.export_alex_derived = (
            self.has_alex_derived and self.export_alex_derived_cb.isChecked()
        )

        return {
            'selected_classes': self.selected_classes,
            'export_unclassified': self.export_unclassified,
            'export_events_summary': self.export_events_summary,
            'export_event_signals': self.export_event_signals,
            'export_params_txt': self.export_params_cb.isChecked(),
            'export_alex_derived': self.export_alex_derived,
        }


class RecalculateWorker(QObject):
    progress_changed = pyqtSignal(int, str)
    finished = pyqtSignal(dict, bool)
    error = pyqtSignal(str)

    def __init__(self, detector, roi_ids, params, overrides):
        super().__init__()
        self.detector = detector
        self.roi_ids = list(roi_ids)
        self.params = params
        self.overrides = dict(overrides)
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        results = {}
        try:
            for index, roi_id in enumerate(self.roi_ids, start=1):
                if self._cancel_requested:
                    break
                override = self.overrides.get(roi_id)
                results[roi_id] = self.detector.detect_events_for_roi(roi_id, self.params, override)
                self.progress_changed.emit(index, f"Processed ID {roi_id}")
            self.finished.emit(results, self._cancel_requested)
        except Exception as exc:
            logger.exception("Background recalculation failed")
            self.error.emit(str(exc))


class ExportWorker(QObject):
    progress_changed = pyqtSignal(int, str)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(
        self,
        df,
        signal_columns,
        signal_channel_order,
        profile,
        source_filepath,
        save_dir,
        base_filename,
        export_jobs,
        event_detection_enabled,
        display_events_by_id,
        override_by_id,
        params,
        alex_correction_params=None,
    ):
        super().__init__()
        self.df = df
        self.signal_columns = dict(signal_columns)
        self.signal_channel_order = list(signal_channel_order)
        self.profile = profile
        self.source_filepath = source_filepath
        self.save_dir = save_dir
        self.base_filename = base_filename
        self.export_jobs = list(export_jobs)
        self.event_detection_enabled = event_detection_enabled
        self.display_events_by_id = display_events_by_id
        self.override_by_id = override_by_id
        self.params = params
        self.alex_correction_params = dict(alex_correction_params or {})
        self.total_steps = self._count_total_steps()

    def _count_total_steps(self):
        total = 0
        for job in self.export_jobs:
            total += 1
            if job["export_events_summary"] and self.event_detection_enabled:
                total += len(job["ids"])
            if job["export_event_signals"] and self.event_detection_enabled:
                total += len(job["ids"])
            if job["export_params_txt"]:
                total += 1
        return max(total, 1)

    def _emit_progress(self, step, message):
        self.progress_changed.emit(step, message)

    def _alex_derived_dataframe(self, df_subset):
        if self.profile != "alex_4ch":
            return pd.DataFrame(index=df_subset.index)
        correction = compute_alex_correction_columns(
            df_subset,
            self.signal_columns,
            alpha=self.alex_correction_params.get("alpha", ALEX_ALPHA_DEFAULT),
            beta=self.alex_correction_params.get("beta", ALEX_BETA_DEFAULT),
            gamma=self.alex_correction_params.get("gamma", ALEX_GAMMA_DEFAULT),
        )
        if correction is None:
            return pd.DataFrame(index=df_subset.index)

        derived = pd.DataFrame(index=df_subset.index)
        for lane in (
            ALEX_CY3_TOTAL_LANE,
            ALEX_CY3_ALIVE_LANE,
            ALEX_S_CORR_LANE,
            ALEX_AF488_INDEX_LANE,
        ):
            column_name = ALEX_CORRECTION_COLUMN_NAMES[lane]
            values = correction[lane]
            if lane == ALEX_CY3_ALIVE_LANE:
                values = values.astype(int)
            derived[column_name] = values
        return derived

    def _export_original_columns_subset(self, df_subset, include_alex_derived=False):
        columns = ["Time_sec", "ROI_ID"]
        for channel in self.signal_channel_order:
            column = self.signal_columns.get(channel)
            if column and column in df_subset.columns:
                columns.append(column)
        result = df_subset[columns].copy()
        if include_alex_derived:
            result = pd.concat([result, self._alex_derived_dataframe(df_subset)], axis=1)
        return result

    def _write_params_txt(self, filepath, ids, class_name):
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write("Data Classification and Event Detection Tool Parameter Notes\n")
            handle.write("=" * 50 + "\n")
            handle.write(f"Class name: {class_name}\n")
            handle.write(f"Exported ID count: {len(ids)}\n")
            handle.write(f"Source file: {self.source_filepath}\n\n")
            handle.write(f"Input profile: {self.profile}\n")
            handle.write("Signal columns:\n")
            for channel in self.signal_channel_order:
                column = self.signal_columns.get(channel)
                if column:
                    handle.write(f"  {channel}: {column}\n")
            if self.profile == "alex_4ch":
                handle.write("ALEX derived display formulas:\n")
                handle.write("  E 532ex = Net_532ex_638 / (Net_532ex_532 + Net_532ex_638)\n")
                handle.write("  R 488ex = Net_488ex_532 / (Net_488ex_488 + Net_488ex_532)\n")
                handle.write("  Total 532ex = Net_532ex_532 + Net_532ex_638\n")
                handle.write("  Total 488ex = Net_488ex_488 + Net_488ex_532\n")
                handle.write("ALEX AF488 index correction:\n")
                handle.write(f"  alpha AF488 bleed-through 488ex_488 -> 488ex_532: {self.alex_correction_params.get('alpha', ALEX_ALPHA_DEFAULT)}\n")
                handle.write(f"  beta Cy3 direct-excitation contribution: {self.alex_correction_params.get('beta', ALEX_BETA_DEFAULT)}\n")
                handle.write(f"  gamma sensitized-emission scale: {self.alex_correction_params.get('gamma', ALEX_GAMMA_DEFAULT)}\n")
                handle.write(f"  beta source: {self.alex_correction_params.get('beta_source', 'manual/default')}\n")
                handle.write("  Cy3 bleach input: Net_532ex_532 + Net_532ex_638\n")
                handle.write("  Cy3_total_532ex = Net_532ex_532 + Net_532ex_638\n")
                handle.write("  S_488ex_532_corr = Net_488ex_532 - alpha * Net_488ex_488 - beta * Net_532ex_532\n")
                handle.write("  AF488_index = Net_488ex_488 + gamma * S_488ex_532_corr_alive\n")
            handle.write("\n")

            handle.write("Global parameters:\n")
            handle.write(f"  Injection time inj_time: {self.params.inj_time}\n")
            handle.write(f"  Minimum dwell time min_dwell: {self.params.min_dwell}\n")
            handle.write(f"  Primary channel deltaMAD: {self.params.delta_mad}\n")
            handle.write(f"  Auxiliary channel deltaMAD: {self.params.aux_delta_mad}\n")
            handle.write(f"  merge_gap: {self.params.merge_gap}\n")
            handle.write(f"  penMAD: {self.params.pen_mad} (currently unused)\n")
            handle.write(f"  Detection mode: {self.params.mode}\n")
            handle.write(f"  Primary channel: {self.params.primary_channel}\n")
            handle.write(f"  Auxiliary channel: {self.params.auxiliary_channel}\n")
            handle.write("  Auto-detection start rule: the primary channel first exceeds the high threshold, then backtracks left to the low-threshold boundary\n")
            handle.write("  Dual-channel validation rule: any point in the auxiliary channel within the primary-event window exceeds its own threshold\n\n")

            handle.write("Manual editing rules:\n")
            handle.write("  1. Manual box selection adds the actual event object.\n")
            handle.write("  2. Two manual actions are provided: add a primary-channel event and mark as dual supported.\n")
            handle.write("  3. In add-primary-event mode, clicking a peak can automatically snap to a short event.\n")
            handle.write("  4. Click-based snapping uses primary-channel low-threshold backtracking to determine both boundaries.\n")
            handle.write("  5. If it overlaps with an existing event, you can choose to modify that event boundaries.\n")
            handle.write("  6. A/D/J/L are supported for fine-tuning the boundary of a single selected event.\n")
            handle.write("  7. In multi-channel display mode, highlighted event regions and the injection time line are shown on all visible channel plots.\n")
            handle.write("  8. Undo for manual actions is supported.\n\n")

            handle.write("ID override parameters used in this class:\n")
            has_override = False
            for roi_id in ids:
                override = self.override_by_id.get(roi_id) or {}
                if override:
                    has_override = True
                    handle.write(f"  ID {roi_id}: {override}\n")
            if not has_override:
                handle.write("  None\n")

    def run(self):
        step = 0
        exported_count = 0

        try:
            for job in self.export_jobs:
                class_name = job["class_name"]
                ids = list(job["ids"])

                df_class = self.df[self.df["ROI_ID"].isin(ids)].copy()
                data_filename = os.path.join(self.save_dir, f"{self.base_filename}_{class_name}_data.csv")
                self._export_original_columns_subset(
                    df_class,
                    include_alex_derived=job.get("export_alex_derived", False),
                ).to_csv(data_filename, sep=",", index=False)
                step += 1
                self._emit_progress(step, f"Writing data file for {class_name}")

                if job["export_events_summary"] and self.event_detection_enabled:
                    events_data = []
                    for roi_id in ids:
                        for event in self.display_events_by_id.get(roi_id, []):
                            events_data.append({
                                "ID": roi_id,
                                "BindTime_s": event["start_time"],
                                "DissTime_s": event["end_time"],
                                "DwellTime_s": event["dwell_time"],
                                "Remark": "right-censored" if event.get("right_censored") else "",
                                "BindTimeRelToInjection_s": event["start_time"] - self.params.inj_time,
                                "EventType": event["classification"],
                                "AuxSupported": event["aux_supported"],
                                "AuxOverlapRatio": event["aux_overlap_ratio"],
                                "EventKey": str(event["event_key"]),
                                "Source": event.get("source", "auto"),
                                "ManualStatus": event.get("manual_status", ""),
                                "CreatedBy": event.get("created_by", "auto"),
                            })
                        step += 1
                        self._emit_progress(step, f"Collecting event summary for {class_name}: ID {roi_id}")

                    if events_data:
                        events_df = pd.DataFrame(events_data)
                        events_filename = os.path.join(self.save_dir, f"{self.base_filename}_{class_name}_events.csv")
                        events_df.to_csv(events_filename, sep=",", index=False)

                if job["export_event_signals"] and self.event_detection_enabled:
                    event_signal_data = []
                    for roi_id in ids:
                        df_id = self.df[self.df["ROI_ID"] == roi_id].copy()
                        if not df_id.empty:
                            alex_derived = (
                                self._alex_derived_dataframe(df_id)
                                if job.get("export_alex_derived", False)
                                else pd.DataFrame(index=df_id.index)
                            )
                            for event_index, event in enumerate(self.display_events_by_id.get(roi_id, []), start=1):
                                start_idx = event["start_idx"]
                                end_idx = event["end_idx"]
                                event_times = df_id["Time_sec"].iloc[start_idx:end_idx + 1].values

                                signal_slices = {}
                                for channel in self.signal_channel_order:
                                    column = self.signal_columns.get(channel)
                                    if column and column in df_id.columns:
                                        signal_slices[column] = df_id[column].iloc[start_idx:end_idx + 1].values
                                for column in alex_derived.columns:
                                    signal_slices[column] = alex_derived[column].iloc[start_idx:end_idx + 1].values

                                for point_index, time_value in enumerate(event_times):
                                    row = {
                                        "ID": roi_id,
                                        "Event_Index": event_index,
                                        "Time_sec": time_value,
                                        "EventType": event["classification"],
                                        "EventKey": str(event["event_key"]),
                                        "Source": event.get("source", "auto"),
                                        "ManualStatus": event.get("manual_status", ""),
                                    }
                                    for column_name, values in signal_slices.items():
                                        row[column_name] = values[point_index]
                                    event_signal_data.append(row)

                        step += 1
                        self._emit_progress(step, f"Collecting event signal data for {class_name}: ID {roi_id}")

                    if event_signal_data:
                        event_signal_df = pd.DataFrame(event_signal_data)
                        event_signal_filename = os.path.join(
                            self.save_dir,
                            f"{self.base_filename}_{class_name}_event_signal.csv"
                        )
                        event_signal_df.to_csv(event_signal_filename, sep=",", index=False)

                if job["export_params_txt"]:
                    params_filename = os.path.join(self.save_dir, f"{self.base_filename}_{class_name}_params.txt")
                    self._write_params_txt(params_filename, ids, class_name)
                    step += 1
                    self._emit_progress(step, f"Writing parameter notes for {class_name}")

                exported_count += 1

            self.finished.emit(exported_count)
        except Exception as exc:
            logger.exception("Background export failed")
            self.error.emit(str(exc))


class SyncedNavigationToolbar(NavigationToolbar):
    mode_changed = pyqtSignal(str)
    view_changed = pyqtSignal()

    def _emit_mode_changed(self):
        self.mode_changed.emit(str(getattr(self, "mode", "")))

    def _emit_view_changed(self):
        self.view_changed.emit()

    def pan(self, *args):
        super().pan(*args)
        self._emit_mode_changed()

    def zoom(self, *args):
        super().zoom(*args)
        self._emit_mode_changed()

    def home(self, *args):
        super().home(*args)
        self._emit_view_changed()
        self._emit_mode_changed()

    def back(self, *args):
        super().back(*args)
        self._emit_view_changed()
        self._emit_mode_changed()

    def forward(self, *args):
        super().forward(*args)
        self._emit_view_changed()
        self._emit_mode_changed()

    def release_pan(self, event):
        super().release_pan(event)
        self._emit_view_changed()
        self._emit_mode_changed()

    def release_zoom(self, event):
        super().release_zoom(event)
        self._emit_view_changed()
        self._emit_mode_changed()


class EventDetectionTool(QMainWindow):
    CHANNEL_ORDER = ['488', '532', '638']
    CHANNEL_COLORS = {'488': 'blue', '532': 'green', '638': 'red'}
    DISPLAY_LANE_COLORS = {
        '488': 'blue',
        '532': 'green',
        '638': 'red',
        'FRET': 'purple',
        '532ex_532': '#00d26a',
        '532ex_638': '#ff5555',
        '488ex_532': '#f5c542',
        '488ex_488': '#55aaff',
        'ALEX_532EX_INTENSITY': '#444444',
        'ALEX_488EX_INTENSITY': '#666666',
        'ALEX_E_532EX': 'purple',
        'ALEX_R_488EX': '#8a5a00',
        ALEX_CY3_TOTAL_LANE: '#444444',
        ALEX_S_CORR_LANE: '#d18f00',
        ALEX_AF488_INDEX_LANE: '#1b66cc',
    }
    FRET_LANE = 'FRET'
    INTENSITY_LANE = '532638'
    ALEX_INTENSITY_532EX_LANE = 'ALEX_532EX_INTENSITY'
    ALEX_INTENSITY_488EX_LANE = 'ALEX_488EX_INTENSITY'
    DISPLAY_MODE_STANDARD = 'standard'
    DISPLAY_MODE_FRET = 'fret'
    DISPLAY_MODE_ALEX_RAW = 'alex_raw'
    DISPLAY_MODE_ALEX_COMPACT = 'alex_compact'
    DISPLAY_MODE_ALEX_DERIVED = 'alex_derived'
    SMOOTHABLE_CURVES = (
        '488', '532', '638', 'FRET',
        '532ex_532', '532ex_638', '488ex_532', '488ex_488',
        'ALEX_E_532EX', 'ALEX_R_488EX', 'ALEX_TOTAL_532EX', 'ALEX_TOTAL_488EX',
        ALEX_CY3_TOTAL_LANE, ALEX_S_CORR_LANE, ALEX_AF488_INDEX_LANE,
    )
    SMOOTH_DEFAULT_WINDOW = 7
    SMOOTH_POLYORDER = 2
    X_SCROLL_ZOOM_BASE = 0.90
    NAVIGATION_DEBOUNCE_MS = 25
    CLASS_BTN_ACTIVE_STYLE = """
        QPushButton {
            background-color: #2d89ef;
            color: white;
            font-weight: bold;
            border: 2px solid #1b5fa7;
            border-radius: 4px;
            padding: 6px;
        }
    """
    CLASS_BTN_DEFAULT_STYLE = ""

    def __init__(self, default_directory=None):
        super().__init__()
        self.setWindowTitle("Data Classification and Event Detection Tool")
        self.setGeometry(100, 100, 1780, 1000)

        self.system = platform.system()
        self.default_directory = default_directory if default_directory else ""
        self.settings = QSettings("Colocal", "ColocalClassifier")
        self.max_recent_files = 5

        self.data_model = DataModel()
        self.detector = EventDetector(self.data_model)
        self.state = AppState()

        self.event_detection_enabled = False

        self.multiselect_modifier_active = False
        self.shift_modifier_active = False

        self.rect_selectors = {}
        self.event_patch_map = {}
        self.class_button_map = {}

        self.channel_axes = {}
        self.visible_channels = []
        self._axes_layout_signature = None
        self._display_mode = self.DISPLAY_MODE_STANDARD
        self.raw_channel_checkboxes = {}
        self.derived_channel_checkboxes = {}
        self.display_mode_buttons = {}
        self.smoothing_checkboxes = {}
        self.smoothing_window_inputs = {}
        self.smoothing_controls = {}
        self._last_beta_estimate = None
        self._beta_source = "manual/default"
        self._syncing_xlim = False
        self._plot_pan_state = None
        self._crosshair_lines = {}
        self._roi_view_state_cache = {}
        self._suspend_view_state_cache = False
        self._crosshair_pending_x = None
        self._last_crosshair_x = None
        self._crosshair_update_timer = QTimer(self)
        self._crosshair_update_timer.setSingleShot(True)
        self._crosshair_update_timer.timeout.connect(self._flush_crosshair_update)

        self.manual_draw_mode = None
        self.manual_click_min_points = 3

        self._status_clear_timer = QTimer(self)
        self._status_clear_timer.setSingleShot(True)
        self._status_clear_timer.timeout.connect(self._clear_transient_status)

        self._suppress_table_selection_signal = False
        self._suppress_table_item_changed_signal = False
        self._last_displayed_id = None
        self._worker_thread = None
        self._worker = None
        self._progress_dialog = None
        self._current_task_name = ""
        self._task_cancelled = False
        self._pending_navigation_delta = 0
        self._navigation_timer = QTimer(self)
        self._navigation_timer.setSingleShot(True)
        self._navigation_timer.timeout.connect(self._flush_pending_navigation)

        self.init_ui()
        self._load_persisted_settings()
        self._setup_shortcuts()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.setStatusBar(QStatusBar(self))

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.fig_main = Figure(figsize=(12, 8))
        self.fig_main.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.96, hspace=0.18)

        self.canvas_main = FigureCanvas(self.fig_main)
        self.canvas_main.setFocusPolicy(Qt.StrongFocus)
        self.canvas_main.setFocus()
        self.canvas_main.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = SyncedNavigationToolbar(self.canvas_main, self)
        left_layout.addWidget(self.toolbar)
        self.toolbar.mode_changed.connect(self._on_toolbar_mode_changed)
        self.toolbar.view_changed.connect(self._on_toolbar_view_changed)

        self.canvas_main.mpl_connect('button_press_event', self.on_main_plot_click)
        self.canvas_main.mpl_connect('button_release_event', self.on_main_plot_release)
        self.canvas_main.mpl_connect('key_press_event', self.on_canvas_key_press)
        self.canvas_main.mpl_connect('key_release_event', self.on_canvas_key_release)
        self.canvas_main.mpl_connect('scroll_event', self.on_main_plot_scroll)
        self.canvas_main.mpl_connect('motion_notify_event', self.on_main_plot_motion)
        self.canvas_main.mpl_connect('figure_leave_event', self.on_main_plot_leave)

        left_layout.addWidget(self.canvas_main, 65)

        event_list_group = QGroupBox("Current ID Event List")
        event_list_layout = QVBoxLayout()

        self.event_table = QTableWidget(0, 10)
        self.event_table.setHorizontalHeaderLabels([
            "Select", "Event Type", "Source", "Start (s)", "End (s)", "Dwell (s)",
            "Aux Support", "Overlap Ratio", "Manual Status", "Key"
        ])
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.event_table.itemChanged.connect(self.on_event_table_item_changed)
        self.event_table.itemSelectionChanged.connect(self.on_event_table_selection_changed)
        event_list_layout.addWidget(self.event_table)

        event_list_btn_layout = QHBoxLayout()
        select_table_all_btn = QPushButton("Select All Rows")
        select_table_all_btn.clicked.connect(self.select_all_events)
        event_list_btn_layout.addWidget(select_table_all_btn)

        invert_table_btn = QPushButton("Invert Row Selection")
        invert_table_btn.clicked.connect(self.invert_selection)
        event_list_btn_layout.addWidget(invert_table_btn)

        delete_table_btn = QPushButton("Delete Selected Events")
        delete_table_btn.clicked.connect(self.delete_selected_events)
        event_list_btn_layout.addWidget(delete_table_btn)

        undo_btn = QPushButton("Undo Last Step")
        undo_btn.clicked.connect(self.undo_last_manual_action)
        event_list_btn_layout.addWidget(undo_btn)

        event_list_layout.addLayout(event_list_btn_layout)
        event_list_group.setLayout(event_list_layout)
        left_layout.addWidget(event_list_group, 35)

        main_layout.addWidget(left_widget, 72)

        right_widget = QWidget()
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setWidget(right_widget)
        right_layout = QVBoxLayout(right_widget)

        box_data = CollapsibleBox("Data Loading", expanded=True)
        self.data_loading_box = box_data
        self.load_btn = QPushButton("Load CSV...")
        self.load_btn.clicked.connect(self.load_csv)
        box_data.addWidget(self.load_btn)

        recent_layout = QHBoxLayout()
        self.recent_files_combo = QComboBox()
        self.recent_files_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.recent_files_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.recent_files_combo.setMinimumContentsLength(18)
        self.recent_files_combo.currentIndexChanged.connect(self._update_recent_files_tooltip)
        recent_layout.addWidget(self.recent_files_combo)
        self.open_recent_btn = QPushButton("Open Recent")
        self.open_recent_btn.clicked.connect(self.open_recent_file)
        recent_layout.addWidget(self.open_recent_btn)
        box_data.addLayout(recent_layout)

        self.save_session_btn = QPushButton("Save Session")
        self.save_session_btn.clicked.connect(lambda: self.save_session(show_feedback=True))
        box_data.addWidget(self.save_session_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        box_data.addWidget(self.file_label)
        right_layout.addWidget(box_data)

        box_nav = CollapsibleBox("Navigation", expanded=False)
        self.navigation_box = box_nav
        self.nav_label = QLabel("Page 0/0")
        self.nav_label.setAlignment(Qt.AlignCenter)
        box_nav.addWidget(self.nav_label)

        nav_btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_id)
        self.prev_btn.setToolTip("Previous ID (Left Arrow)")
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_id)
        self.next_btn.setToolTip("Next ID (Right Arrow)")
        nav_btn_layout.addWidget(self.prev_btn)
        nav_btn_layout.addWidget(self.next_btn)
        box_nav.addLayout(nav_btn_layout)

        self.total_label = QLabel("Visible IDs: 0")
        self.total_label.setAlignment(Qt.AlignCenter)
        box_nav.addWidget(self.total_label)

        self.stats_label = QLabel("Total IDs: 0 | Classified: 0 | Unclassified: 0")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setWordWrap(True)
        box_nav.addWidget(self.stats_label)

        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("Jump to ID:"))
        self.jump_id_input = QLineEdit()
        self.jump_id_input.setPlaceholderText("Enter exact ID")
        self.jump_id_input.returnPressed.connect(self.jump_to_id)
        jump_layout.addWidget(self.jump_id_input)
        jump_btn = QPushButton("Jump")
        jump_btn.clicked.connect(self.jump_to_id)
        jump_layout.addWidget(jump_btn)
        box_nav.addLayout(jump_layout)

        filter_layout = QVBoxLayout()
        filter_layout.addWidget(QLabel("Navigation Filter:"))

        self.filter_all_radio = QRadioButton("All IDs")
        self.filter_unclassified_radio = QRadioButton("Unclassified Only")
        self.filter_class_radio = QRadioButton("By Class Only")
        self.filter_all_radio.setChecked(True)

        self.filter_group = QButtonGroup()
        self.filter_group.addButton(self.filter_all_radio)
        self.filter_group.addButton(self.filter_unclassified_radio)
        self.filter_group.addButton(self.filter_class_radio)

        self.filter_all_radio.toggled.connect(self.apply_navigation_filter)
        self.filter_unclassified_radio.toggled.connect(self.apply_navigation_filter)
        self.filter_class_radio.toggled.connect(self.apply_navigation_filter)

        self.filter_class_combo = QComboBox()
        self.filter_class_combo.currentIndexChanged.connect(self.apply_navigation_filter)

        filter_layout.addWidget(self.filter_all_radio)
        filter_layout.addWidget(self.filter_unclassified_radio)

        class_filter_line = QHBoxLayout()
        class_filter_line.addWidget(self.filter_class_radio)
        class_filter_line.addWidget(self.filter_class_combo)
        filter_layout.addLayout(class_filter_line)

        box_nav.addLayout(filter_layout)
        right_layout.addWidget(box_nav)

        box_class = CollapsibleBox("Classification", expanded=True)
        self.classification_box = box_class
        self.current_id_label = QLabel("Current ID: -")
        box_class.addWidget(self.current_id_label)

        self.current_class_label = QLabel("Current class: Unclassified")
        box_class.addWidget(self.current_class_label)

        shortcut_hint = QLabel("Use keys 1-9 to classify or unclassify quickly. Use Left and Right to change pages.")
        shortcut_hint.setStyleSheet("color: #0066cc; font-style: italic; font-size: 11px;")
        shortcut_hint.setWordWrap(True)
        shortcut_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        box_class.addWidget(shortcut_hint)

        self.auto_advance_checkbox = QCheckBox("Auto-advance")
        self.auto_advance_checkbox.setChecked(True)
        self.auto_advance_checkbox.toggled.connect(self._persist_auto_advance_setting)
        self.auto_advance_checkbox.setToolTip("Automatically move to the next visible ID after classification")
        box_class.addWidget(self.auto_advance_checkbox)

        self.class_buttons_container = QWidget()
        self.class_buttons_layout = QVBoxLayout(self.class_buttons_container)
        self.class_buttons_layout.setContentsMargins(0, 0, 0, 0)
        box_class.addWidget(self.class_buttons_container)

        self.add_class_btn = QPushButton("+ Add New Class")
        self.add_class_btn.clicked.connect(self.add_classification)
        box_class.addWidget(self.add_class_btn)

        classification_history_layout = QHBoxLayout()
        self.undo_classification_btn = QPushButton("Undo")
        self.undo_classification_btn.clicked.connect(self.undo_classification)
        self.undo_classification_btn.setToolTip("Undo the last classification action")
        classification_history_layout.addWidget(self.undo_classification_btn)

        self.redo_classification_btn = QPushButton("Redo")
        self.redo_classification_btn.clicked.connect(self.redo_classification)
        self.redo_classification_btn.setToolTip("Redo the last undone classification action")
        classification_history_layout.addWidget(self.redo_classification_btn)
        box_class.addLayout(classification_history_layout)

        self.bulk_assign_btn = QPushButton("Classify Visible IDs...")
        self.bulk_assign_btn.clicked.connect(self.assign_current_filter_to_class)
        self.bulk_assign_btn.setToolTip("Assign all currently visible IDs to one class")
        box_class.addWidget(self.bulk_assign_btn)

        self.bulk_clear_btn = QPushButton("Clear Visible IDs")
        self.bulk_clear_btn.clicked.connect(self.clear_classifications_in_current_filter)
        self.bulk_clear_btn.setToolTip("Clear classifications for all currently visible IDs")
        box_class.addWidget(self.bulk_clear_btn)

        self.export_btn = QPushButton("Export All Classes")
        self.export_btn.clicked.connect(self.export_classifications)
        box_class.addWidget(self.export_btn)

        right_layout.addWidget(box_class)

        box_channel = CollapsibleBox("Channel Display", expanded=False)
        self.channel_display_box = box_channel
        box_channel.addWidget(QLabel("Display Mode"))
        display_mode_layout = QVBoxLayout()
        self.display_standard_radio = QRadioButton("Standard 488/532/638")
        self.display_fret_radio = QRadioButton("FRET 488 + 532/638 + FRET")
        self.display_alex_raw_radio = QRadioButton("ALEX raw four-channel")
        self.display_alex_compact_radio = QRadioButton("ALEX grouped by excitation")
        self.display_alex_derived_radio = QRadioButton("ALEX grouped + derived ratios")
        self.display_standard_radio.setChecked(True)
        self.display_mode_group = QButtonGroup(self)
        self.display_mode_group.addButton(self.display_standard_radio)
        self.display_mode_group.addButton(self.display_fret_radio)
        self.display_mode_group.addButton(self.display_alex_raw_radio)
        self.display_mode_group.addButton(self.display_alex_compact_radio)
        self.display_mode_group.addButton(self.display_alex_derived_radio)
        self.display_mode_buttons = {
            self.DISPLAY_MODE_STANDARD: self.display_standard_radio,
            self.DISPLAY_MODE_FRET: self.display_fret_radio,
            self.DISPLAY_MODE_ALEX_RAW: self.display_alex_raw_radio,
            self.DISPLAY_MODE_ALEX_COMPACT: self.display_alex_compact_radio,
            self.DISPLAY_MODE_ALEX_DERIVED: self.display_alex_derived_radio,
        }
        self.display_standard_radio.toggled.connect(self.on_display_mode_changed)
        self.display_fret_radio.toggled.connect(self.on_display_mode_changed)
        self.display_alex_raw_radio.toggled.connect(self.on_display_mode_changed)
        self.display_alex_compact_radio.toggled.connect(self.on_display_mode_changed)
        self.display_alex_derived_radio.toggled.connect(self.on_display_mode_changed)
        display_mode_layout.addWidget(self.display_standard_radio)
        display_mode_layout.addWidget(self.display_fret_radio)
        display_mode_layout.addWidget(self.display_alex_raw_radio)
        display_mode_layout.addWidget(self.display_alex_compact_radio)
        display_mode_layout.addWidget(self.display_alex_derived_radio)
        box_channel.addLayout(display_mode_layout)

        self.channel_488_checkbox = QCheckBox("Show 488nm")
        self.channel_488_checkbox.setChecked(True)
        self.channel_488_checkbox.stateChanged.connect(self.update_plots)
        box_channel.addWidget(self.channel_488_checkbox)
        self.raw_channel_checkboxes['488'] = self.channel_488_checkbox

        self.channel_532_checkbox = QCheckBox("Show 532nm")
        self.channel_532_checkbox.setChecked(True)
        self.channel_532_checkbox.stateChanged.connect(self.update_plots)
        box_channel.addWidget(self.channel_532_checkbox)
        self.raw_channel_checkboxes['532'] = self.channel_532_checkbox

        self.channel_638_checkbox = QCheckBox("Show 638nm")
        self.channel_638_checkbox.setChecked(True)
        self.channel_638_checkbox.stateChanged.connect(self.update_plots)
        box_channel.addWidget(self.channel_638_checkbox)
        self.raw_channel_checkboxes['638'] = self.channel_638_checkbox

        for channel in ALEX_LOGICAL_CHANNELS:
            label = channel.replace("ex_", "ex->")
            checkbox = QCheckBox(f"Show {label}")
            checkbox.setChecked(True)
            checkbox.setEnabled(False)
            checkbox.stateChanged.connect(self.update_plots)
            self.raw_channel_checkboxes[channel] = checkbox
            box_channel.addWidget(checkbox)

        self.channel_fret_checkbox = QCheckBox("Show FRET")
        self.channel_fret_checkbox.setChecked(True)
        self.channel_fret_checkbox.stateChanged.connect(self.update_plots)
        box_channel.addWidget(self.channel_fret_checkbox)
        self.derived_channel_checkboxes[self.FRET_LANE] = self.channel_fret_checkbox

        self.channel_alex_e_checkbox = QCheckBox("Show E 532ex")
        self.channel_alex_e_checkbox.setChecked(True)
        self.channel_alex_e_checkbox.setEnabled(False)
        self.channel_alex_e_checkbox.stateChanged.connect(self.update_plots)
        self.derived_channel_checkboxes['ALEX_E_532EX'] = self.channel_alex_e_checkbox
        box_channel.addWidget(self.channel_alex_e_checkbox)

        self.channel_alex_r_checkbox = QCheckBox("Show R 488ex")
        self.channel_alex_r_checkbox.setChecked(True)
        self.channel_alex_r_checkbox.setEnabled(False)
        self.channel_alex_r_checkbox.stateChanged.connect(self.update_plots)
        self.derived_channel_checkboxes['ALEX_R_488EX'] = self.channel_alex_r_checkbox
        box_channel.addWidget(self.channel_alex_r_checkbox)

        self.channel_alex_cy3_total_checkbox = QCheckBox("Show Cy3 total 532ex")
        self.channel_alex_cy3_total_checkbox.setChecked(True)
        self.channel_alex_cy3_total_checkbox.setEnabled(False)
        self.channel_alex_cy3_total_checkbox.stateChanged.connect(self.update_plots)
        self.derived_channel_checkboxes[ALEX_CY3_TOTAL_LANE] = self.channel_alex_cy3_total_checkbox
        box_channel.addWidget(self.channel_alex_cy3_total_checkbox)

        self.channel_alex_s_corr_checkbox = QCheckBox("Show S corrected")
        self.channel_alex_s_corr_checkbox.setChecked(True)
        self.channel_alex_s_corr_checkbox.setEnabled(False)
        self.channel_alex_s_corr_checkbox.stateChanged.connect(self.update_plots)
        self.derived_channel_checkboxes[ALEX_S_CORR_LANE] = self.channel_alex_s_corr_checkbox
        box_channel.addWidget(self.channel_alex_s_corr_checkbox)

        self.channel_alex_af488_index_checkbox = QCheckBox("Show AF488 index")
        self.channel_alex_af488_index_checkbox.setChecked(True)
        self.channel_alex_af488_index_checkbox.setEnabled(False)
        self.channel_alex_af488_index_checkbox.stateChanged.connect(self.update_plots)
        self.derived_channel_checkboxes[ALEX_AF488_INDEX_LANE] = self.channel_alex_af488_index_checkbox
        box_channel.addWidget(self.channel_alex_af488_index_checkbox)

        right_layout.addWidget(box_channel)

        box_alex_correction = CollapsibleBox("ALEX Correction / AF488 Index", expanded=False)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("alpha:"))
        self.alex_alpha_input = QDoubleSpinBox()
        self.alex_alpha_input.setRange(-100.0, 100.0)
        self.alex_alpha_input.setDecimals(4)
        self.alex_alpha_input.setSingleStep(0.01)
        self.alex_alpha_input.setValue(ALEX_ALPHA_DEFAULT)
        self.alex_alpha_input.valueChanged.connect(self.on_alex_correction_changed)
        alpha_layout.addWidget(self.alex_alpha_input)
        box_alex_correction.addLayout(alpha_layout)

        beta_layout = QHBoxLayout()
        beta_layout.addWidget(QLabel("beta:"))
        self.alex_beta_input = QDoubleSpinBox()
        self.alex_beta_input.setRange(-100.0, 100.0)
        self.alex_beta_input.setDecimals(4)
        self.alex_beta_input.setSingleStep(0.01)
        self.alex_beta_input.setValue(ALEX_BETA_DEFAULT)
        self.alex_beta_input.valueChanged.connect(self.on_alex_correction_changed)
        beta_layout.addWidget(self.alex_beta_input)
        box_alex_correction.addLayout(beta_layout)

        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("gamma:"))
        self.alex_gamma_input = QDoubleSpinBox()
        self.alex_gamma_input.setRange(-100.0, 100.0)
        self.alex_gamma_input.setDecimals(4)
        self.alex_gamma_input.setSingleStep(0.01)
        self.alex_gamma_input.setValue(ALEX_GAMMA_DEFAULT)
        self.alex_gamma_input.valueChanged.connect(self.on_alex_correction_changed)
        gamma_layout.addWidget(self.alex_gamma_input)
        box_alex_correction.addLayout(gamma_layout)

        estimate_layout = QHBoxLayout()
        self.estimate_beta_btn = QPushButton("Estimate beta")
        self.estimate_beta_btn.clicked.connect(self.estimate_alex_beta)
        estimate_layout.addWidget(self.estimate_beta_btn)
        self.apply_beta_estimate_btn = QPushButton("Apply estimate")
        self.apply_beta_estimate_btn.clicked.connect(self.apply_beta_estimate)
        self.apply_beta_estimate_btn.setEnabled(False)
        estimate_layout.addWidget(self.apply_beta_estimate_btn)
        box_alex_correction.addLayout(estimate_layout)

        self.beta_estimate_label = QLabel("Estimated beta: n/a")
        self.beta_estimate_label.setStyleSheet("color: gray; font-style: italic;")
        self.beta_estimate_label.setWordWrap(True)
        box_alex_correction.addWidget(self.beta_estimate_label)

        self.alex_correction_box = box_alex_correction
        right_layout.addWidget(box_alex_correction)

        box_smoothing = CollapsibleBox("Smoothing (Display Only)", expanded=False)
        self.smoothing_box = box_smoothing
        self._add_smoothing_controls(box_smoothing, '488', '488nm')
        self._add_smoothing_controls(box_smoothing, '532', '532nm')
        self._add_smoothing_controls(box_smoothing, '638', '638nm')
        self._add_smoothing_controls(box_smoothing, self.FRET_LANE, 'FRET')
        for channel in ALEX_LOGICAL_CHANNELS:
            self._add_smoothing_controls(box_smoothing, channel, channel.replace("ex_", "ex->"))
        self._add_smoothing_controls(box_smoothing, 'ALEX_E_532EX', 'E 532ex')
        self._add_smoothing_controls(box_smoothing, 'ALEX_R_488EX', 'R 488ex')
        self._add_smoothing_controls(box_smoothing, ALEX_CY3_TOTAL_LANE, 'Cy3 total 532ex')
        self._add_smoothing_controls(box_smoothing, ALEX_S_CORR_LANE, 'S corrected')
        self._add_smoothing_controls(box_smoothing, ALEX_AF488_INDEX_LANE, 'AF488 index')
        right_layout.addWidget(box_smoothing)

        box_detect = CollapsibleBox("Event Detection (Optional)", expanded=True)
        self.event_detection_box = box_detect

        self.enable_event_detection_checkbox = QCheckBox("Enable Event Detection")
        self.enable_event_detection_checkbox.setChecked(False)
        self.enable_event_detection_checkbox.stateChanged.connect(self.toggle_event_detection)
        box_detect.addWidget(self.enable_event_detection_checkbox)

        box_detect.addWidget(QLabel("Detection Mode"))
        mode_layout = QVBoxLayout()
        self.mode_single_radio = QRadioButton("Single-channel Mode")
        self.mode_dual_radio = QRadioButton("Primary Channel + Auxiliary Channel Mode")
        self.mode_single_radio.setChecked(True)
        self.mode_single_radio.toggled.connect(self.on_detection_mode_changed)
        self.mode_dual_radio.toggled.connect(self.on_detection_mode_changed)
        mode_layout.addWidget(self.mode_single_radio)
        mode_layout.addWidget(self.mode_dual_radio)
        box_detect.addLayout(mode_layout)

        box_detect.addWidget(QLabel("Basic Parameters"))

        inj_layout = QHBoxLayout()
        inj_layout.addWidget(QLabel("Injection Time (s):"))
        self.inj_time_input = QDoubleSpinBox()
        self.inj_time_input.setRange(0, 1000)
        self.inj_time_input.setValue(0)
        self.inj_time_input.setSingleStep(0.1)
        self.inj_time_input.valueChanged.connect(self.update_global_params)
        inj_layout.addWidget(self.inj_time_input)
        box_detect.addLayout(inj_layout)

        min_dwell_layout = QHBoxLayout()
        min_dwell_layout.addWidget(QLabel("Minimum Dwell Time (s):"))
        self.min_dwell_input = QDoubleSpinBox()
        self.min_dwell_input.setRange(0, 100)
        self.min_dwell_input.setValue(0.5)
        self.min_dwell_input.setSingleStep(0.1)
        self.min_dwell_input.valueChanged.connect(self.update_global_params)
        min_dwell_layout.addWidget(self.min_dwell_input)
        box_detect.addLayout(min_dwell_layout)

        delta_mad_layout = QHBoxLayout()
        delta_mad_layout.addWidget(QLabel("Primary Channel deltaMAD:"))
        self.delta_mad_input = QDoubleSpinBox()
        self.delta_mad_input.setRange(0, 100)
        self.delta_mad_input.setValue(3.0)
        self.delta_mad_input.setSingleStep(0.1)
        self.delta_mad_input.valueChanged.connect(self.update_global_params)
        delta_mad_layout.addWidget(self.delta_mad_input)
        box_detect.addLayout(delta_mad_layout)

        aux_delta_layout = QHBoxLayout()
        aux_delta_layout.addWidget(QLabel("Auxiliary Channel deltaMAD:"))
        self.aux_delta_mad_input = QDoubleSpinBox()
        self.aux_delta_mad_input.setRange(0, 100)
        self.aux_delta_mad_input.setValue(3.0)
        self.aux_delta_mad_input.setSingleStep(0.1)
        self.aux_delta_mad_input.valueChanged.connect(self.update_global_params)
        aux_delta_layout.addWidget(self.aux_delta_mad_input)
        box_detect.addLayout(aux_delta_layout)

        merge_gap_layout = QHBoxLayout()
        merge_gap_layout.addWidget(QLabel("Merge Gap Between Adjacent Events (s):"))
        self.merge_gap_input = QDoubleSpinBox()
        self.merge_gap_input.setRange(0, 10)
        self.merge_gap_input.setValue(0.2)
        self.merge_gap_input.setSingleStep(0.1)
        self.merge_gap_input.valueChanged.connect(self.update_global_params)
        merge_gap_layout.addWidget(self.merge_gap_input)
        box_detect.addLayout(merge_gap_layout)

        pen_mad_layout = QHBoxLayout()
        pen_mad_layout.addWidget(QLabel("penMAD:"))
        self.pen_mad_input = QDoubleSpinBox()
        self.pen_mad_input.setRange(0, 100)
        self.pen_mad_input.setValue(3.0)
        self.pen_mad_input.setSingleStep(0.1)
        self.pen_mad_input.valueChanged.connect(self.update_global_params)
        pen_mad_layout.addWidget(self.pen_mad_input)
        box_detect.addLayout(pen_mad_layout)

        pen_hint = QLabel("Note: penMAD is currently unused and retained only for future extension.")
        pen_hint.setStyleSheet("color: gray; font-style: italic;")
        box_detect.addWidget(pen_hint)

        box_detect.addWidget(QLabel("Channel Settings"))

        primary_layout = QHBoxLayout()
        primary_layout.addWidget(QLabel("Primary Channel:"))
        self.primary_channel_combo = QComboBox()
        self.primary_channel_combo.currentIndexChanged.connect(self.on_detection_channel_changed)
        primary_layout.addWidget(self.primary_channel_combo)
        box_detect.addLayout(primary_layout)

        aux_layout = QHBoxLayout()
        aux_layout.addWidget(QLabel("Auxiliary Channel:"))
        self.aux_channel_combo = QComboBox()
        self.aux_channel_combo.currentIndexChanged.connect(self.on_detection_channel_changed)
        aux_layout.addWidget(self.aux_channel_combo)
        box_detect.addLayout(aux_layout)

        recalc_btn = QPushButton("Recalculate (All IDs)")
        recalc_btn.clicked.connect(self.recalculate_all)
        self.recalc_all_btn = recalc_btn
        box_detect.addWidget(recalc_btn)

        box_detect.addWidget(QLabel("Manual Event Editing"))

        manual_btn_layout = QHBoxLayout()
        self.add_event_btn = QPushButton("Add Primary-channel Event")
        self.add_event_btn.setCheckable(True)
        self.add_event_btn.clicked.connect(self.activate_add_event_mode)
        manual_btn_layout.addWidget(self.add_event_btn)

        self.mark_bound_btn = QPushButton("Mark as Dual Supported")
        self.mark_bound_btn.setCheckable(True)
        self.mark_bound_btn.clicked.connect(self.activate_mark_bound_mode)
        manual_btn_layout.addWidget(self.mark_bound_btn)
        box_detect.addLayout(manual_btn_layout)

        self.manual_mode_label = QLabel("Manual mode: None")
        self.manual_mode_label.setStyleSheet("color: #aa5500;")
        box_detect.addWidget(self.manual_mode_label)

        self.manual_hint_label = QLabel(
            "Hint: In add-primary-event mode, drag with the left mouse button to add an event; click a peak to snap to a short event automatically; "
            "after selecting an event, use A/D to adjust the left boundary and J/L to adjust the right boundary"
        )
        self.manual_hint_label.setWordWrap(True)
        self.manual_hint_label.setStyleSheet("color: gray; font-style: italic;")
        box_detect.addWidget(self.manual_hint_label)

        undo_manual_btn = QPushButton("Undo Manual Action")
        undo_manual_btn.clicked.connect(self.undo_last_manual_action)
        box_detect.addWidget(undo_manual_btn)

        right_layout.addWidget(box_detect)

        box_override = CollapsibleBox("Current ID Override", expanded=False)
        self.override_checkbox = QCheckBox("Enable Current ID Override")
        self.override_checkbox.stateChanged.connect(self.toggle_override)
        box_override.addWidget(self.override_checkbox)

        box_override.addWidget(QLabel("Threshold Override"))
        self.override_mode_group = QButtonGroup()
        self.override_delta_radio = QRadioButton("deltaMAD Override")
        self.override_absolute_radio = QRadioButton("Absolute Threshold (a.u.)")
        self.override_mode_group.addButton(self.override_delta_radio)
        self.override_mode_group.addButton(self.override_absolute_radio)
        self.override_delta_radio.setChecked(True)
        box_override.addWidget(self.override_delta_radio)
        box_override.addWidget(self.override_absolute_radio)

        delta_override_layout = QHBoxLayout()
        delta_override_layout.addWidget(QLabel("deltaMAD:"))
        self.delta_override_input = QDoubleSpinBox()
        self.delta_override_input.setRange(0, 100)
        self.delta_override_input.setValue(3.0)
        self.delta_override_input.setSingleStep(0.1)
        delta_override_layout.addWidget(self.delta_override_input)
        box_override.addLayout(delta_override_layout)

        absolute_override_layout = QHBoxLayout()
        absolute_override_layout.addWidget(QLabel("Absolute Threshold:"))
        self.absolute_override_input = QDoubleSpinBox()
        self.absolute_override_input.setRange(0, 10000)
        self.absolute_override_input.setValue(5)
        self.absolute_override_input.setSingleStep(1)
        absolute_override_layout.addWidget(self.absolute_override_input)
        box_override.addLayout(absolute_override_layout)

        box_override.addWidget(QLabel("Minimum Dwell Override"))
        self.override_min_dwell_checkbox = QCheckBox("Override Minimum Dwell Time")
        box_override.addWidget(self.override_min_dwell_checkbox)

        min_dwell_override_layout = QHBoxLayout()
        min_dwell_override_layout.addWidget(QLabel("Minimum Dwell Time (s):"))
        self.min_dwell_override_input = QDoubleSpinBox()
        self.min_dwell_override_input.setRange(0, 100)
        self.min_dwell_override_input.setValue(0.5)
        self.min_dwell_override_input.setSingleStep(0.1)
        min_dwell_override_layout.addWidget(self.min_dwell_override_input)
        box_override.addLayout(min_dwell_override_layout)

        box_override.addWidget(QLabel("Merge Gap Override"))
        self.override_merge_gap_checkbox = QCheckBox("Override Merge Gap Between Adjacent Events")
        box_override.addWidget(self.override_merge_gap_checkbox)

        merge_gap_override_layout = QHBoxLayout()
        merge_gap_override_layout.addWidget(QLabel("Merge Gap (s):"))
        self.merge_gap_override_input = QDoubleSpinBox()
        self.merge_gap_override_input.setRange(0, 10)
        self.merge_gap_override_input.setValue(0.2)
        self.merge_gap_override_input.setSingleStep(0.1)
        merge_gap_override_layout.addWidget(self.merge_gap_override_input)
        box_override.addLayout(merge_gap_override_layout)

        apply_override_btn = QPushButton("Apply to Current ID")
        apply_override_btn.clicked.connect(self.apply_override)
        box_override.addWidget(apply_override_btn)

        clear_override_btn = QPushButton("Clear Current ID Override")
        clear_override_btn.clicked.connect(self.clear_override)
        box_override.addWidget(clear_override_btn)

        recover_manual_btn = QPushButton("Restore Manually Deleted Events")
        recover_manual_btn.clicked.connect(self.restore_manual_deletions_current_id)
        box_override.addWidget(recover_manual_btn)

        redetect_current_btn = QPushButton("Redetect Current ID with Current Parameters")
        redetect_current_btn.clicked.connect(self.redetect_current_id)
        box_override.addWidget(redetect_current_btn)

        selection_btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_events)
        selection_btn_layout.addWidget(select_all_btn)

        invert_selection_btn = QPushButton("Invert Selection")
        invert_selection_btn.clicked.connect(self.invert_selection)
        selection_btn_layout.addWidget(invert_selection_btn)
        box_override.addLayout(selection_btn_layout)

        delete_selected_btn = QPushButton("Delete Selected Events")
        delete_selected_btn.clicked.connect(self.delete_selected_events)
        box_override.addWidget(delete_selected_btn)

        self.selection_label = QLabel("Selected: 0 events")
        box_override.addWidget(self.selection_label)

        modifier_key = "Command" if self.system == "Darwin" else "Ctrl"
        hint_label = QLabel(
            f"Hold {modifier_key} and click for multi-select. Hold Shift for range selection. Drag to box-select; "
            f"in manual mode, dragging performs the active manual action"
        )
        hint_label.setStyleSheet("color: gray; font-style: italic;")
        hint_label.setWordWrap(True)
        box_override.addWidget(hint_label)

        self.override_box = box_override
        right_layout.addWidget(box_override)

        right_layout.addStretch()
        main_layout.addWidget(self.right_scroll, 28)

        self.update_classification_buttons()
        self._update_stats_summary()
        self._update_classification_history_buttons()
        self._refresh_recent_files_ui()
        self.set_event_detection_controls_enabled(False)
        self.set_override_controls_enabled(False)
        self._sync_display_mode_controls()
        self._apply_default_sidebar_layout()
        self.save_session_btn.setEnabled(False)

    def _apply_default_sidebar_layout(self):
        default_layout = (
            ("data_loading_box", True),
            ("navigation_box", False),
            ("classification_box", True),
            ("channel_display_box", False),
            ("alex_correction_box", False),
            ("smoothing_box", False),
            ("event_detection_box", True),
            ("override_box", False),
        )
        for attr_name, expanded in default_layout:
            box = getattr(self, attr_name, None)
            if box is not None:
                box.set_expanded(expanded)

    def _setup_shortcuts(self):
        self.prev_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.prev_shortcut.activated.connect(self.prev_id)

        self.next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.next_shortcut.activated.connect(self.next_id)

        self.classification_shortcuts = []
        for class_num in range(1, 10):
            shortcut = QShortcut(QKeySequence(str(class_num)), self.canvas_main)
            shortcut.activated.connect(lambda c=class_num: self._assign_classification_shortcut(c))
            self.classification_shortcuts.append(shortcut)

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_last_manual_action)

        self.undo_classification_shortcut = QShortcut(QKeySequence("Ctrl+Alt+Z"), self)
        self.undo_classification_shortcut.activated.connect(self.undo_classification)

        self.redo_classification_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.redo_classification_shortcut.activated.connect(self.redo_classification)

        self.shortcut_a = QShortcut(QKeySequence("A"), self)
        self.shortcut_a.activated.connect(lambda: self.nudge_selected_event_boundary("left", -1))

        self.shortcut_d = QShortcut(QKeySequence("D"), self)
        self.shortcut_d.activated.connect(lambda: self.nudge_selected_event_boundary("left", +1))

        self.shortcut_j = QShortcut(QKeySequence("J"), self)
        self.shortcut_j.activated.connect(lambda: self.nudge_selected_event_boundary("right", -1))

        self.shortcut_l = QShortcut(QKeySequence("L"), self)
        self.shortcut_l.activated.connect(lambda: self.nudge_selected_event_boundary("right", +1))

    def _assign_classification_shortcut(self, class_num):
        if class_num in self.state.classifications:
            self.assign_classification(class_num)

    def _show_transient_status(self, message: str, duration_ms: int = 200):
        self.statusBar().showMessage(message)
        self._status_clear_timer.stop()
        self._status_clear_timer.start(max(1, int(duration_ms)))

    def _clear_transient_status(self):
        self.statusBar().clearMessage()

    def _display_mode_is_fret(self):
        return self._current_display_mode_from_radios() == self.DISPLAY_MODE_FRET

    def _current_display_mode_from_radios(self):
        for mode, button in getattr(self, "display_mode_buttons", {}).items():
            if button.isChecked():
                return mode
        return self.DISPLAY_MODE_STANDARD

    def _set_display_mode_radio(self, mode):
        buttons = list(getattr(self, "display_mode_buttons", {}).values())
        previous_states = [(button, button.blockSignals(True)) for button in buttons]
        try:
            button = self.display_mode_buttons.get(mode)
            if button is not None:
                button.setChecked(True)
        finally:
            for button, previous_state in previous_states:
                button.blockSignals(previous_state)
        self._display_mode = mode

    def _sync_display_mode_controls(self):
        is_alex = self.data_model.is_alex_profile()
        current_mode = self._current_display_mode_from_radios()
        if is_alex and current_mode not in {
            self.DISPLAY_MODE_ALEX_RAW,
            self.DISPLAY_MODE_ALEX_COMPACT,
            self.DISPLAY_MODE_ALEX_DERIVED,
        }:
            current_mode = self.DISPLAY_MODE_ALEX_RAW
            self._set_display_mode_radio(current_mode)
        elif not is_alex and current_mode in {
            self.DISPLAY_MODE_ALEX_RAW,
            self.DISPLAY_MODE_ALEX_COMPACT,
            self.DISPLAY_MODE_ALEX_DERIVED,
        }:
            current_mode = self.DISPLAY_MODE_STANDARD
            self._set_display_mode_radio(current_mode)
        else:
            self._display_mode = current_mode

        available = set(self.data_model.get_available_channels())
        for mode, button in self.display_mode_buttons.items():
            if mode in {self.DISPLAY_MODE_ALEX_RAW, self.DISPLAY_MODE_ALEX_COMPACT, self.DISPLAY_MODE_ALEX_DERIVED}:
                button.setEnabled(is_alex)
            else:
                button.setEnabled(not is_alex)

        has_standard_fret_inputs = {'532', '638'}.issubset(available)
        self.channel_fret_checkbox.setEnabled(current_mode == self.DISPLAY_MODE_FRET and has_standard_fret_inputs)
        if current_mode != self.DISPLAY_MODE_FRET:
            self.channel_fret_checkbox.setToolTip("Enable FRET display mode to show the FRET lane.")
        elif not has_standard_fret_inputs:
            self.channel_fret_checkbox.setToolTip("FRET requires both 532nm and 638nm channels.")
        else:
            self.channel_fret_checkbox.setToolTip("Show the display-only FRET lane computed from 532nm and 638nm.")

        alex_mode = current_mode in {
            self.DISPLAY_MODE_ALEX_RAW,
            self.DISPLAY_MODE_ALEX_COMPACT,
            self.DISPLAY_MODE_ALEX_DERIVED,
        }
        for channel, checkbox in self.raw_channel_checkboxes.items():
            is_available = channel in available
            checkbox.setEnabled(is_available)
            if channel in ALEX_LOGICAL_CHANNELS:
                checkbox.setVisible(is_alex)
            else:
                checkbox.setVisible(not is_alex)

        for lane, checkbox in self.derived_channel_checkboxes.items():
            if lane == self.FRET_LANE:
                checkbox.setVisible(not is_alex)
                continue
            checkbox.setVisible(is_alex)
            checkbox.setEnabled(alex_mode and current_mode == self.DISPLAY_MODE_ALEX_DERIVED)

        if hasattr(self, "alex_correction_box"):
            self.alex_correction_box.setVisible(is_alex)
            self.alex_correction_box.setEnabled(is_alex)

        visible_smoothing = set(available)
        if is_alex:
            visible_smoothing.update({'ALEX_E_532EX', 'ALEX_R_488EX'})
            visible_smoothing.update(ALEX_CORRECTION_DISPLAY_LANES)
        else:
            visible_smoothing.add(self.FRET_LANE)
        for curve_key, widgets in self.smoothing_controls.items():
            should_show = curve_key in visible_smoothing
            for widget in widgets:
                widget.setVisible(should_show)


    def on_display_mode_changed(self, checked=False):
        sender = self.sender()
        if isinstance(sender, QRadioButton) and not checked:
            return

        new_mode = self._current_display_mode_from_radios()
        mode_changed = new_mode != self._display_mode
        self._display_mode = new_mode
        self._sync_display_mode_controls()

        if mode_changed:
            self._reset_plot_state_for_layout_change()
            self.update_plots(preserve_view=False, reset_home=True)

    def _add_smoothing_controls(self, container, curve_key, label):
        row = QHBoxLayout()

        checkbox = QCheckBox(f"Smooth {label}")
        checkbox.setChecked(False)

        window_input = QSpinBox()
        window_input.setRange(1, 501)
        window_input.setSingleStep(2)
        window_input.setValue(self.SMOOTH_DEFAULT_WINDOW)
        window_input.setSuffix(" frames")
        window_input.setToolTip(
            "Savitzky-Golay display smoothing window. Even values are shown with the next odd window."
        )

        if savgol_filter is None:
            tooltip = "SciPy is unavailable; display smoothing is disabled."
            checkbox.setEnabled(False)
            window_input.setEnabled(False)
            checkbox.setToolTip(tooltip)
            window_input.setToolTip(tooltip)
            checkbox.setStatusTip(tooltip)
            window_input.setStatusTip(tooltip)
        else:
            tooltip = "Display only; does not change raw data, FRET calculation, detection, or export."
            checkbox.setToolTip(tooltip)
            checkbox.setStatusTip(tooltip)
            window_input.setStatusTip(tooltip)

        checkbox.stateChanged.connect(lambda _state, key=curve_key: self._on_smoothing_control_changed(key))
        window_input.valueChanged.connect(lambda _value, key=curve_key: self._on_smoothing_control_changed(key))

        self.smoothing_checkboxes[curve_key] = checkbox
        self.smoothing_window_inputs[curve_key] = window_input
        self.smoothing_controls[curve_key] = (checkbox, window_input)

        row.addWidget(checkbox)
        row.addWidget(window_input)
        container.addLayout(row)

    def _on_smoothing_control_changed(self, curve_key):
        if savgol_filter is None:
            self._show_transient_status("SciPy is unavailable; display smoothing is disabled.", duration_ms=1500)
            return
        self.update_plots(preserve_view=True)

    def _smoothing_enabled(self, curve_key):
        checkbox = self.smoothing_checkboxes.get(curve_key)
        return bool(checkbox and checkbox.isEnabled() and checkbox.isChecked() and savgol_filter is not None)

    def _smoothing_window(self, curve_key):
        window_input = self.smoothing_window_inputs.get(curve_key)
        if window_input is None:
            return 1
        window = int(window_input.value())
        if window % 2 == 0:
            window += 1
        return window

    def _smooth_display_signal(self, curve_key, signal):
        if signal is None:
            return None

        values = np.asarray(signal, dtype=float)
        if values.size == 0 or not self._smoothing_enabled(curve_key):
            return values

        finite_mask = np.isfinite(values)
        if finite_mask.sum() != values.size:
            return values

        window = self._smoothing_window(curve_key)
        if (
            window < 3
            or window <= self.SMOOTH_POLYORDER
            or window > values.size
        ):
            return values

        try:
            return savgol_filter(
                values,
                window_length=window,
                polyorder=self.SMOOTH_POLYORDER,
                mode='interp'
            )
        except Exception:
            return values

    def _cancel_pending_navigation(self):
        self._pending_navigation_delta = 0
        if self._navigation_timer.isActive():
            self._navigation_timer.stop()

    def _queue_navigation(self, delta):
        if not self.state.filtered_id_list:
            self._cancel_pending_navigation()
            return

        if delta > 0:
            current_id = self.state.get_current_id()
            if current_id is not None and current_id not in self.state.id_classifications:
                self._show_transient_status("Current ID is still unclassified.", duration_ms=1200)

        max_index = len(self.state.filtered_id_list) - 1
        current_index = self.state.current_id_index
        requested_index = current_index + self._pending_navigation_delta + int(delta)
        target_index = max(0, min(max_index, requested_index))
        self._pending_navigation_delta = target_index - current_index

        if self._pending_navigation_delta == 0:
            if self._navigation_timer.isActive():
                self._navigation_timer.stop()
            return

        self._navigation_timer.start(self.NAVIGATION_DEBOUNCE_MS)

    def _flush_pending_navigation(self):
        if not self.state.filtered_id_list:
            self._cancel_pending_navigation()
            return

        delta = self._pending_navigation_delta
        self._pending_navigation_delta = 0
        if delta == 0:
            return

        max_index = len(self.state.filtered_id_list) - 1
        target_index = max(0, min(max_index, self.state.current_id_index + delta))
        if target_index == self.state.current_id_index:
            return

        self.state.current_id_index = target_index
        self.update_current_id_display(reset_view=True)

    def _update_selection_label(self):
        count = len(self.state.selected_event_keys)
        noun = "event" if count == 1 else "events"
        self.selection_label.setText(f"Selected: {count} {noun}")

    def _set_recent_classification_feedback(self, message, roi_id=None, class_num=None):
        del roi_id
        del class_num
        self._show_transient_status(message, duration_ms=2500)

    def _clear_recent_classification_feedback(self):
        pass

    def _toolbar_mode_active(self):
        return bool(str(getattr(self.toolbar, "mode", "")).strip())

    def _selectors_should_be_active(self):
        return (
            self.event_detection_enabled
            and not self._toolbar_mode_active()
            and self.manual_draw_mode in ("add_event", "mark_dual")
        )

    def _on_toolbar_mode_changed(self, _mode):
        self._set_selectors_active(self._selectors_should_be_active())
        if self._toolbar_mode_active():
            self._hide_crosshair(draw=False)

    def _on_toolbar_view_changed(self):
        self._cache_current_view_state()
        self._hide_crosshair(draw=False)
        self.canvas_main.draw_idle()

    def _cache_current_view_state(self, roi_id=None, force=False):
        if self._suspend_view_state_cache and not force:
            return

        target_id = self.state.get_current_id() if roi_id is None else roi_id
        if target_id is None:
            return

        view_state = self._capture_current_view_limits()
        if not view_state:
            return

        self._roi_view_state_cache[target_id] = {
            "channels": tuple(self.visible_channels),
            "view": view_state,
        }

    def _get_cached_view_state(self, roi_id):
        cached = self._roi_view_state_cache.get(roi_id)
        if not cached:
            return None
        if tuple(self.visible_channels) != tuple(cached.get("channels", ())):
            return None
        return cached.get("view")

    def _on_axis_xlim_changed(self, changed_ax):
        if self._syncing_xlim:
            return
        if changed_ax not in self.channel_axes.values():
            return

        sibling_axes = [ax for ax in self.channel_axes.values() if ax is not changed_ax]
        shared_axes = changed_ax.get_shared_x_axes()
        needs_manual_sync = any(
            not shared_axes.joined(changed_ax, ax)
            for ax in sibling_axes
        )

        if needs_manual_sync:
            self._syncing_xlim = True
            try:
                xlim = changed_ax.get_xlim()
                for ax in sibling_axes:
                    ax.set_xlim(xlim, emit=False)
            finally:
                self._syncing_xlim = False

        self._cache_current_view_state()

    def _ensure_axes_layout(self, visible_channels):
        signature = tuple(visible_channels)
        if signature != self._axes_layout_signature:
            self._rebuild_axes_layout(visible_channels)
            return

        self.event_patch_map = {}
        self.visible_channels = list(visible_channels)
        self._crosshair_lines = {}

        for ch in visible_channels:
            ax = self.channel_axes[ch]
            ax.clear()
            callback_id = getattr(ax, "_colocal_xlim_callback_id", None)
            if callback_id is not None:
                try:
                    ax.callbacks.disconnect(callback_id)
                except Exception:
                    pass
            ax._colocal_xlim_callback_id = ax.callbacks.connect("xlim_changed", self._on_axis_xlim_changed)

    def _rebuild_axes_layout(self, visible_channels):
        self.fig_main.clear()
        self.channel_axes = {}
        self.event_patch_map = {}
        self._crosshair_lines = {}
        self.visible_channels = list(visible_channels)
        self._axes_layout_signature = tuple(visible_channels)

        n_axes = len(visible_channels)
        shared_x_axis = None
        for index, channel in enumerate(visible_channels, start=1):
            subplot_kwargs = {"sharex": shared_x_axis} if shared_x_axis is not None else {}
            ax = self.fig_main.add_subplot(n_axes, 1, index, **subplot_kwargs)
            ax._colocal_xlim_callback_id = ax.callbacks.connect("xlim_changed", self._on_axis_xlim_changed)
            self.channel_axes[channel] = ax
            if shared_x_axis is None:
                shared_x_axis = ax

        self.fig_main.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.96, hspace=0.18)

    def _reset_plot_state_for_layout_change(self):
        self._clear_selectors()
        self._hide_crosshair(draw=False)
        self._plot_pan_state = None
        self._roi_view_state_cache.clear()
        self.event_patch_map = {}
        self.channel_axes = {}
        self.visible_channels = []
        self._axes_layout_signature = None
        self._crosshair_lines = {}
        self._crosshair_pending_x = None
        self._last_crosshair_x = None
        self._suspend_view_state_cache = False

        self.fig_main.clear()
        try:
            self.toolbar._nav_stack.clear()
            self.toolbar.set_history_buttons()
        except Exception:
            pass
        self.canvas_main.draw_idle()

    def _reset_plot_state_for_new_data(self):
        self._reset_plot_state_for_layout_change()
        self._last_displayed_id = None

    def _hide_crosshair(self, draw=True):
        self._crosshair_update_timer.stop()
        self._crosshair_pending_x = None
        self._last_crosshair_x = None
        updated = False
        for line in self._crosshair_lines.values():
            if line.get_visible():
                line.set_visible(False)
                updated = True
        if draw and updated:
            self.canvas_main.draw_idle()

    def _queue_crosshair_update(self, xdata):
        if xdata is None:
            self._crosshair_pending_x = None
        else:
            self._crosshair_pending_x = float(xdata)

        if not self._crosshair_update_timer.isActive():
            self._crosshair_update_timer.start(16)

    def _flush_crosshair_update(self):
        if not self._crosshair_lines:
            self._crosshair_pending_x = None
            self._last_crosshair_x = None
            return

        if self._crosshair_pending_x is None:
            updated = False
            for line in self._crosshair_lines.values():
                if line.get_visible():
                    line.set_visible(False)
                    updated = True
            self._last_crosshair_x = None
            if updated:
                self.canvas_main.draw_idle()
            return

        all_visible = all(line.get_visible() for line in self._crosshair_lines.values())
        if all_visible and self._last_crosshair_x is not None and abs(self._last_crosshair_x - self._crosshair_pending_x) < 1e-9:
            return

        for line in self._crosshair_lines.values():
            line.set_xdata([self._crosshair_pending_x, self._crosshair_pending_x])
            line.set_visible(True)
        self._last_crosshair_x = self._crosshair_pending_x
        self.canvas_main.draw_idle()

    def on_main_plot_motion(self, event):
        if self._plot_pan_state is not None:
            self._update_plot_pan(event)
            return
        if self._toolbar_mode_active():
            self._hide_crosshair()
            return
        if event.inaxes not in self.channel_axes.values() or event.xdata is None:
            self._queue_crosshair_update(None)
            return

        self._queue_crosshair_update(event.xdata)

    def on_main_plot_leave(self, _event):
        self._finish_plot_pan()
        self._hide_crosshair()

    def _get_current_time_bounds(self):
        current_id = self.state.get_current_id()
        df_id = self.data_model.get_roi_df(current_id)
        if df_id is None or df_id.empty:
            return None

        time_values = df_id["Time_sec"].values
        if len(time_values) == 0:
            return None
        return float(time_values[0]), float(time_values[-1]), len(time_values)

    def _clamp_xlim_to_time_bounds(self, left, right, bounds):
        min_time, max_time, n_points = bounds
        total_span = max(max_time - min_time, 1e-9)
        min_span = max(total_span / max(n_points, 1), 1e-6)
        span = max(float(right - left), min_span)

        if span >= total_span:
            return min_time, max_time

        if left < min_time:
            right += min_time - left
            left = min_time
        if right > max_time:
            left -= right - max_time
            right = max_time

        left = max(left, min_time)
        right = min(right, max_time)
        if right - left < min_span:
            center = min(max((left + right) / 2.0, min_time), max_time)
            left = center - min_span / 2.0
            right = center + min_span / 2.0
            if left < min_time:
                right += min_time - left
                left = min_time
            if right > max_time:
                left -= right - max_time
                right = max_time
        return left, right

    def _set_synced_xlim(self, left, right):
        first_axis = next(iter(self.channel_axes.values()), None)
        if first_axis is None:
            return

        first_axis.set_xlim(left, right)
        for ax in self.channel_axes.values():
            if ax is not first_axis:
                ax.set_xlim(left, right, emit=False)

    def _reset_current_roi_view(self):
        current_id = self.state.get_current_id()
        if current_id is not None:
            self._roi_view_state_cache.pop(current_id, None)
        self._plot_pan_state = None
        self.update_plots(preserve_view=False, reset_home=True)

    def _start_plot_pan(self, event):
        if event.inaxes not in self.channel_axes.values() or event.xdata is None:
            return
        bounds = self._get_current_time_bounds()
        if bounds is None:
            return
        left, right = event.inaxes.get_xlim()
        self._plot_pan_state = {
            "axis": event.inaxes,
            "start_x": float(event.x),
            "start_xlim": (float(left), float(right)),
            "bounds": bounds,
            "moved": False,
        }
        self._hide_crosshair(draw=False)

    def _update_plot_pan(self, event):
        state = self._plot_pan_state
        if state is None or event.x is None:
            return

        axis = state["axis"]
        bbox_width = max(float(axis.bbox.width), 1.0)
        start_left, start_right = state["start_xlim"]
        span = max(start_right - start_left, 1e-9)
        data_delta = (float(event.x) - state["start_x"]) * span / bbox_width

        new_left = start_left - data_delta
        new_right = start_right - data_delta
        new_left, new_right = self._clamp_xlim_to_time_bounds(new_left, new_right, state["bounds"])
        self._set_synced_xlim(new_left, new_right)

        if abs(float(event.x) - state["start_x"]) > 2:
            state["moved"] = True
        self._cache_current_view_state()
        self.canvas_main.draw_idle()

    def _finish_plot_pan(self):
        state = self._plot_pan_state
        self._plot_pan_state = None
        if not state or not state.get("moved"):
            return
        self._cache_current_view_state()
        try:
            self.toolbar.push_current()
        except Exception:
            pass
        self.canvas_main.draw_idle()

    def on_main_plot_release(self, event):
        if event.button == 1:
            self._finish_plot_pan()

    def on_main_plot_scroll(self, event):
        if event.inaxes not in self.channel_axes.values():
            return
        if event.xdata is None or self._toolbar_mode_active():
            return

        bounds = self._get_current_time_bounds()
        if bounds is None:
            return

        current_left, current_right = event.inaxes.get_xlim()
        step = getattr(event, "step", None)
        if step is None:
            step = 1 if event.button == "up" else -1
        scale_factor = self.X_SCROLL_ZOOM_BASE ** float(step)

        new_left = event.xdata - (event.xdata - current_left) * scale_factor
        new_right = event.xdata + (current_right - event.xdata) * scale_factor
        new_left, new_right = self._clamp_xlim_to_time_bounds(new_left, new_right, bounds)

        self._set_synced_xlim(new_left, new_right)
        self._cache_current_view_state()
        try:
            self.toolbar.push_current()
        except Exception:
            pass
        self.canvas_main.draw_idle()

    def _settings_bool(self, key, default=False):
        value = self.settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _get_recent_files(self):
        value = self.settings.value("recent_files", [])
        if isinstance(value, str):
            value = [value] if value else []
        recent_files = []
        for path in value or []:
            normalized = os.path.normpath(str(path))
            if normalized not in recent_files:
                recent_files.append(normalized)
        return recent_files[:self.max_recent_files]

    def _save_recent_files(self, recent_files):
        self.settings.setValue("recent_files", list(recent_files[:self.max_recent_files]))
        self.settings.sync()

    def _recent_file_display_label(self, path):
        filename = os.path.basename(path)
        if len(filename) <= 32:
            return filename
        return f"{filename[:16]}...{filename[-13:]}"

    def _update_recent_files_tooltip(self):
        selected_path = self.recent_files_combo.currentData()
        self.recent_files_combo.setToolTip(selected_path if selected_path else "")

    def _refresh_recent_files_ui(self):
        current_path = self.recent_files_combo.currentData()
        recent_files = self._get_recent_files()

        self.recent_files_combo.blockSignals(True)
        self.recent_files_combo.clear()
        for path in recent_files:
            label = self._recent_file_display_label(path)
            self.recent_files_combo.addItem(label, path)

        if current_path:
            index = self.recent_files_combo.findData(current_path)
            if index >= 0:
                self.recent_files_combo.setCurrentIndex(index)
        self.recent_files_combo.blockSignals(False)
        self._update_recent_files_tooltip()
        self.open_recent_btn.setEnabled(self.recent_files_combo.count() > 0)

    def _persist_auto_advance_setting(self, checked):
        self.settings.setValue("auto_advance_enabled", bool(checked))
        self.settings.sync()

    def _load_persisted_settings(self):
        self.auto_advance_checkbox.blockSignals(True)
        self.auto_advance_checkbox.setChecked(self._settings_bool("auto_advance_enabled", True))
        self.auto_advance_checkbox.blockSignals(False)
        self._refresh_recent_files_ui()

    def _get_default_open_directory(self):
        last_open_dir = self.settings.value("last_open_dir", "", type=str)
        if last_open_dir and os.path.isdir(last_open_dir):
            return last_open_dir
        if self.default_directory and os.path.isdir(self.default_directory):
            return self.default_directory
        return os.getcwd()

    def _get_default_export_directory(self):
        if self.data_model.filepath:
            csv_dir = os.path.dirname(self.data_model.filepath)
            if csv_dir and os.path.isdir(csv_dir):
                return csv_dir

        last_export_dir = self.settings.value("last_export_dir", "", type=str)
        if last_export_dir and os.path.isdir(last_export_dir):
            return last_export_dir

        last_open_dir = self.settings.value("last_open_dir", "", type=str)
        if last_open_dir and os.path.isdir(last_open_dir):
            return last_open_dir

        if self.default_directory and os.path.isdir(self.default_directory):
            return self.default_directory

        return os.getcwd()

    def _remember_loaded_file(self, filepath):
        filepath = os.path.normpath(filepath)
        directory = os.path.dirname(filepath)

        self.settings.setValue("last_open_dir", directory)
        self.settings.setValue("last_open_file", filepath)

        recent_files = [path for path in self._get_recent_files() if path != filepath]
        recent_files.insert(0, filepath)
        self._save_recent_files(recent_files)
        self._refresh_recent_files_ui()

    def _remember_export_directory(self, directory):
        self.settings.setValue("last_export_dir", os.path.normpath(directory))
        self.settings.sync()

    def _update_stats_summary(self):
        total = len(self.state.full_id_list)
        classified = len(self.state.id_classifications)
        unclassified = max(total - classified, 0)

        parts = [
            f"Total IDs: {total}",
            f"Classified: {classified}",
            f"Unclassified: {unclassified}",
        ]
        for class_num in self.state.classifications:
            count = sum(1 for value in self.state.id_classifications.values() if value == class_num)
            parts.append(f"Class {class_num}: {count}")

        self.stats_label.setText(" | ".join(parts))
        self.total_label.setText(f"Visible IDs: {len(self.state.filtered_id_list)}")
        self.save_session_btn.setEnabled(bool(self.data_model.filepath))

    def _update_classification_history_buttons(self):
        self.undo_classification_btn.setEnabled(self.state.can_undo_classification())
        self.redo_classification_btn.setEnabled(self.state.can_redo_classification())

    def _session_sidecar_path(self, csv_path=None):
        source_path = csv_path or self.data_model.filepath
        if not source_path:
            return None
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        return os.path.join(os.path.dirname(source_path), f"{base_name}.colocal.session.json")

    def _build_session_payload(self):
        if not self.data_model.filepath:
            return None

        payload = self.state.serialize_session_payload()
        payload.update({
            "source_csv_path": os.path.normpath(self.data_model.filepath),
            "source_csv_mtime": os.path.getmtime(self.data_model.filepath),
            "source_signal_signature": self.data_model.signal_signature(),
            "alex_correction_params": self.get_alex_correction_params(),
            "auto_advance_enabled": self.auto_advance_checkbox.isChecked(),
        })
        return payload

    def _load_session_payload(self, csv_path):
        sidecar_path = self._session_sidecar_path(csv_path)
        if not sidecar_path or not os.path.exists(sidecar_path):
            return None

        try:
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            logger.exception("Failed to read session sidecar")
            QMessageBox.warning(self, "Session Warning", f"Failed to read session file:\n{sidecar_path}")
            return None

        expected_path = os.path.normpath(csv_path)
        stored_path = os.path.normpath(str(payload.get("source_csv_path", "")))
        stored_mtime = payload.get("source_csv_mtime")
        current_mtime = os.path.getmtime(csv_path)

        if stored_path != expected_path:
            return None
        if stored_mtime is None or abs(float(stored_mtime) - float(current_mtime)) > 1e-6:
            return None

        stored_signature = payload.get("source_signal_signature")
        current_signature = self.data_model.signal_signature()
        if stored_signature:
            if stored_signature != current_signature:
                return None
        elif self.data_model.is_alex_profile():
            return None

        return payload

    def _sync_filter_controls_from_state(self):
        self.filter_all_radio.blockSignals(True)
        self.filter_unclassified_radio.blockSignals(True)
        self.filter_class_radio.blockSignals(True)
        self.filter_class_combo.blockSignals(True)

        if self.state.navigation_filter_mode == "unclassified":
            self.filter_unclassified_radio.setChecked(True)
        elif self.state.navigation_filter_mode == "class":
            self.filter_class_radio.setChecked(True)
            if self.state.navigation_filter_class is not None:
                class_text = str(self.state.navigation_filter_class)
                index = self.filter_class_combo.findText(class_text)
                if index >= 0:
                    self.filter_class_combo.setCurrentIndex(index)
        else:
            self.filter_all_radio.setChecked(True)

        self.filter_all_radio.blockSignals(False)
        self.filter_unclassified_radio.blockSignals(False)
        self.filter_class_radio.blockSignals(False)
        self.filter_class_combo.blockSignals(False)

    def save_session(self, show_feedback=True):
        payload = self._build_session_payload()
        if payload is None:
            QMessageBox.warning(self, "Warning", "Load a CSV file before saving a session.")
            return False

        sidecar_path = self._session_sidecar_path()
        try:
            with open(sidecar_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self.state.mark_clean()
            if show_feedback:
                QMessageBox.information(self, "Session Saved", f"Session saved to:\n{sidecar_path}")
            return True
        except Exception as exc:
            logger.exception("Failed to save session")
            QMessageBox.critical(self, "Error", f"Failed to save session: {exc}")
            return False

    def _prompt_to_restore_session(self, payload):
        message_box = QMessageBox(self)
        message_box.setWindowTitle("Restore Session")
        message_box.setText("A saved session was found for this CSV file.")
        message_box.setInformativeText("Do you want to restore the saved session state?")
        restore_button = message_box.addButton("Restore Session", QMessageBox.AcceptRole)
        message_box.addButton("Ignore", QMessageBox.RejectRole)
        message_box.setDefaultButton(restore_button)
        message_box.exec_()
        return message_box.clickedButton() == restore_button

    def _restore_session_payload(self, payload):
        self.state.restore_session_payload(payload)
        self.update_classification_buttons()

        correction = payload.get("alex_correction_params") or {}
        if correction and hasattr(self, "alex_alpha_input"):
            widgets = [self.alex_alpha_input, self.alex_beta_input, self.alex_gamma_input]
            previous_states = [(widget, widget.blockSignals(True)) for widget in widgets]
            try:
                self.alex_alpha_input.setValue(float(correction.get("alpha", ALEX_ALPHA_DEFAULT)))
                self.alex_beta_input.setValue(float(correction.get("beta", ALEX_BETA_DEFAULT)))
                self.alex_gamma_input.setValue(float(correction.get("gamma", ALEX_GAMMA_DEFAULT)))
                self._beta_source = str(correction.get("beta_source", "manual/default"))
            finally:
                for widget, previous_state in previous_states:
                    widget.blockSignals(previous_state)

        self.auto_advance_checkbox.blockSignals(True)
        self.auto_advance_checkbox.setChecked(bool(payload.get("auto_advance_enabled", True)))
        self.auto_advance_checkbox.blockSignals(False)
        self._persist_auto_advance_setting(self.auto_advance_checkbox.isChecked())

        self._sync_filter_controls_from_state()
        self._update_stats_summary()
        self._update_classification_history_buttons()

    def _prompt_to_handle_unsaved_session(self):
        if not self.data_model.filepath or not self.state.is_dirty:
            return True

        message_box = QMessageBox(self)
        message_box.setWindowTitle("Unsaved Session")
        message_box.setText("The current session has unsaved changes.")
        message_box.setInformativeText("Do you want to save the session before continuing?")
        save_button = message_box.addButton("Save Session", QMessageBox.AcceptRole)
        discard_button = message_box.addButton("Discard", QMessageBox.DestructiveRole)
        cancel_button = message_box.addButton("Cancel", QMessageBox.RejectRole)
        message_box.setDefaultButton(save_button)
        message_box.exec_()

        clicked = message_box.clickedButton()
        if clicked == save_button:
            return self.save_session(show_feedback=False)
        if clicked == discard_button:
            return True
        return clicked != cancel_button

    def _set_busy_state(self, busy, task_name=""):
        self.centralWidget().setEnabled(not busy)
        self._current_task_name = task_name if busy else ""

    def _cleanup_worker(self):
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None

        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread = None

        self._worker = None
        self._set_busy_state(False)

    def _start_recalculation_task(self, show_completion_message=False, allow_cancel=True):
        if not self.state.full_id_list or self._worker_thread is not None:
            return

        params = self.get_detection_params()
        overrides = {
            roi_id: dict(self.state.get_override(roi_id))
            for roi_id in self.state.full_id_list
        }

        self._task_cancelled = False
        self._set_busy_state(True, "Recalculate Events")

        worker = RecalculateWorker(self.detector, self.state.full_id_list, params, overrides)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        progress = QProgressDialog("Recalculating events...", "Cancel" if allow_cancel else "", 0, len(self.state.full_id_list), self)
        progress.setWindowTitle("Recalculate Events")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        if allow_cancel:
            progress.canceled.connect(worker.cancel)
        else:
            progress.setCancelButton(None)

        worker.progress_changed.connect(lambda value, text: (progress.setValue(value), progress.setLabelText(text)))

        def finish_handler(results, cancelled):
            for roi_id, events in results.items():
                self.state.set_auto_events(roi_id, events)

            self.state.clear_event_selection()
            self._update_selection_label()
            self.update_current_id_display(reset_view=False)
            self._cleanup_worker()

            if cancelled:
                QMessageBox.information(
                    self,
                    "Recalculation Interrupted",
                    "Recalculation was cancelled. Finished IDs were updated and remaining IDs kept their previous results."
                )
            elif show_completion_message:
                QMessageBox.information(self, "Done", "Recalculated events for all IDs.")

        def error_handler(message):
            self._cleanup_worker()
            QMessageBox.critical(self, "Error", f"Recalculation failed: {message}")

        worker.finished.connect(finish_handler)
        worker.error.connect(error_handler)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)

        self._worker = worker
        self._worker_thread = thread
        self._progress_dialog = progress
        thread.start()
        progress.show()

    def _start_export_task(self, save_dir, base_filename, export_jobs):
        if self._worker_thread is not None:
            return

        display_events_by_id = {
            roi_id: [event.to_dict() for event in self.state.get_display_events(roi_id)]
            for job in export_jobs
            for roi_id in job["ids"]
        }
        override_by_id = {
            roi_id: dict(self.state.get_override(roi_id))
            for job in export_jobs
            for roi_id in job["ids"]
        }

        worker = ExportWorker(
            df=self.data_model.df.copy(),
            signal_columns=dict(self.data_model.loaded.original_signal_columns),
            signal_channel_order=self.data_model.get_available_channels(),
            profile=self.data_model.profile,
            source_filepath=self.data_model.filepath,
            save_dir=save_dir,
            base_filename=base_filename,
            export_jobs=export_jobs,
            event_detection_enabled=self.event_detection_enabled,
            display_events_by_id=display_events_by_id,
            override_by_id=override_by_id,
            params=self.get_detection_params(),
            alex_correction_params=self.get_alex_correction_params(),
        )

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        progress = QProgressDialog("Exporting files...", None, 0, worker.total_steps, self)
        progress.setWindowTitle("Export Classes")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setCancelButton(None)

        worker.progress_changed.connect(lambda value, text: (progress.setValue(value), progress.setLabelText(text)))

        def finish_handler(exported_count):
            self._cleanup_worker()
            self._remember_export_directory(save_dir)
            if exported_count > 0:
                QMessageBox.information(self, "Success", f"Successfully exported {exported_count} classes to:\n{save_dir}")
            else:
                QMessageBox.information(self, "Notice", "No classes were selected for export.")

        def error_handler(message):
            self._cleanup_worker()
            QMessageBox.critical(self, "Error", f"Export failed: {message}")

        worker.finished.connect(finish_handler)
        worker.error.connect(error_handler)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)

        self._set_busy_state(True, "Export Classes")
        self._worker = worker
        self._worker_thread = thread
        self._progress_dialog = progress
        thread.start()
        progress.show()

    def _apply_navigation_filter_state(self):
        if self.filter_all_radio.isChecked():
            self.state.navigation_filter_mode = "all"
            self.state.navigation_filter_class = None
        elif self.filter_unclassified_radio.isChecked():
            self.state.navigation_filter_mode = "unclassified"
            self.state.navigation_filter_class = None
        else:
            self.state.navigation_filter_mode = "class"
            txt = self.filter_class_combo.currentText().strip()
            self.state.navigation_filter_class = int(txt) if txt.isdigit() else None

        self.state.apply_navigation_filter()

    def _resolve_exact_roi_id(self, raw_text):
        text = str(raw_text).strip()
        if not text:
            return None

        for roi_id in self.state.full_id_list:
            if str(roi_id) == text:
                return roi_id

        return None

    def _advance_after_classification(self, previous_id, previous_index):
        filtered_ids = self.state.filtered_id_list
        if not filtered_ids:
            self.state.current_id_index = 0
            return False

        if previous_id in filtered_ids:
            current_index = filtered_ids.index(previous_id)
            target_index = current_index + 1
            if target_index < len(filtered_ids):
                self.state.current_id_index = target_index
                return True

            self.state.current_id_index = current_index
            return False

        fallback_index = min(max(0, previous_index), len(filtered_ids) - 1)
        self.state.current_id_index = fallback_index
        return True

    def _show_classification_boundary_status(self):
        self._show_transient_status("Reached the last visible ID.", duration_ms=1500)

    def get_detection_params(self) -> DetectionParams:
        mode = "dual" if self.mode_dual_radio.isChecked() else "single"

        primary = self.primary_channel_combo.currentText() if self.primary_channel_combo.count() > 0 else "532"
        auxiliary = self.aux_channel_combo.currentText() if self.aux_channel_combo.count() > 0 else None
        if mode == "single":
            auxiliary = None

        return DetectionParams(
            inj_time=self.inj_time_input.value(),
            min_dwell=self.min_dwell_input.value(),
            delta_mad=self.delta_mad_input.value(),
            pen_mad=self.pen_mad_input.value(),
            merge_gap=self.merge_gap_input.value(),
            hysteresis_ratio=0.85,
            mode=mode,
            primary_channel=primary,
            auxiliary_channel=auxiliary,
            aux_delta_mad=self.aux_delta_mad_input.value(),
            overlap_ratio_threshold=0.0,
            alex_alpha=self.alex_alpha_input.value() if hasattr(self, "alex_alpha_input") else ALEX_ALPHA_DEFAULT,
            alex_beta=self.alex_beta_input.value() if hasattr(self, "alex_beta_input") else ALEX_BETA_DEFAULT,
            alex_gamma=self.alex_gamma_input.value() if hasattr(self, "alex_gamma_input") else ALEX_GAMMA_DEFAULT,
        )

    def update_global_params(self):
        if self.event_detection_enabled and self.data_model.id_list:
            self.update_plots(preserve_view=True)

    def get_alex_correction_params(self):
        return {
            "alpha": self.alex_alpha_input.value() if hasattr(self, "alex_alpha_input") else ALEX_ALPHA_DEFAULT,
            "beta": self.alex_beta_input.value() if hasattr(self, "alex_beta_input") else ALEX_BETA_DEFAULT,
            "gamma": self.alex_gamma_input.value() if hasattr(self, "alex_gamma_input") else ALEX_GAMMA_DEFAULT,
            "beta_source": self._beta_source,
        }

    def get_alex_correction_numeric_params(self):
        params = self.get_alex_correction_params()
        return {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "gamma": params["gamma"],
        }

    def on_alex_correction_changed(self):
        if self.sender() is getattr(self, "alex_beta_input", None):
            self._beta_source = "manual"
        if self.data_model.is_alex_profile():
            if self.event_detection_enabled and self.state.full_id_list:
                self.calculate_all_events(show_completion_message=False, allow_cancel=True)
            self.update_current_id_display(reset_view=False)

    def estimate_alex_beta(self):
        if not self.data_model.is_alex_profile():
            self.beta_estimate_label.setText("Estimated beta: load ALEX data first.")
            return

        estimate = self.data_model.estimate_alex_beta(alpha=self.alex_alpha_input.value())
        if not estimate:
            self._last_beta_estimate = None
            self.apply_beta_estimate_btn.setEnabled(False)
            self.beta_estimate_label.setText("Estimated beta: unavailable for this dataset.")
            return

        self._last_beta_estimate = estimate
        self.apply_beta_estimate_btn.setEnabled(True)
        self.beta_estimate_label.setText(
            f"Estimated beta: {estimate['beta']:.4f} "
            f"(q={estimate['quantile']:.2f}, n={estimate['count']})"
        )

    def apply_beta_estimate(self):
        if not self._last_beta_estimate:
            return
        self.alex_beta_input.setValue(float(self._last_beta_estimate["beta"]))
        self._beta_source = (
            f"estimated q={self._last_beta_estimate['quantile']:.2f}; "
            f"n={self._last_beta_estimate['count']}"
        )
        self.on_alex_correction_changed()

    def _auto_apply_alex_beta_estimate(self):
        if not self.data_model.is_alex_profile():
            return

        estimate = self.data_model.estimate_alex_beta(alpha=self.alex_alpha_input.value())
        if not estimate:
            self._last_beta_estimate = None
            self.apply_beta_estimate_btn.setEnabled(False)
            self._beta_source = "auto-estimate unavailable; default beta"
            self.beta_estimate_label.setText("Estimated beta: unavailable; beta kept at default.")
            return

        self._last_beta_estimate = estimate
        previous_state = self.alex_beta_input.blockSignals(True)
        try:
            self.alex_beta_input.setValue(float(estimate["beta"]))
        finally:
            self.alex_beta_input.blockSignals(previous_state)
        self._beta_source = f"auto-estimated q={estimate['quantile']:.2f}; n={estimate['count']}"
        self.apply_beta_estimate_btn.setEnabled(False)
        self.beta_estimate_label.setText(
            f"Auto-applied beta: {estimate['beta']:.4f} "
            f"(q={estimate['quantile']:.2f}, n={estimate['count']})"
        )

    def _reset_alex_correction_defaults(self):
        widgets = (
            (self.alex_alpha_input, ALEX_ALPHA_DEFAULT),
            (self.alex_beta_input, ALEX_BETA_DEFAULT),
            (self.alex_gamma_input, ALEX_GAMMA_DEFAULT),
        )
        previous_states = [(widget, widget.blockSignals(True)) for widget, _value in widgets]
        try:
            for widget, value in widgets:
                widget.setValue(float(value))
        finally:
            for widget, previous_state in previous_states:
                widget.blockSignals(previous_state)
        self._beta_source = "manual/default"

    def _apply_default_alex_display_selection(self):
        if not self.data_model.is_alex_profile():
            return

        desired = {
            "532ex_532": True,
            "532ex_638": True,
            "488ex_532": True,
            "488ex_488": True,
            "ALEX_E_532EX": True,
            "ALEX_R_488EX": False,
            ALEX_CY3_TOTAL_LANE: False,
            ALEX_S_CORR_LANE: False,
            ALEX_AF488_INDEX_LANE: True,
        }
        checkboxes = {}
        checkboxes.update(self.raw_channel_checkboxes)
        checkboxes.update(self.derived_channel_checkboxes)
        previous_states = [
            (checkbox, checkbox.blockSignals(True))
            for checkbox in checkboxes.values()
        ]
        try:
            for lane, checked in desired.items():
                checkbox = checkboxes.get(lane)
                if checkbox is not None:
                    checkbox.setChecked(bool(checked))
        finally:
            for checkbox, previous_state in previous_states:
                checkbox.blockSignals(previous_state)

    def _set_manual_mode(self, mode):
        self.manual_draw_mode = mode
        self.add_event_btn.setChecked(mode == "add_event")
        self.mark_bound_btn.setChecked(mode == "mark_dual")

        if mode == "add_event":
            self.manual_mode_label.setText("Manual mode: Add primary-channel event")
        elif mode == "mark_dual":
            self.manual_mode_label.setText("Manual mode: Mark as dual supported")
        else:
            self.manual_mode_label.setText("Manual mode: None")

        self._set_selectors_active(self._selectors_should_be_active())

    def activate_add_event_mode(self):
        if self.add_event_btn.isChecked():
            self._set_manual_mode("add_event")
        else:
            self._set_manual_mode(None)

    def activate_mark_bound_mode(self):
        if self.mark_bound_btn.isChecked():
            self._set_manual_mode("mark_dual")
        else:
            self._set_manual_mode(None)

    def on_canvas_key_press(self, event):
        if event.key in ('control', 'cmd', 'super', 'meta', 'command'):
            self.multiselect_modifier_active = True
        elif event.key == 'shift':
            self.shift_modifier_active = True

    def on_canvas_key_release(self, event):
        if event.key in ('control', 'cmd', 'super', 'meta', 'command'):
            self.multiselect_modifier_active = False
        elif event.key == 'shift':
            self.shift_modifier_active = False

    def keyPressEvent(self, event):
        key = event.key()

        if Qt.Key_1 <= key <= Qt.Key_9:
            class_num = key - Qt.Key_0
            if class_num in self.state.classifications:
                self.assign_classification(class_num)
                event.accept()
                return

        if key == Qt.Key_Shift:
            self.shift_modifier_active = True
        elif key in (Qt.Key_Control, Qt.Key_Meta):
            self.multiselect_modifier_active = True

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()
        if key == Qt.Key_Shift:
            self.shift_modifier_active = False
        elif key in (Qt.Key_Control, Qt.Key_Meta):
            self.multiselect_modifier_active = False
        super().keyReleaseEvent(event)

    def set_event_detection_controls_enabled(self, enabled):
        widgets = [
            self.inj_time_input, self.min_dwell_input, self.delta_mad_input,
            self.pen_mad_input, self.merge_gap_input, self.primary_channel_combo,
            self.aux_channel_combo, self.aux_delta_mad_input, self.recalc_all_btn,
            self.mode_single_radio, self.mode_dual_radio,
            self.add_event_btn, self.mark_bound_btn
        ]
        for w in widgets:
            w.setEnabled(enabled)

    def set_override_controls_enabled(self, enabled):
        def walk_layout(layout):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget():
                    widget = item.widget()
                    if widget is not self.override_checkbox:
                        widget.setEnabled(enabled)
                elif item.layout():
                    walk_layout(item.layout())

        walk_layout(self.override_box.content_layout)
        self.override_checkbox.setEnabled(enabled)
        self.toggle_override(self.override_checkbox.checkState())

    def load_csv(self):
        if self._worker_thread is not None:
            QMessageBox.information(self, "Busy", f"Please wait until {self._current_task_name} finishes.")
            return
        if not self._prompt_to_handle_unsaved_session():
            return

        initial_dir = self._get_default_open_directory()
        filepath, _ = QFileDialog.getOpenFileName(self, "Select CSV File", initial_dir, "CSV Files (*.csv)")
        if not filepath:
            return

        self.load_csv_file(filepath)

    def open_recent_file(self):
        filepath = self.recent_files_combo.currentData()
        if not filepath:
            return
        if not os.path.exists(filepath):
            recent_files = [path for path in self._get_recent_files() if path != filepath]
            self._save_recent_files(recent_files)
            self._refresh_recent_files_ui()
            QMessageBox.warning(self, "Missing File", f"The recent file does not exist anymore:\n{filepath}")
            return
        if not self._prompt_to_handle_unsaved_session():
            return

        self.load_csv_file(filepath)

    def load_csv_file(self, filepath):
        filepath = os.path.normpath(filepath)
        self._cancel_pending_navigation()

        try:
            loaded = self.data_model.load_csv(filepath)
            self.state.reset_for_new_data(loaded.id_list)
            self._reset_plot_state_for_new_data()
            self._remember_loaded_file(filepath)
            self._clear_recent_classification_feedback()
            self.update_classification_buttons()
            self._sync_filter_controls_from_state()

            self.file_label.setText(f"Loaded: {os.path.basename(filepath)}")
            self.jump_id_input.clear()

            available = self.data_model.get_available_channels()
            self.populate_channel_combos(available)

            display_checkboxes = {}
            display_checkboxes.update(self.raw_channel_checkboxes)
            display_checkboxes.update(self.derived_channel_checkboxes)
            previous_signal_states = [
                (checkbox, checkbox.blockSignals(True))
                for checkbox in display_checkboxes.values()
            ]
            try:
                available_set = set(available)
                lane_availability = {
                    lane: lane in available_set
                    for lane in self.raw_channel_checkboxes
                }
                lane_availability[self.FRET_LANE] = {'532', '638'}.issubset(available_set)
                lane_availability['ALEX_E_532EX'] = {'532ex_532', '532ex_638'}.issubset(available_set)
                lane_availability['ALEX_R_488EX'] = {'488ex_488', '488ex_532'}.issubset(available_set)
                for lane in ALEX_CORRECTION_DISPLAY_LANES:
                    lane_availability[lane] = set(ALEX_LOGICAL_CHANNELS).issubset(available_set)
                for lane, checkbox in display_checkboxes.items():
                    is_available = lane_availability.get(lane, False)
                    checkbox.setEnabled(is_available)
                    checkbox.setChecked(is_available)
            finally:
                for checkbox, previous_state in previous_signal_states:
                    checkbox.blockSignals(previous_state)
            if self.data_model.is_alex_profile():
                self._last_beta_estimate = None
                self._reset_alex_correction_defaults()
                self.apply_beta_estimate_btn.setEnabled(False)
                self.beta_estimate_label.setText("Estimated beta: n/a")
                self._apply_default_alex_display_selection()
                self._auto_apply_alex_beta_estimate()
                self._set_display_mode_radio(self.DISPLAY_MODE_ALEX_DERIVED)
            else:
                self._set_display_mode_radio(self.DISPLAY_MODE_STANDARD)
            self._sync_display_mode_controls()

            restored_session = False
            payload = self._load_session_payload(filepath)
            if payload and self._prompt_to_restore_session(payload):
                self._restore_session_payload(payload)
                restored_session = True
            else:
                self.state.apply_navigation_filter()
                self.state.mark_clean()

            self.apply_navigation_filter()
            self.update_current_id_display(reset_view=True)
            self._update_stats_summary()
            self._update_classification_history_buttons()
            self._apply_default_sidebar_layout()

            if self.event_detection_enabled:
                self.calculate_all_events(show_completion_message=False, allow_cancel=True)

            QMessageBox.information(
                self, "Success",
                f"File loaded successfully!\n"
                f"Separator: {'Tab' if loaded.separator == chr(9) else 'Comma'}\n"
                f"Experiment type: {self.data_model.get_profile_label()}\n"
                f"ID count: {len(loaded.id_list)}"
            )

            if restored_session:
                self.state.mark_dirty()

        except Exception as e:
            logger.exception("Failed to load CSV")
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def populate_channel_combos(self, available):
        self.primary_channel_combo.blockSignals(True)
        self.aux_channel_combo.blockSignals(True)

        self.primary_channel_combo.clear()
        self.aux_channel_combo.clear()

        detection_channels = list(self.data_model.get_detection_channels()) or list(available)
        for ch in detection_channels:
            self.primary_channel_combo.addItem(ch)
            self.aux_channel_combo.addItem(ch)

        if ALEX_AF488_INDEX_LANE in detection_channels:
            self.primary_channel_combo.setCurrentText(ALEX_AF488_INDEX_LANE)
        elif '532ex_532' in detection_channels:
            self.primary_channel_combo.setCurrentText('532ex_532')
        elif '532' in detection_channels:
            self.primary_channel_combo.setCurrentText('532')
        elif detection_channels:
            self.primary_channel_combo.setCurrentIndex(0)

        if ALEX_AF488_INDEX_LANE in detection_channels and '532ex_638' in detection_channels:
            self.primary_channel_combo.setCurrentText(ALEX_AF488_INDEX_LANE)
            self.aux_channel_combo.setCurrentText('532ex_638')
        elif '532ex_532' in detection_channels and '532ex_638' in detection_channels:
            self.primary_channel_combo.setCurrentText('532ex_532')
            self.aux_channel_combo.setCurrentText('532ex_638')
        elif '488' in detection_channels and '532' in detection_channels:
            self.primary_channel_combo.setCurrentText('488')
            self.aux_channel_combo.setCurrentText('532')
        elif len(detection_channels) >= 2:
            self.aux_channel_combo.setCurrentIndex(1 if self.primary_channel_combo.currentIndex() == 0 else 0)

        self.primary_channel_combo.blockSignals(False)
        self.aux_channel_combo.blockSignals(False)

    def toggle_event_detection(self, state):
        self.event_detection_enabled = (state == Qt.Checked)
        self.set_event_detection_controls_enabled(self.event_detection_enabled)
        self.set_override_controls_enabled(self.event_detection_enabled)
        self._set_selectors_active(self._selectors_should_be_active())

        if self.event_detection_enabled and self.state.full_id_list:
            self.calculate_all_events(show_completion_message=False, allow_cancel=True)

        self.update_current_id_display(reset_view=False)

    def on_detection_mode_changed(self):
        if self.event_detection_enabled and self.state.full_id_list:
            self.calculate_all_events(show_completion_message=False, allow_cancel=True)
            self.update_current_id_display(reset_view=False)

    def on_detection_channel_changed(self):
        if self.event_detection_enabled and self.state.full_id_list:
            self.calculate_all_events(show_completion_message=False, allow_cancel=True)
            self.update_current_id_display(reset_view=False)

    def calculate_all_events(self, show_completion_message=False, allow_cancel=True):
        if not self.state.full_id_list:
            return

        self._start_recalculation_task(
            show_completion_message=show_completion_message,
            allow_cancel=allow_cancel
        )
        logger.info("Started background recalculation for all ROIs")

    def redetect_current_id(self):
        if not self.event_detection_enabled:
            return
        roi_id = self.state.get_current_id()
        if roi_id is None:
            return

        params = self.get_detection_params()
        override = self.state.get_override(roi_id)
        events = self.detector.detect_events_for_roi(roi_id, params, override)
        self.state.set_auto_events(roi_id, events)

        self.state.clear_event_selection()
        self._update_selection_label()
        self.update_current_id_display(reset_view=False)

    def recalculate_all(self):
        if not self.state.full_id_list or not self.event_detection_enabled:
            return
        self.calculate_all_events(show_completion_message=True, allow_cancel=True)
    def toggle_override(self, state):
        enabled = (state == Qt.Checked) and self.event_detection_enabled
        widgets = [
            self.override_delta_radio,
            self.override_absolute_radio,
            self.delta_override_input,
            self.absolute_override_input,
            self.override_min_dwell_checkbox,
            self.min_dwell_override_input,
            self.override_merge_gap_checkbox,
            self.merge_gap_override_input,
        ]
        for w in widgets:
            w.setEnabled(enabled)

    def apply_override(self):
        if not self.state.full_id_list or not self.event_detection_enabled:
            return

        current_id = self.state.get_current_id()
        if current_id is None:
            return

        if not self.override_checkbox.isChecked():
            QMessageBox.information(self, "Notice", 'Please enable "Enable Current ID Override" first.')
            return

        override = {}

        if self.override_delta_radio.isChecked():
            override['threshold'] = {
                'mode': 'deltaMAD',
                'value': self.delta_override_input.value()
            }
        else:
            override['threshold'] = {
                'mode': 'absolute',
                'value': self.absolute_override_input.value()
            }

        if self.override_min_dwell_checkbox.isChecked():
            override['min_dwell'] = self.min_dwell_override_input.value()

        if self.override_merge_gap_checkbox.isChecked():
            override['merge_gap'] = self.merge_gap_override_input.value()

        self.state.set_override(current_id, override)
        self.redetect_current_id()
    def clear_override(self):
        if not self.state.full_id_list or not self.event_detection_enabled:
            return

        current_id = self.state.get_current_id()
        if current_id is None:
            return

        self.state.clear_override(current_id)
        self.override_checkbox.setChecked(False)
        self.override_delta_radio.setChecked(True)
        self.redetect_current_id()

    def restore_manual_deletions_current_id(self):
        if not self.event_detection_enabled:
            return

        current_id = self.state.get_current_id()
        if current_id is None:
            return

        self.state.restore_hidden_events(current_id)
        self.state.clear_event_selection()
        self._update_selection_label()
        self.update_current_id_display(reset_view=False)

    def apply_navigation_filter(self):
        self._cancel_pending_navigation()
        previous_id = self.state.get_current_id()
        self._apply_navigation_filter_state()
        if previous_id in self.state.filtered_id_list:
            self.state.current_id_index = self.state.filtered_id_list.index(previous_id)
        self._update_stats_summary()
        self.update_current_id_display(reset_view=True)

    def jump_to_id(self):
        self._cancel_pending_navigation()
        if not self.state.full_id_list:
            return
        target_text = self.jump_id_input.text().strip()
        if not target_text:
            return
        target = self._resolve_exact_roi_id(target_text)
        if target is None:
            QMessageBox.warning(self, "Notice", f'ID "{target_text}" was not found.')
            return
        if target in self.state.filtered_id_list:
            self.state.current_id_index = self.state.filtered_id_list.index(target)
            self.update_current_id_display(reset_view=True)
        else:
            message_box = QMessageBox(self)
            message_box.setWindowTitle("Hidden by Filter")
            message_box.setText(f'ID "{target_text}" exists but is hidden by the current filter.')
            message_box.setInformativeText("Do you want to switch to All IDs and jump to it?")
            jump_button = message_box.addButton("Show in All IDs and Jump", QMessageBox.AcceptRole)
            message_box.addButton("Cancel", QMessageBox.RejectRole)
            message_box.setDefaultButton(jump_button)
            message_box.exec_()
            if message_box.clickedButton() == jump_button:
                self.filter_all_radio.setChecked(True)
                self.apply_navigation_filter()
                if target in self.state.filtered_id_list:
                    self.state.current_id_index = self.state.filtered_id_list.index(target)
                    self.update_current_id_display(reset_view=True)

    def prev_id(self):
        self._queue_navigation(-1)

    def next_id(self):
        self._queue_navigation(1)

    def update_classification_buttons(self):
        while self.class_buttons_layout.count():
            child = self.class_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.class_button_map = {}
        selected_filter_text = self.filter_class_combo.currentText().strip()
        self.filter_class_combo.blockSignals(True)
        self.filter_class_combo.clear()
        current_id = self.state.get_current_id()
        current_class = self.state.id_classifications.get(current_id) if current_id is not None else None
        for class_num in self.state.classifications:
            count = sum(1 for c in self.state.id_classifications.values() if c == class_num)
            btn = QPushButton(f"Class {class_num} ({count}) [Shortcut: {class_num}]")
            btn.clicked.connect(lambda checked, c=class_num: self.assign_classification(c))
            if current_class == class_num:
                btn.setStyleSheet(self.CLASS_BTN_ACTIVE_STYLE)
            else:
                btn.setStyleSheet(self.CLASS_BTN_DEFAULT_STYLE)
            self.class_buttons_layout.addWidget(btn)
            self.class_button_map[class_num] = btn
            self.filter_class_combo.addItem(str(class_num))
        restore_text = selected_filter_text
        if not restore_text and self.state.navigation_filter_class is not None:
            restore_text = str(self.state.navigation_filter_class)
        restore_index = self.filter_class_combo.findText(restore_text) if restore_text else -1
        if restore_index >= 0:
            self.filter_class_combo.setCurrentIndex(restore_index)
        elif self.filter_class_combo.count() > 0:
            self.filter_class_combo.setCurrentIndex(0)
        self.filter_class_combo.blockSignals(False)
        self._update_classification_history_buttons()

    def assign_classification(self, class_num):
        self._cancel_pending_navigation()
        current_id = self.state.get_current_id()
        if current_id is None:
            return

        previous_index = self.state.current_id_index
        new_class = self.state.toggle_classification(current_id, class_num)
        self.update_classification_buttons()
        self._apply_navigation_filter_state()

        if new_class is not None and self.auto_advance_checkbox.isChecked():
            if self._advance_after_classification(current_id, previous_index):
                next_id = self.state.get_current_id()
                self._set_recent_classification_feedback(
                    f"Last action: ID {current_id} assigned to Class {new_class}. Auto-advanced to ID {next_id}.",
                    roi_id=current_id,
                    class_num=new_class,
                )
            else:
                self._show_classification_boundary_status()
                self._set_recent_classification_feedback(
                    f"Last action: ID {current_id} assigned to Class {new_class}. Reached the last visible ID.",
                    roi_id=current_id,
                    class_num=new_class,
                )
        elif new_class is not None:
            self._set_recent_classification_feedback(
                f"Last action: ID {current_id} assigned to Class {new_class}.",
                roi_id=current_id,
                class_num=new_class,
            )
        elif current_id in self.state.filtered_id_list:
            self.state.current_id_index = self.state.filtered_id_list.index(current_id)
            self._set_recent_classification_feedback(
                f"Last action: ID {current_id} was unclassified.",
                roi_id=current_id,
                class_num=None,
            )
        else:
            self._set_recent_classification_feedback(
                f"Last action: ID {current_id} was unclassified.",
                roi_id=current_id,
                class_num=None,
            )

        self._update_stats_summary()
        self._update_classification_history_buttons()
        self.update_current_id_display(reset_view=False)

    def undo_classification(self):
        self._cancel_pending_navigation()
        current_id = self.state.get_current_id()
        if not self.state.undo_classification():
            QMessageBox.information(self, "Notice", "No classification action is available to undo.")
            return

        self._clear_recent_classification_feedback()
        self._set_recent_classification_feedback("Last action: Classification undo applied.")
        self.update_classification_buttons()
        self._apply_navigation_filter_state()
        if current_id in self.state.filtered_id_list:
            self.state.current_id_index = self.state.filtered_id_list.index(current_id)
        self._update_stats_summary()
        self.update_current_id_display(reset_view=False)

    def redo_classification(self):
        self._cancel_pending_navigation()
        current_id = self.state.get_current_id()
        if not self.state.redo_classification():
            QMessageBox.information(self, "Notice", "No classification action is available to redo.")
            return

        self._set_recent_classification_feedback("Last action: Classification redo applied.")
        self.update_classification_buttons()
        self._apply_navigation_filter_state()
        if current_id in self.state.filtered_id_list:
            self.state.current_id_index = self.state.filtered_id_list.index(current_id)
        self._update_stats_summary()
        self.update_current_id_display(reset_view=False)

    def assign_current_filter_to_class(self):
        self._cancel_pending_navigation()
        if not self.state.filtered_id_list:
            QMessageBox.information(self, "Notice", "There are no visible IDs in the current filter.")
            return

        items = [str(class_num) for class_num in self.state.classifications]
        selected, ok = QInputDialog.getItem(self, "Assign Current Filter", "Choose a class:", items, 0, False)
        if not ok or not selected:
            return

        target_class = int(selected)
        changes = {
            roi_id: {
                "before": self.state.id_classifications.get(roi_id),
                "after": target_class,
            }
            for roi_id in self.state.filtered_id_list
        }
        if not self.state.apply_classification_changes(changes, record_history=True, label="bulk_assign_filter"):
            QMessageBox.information(self, "Notice", "All visible IDs are already assigned to that class.")
            return

        affected_count = sum(1 for change in changes.values() if change["before"] != change["after"])
        self._set_recent_classification_feedback(
            f"Last action: Assigned {affected_count} visible IDs to Class {target_class}.",
            class_num=target_class,
        )
        self.update_classification_buttons()
        self._apply_navigation_filter_state()
        self._update_stats_summary()
        self.update_current_id_display(reset_view=False)

    def clear_classifications_in_current_filter(self):
        self._cancel_pending_navigation()
        if not self.state.filtered_id_list:
            QMessageBox.information(self, "Notice", "There are no visible IDs in the current filter.")
            return

        changes = {
            roi_id: {
                "before": self.state.id_classifications.get(roi_id),
                "after": None,
            }
            for roi_id in self.state.filtered_id_list
        }
        if not self.state.apply_classification_changes(changes, record_history=True, label="bulk_clear_filter"):
            QMessageBox.information(self, "Notice", "There are no classifications to clear in the current filter.")
            return

        affected_count = sum(1 for change in changes.values() if change["before"] is not None)
        self._set_recent_classification_feedback(
            f"Last action: Cleared classifications for {affected_count} visible IDs."
        )
        self.update_classification_buttons()
        self._apply_navigation_filter_state()
        self._update_stats_summary()
        self.update_current_id_display(reset_view=False)

    def add_classification(self):
        try:
            new_class = self.state.add_classification()
            self._set_recent_classification_feedback(f"Last action: Added Class {new_class}.")
            self.update_classification_buttons()
            self._update_stats_summary()
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))

    def update_current_id_display(self, reset_view=True):
        current_id = self.state.get_current_id()
        if current_id is None:
            self._cache_current_view_state(self._last_displayed_id, force=True)
            self._last_displayed_id = None
            self.state.clear_event_selection()
            self._update_selection_label()
            self.nav_label.setText("Page 0/0")
            self.current_id_label.setText("Current ID: -")
            self.current_class_label.setText("Current class: Unclassified")
            self.jump_id_input.clear()
            self.fig_main.clear()
            self.channel_axes = {}
            self.visible_channels = []
            self._axes_layout_signature = None
            self._crosshair_lines = {}
            self.canvas_main.draw_idle()
            self.refresh_event_table([])
            self.update_classification_buttons()
            self._update_stats_summary()
            return

        previous_displayed_id = self._last_displayed_id
        if current_id != previous_displayed_id:
            self._cache_current_view_state(previous_displayed_id, force=True)
            self.state.clear_event_selection()
            self._update_selection_label()

        self.nav_label.setText(f"Page {self.state.current_id_index + 1}/{len(self.state.filtered_id_list)}")
        self.current_id_label.setText(f"Current ID: {current_id}")
        self.jump_id_input.blockSignals(True)
        self.jump_id_input.setText(str(current_id))
        self.jump_id_input.blockSignals(False)
        if current_id in self.state.id_classifications:
            class_num = self.state.id_classifications[current_id]
            self.current_class_label.setText(f"Current class: Class {class_num}")
        else:
            self.current_class_label.setText("Current class: Unclassified")
        override = self.state.get_override(current_id)
        if override:
            self.override_checkbox.setChecked(True)
            if 'threshold' in override:
                if override['threshold']['mode'] == 'deltaMAD':
                    self.override_delta_radio.setChecked(True)
                    self.delta_override_input.setValue(override['threshold']['value'])
                else:
                    self.override_absolute_radio.setChecked(True)
                    self.absolute_override_input.setValue(override['threshold']['value'])
            else:
                self.override_delta_radio.setChecked(True)
            if 'min_dwell' in override:
                self.override_min_dwell_checkbox.setChecked(True)
                self.min_dwell_override_input.setValue(override['min_dwell'])
            else:
                self.override_min_dwell_checkbox.setChecked(False)
                self.min_dwell_override_input.setValue(self.min_dwell_input.value())
            if 'merge_gap' in override:
                self.override_merge_gap_checkbox.setChecked(True)
                self.merge_gap_override_input.setValue(override['merge_gap'])
            else:
                self.override_merge_gap_checkbox.setChecked(False)
                self.merge_gap_override_input.setValue(self.merge_gap_input.value())
        else:
            self.override_checkbox.setChecked(False)
            self.override_delta_radio.setChecked(True)
            self.override_min_dwell_checkbox.setChecked(False)
            self.override_merge_gap_checkbox.setChecked(False)
            self.min_dwell_override_input.setValue(self.min_dwell_input.value())
            self.merge_gap_override_input.setValue(self.merge_gap_input.value())
        self.update_classification_buttons()
        self.update_plots(preserve_view=not reset_view, reset_home=reset_view)
        self._last_displayed_id = current_id

    def _event_source_text(self, ev):
        if getattr(ev, "source", "auto") == "auto":
            return "Auto"
        if getattr(ev, "source", "") == "manual_added":
            return "Manual Added"
        if getattr(ev, "source", "") == "manual_modified":
            return "Manual Modified"
        return getattr(ev, "source", "")

    def _event_manual_status_text(self, ev):
        status = getattr(ev, "manual_status", "")
        mapping = {
            "": "",
            "manual_added": "Manual Added",
            "manual_bound": "Manual Reclassified",
            "manual_boundary_modified": "Boundary Adjusted"
        }
        return mapping.get(status, status)

    def _get_roi_time_and_len(self, roi_id):
        df_id = self.data_model.get_roi_df(roi_id)
        if df_id is None or df_id.empty:
            return None, 0
        return df_id['Time_sec'].values, len(df_id)

    def _time_to_nearest_idx(self, time_array, x):
        if time_array is None or len(time_array) == 0:
            return None
        return int(min(range(len(time_array)), key=lambda i: abs(time_array[i] - x)))

    def _find_overlapping_visible_event(self, roi_id, start_idx, end_idx, exclude_event_key=None):
        events = self.state.get_display_events(roi_id)
        for ev in events:
            if exclude_event_key is not None and ev.event_key == exclude_event_key:
                continue
            if not (ev.end_idx < start_idx or ev.start_idx > end_idx):
                return ev
        return None

    def _replace_boundary_of_existing_event(self, roi_id, old_event: EventRecord, new_start_idx: int, new_end_idx: int):
        params = self.get_detection_params()
        new_ev = self.detector.recompute_event_metrics_for_roi(
            roi_id=roi_id,
            params=params,
            start_idx=new_start_idx,
            end_idx=new_end_idx,
            source="manual_modified" if getattr(old_event, "source", "auto") == "auto" else getattr(old_event, "source", "manual_modified"),
            manual_status="manual_boundary_modified",
            preserve_right_censored=getattr(old_event, "right_censored", False)
        )
        if new_ev is None:
            return False

        if getattr(old_event, "manual_status", "") == "manual_bound":
            new_ev = self.detector.force_mark_event_as_dual_supported(new_ev)
            new_ev.manual_status = "manual_bound"

        replaced = self.state.replace_event_by_key(roi_id, old_event.event_key, new_ev)
        if replaced:
            self.state.push_undo(roi_id, {
                "type": "modify_boundary",
                "old_event": old_event,
                "new_event": new_ev
            })
            self.state.selected_event_keys.discard(old_event.event_key)
            self.state.selected_event_keys.add(new_ev.event_key)
            self.state.last_selected_event_key = new_ev.event_key
        return replaced

    def _get_event_by_key(self, roi_id, event_key):
        for ev in self.state.get_display_events(roi_id):
            if ev.event_key == event_key:
                return ev
        return None

    def _get_single_selected_event(self, roi_id):
        if len(self.state.selected_event_keys) != 1:
            return None
        key = next(iter(self.state.selected_event_keys))
        return self._get_event_by_key(roi_id, key)

    def _can_adjust_selected_event_boundary(self):
        if not self.event_detection_enabled:
            return False
        current_id = self.state.get_current_id()
        if current_id is None:
            return False
        return len(self.state.selected_event_keys) == 1

    def _event_overlap_except_self(self, roi_id, start_idx, end_idx, self_event_key):
        return self._find_overlapping_visible_event(
            roi_id=roi_id,
            start_idx=start_idx,
            end_idx=end_idx,
            exclude_event_key=self_event_key
        )

    def _get_visible_channels(self, df_id):
        channels = []
        available = set(self.data_model.get_available_channels())
        if self.data_model.is_alex_profile():
            mode = self._current_display_mode_from_radios()
            if mode == self.DISPLAY_MODE_ALEX_RAW:
                for channel in ALEX_LOGICAL_CHANNELS:
                    checkbox = self.raw_channel_checkboxes.get(channel)
                    col = self.data_model.get_signal_column(channel)
                    if checkbox and checkbox.isChecked() and channel in available and col in df_id.columns:
                        channels.append(channel)
                return channels

            groups = [
                (self.ALEX_INTENSITY_532EX_LANE, ('532ex_532', '532ex_638')),
                (self.ALEX_INTENSITY_488EX_LANE, ('488ex_488', '488ex_532')),
            ]
            for lane, group_channels in groups:
                has_visible = False
                for channel in group_channels:
                    checkbox = self.raw_channel_checkboxes.get(channel)
                    col = self.data_model.get_signal_column(channel)
                    if checkbox and checkbox.isChecked() and channel in available and col in df_id.columns:
                        has_visible = True
                if has_visible:
                    channels.append(lane)

            if mode == self.DISPLAY_MODE_ALEX_DERIVED:
                for lane in (
                    'ALEX_E_532EX',
                    'ALEX_R_488EX',
                    ALEX_CY3_TOTAL_LANE,
                    ALEX_S_CORR_LANE,
                    ALEX_AF488_INDEX_LANE,
                ):
                    checkbox = self.derived_channel_checkboxes.get(lane)
                    if (
                        checkbox
                        and checkbox.isChecked()
                        and self.data_model.compute_derived_signal(
                            lane,
                            df_id,
                            **self.get_alex_correction_numeric_params()
                        ) is not None
                    ):
                        channels.append(lane)
            return channels

        if self.channel_488_checkbox.isChecked():
            col = self.data_model.get_signal_column('488')
            if col and col in df_id.columns:
                channels.append('488')

        if self._display_mode_is_fret():
            has_fret_intensity_channel = False
            for ch, checkbox in (
                ('532', self.channel_532_checkbox),
                ('638', self.channel_638_checkbox),
            ):
                col = self.data_model.get_signal_column(ch)
                if checkbox.isChecked() and col and col in df_id.columns:
                    has_fret_intensity_channel = True
            if has_fret_intensity_channel:
                channels.append(self.INTENSITY_LANE)

            if self.channel_fret_checkbox.isChecked():
                donor_col = self.data_model.get_signal_column('532')
                acceptor_col = self.data_model.get_signal_column('638')
                if (
                    donor_col and acceptor_col
                    and donor_col in df_id.columns
                    and acceptor_col in df_id.columns
                ):
                    channels.append(self.FRET_LANE)
        else:
            for ch, checkbox in (
                ('532', self.channel_532_checkbox),
                ('638', self.channel_638_checkbox),
            ):
                col = self.data_model.get_signal_column(ch)
                if checkbox.isChecked() and col and col in df_id.columns:
                    channels.append(ch)

        return channels

    def _clear_selectors(self):
        for selector in self.rect_selectors.values():
            try:
                selector.set_active(False)
            except Exception:
                pass
            try:
                selector.disconnect_events()
            except Exception:
                pass
        self.rect_selectors = {}

    def _set_selectors_active(self, active: bool):
        for selector in self.rect_selectors.values():
            try:
                selector.set_active(active)
            except Exception:
                pass

    def _build_selectors_for_visible_axes(self):
        self._clear_selectors()
        if not self.event_detection_enabled:
            return
        for ch, ax in self.channel_axes.items():
            selector = RectangleSelector(
                ax,
                self.on_rectangle_select,
                useblit=False,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                interactive=False
            )
            selector.set_active(self._selectors_should_be_active())
            self.rect_selectors[ch] = selector

    def _iter_all_event_patches(self):
        for patches in self.event_patch_map.values():
            for patch in patches:
                yield patch

    def _find_clicked_event_key_from_all_axes(self, event):
        for patch in self._iter_all_event_patches():
            try:
                contains, _ = patch.contains(event)
            except Exception:
                contains = False
            if contains and hasattr(patch, 'event_key'):
                return patch.event_key
        return None

    def _compute_channel_ylim(self, signal):
        if signal is None or len(signal) == 0:
            return (0, 1)

        finite = np.asarray(signal, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (0, 1)

        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if y_min == y_max:
            pad = 1.0 if y_min == 0 else abs(y_min) * 0.1
        else:
            pad = (y_max - y_min) * 0.08
        return y_min - pad, y_max + pad

    def _compute_multi_channel_ylim(self, signals):
        non_empty = [
            np.asarray(signal, dtype=float)
            for signal in signals
            if signal is not None and len(signal) > 0
        ]
        if not non_empty:
            return (0, 1)

        finite_signals = [signal[np.isfinite(signal)] for signal in non_empty]
        finite_signals = [signal for signal in finite_signals if signal.size > 0]
        if not finite_signals:
            return (0, 1)

        y_min = min(float(np.min(signal)) for signal in finite_signals)
        y_max = max(float(np.max(signal)) for signal in finite_signals)
        if y_min == y_max:
            pad = 1.0 if y_min == 0 else abs(y_min) * 0.1
        else:
            pad = (y_max - y_min) * 0.08
        return y_min - pad, y_max + pad

    def _compute_fret_ylim(self, signal):
        if signal is None or len(signal) == 0:
            return (-0.05, 1.05)

        finite = signal[np.isfinite(signal)]
        if finite.size == 0:
            return (-0.05, 1.05)

        y_min = min(0.0, float(np.min(finite)))
        y_max = max(1.0, float(np.max(finite)))
        if y_min == y_max:
            pad = 0.05
        else:
            pad = (y_max - y_min) * 0.05
        return y_min - pad, y_max + pad

    def _plot_display_lane(self, ax, lane, time, df_id):
        if lane in self.data_model.get_available_channels():
            col = self.data_model.get_signal_column(lane)
            raw_signal = df_id[col].values if col and col in df_id.columns else None
            signal = self._smooth_display_signal(lane, raw_signal)
            ax.plot(
                time,
                signal,
                color=self.data_model.get_channel_color(lane),
                label=self.data_model.get_channel_label(lane),
                linewidth=1.2,
                alpha=0.8
            )
            ax.set_ylabel(f'{self.data_model.get_channel_label(lane)}\nIntensity')
            ax.set_ylim(*self._compute_channel_ylim(signal))
            return {lane}

        if lane == self.INTENSITY_LANE:
            plotted_channels = set()
            signals = []
            for channel, checkbox in (
                ('532', self.channel_532_checkbox),
                ('638', self.channel_638_checkbox),
            ):
                col = self.data_model.get_signal_column(channel)
                if checkbox.isChecked() and col and col in df_id.columns:
                    raw_signal = df_id[col].values
                    signal = self._smooth_display_signal(channel, raw_signal)
                    signals.append(signal)
                    plotted_channels.add(channel)
                    ax.plot(
                        time,
                        signal,
                        color=self.DISPLAY_LANE_COLORS[channel],
                        label=f'{channel}nm',
                        linewidth=1.2,
                        alpha=0.8
                    )
            ax.set_ylabel('532/638\nIntensity')
            ax.set_ylim(*self._compute_multi_channel_ylim(signals))
            return plotted_channels

        if lane in {self.ALEX_INTENSITY_532EX_LANE, self.ALEX_INTENSITY_488EX_LANE}:
            group_channels = (
                ('532ex_532', '532ex_638')
                if lane == self.ALEX_INTENSITY_532EX_LANE
                else ('488ex_488', '488ex_532')
            )
            plotted_channels = set()
            signals = []
            for channel in group_channels:
                checkbox = self.raw_channel_checkboxes.get(channel)
                col = self.data_model.get_signal_column(channel)
                if checkbox and checkbox.isChecked() and col and col in df_id.columns:
                    raw_signal = df_id[col].values
                    signal = self._smooth_display_signal(channel, raw_signal)
                    signals.append(signal)
                    plotted_channels.add(channel)
                    ax.plot(
                        time,
                        signal,
                        color=self.data_model.get_channel_color(channel),
                        label=self.data_model.get_channel_label(channel),
                        linewidth=1.2,
                        alpha=0.8
                    )
            ylabel = '532ex\nIntensity' if lane == self.ALEX_INTENSITY_532EX_LANE else '488ex\nIntensity'
            ax.set_ylabel(ylabel)
            ax.set_ylim(*self._compute_multi_channel_ylim(signals))
            return plotted_channels

        if lane == self.FRET_LANE:
            raw_signal = self.data_model.compute_fret(df_id)
            signal = self._smooth_display_signal(self.FRET_LANE, raw_signal)
            if signal is None:
                return set()
            ax.plot(
                time,
                signal,
                color=self.DISPLAY_LANE_COLORS[self.FRET_LANE],
                label='FRET',
                linewidth=1.2,
                alpha=0.9
            )
            ax.set_ylabel('FRET')
            ax.set_ylim(*self._compute_fret_ylim(signal))
            return set()

        if lane in ALEX_DERIVED_LANES:
            raw_signal = self.data_model.compute_derived_signal(
                lane,
                df_id,
                **self.get_alex_correction_numeric_params()
            )
            signal = self._smooth_display_signal(lane, raw_signal)
            if signal is None:
                return set()
            spec = ALEX_DERIVED_LANES[lane]
            ax.plot(
                time,
                signal,
                color=spec.get('color', self.DISPLAY_LANE_COLORS.get(lane, 'black')),
                label=spec.get('label', lane),
                linewidth=1.2,
                alpha=0.9
            )
            ax.set_ylabel(spec.get('label', lane))
            if spec.get('kind') == 'ratio':
                ax.set_ylim(*self._compute_fret_ylim(signal))
            else:
                ax.set_ylim(*self._compute_channel_ylim(signal))
            return {lane}

        return set()

    def _display_lane_contains_channel(self, lane, channel):
        if lane in self.data_model.get_available_channels():
            return channel == lane
        if lane == self.INTENSITY_LANE:
            return channel in {'532', '638'}
        if lane == self.ALEX_INTENSITY_532EX_LANE:
            return channel in {'532ex_532', '532ex_638'}
        if lane == self.ALEX_INTENSITY_488EX_LANE:
            return channel in {'488ex_488', '488ex_532'}
        if lane in ALEX_DERIVED_LANES:
            return channel == lane
        return False

    def _format_overall_title(self, current_id, displayed_events, params):
        mode_text = "Single-channel" if params.mode == "single" else f"Dual-channel({params.primary_channel}+{params.auxiliary_channel})"
        candidate_count = sum(1 for e in displayed_events if e.classification == "candidate")
        supported_count = sum(1 for e in displayed_events if e.classification == "dual_supported")
        manual_count = sum(1 for e in displayed_events if getattr(e, "source", "auto") != "auto")

        override_info = ""
        override = self.state.get_override(current_id)
        if override:
            parts = []
            if 'threshold' in override:
                parts.append("threshold")
            if 'min_dwell' in override:
                parts.append("dwell")
            if 'merge_gap' in override:
                parts.append("merge")
            if parts:
                override_info = f" [override: {', '.join(parts)}]"

        return (
            f'ID {current_id} | Mode: {mode_text} | '
            f'Candidate events: {candidate_count} | Dual-supported events: {supported_count} | '
            f'Manual events/reclassified: {manual_count}{override_info}'
        )

    def _capture_current_view_limits(self):
        if not self.channel_axes:
            return None

        xlim = None
        ylims = {}
        for ch, ax in self.channel_axes.items():
            try:
                if xlim is None:
                    xlim = ax.get_xlim()
                ylims[ch] = ax.get_ylim()
            except Exception:
                pass

        if xlim is None:
            return None

        return {'xlim': xlim, 'ylims': ylims}

    def _restore_view_limits(self, view_state):
        if not view_state:
            return

        xlim = view_state.get('xlim')
        ylims = view_state.get('ylims', {})

        first_axis = next(iter(self.channel_axes.values()), None)
        if first_axis is not None and xlim is not None:
            try:
                first_axis.set_xlim(xlim)
            except Exception:
                pass

        for ch, ax in self.channel_axes.items():
            try:
                if ax is not first_axis and xlim is not None:
                    ax.set_xlim(xlim, emit=False)
                if ch in ylims:
                    ax.set_ylim(ylims[ch])
            except Exception:
                pass

    def update_plots(self, preserve_view=True, reset_home=False):
        current_id = self.state.get_current_id()
        if current_id is None:
            self.fig_main.clear()
            self.channel_axes = {}
            self.visible_channels = []
            self._axes_layout_signature = None
            self._crosshair_lines = {}
            self.canvas_main.draw_idle()
            self.refresh_event_table([])
            return

        df_id = self.data_model.get_roi_df(current_id)
        if df_id is None or df_id.empty:
            self.fig_main.clear()
            self.channel_axes = {}
            self.visible_channels = []
            self._axes_layout_signature = None
            self._crosshair_lines = {}
            self.canvas_main.draw_idle()
            self.refresh_event_table([])
            return

        self._hide_crosshair(draw=False)

        same_roi = current_id == self._last_displayed_id
        previous_view = self._capture_current_view_limits() if (preserve_view and same_roi) else None

        time = df_id['Time_sec'].values
        visible_channels = self._get_visible_channels(df_id)

        if not visible_channels:
            self.fig_main.clear()
            self.channel_axes = {}
            self.visible_channels = []
            self._axes_layout_signature = None
            self._crosshair_lines = {}
            ax = self.fig_main.add_subplot(111)
            ax.text(0.5, 0.5, "No channels are currently visible", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas_main.draw_idle()
            self.refresh_event_table([])
            self._clear_selectors()
            return

        displayed_events = []
        params = self.get_detection_params() if self.event_detection_enabled else None
        thresholds = self.detector.get_channel_thresholds(
            current_id, params, self.state.get_override(current_id)
        ) if self.event_detection_enabled else {}

        if self.event_detection_enabled:
            displayed_events = self.state.get_display_events(current_id)

        self._suspend_view_state_cache = True
        try:
            self._ensure_axes_layout(visible_channels)
            axes = [self.channel_axes[ch] for ch in visible_channels]

            for idx, ch in enumerate(visible_channels):
                ax = self.channel_axes[ch]
                lane_channels = self._plot_display_lane(ax, ch, time, df_id)
                ax.grid(True, alpha=0.3)

                if idx < len(visible_channels) - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel('Time (s)')

                if self.event_detection_enabled:
                    ylim = ax.get_ylim()

                    for ev in displayed_events:
                        if ev.event_key in self.state.selected_event_keys:
                            edgecolor = 'darkred'
                            linewidth = 2.0
                            alpha = 0.35
                        else:
                            edgecolor = 'black'
                            linewidth = 0.8
                            alpha = 0.22

                        facecolor = 'lime' if ev.classification == "dual_supported" else 'yellow'

                        if getattr(ev, "source", "auto") != "auto":
                            edgecolor = 'blue' if ev.event_key not in self.state.selected_event_keys else 'darkred'
                            linewidth = max(linewidth, 1.6)

                        rect = Rectangle(
                            (ev.start_time, ylim[0]),
                            max(ev.end_time - ev.start_time, 1e-9),
                            ylim[1] - ylim[0],
                            facecolor=facecolor,
                            alpha=alpha,
                            edgecolor=edgecolor,
                            linewidth=linewidth,
                            picker=True
                        )
                        rect.event_key = ev.event_key
                        rect.channel = ch
                        ax.add_patch(rect)
                        self.event_patch_map.setdefault(ev.event_key, []).append(rect)

                    ax.axvline(params.inj_time, color='green', linestyle='--', linewidth=1.0, label='Injection')

                    if (
                        'primary' in thresholds
                        and thresholds['primary']['channel'] in lane_channels
                        and self._display_lane_contains_channel(ch, thresholds['primary']['channel'])
                    ):
                        t = thresholds['primary']
                        ax.axhline(
                            t['baseline'],
                            color='gray',
                            linestyle='-',
                            linewidth=1.0,
                            label=f'Primary Baseline ({t["channel"]})'
                        )
                        ax.axhline(
                            t['high'],
                            color='magenta',
                            linestyle='--',
                            linewidth=1.0,
                            label=f'Primary Threshold ({t["channel"]})'
                        )
                        ax.axhline(
                            t['low'],
                            color='orange',
                            linestyle=':',
                            linewidth=1.0,
                            label=f'Primary Low ({t["channel"]})'
                        )

                    if (
                        'auxiliary' in thresholds
                        and thresholds['auxiliary']['channel'] in lane_channels
                        and self._display_lane_contains_channel(ch, thresholds['auxiliary']['channel'])
                    ):
                        t = thresholds['auxiliary']
                        ax.axhline(
                            t['high'],
                            color='cyan',
                            linestyle='--',
                            linewidth=1.0,
                            label=f'Aux Threshold ({t["channel"]})'
                        )

                legend = ax.legend(loc='upper right')
                if legend:
                    legend.set_zorder(10)

                crosshair_line = ax.axvline(
                    time[0] if len(time) > 0 else 0.0,
                    color='#555555',
                    linestyle=':',
                    linewidth=0.9,
                    alpha=0.6,
                    visible=False,
                    zorder=12
                )
                self._crosshair_lines[ch] = crosshair_line

            overall_title = f'ID {current_id}'
            if self.event_detection_enabled:
                overall_title = self._format_overall_title(current_id, displayed_events, params)

            axes[0].set_title(overall_title)
            self.fig_main.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.96, hspace=0.18)

            restore_view = None
            if not reset_home:
                restore_view = previous_view if previous_view else self._get_cached_view_state(current_id)
            if restore_view:
                self._restore_view_limits(restore_view)
            else:
                default_xlim = axes[0].get_xlim()
                for ax in axes[1:]:
                    ax.set_xlim(default_xlim, emit=False)
        finally:
            self._suspend_view_state_cache = False

        self._cache_current_view_state(current_id, force=True)

        self._build_selectors_for_visible_axes()
        self.canvas_main.draw_idle()

        if reset_home:
            try:
                self.toolbar.update()
                self.toolbar.push_current()
            except Exception:
                pass

        self.refresh_event_table(displayed_events)

    def refresh_event_table(self, events):
        valid_keys = {event.event_key for event in events}
        self.state.selected_event_keys.intersection_update(valid_keys)
        if self.state.last_selected_event_key not in self.state.selected_event_keys:
            self.state.last_selected_event_key = None
        self._update_selection_label()

        self._suppress_table_selection_signal = True
        self._suppress_table_item_changed_signal = True
        self.event_table.blockSignals(True)

        self.event_table.setRowCount(0)

        for row, ev in enumerate(events):
            self.event_table.insertRow(row)

            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if ev.event_key in self.state.selected_event_keys else Qt.Unchecked)
            chk_item.setData(Qt.UserRole, ev.event_key)
            self.event_table.setItem(row, 0, chk_item)

            self.event_table.setItem(row, 1, QTableWidgetItem("Dual Supported" if ev.classification == "dual_supported" else "Candidate"))
            self.event_table.setItem(row, 2, QTableWidgetItem(self._event_source_text(ev)))
            self.event_table.setItem(row, 3, QTableWidgetItem(f"{ev.start_time:.3f}"))
            self.event_table.setItem(row, 4, QTableWidgetItem(f"{ev.end_time:.3f}"))
            self.event_table.setItem(row, 5, QTableWidgetItem(f"{ev.dwell_time:.3f}"))
            self.event_table.setItem(row, 6, QTableWidgetItem("Yes" if ev.aux_supported else "No"))
            self.event_table.setItem(row, 7, QTableWidgetItem(f"{ev.aux_overlap_ratio:.3f}"))
            self.event_table.setItem(row, 8, QTableWidgetItem(self._event_manual_status_text(ev)))
            self.event_table.setItem(row, 9, QTableWidgetItem(str(ev.event_key)))

            if ev.event_key in self.state.selected_event_keys:
                self.event_table.selectRow(row)

        self.event_table.blockSignals(False)
        self._suppress_table_item_changed_signal = False
        self._suppress_table_selection_signal = False

    def on_event_table_item_changed(self, item):
        if self._suppress_table_item_changed_signal:
            return
        if item.column() != 0:
            return
        event_key = item.data(Qt.UserRole)
        if event_key is None:
            return
        if item.checkState() == Qt.Checked:
            self.state.selected_event_keys.add(event_key)
            self.state.last_selected_event_key = event_key
        else:
            self.state.selected_event_keys.discard(event_key)
            if self.state.last_selected_event_key == event_key:
                self.state.last_selected_event_key = None
        self._update_selection_label()
        self.update_plots(preserve_view=True)

    def on_event_table_selection_changed(self):
        if self._suppress_table_selection_signal:
            return
        selected_rows = {idx.row() for idx in self.event_table.selectedIndexes()}
        current_id = self.state.get_current_id()
        events = self.state.get_display_events(current_id) if current_id is not None else []
        if not selected_rows:
            self.state.selected_event_keys.clear()
            self.state.last_selected_event_key = None
            self._update_selection_label()
            self.update_plots(preserve_view=True)
            return
        selected_keys = set()
        for r in selected_rows:
            if 0 <= r < len(events):
                selected_keys.add(events[r].event_key)
        if self._is_multiselect_active():
            self.state.selected_event_keys.update(selected_keys)
        else:
            self.state.selected_event_keys = selected_keys
        if len(selected_keys) == 1:
            self.state.last_selected_event_key = next(iter(selected_keys))
        elif not self.state.selected_event_keys:
            self.state.last_selected_event_key = None
        self._update_selection_label()
        self.update_plots(preserve_view=True)

    def _is_multiselect_active(self):
        if self.multiselect_modifier_active:
            return True

        modifiers = QApplication.keyboardModifiers()
        if self.system == 'Darwin':
            return bool(modifiers & Qt.MetaModifier)
        return bool(modifiers & Qt.ControlModifier)

    def _is_shift_active(self):
        if self.shift_modifier_active:
            return True
        modifiers = QApplication.keyboardModifiers()
        return bool(modifiers & Qt.ShiftModifier)

    def on_rectangle_select(self, eclick, erelease):
        if not self.event_detection_enabled:
            return
        if self._toolbar_mode_active():
            return
        if eclick.xdata is None or erelease.xdata is None:
            return

        current_id = self.state.get_current_id()
        if current_id is None:
            return

        time_arr, n = self._get_roi_time_and_len(current_id)
        if time_arr is None or n == 0:
            return

        x1, x2 = sorted([eclick.xdata, erelease.xdata])

        if self.manual_draw_mode in ("add_event", "mark_dual"):
            start_idx = self._time_to_nearest_idx(time_arr, x1)
            end_idx = self._time_to_nearest_idx(time_arr, x2)
            if start_idx is None or end_idx is None:
                return
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx

            if start_idx == end_idx:
                QMessageBox.information(self, "Notice", "The selected range is too short. Cover at least two time points; click the peak directly for short events.")
                return

            if self.manual_draw_mode == "add_event":
                self.handle_manual_add_event(current_id, start_idx, end_idx)
            elif self.manual_draw_mode == "mark_dual":
                self.handle_manual_mark_dual(current_id, start_idx, end_idx)
            return

        events = self.state.get_display_events(current_id)
        matched = set()
        for ev in events:
            overlap = not (ev.end_time < x1 or ev.start_time > x2)
            if overlap:
                matched.add(ev.event_key)

        if self._is_multiselect_active():
            self.state.selected_event_keys.update(matched)
        else:
            self.state.selected_event_keys = matched

        if len(matched) == 1:
            self.state.last_selected_event_key = next(iter(matched))

        self._update_selection_label()
        self.update_plots(preserve_view=True)

    def handle_manual_add_event(self, roi_id, start_idx, end_idx):
        overlap_ev = self._find_overlapping_visible_event(roi_id, start_idx, end_idx)
        if overlap_ev is not None:
            reply = QMessageBox.question(
                self,
                "Overlapping Event Detected",
                f"The selected range overlaps with existing event {overlap_ev.event_key}.\nDo you want to update it to the new boundaries?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                ok = self._replace_boundary_of_existing_event(roi_id, overlap_ev, start_idx, end_idx)
                if ok:
                    self._update_selection_label()
                    self.update_current_id_display(reset_view=False)
                return
            else:
                return

        params = self.get_detection_params()
        new_ev = self.detector.recompute_event_metrics_for_roi(
            roi_id=roi_id,
            params=params,
            start_idx=start_idx,
            end_idx=end_idx,
            source="manual_added",
            manual_status="manual_added"
        )

        if new_ev is None:
            QMessageBox.warning(self, "Error", "Failed to add manual event.")
            return

        self.state.add_manual_event(roi_id, new_ev)
        self.state.push_undo(roi_id, {
            "type": "add_manual_event",
            "event": new_ev
        })
        self.state.selected_event_keys = {new_ev.event_key}
        self.state.last_selected_event_key = new_ev.event_key
        self._update_selection_label()
        self.update_current_id_display(reset_view=False)
    def handle_manual_mark_dual(self, roi_id, start_idx, end_idx):
        overlap_ev = self._find_overlapping_visible_event(roi_id, start_idx, end_idx)

        if overlap_ev is not None:
            old_copy = EventRecord(**overlap_ev.to_dict())
            self.detector.force_mark_event_as_dual_supported(overlap_ev)
            replaced = self.state.replace_event_by_key(roi_id, old_copy.event_key, overlap_ev)
            if replaced:
                self.state.push_undo(roi_id, {
                    "type": "mark_dual",
                    "old_event": old_copy,
                    "new_event": overlap_ev
                })
                self.state.selected_event_keys = {overlap_ev.event_key}
                self.state.last_selected_event_key = overlap_ev.event_key
                self._update_selection_label()
                self.update_current_id_display(reset_view=False)
                return

        params = self.get_detection_params()
        new_ev = self.detector.recompute_event_metrics_for_roi(
            roi_id=roi_id,
            params=params,
            start_idx=start_idx,
            end_idx=end_idx,
            source="manual_added",
            manual_status="manual_added"
        )
        if new_ev is None:
            QMessageBox.warning(self, "Error", "Failed to manually reclassify event.")
            return

        new_ev = self.detector.force_mark_event_as_dual_supported(new_ev)
        self.state.add_manual_event(roi_id, new_ev)
        self.state.push_undo(roi_id, {
            "type": "add_and_mark_dual",
            "event": new_ev
        })
        self.state.selected_event_keys = {new_ev.event_key}
        self.state.last_selected_event_key = new_ev.event_key
        self._update_selection_label()
        self.update_current_id_display(reset_view=False)
    def handle_manual_add_event_by_click(self, roi_id, clicked_idx):
        params = self.get_detection_params()
        override = self.state.get_override(roi_id)

        inferred = self.detector.infer_event_bounds_from_peak_click(
            roi_id=roi_id,
            params=params,
            clicked_idx=clicked_idx,
            override=override,
            min_points=self.manual_click_min_points
        )
        if inferred is None:
            QMessageBox.warning(self, "Error", "Could not infer event bounds from the clicked position.")
            return

        start_idx, end_idx = inferred
        self.handle_manual_add_event(roi_id, start_idx, end_idx)
    def nudge_selected_event_boundary(self, side, delta):
        if not self._can_adjust_selected_event_boundary():
            return

        current_id = self.state.get_current_id()
        ev = self._get_single_selected_event(current_id)
        if ev is None:
            return

        time_arr, n = self._get_roi_time_and_len(current_id)
        if time_arr is None or n == 0:
            return

        new_start = ev.start_idx
        new_end = ev.end_idx

        if side == "left":
            new_start = max(0, min(ev.start_idx + delta, ev.end_idx))
        elif side == "right":
            new_end = min(n - 1, max(ev.start_idx, ev.end_idx + delta))
        else:
            return

        if new_start == ev.start_idx and new_end == ev.end_idx:
            return

        overlap_ev = self._event_overlap_except_self(current_id, new_start, new_end, ev.event_key)
        if overlap_ev is not None:
            QMessageBox.information(
                self,
                "Notice",
                f"The adjustment would overlap with event {overlap_ev.event_key}. The adjustment was cancelled."
            )
            return

        ok = self._replace_boundary_of_existing_event(current_id, ev, new_start, new_end)
        if ok:
            self._update_selection_label()
            self.update_current_id_display(reset_view=False)
    def on_main_plot_click(self, event):
        if event.inaxes not in self.channel_axes.values():
            return
        if self._toolbar_mode_active():
            return

        if event.button == 1:
            self.canvas_main.setFocus()

            if getattr(event, "dblclick", False):
                self._reset_current_roi_view()
                return

            if self.event_detection_enabled:
                clicked_event_key = self._find_clicked_event_key_from_all_axes(event)

                if clicked_event_key is not None:
                    current_id = self.state.get_current_id()
                    events = self.state.get_display_events(current_id)
                    event_keys_in_order = [ev.event_key for ev in events]

                    if self._is_shift_active() and self.state.last_selected_event_key in event_keys_in_order:
                        a = event_keys_in_order.index(self.state.last_selected_event_key)
                        b = event_keys_in_order.index(clicked_event_key)
                        start, end = sorted([a, b])
                        self.state.selected_event_keys.update(event_keys_in_order[start:end + 1])
                    elif self._is_multiselect_active():
                        if clicked_event_key in self.state.selected_event_keys:
                            self.state.selected_event_keys.remove(clicked_event_key)
                        else:
                            self.state.selected_event_keys.add(clicked_event_key)
                    else:
                        self.state.selected_event_keys = {clicked_event_key}

                    self.state.last_selected_event_key = clicked_event_key
                    self._update_selection_label()
                    self.update_plots(preserve_view=True)
                    return

                if self.manual_draw_mode == "add_event" and event.xdata is not None:
                    current_id = self.state.get_current_id()
                    if current_id is None:
                        return

                    time_arr, n = self._get_roi_time_and_len(current_id)
                    if time_arr is None or n == 0:
                        return

                    clicked_idx = self._time_to_nearest_idx(time_arr, event.xdata)
                    if clicked_idx is None:
                        return

                    self.handle_manual_add_event_by_click(current_id, clicked_idx)
                    return

                if self.manual_draw_mode in ("add_event", "mark_dual"):
                    return

            self._start_plot_pan(event)
            return

        elif event.button == 3 and event.inaxes in self.channel_axes.values():
            if self.event_detection_enabled:
                clicked_event_key = self._find_clicked_event_key_from_all_axes(event)
                if clicked_event_key is not None:
                    self.show_delete_event_menu(event, clicked_event_key)

    def show_delete_event_menu(self, event, event_key):
        menu = QMenu()
        delete_action = menu.addAction("Delete This Event")

        global_pos = self.canvas_main.mapToGlobal(QPoint(int(event.x), int(self.canvas_main.height() - event.y)))
        action = menu.exec_(global_pos)

        if action == delete_action:
            self.delete_event(event_key)

    def delete_event(self, event_key):
        current_id = self.state.get_current_id()
        if current_id is None:
            return

        self.state.hide_event_keys(current_id, {event_key})
        self.state.push_undo(current_id, {
            "type": "hide_events",
            "event_keys": {event_key}
        })
        self.state.selected_event_keys.discard(event_key)
        self._update_selection_label()
        self.update_current_id_display(reset_view=False)

    def delete_selected_events(self):
        if not self.state.selected_event_keys:
            QMessageBox.warning(self, "Notice", "No events are selected.")
            return

        current_id = self.state.get_current_id()
        if current_id is None:
            return

        keys = set(self.state.selected_event_keys)
        self.state.hide_event_keys(current_id, keys)
        self.state.push_undo(current_id, {
            "type": "hide_events",
            "event_keys": keys
        })
        self.state.selected_event_keys.clear()
        self.state.last_selected_event_key = None
        self._update_selection_label()
        self.update_current_id_display(reset_view=False)

    def undo_last_manual_action(self):
        current_id = self.state.get_current_id()
        if current_id is None:
            return

        action = self.state.pop_undo(current_id)
        if not action:
            QMessageBox.information(self, "Notice", "No actions to undo.")
            return

        action_type = action.get("type")

        if action_type == "add_manual_event":
            ev = action["event"]
            self.state.remove_manual_event_by_key(current_id, ev.event_key)
            self.state.selected_event_keys.discard(ev.event_key)

        elif action_type == "add_and_mark_dual":
            ev = action["event"]
            self.state.remove_manual_event_by_key(current_id, ev.event_key)
            self.state.selected_event_keys.discard(ev.event_key)

        elif action_type == "mark_dual":
            old_event = action["old_event"]
            new_event = action["new_event"]
            self.state.replace_event_by_key(current_id, new_event.event_key, old_event)
            self.state.selected_event_keys.discard(new_event.event_key)
            self.state.selected_event_keys.add(old_event.event_key)
            self.state.last_selected_event_key = old_event.event_key

        elif action_type == "modify_boundary":
            old_event = action["old_event"]
            new_event = action["new_event"]
            self.state.replace_event_by_key(current_id, new_event.event_key, old_event)
            self.state.selected_event_keys.discard(new_event.event_key)
            self.state.selected_event_keys.add(old_event.event_key)
            self.state.last_selected_event_key = old_event.event_key

        elif action_type == "hide_events":
            self.state.unhide_event_keys(current_id, action["event_keys"])

        self._update_selection_label()
        self.update_current_id_display(reset_view=False)

    def select_all_events(self):
        current_id = self.state.get_current_id()
        if current_id is None or not self.event_detection_enabled:
            return

        events = self.state.get_display_events(current_id)
        self.state.selected_event_keys = {ev.event_key for ev in events}
        self.state.last_selected_event_key = None
        self._update_selection_label()
        self.update_plots(preserve_view=True)

    def invert_selection(self):
        current_id = self.state.get_current_id()
        if current_id is None or not self.event_detection_enabled:
            return

        events = self.state.get_display_events(current_id)
        all_keys = {ev.event_key for ev in events}
        self.state.selected_event_keys = all_keys - self.state.selected_event_keys
        self.state.last_selected_event_key = None
        self._update_selection_label()
        self.update_plots(preserve_view=True)

    def export_classifications(self):
        if not self.state.full_id_list or not self.data_model.filepath:
            QMessageBox.warning(self, "Warning", "Load a data file first.")
            return
        if self._worker_thread is not None:
            QMessageBox.information(self, "Busy", f"Please wait until {self._current_task_name} finishes.")
            return

        classified_ids = {}
        for roi_id, class_num in self.state.id_classifications.items():
            classified_ids.setdefault(class_num, []).append(roi_id)

        unclassified_ids = [roi_id for roi_id in self.state.full_id_list if roi_id not in self.state.id_classifications]

        dialog = ExportOptionsDialog(
            classifications=self.state.classifications,
            has_unclassified=len(unclassified_ids) > 0,
            has_alex_derived=self.data_model.is_alex_profile(),
            parent=self
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        options = dialog.get_options()

        base_filename = os.path.splitext(os.path.basename(self.data_model.filepath))[0]
        initial_dir = self._get_default_export_directory()
        save_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", initial_dir)
        if not save_dir:
            return

        export_jobs = []
        for class_num in sorted(options["selected_classes"]):
            ids = classified_ids.get(class_num, [])
            if ids:
                export_jobs.append({
                    "class_name": f"Class{class_num}",
                    "ids": ids,
                    "export_events_summary": options["export_events_summary"],
                    "export_event_signals": options["export_event_signals"],
                    "export_params_txt": options["export_params_txt"],
                    "export_alex_derived": options["export_alex_derived"],
                })

        if options["export_unclassified"] and unclassified_ids:
            export_jobs.append({
                "class_name": "Unclassified",
                "ids": unclassified_ids,
                "export_events_summary": options["export_events_summary"],
                "export_event_signals": options["export_event_signals"],
                "export_params_txt": options["export_params_txt"],
                "export_alex_derived": options["export_alex_derived"],
            })

        if not export_jobs:
            QMessageBox.information(self, "Notice", "No classes were selected for export.")
            return

        self._start_export_task(save_dir, base_filename, export_jobs)

    def closeEvent(self, event):
        if self._worker_thread is not None:
            QMessageBox.information(self, "Busy", f"Please wait until {self._current_task_name} finishes.")
            event.ignore()
            return
        if not self._prompt_to_handle_unsaved_session():
            event.ignore()
            return

        logger.info("Application closed")
        super().closeEvent(event)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Classification and Event Detection Tool')
    parser.add_argument('--directory', type=str, default='', help='Default working directory path')
    args = parser.parse_args()

    if not args.directory:
        docs_dir = pathlib.Path.home() / "Documents"
        if docs_dir.exists():
            args.directory = str(docs_dir)
        else:
            args.directory = os.getcwd()

    app = QApplication(sys.argv)
    window = EventDetectionTool(default_directory=args.directory)
    window.show()
    sys.exit(app.exec_())
