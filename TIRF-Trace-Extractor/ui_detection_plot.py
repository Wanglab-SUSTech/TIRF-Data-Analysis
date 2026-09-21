from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from channel_profiles import physical_channel_name
from controllers import DetectionPreviewController

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except Exception:  # pragma: no cover - fallback path depends on environment
    pg = None
    HAS_PYQTGRAPH = False

if not HAS_PYQTGRAPH:  # pragma: no cover - fallback path depends on environment
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


CHANNEL_COLORS = {
    "532": "#00d26a",
    "638": "#ff5555",
    "488": "#55aaff",
    "532ex_532": "#00d26a",
    "532ex_638": "#ff5555",
    "488ex_532": "#f5c542",
    "488ex_488": "#55aaff",
}


def channel_color(channel_name: str) -> str:
    name = str(channel_name)
    return CHANNEL_COLORS.get(name, CHANNEL_COLORS.get(physical_channel_name(name), "#cccccc"))


def channel_label(channel_name: str) -> str:
    name = str(channel_name)
    return name if "ex_" in name else f"{name} nm"


class QtLogHandler(logging.Handler):
    class _Emitter(QObject):
        log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.emitter = self._Emitter()

    def emit(self, record):
        try:
            self.emitter.log_signal.emit(self.format(record))
        except Exception:
            pass


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        header = QHBoxLayout()
        header.addWidget(QLabel("Log"))
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        header.addWidget(clear_btn)
        header.addStretch()
        layout.addLayout(header)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(5000)
        self.text_edit.setStyleSheet(
            "QPlainTextEdit { font-family: monospace; font-size: 9pt; background: #1e1e1e; color: #d4d4d4; }"
        )
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def append_log(self, msg):
        self.text_edit.appendPlainText(msg)

    def clear(self):
        self.text_edit.clear()


class IntensityPlotWidget(QWidget):
    frame_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_frame = 0
        self.time_data = None
        self.intensity_data = None
        self.channel_names = []
        self.current_molecule_id = None
        self._line_objects = {}
        self._channel_visible = {}
        self._hover_proxy = None
        self._click_proxy = None
        self._frame_marker_item = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel("Interactive trace view")
        layout.addWidget(self.info_label)

        if HAS_PYQTGRAPH:
            pg.setConfigOptions(antialias=True)
            self.plot_widget = pg.PlotWidget(background="#111111")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
            self.plot_widget.setLabel("bottom", "Time", units="s")
            self.plot_widget.setLabel("left", "Intensity")
            self._ensure_pg_legend()
            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.plot_widget.getViewBox().setMenuEnabled(False)
            layout.addWidget(self.plot_widget, 1)
            self._frame_marker_item = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#00d9ff", width=1))
            self.plot_widget.addItem(self._frame_marker_item)
            self._frame_marker_item.hide()
            self._hover_proxy = pg.SignalProxy(
                self.plot_widget.scene().sigMouseMoved,
                rateLimit=60,
                slot=self._on_pg_mouse_moved,
            )
            self._click_proxy = pg.SignalProxy(
                self.plot_widget.scene().sigMouseClicked,
                rateLimit=30,
                slot=self._on_pg_mouse_clicked,
            )
        else:  # pragma: no cover - fallback path depends on environment
            self.figure = Figure(figsize=(8, 3))
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.ax = self.figure.add_subplot(111)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, 1)

        self._checkbox_layout = QHBoxLayout()
        self._checkbox_layout.addWidget(QLabel("Channels:"))
        self._checkbox_layout.addStretch()
        self._checkboxes = {}
        layout.addLayout(self._checkbox_layout)

        button_row = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        button_row.addWidget(reset_btn)
        export_btn = QPushButton("Export Trace")
        export_btn.clicked.connect(self._export_current_trace)
        button_row.addWidget(export_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.setLayout(layout)
        self.clear_plot()

    def _rebuild_checkboxes(self):
        for checkbox in self._checkboxes.values():
            self._checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self._checkboxes.clear()

        for channel_name in self.channel_names:
            checkbox = QCheckBox(channel_label(channel_name))
            checkbox.setChecked(self._channel_visible.get(channel_name, True))
            checkbox.setStyleSheet(f"QCheckBox {{ color: {channel_color(channel_name)}; }}")
            checkbox.stateChanged.connect(lambda state, ch=channel_name: self._on_channel_toggled(ch, state))
            self._checkbox_layout.insertWidget(self._checkbox_layout.count() - 1, checkbox)
            self._checkboxes[channel_name] = checkbox

    def _on_channel_toggled(self, channel_name, state):
        self._channel_visible[channel_name] = bool(state)
        if channel_name in self._line_objects:
            obj = self._line_objects[channel_name]
            if HAS_PYQTGRAPH:
                obj.setVisible(bool(state))
            else:  # pragma: no cover
                obj.set_visible(bool(state))
                self.canvas.draw_idle()

    def clear_plot(self):
        self.current_molecule_id = None
        self.time_data = None
        self.intensity_data = None
        self.channel_names = []
        self._line_objects.clear()
        self._rebuild_checkboxes()
        self.info_label.setText("No molecule selected")

        if HAS_PYQTGRAPH:
            self.plot_widget.clear()
            self._reset_pg_legend()
            self._frame_marker_item = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#00d9ff", width=1))
            self.plot_widget.addItem(self._frame_marker_item)
            self._frame_marker_item.hide()
        else:  # pragma: no cover
            self.ax.clear()
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Intensity")
            self.ax.set_title("No molecule selected")
            self.canvas.draw()

    def plot_molecule(self, time, intensities, channel_names, molecule_id):
        self.current_molecule_id = molecule_id
        self.time_data = np.asarray(time, dtype=np.float32)
        self.intensity_data = dict(intensities or {})
        self.channel_names = list(channel_names)
        for channel_name in self.channel_names:
            self._channel_visible.setdefault(channel_name, True)
        self._rebuild_checkboxes()
        self.info_label.setText(f"Molecule {molecule_id} trace")

        if HAS_PYQTGRAPH:
            self.plot_widget.clear()
            self._reset_pg_legend()
            self._line_objects.clear()
            for channel_name in channel_names:
                if channel_name not in self.intensity_data:
                    continue
                pen = pg.mkPen(channel_color(channel_name), width=1.6)
                item = self.plot_widget.plot(
                    self.time_data,
                    np.asarray(self.intensity_data[channel_name], dtype=np.float32),
                    pen=pen,
                    name=channel_label(channel_name),
                )
                item.setVisible(self._channel_visible.get(channel_name, True))
                self._line_objects[channel_name] = item
            self._frame_marker_item = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#00d9ff", width=1))
            self.plot_widget.addItem(self._frame_marker_item)
            self.update_frame_marker(self.current_frame)
            self.reset_view()
        else:  # pragma: no cover
            self.ax.clear()
            self._line_objects.clear()
            for channel_name in channel_names:
                if channel_name not in self.intensity_data:
                    continue
                line, = self.ax.plot(
                    self.time_data,
                    self.intensity_data[channel_name],
                    color=channel_color(channel_name),
                    label=channel_label(channel_name),
                    visible=self._channel_visible.get(channel_name, True),
                )
                self._line_objects[channel_name] = line
            self.ax.legend()
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Intensity")
            self.ax.set_title(f"Molecule {molecule_id} trace")
            self._draw_frame_marker_matplotlib()
            self.canvas.draw()

    def update_frame_marker(self, frame_index):
        self.current_frame = int(frame_index)
        if self.current_molecule_id is None or self.time_data is None or len(self.time_data) == 0:
            return
        x_value = float(self.time_data[min(max(self.current_frame, 0), len(self.time_data) - 1)])
        if HAS_PYQTGRAPH:
            self._frame_marker_item.show()
            self._frame_marker_item.setValue(x_value)
        else:  # pragma: no cover
            self._draw_frame_marker_matplotlib()
            self.canvas.draw_idle()

    def _draw_frame_marker_matplotlib(self):  # pragma: no cover - fallback path
        try:
            for line in list(self.ax.lines):
                if getattr(line, "_is_frame_marker", False):
                    line.remove()
        except Exception:
            pass
        if self.time_data is None or len(self.time_data) == 0:
            return
        x_value = float(self.time_data[min(max(self.current_frame, 0), len(self.time_data) - 1)])
        line = self.ax.axvline(x=x_value, color="cyan", linestyle="--", linewidth=1.0)
        line._is_frame_marker = True

    def reset_view(self):
        if HAS_PYQTGRAPH:
            self.plot_widget.enableAutoRange()
            self.plot_widget.autoRange()
        else:  # pragma: no cover
            self.ax.relim()
            self.ax.autoscale()
            self.canvas.draw_idle()

    def _nearest_frame_from_time(self, x_value: float) -> int | None:
        if self.time_data is None or len(self.time_data) == 0:
            return None
        index = int(np.argmin(np.abs(self.time_data - float(x_value))))
        return max(0, min(index, len(self.time_data) - 1))

    def _on_pg_mouse_moved(self, event):
        if self.time_data is None or self.current_molecule_id is None:
            return
        pos = event[0]
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
        frame_index = self._nearest_frame_from_time(mouse_point.x())
        if frame_index is None:
            return
        values = []
        for channel_name in self.channel_names:
            arr = self.intensity_data.get(channel_name)
            if arr is None or frame_index >= len(arr):
                continue
            values.append(f"{channel_name}: {float(arr[frame_index]):.2f}")
        self.info_label.setText(
            f"Molecule {self.current_molecule_id} | t={float(self.time_data[frame_index]):.3f}s | " + ", ".join(values)
        )

    def _on_pg_mouse_clicked(self, event):
        if self.time_data is None:
            return
        mouse_event = event[0]
        if mouse_event.button() != Qt.LeftButton:
            return
        view_pos = self.plot_widget.getPlotItem().vb.mapSceneToView(mouse_event.scenePos())
        frame_index = self._nearest_frame_from_time(view_pos.x())
        if frame_index is not None:
            self.frame_selected.emit(frame_index)

    def _export_current_trace(self):
        if self.time_data is None or self.intensity_data is None or self.current_molecule_id is None:
            return
        default_name = f"molecule_{self.current_molecule_id}_trace.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export Trace", default_name, "CSV (*.csv)")
        if not path:
            return
        try:
            with Path(path).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Time_sec"] + [f"Net_{channel_name}" for channel_name in self.channel_names])
                for index, time_value in enumerate(self.time_data):
                    row = [f"{float(time_value):.4f}"]
                    for channel_name in self.channel_names:
                        values = self.intensity_data.get(channel_name)
                        if values is None or index >= len(values):
                            row.append("")
                            continue
                        value = float(values[index])
                        row.append("" if np.isnan(value) else f"{value:.4f}")
                    writer.writerow(row)
            QMessageBox.information(self, "Success", f"Trace exported to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Export failed: {exc}")

    def _ensure_pg_legend(self):
        if not HAS_PYQTGRAPH:
            return
        if getattr(self.plot_widget.plotItem, "legend", None) is None:
            self.plot_widget.addLegend()

    def _reset_pg_legend(self):
        if not HAS_PYQTGRAPH:
            return
        legend = getattr(self.plot_widget.plotItem, "legend", None)
        if legend is not None and legend.scene() is not None:
            legend.scene().removeItem(legend)
            self.plot_widget.plotItem.legend = None
        self._ensure_pg_legend()


class DetectionPreviewDialog(QDialog):
    def __init__(self, first_frame, parent=None, initial_params=None):
        super().__init__(parent)
        self.setWindowTitle("Molecule Detection Preview")
        self.resize(1200, 800)
        self.first_frame = first_frame
        self.detected_positions = []
        self.detected_results = []
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self._accepted_final = False
        self._controller_shutdown = False
        self.setup_ui()
        self.controller = DetectionPreviewController(self, first_frame)
        if initial_params:
            self.apply_params(initial_params)
        QApplication.processEvents()
        self.schedule_update()

    def _shutdown_controller(self, *, timeout_ms=None):
        if self._controller_shutdown:
            return
        self._controller_shutdown = True
        try:
            if hasattr(self, "controller") and self.controller is not None:
                self.controller.shutdown_with_timeout(timeout_ms=timeout_ms)
        except Exception:
            logger.exception("Failed to shutdown detection preview controller")

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        img_vbox = QVBoxLayout()
        self.status_label = QLabel("Preparing detection...")
        img_vbox.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        img_vbox.addWidget(self.progress_bar)
        self.image_label = QLabel()
        self.image_label.setStyleSheet("background: black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        img_vbox.addWidget(self.image_label, 1)
        fit_btn = QPushButton("Fit to Window")
        fit_btn.clicked.connect(self.fit_to_window)
        img_vbox.addWidget(fit_btn)
        main_layout.addLayout(img_vbox, 3)

        ctrl_vbox = QVBoxLayout()
        param_group = QGroupBox("Detection Parameters")
        grid = QGridLayout()
        self.sigma_spin = self._add_spin(grid, "Sigma:", 0, 0.5, 10.0, 3.5, 0.1)
        self.snr_spin = self._add_spin(grid, "Min SNR:", 1, 1.0, 20.0, 4.0, 0.5)
        self.r2_spin = self._add_spin(grid, "Min R2:", 2, -1.0, 0.99, 0.0, 0.05)
        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(0, 100)
        self.min_dist_spin.setSpecialValueText("Auto")
        self.min_dist_spin.valueChanged.connect(self.schedule_update)
        grid.addWidget(QLabel("Min Distance (px):"), 3, 0)
        grid.addWidget(self.min_dist_spin, 3, 1)
        self.bg_check = QCheckBox("Background Subtraction")
        self.bg_check.setChecked(True)
        self.bg_check.stateChanged.connect(self.schedule_update)
        grid.addWidget(self.bg_check, 4, 0, 1, 2)
        self.bg_radius_spin = QSpinBox()
        self.bg_radius_spin.setRange(0, 200)
        self.bg_radius_spin.setSpecialValueText("Auto")
        self.bg_radius_spin.valueChanged.connect(self.schedule_update)
        grid.addWidget(QLabel("Rolling Ball Radius (px):"), 5, 0)
        grid.addWidget(self.bg_radius_spin, 5, 1)
        param_group.setLayout(grid)
        ctrl_vbox.addWidget(param_group)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        ctrl_vbox.addWidget(QLabel("Detection Statistics"))
        ctrl_vbox.addWidget(self.stats_text)

        self.result_label = QLabel("Detected: 0 molecules")
        self.result_label.setStyleSheet("font-weight: bold; color: green; font-size: 12pt;")
        ctrl_vbox.addWidget(self.result_label)

        buttons = QHBoxLayout()
        self.ok_btn = QPushButton("Confirm")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.ok_btn)
        buttons.addWidget(self.cancel_btn)
        ctrl_vbox.addLayout(buttons)
        main_layout.addLayout(ctrl_vbox, 1)

    def _add_spin(self, grid, label, row, min_v, max_v, default, step):
        grid.addWidget(QLabel(label), row, 0)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.valueChanged.connect(self.schedule_update)
        grid.addWidget(spin, row, 1)
        return spin

    def schedule_update(self):
        self.controller.schedule_preview(self.get_params())

    def get_params(self):
        return {
            "sigma": self.sigma_spin.value(),
            "min_snr": self.snr_spin.value(),
            "min_r2": self.r2_spin.value(),
            "min_distance": self.min_dist_spin.value() if self.min_dist_spin.value() > 0 else None,
            "auto_bg": self.bg_check.isChecked(),
            "bg_radius": self.bg_radius_spin.value() if self.bg_radius_spin.value() > 0 else None,
        }

    def apply_params(self, params):
        widgets = (
            self.sigma_spin,
            self.snr_spin,
            self.r2_spin,
            self.min_dist_spin,
            self.bg_check,
            self.bg_radius_spin,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.sigma_spin.setValue(params.get("sigma", 3.5))
            self.snr_spin.setValue(params.get("min_snr", 4.0))
            self.r2_spin.setValue(params.get("min_r2", 0.0))
            self.min_dist_spin.setValue(int(params.get("min_distance") or 0))
            self.bg_check.setChecked(params.get("auto_bg", True))
            self.bg_radius_spin.setValue(int(params.get("bg_radius") or 0))
        finally:
            for widget in reversed(widgets):
                widget.blockSignals(False)

    def display_result(self):
        if self.first_frame is None or self.image_label.width() <= 0 or self.image_label.height() <= 0:
            return
        image = self.first_frame.astype(np.float32)
        image = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for x, y in self.detected_positions:
            cv2.circle(display, (int(x), int(y)), 4, (255, 255, 0), 1)
        height, width = display.shape[:2]
        target_w = max(1, int(width * self.zoom_factor))
        target_h = max(1, int(height * self.zoom_factor))
        display = cv2.resize(display, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        qimg = QImage(np.ascontiguousarray(display).data, target_w, target_h, target_w * 3, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        canvas = QPixmap(self.image_label.size())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        offset_x = (canvas.width() - target_w) // 2 + self.pan_offset[0]
        offset_y = (canvas.height() - target_h) // 2 + self.pan_offset[1]
        painter.drawPixmap(offset_x, offset_y, pixmap)
        painter.end()
        self.image_label.setPixmap(canvas)

    def fit_to_window(self):
        if self.first_frame is None or self.image_label.width() <= 0 or self.image_label.height() <= 0:
            return
        height, width = self.first_frame.shape
        label_w, label_h = self.image_label.width(), self.image_label.height()
        self.zoom_factor = min(label_w / width, label_h / height) * 0.9
        self.pan_offset = [0, 0]
        self.display_result()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_result()

    def get_detected_positions(self):
        return self.detected_positions

    def get_detected_results(self):
        return list(self.detected_results)

    def set_detected_results(self, records):
        self.detected_results = [dict(item or {}) for item in list(records or [])]

    def set_status(self, text):
        self.status_label.setText(str(text))

    def set_busy(self, busy):
        self.progress_bar.setVisible(bool(busy))
        if busy and self.progress_bar.maximum() == 100 and self.progress_bar.value() == 0:
            self.progress_bar.setRange(0, 100)

    def set_progress(self, value):
        if value is None:
            self.progress_bar.setRange(0, 0)
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(value))

    def set_detected_positions(self, positions):
        self.detected_positions = list(positions or [])
        self.display_result()

    def set_result_count(self, count):
        self.result_label.setText(f"Detected: {int(count)} molecules")

    def set_stats_text(self, text):
        self.stats_text.setText(str(text))

    def set_confirm_enabled(self, enabled):
        self.ok_btn.setEnabled(bool(enabled))

    def set_cancel_enabled(self, enabled):
        self.cancel_btn.setEnabled(bool(enabled))

    def complete_accept(self):
        self._shutdown_controller(timeout_ms=None)
        self._accepted_final = True
        super().accept()

    def accept(self):
        if self._accepted_final:
            super().accept()
            return
        self.controller.confirm_current_or_latest(self.get_params())

    def reject(self):
        self._shutdown_controller(timeout_ms=None)
        super().reject()

    def closeEvent(self, event):
        self._shutdown_controller(timeout_ms=None)
        super().closeEvent(event)
