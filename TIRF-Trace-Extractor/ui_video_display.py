from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui_lut_cache import DebounceTimer, LUTControlPanel

logger = logging.getLogger(__name__)

_SCENE_MARGIN = 5000


class SyncZoomManager:
    def __init__(self):
        self.video_widgets = []
        self.is_syncing = False

    def _prune_dead_widgets(self):
        alive = []
        for widget in self.video_widgets:
            try:
                _ = widget.isVisible()
                alive.append(widget)
            except (RuntimeError, AttributeError):
                pass
        self.video_widgets = alive

    def register(self, widget):
        self._prune_dead_widgets()
        if widget not in self.video_widgets:
            self.video_widgets.append(widget)

    def unregister(self, widget):
        self._prune_dead_widgets()
        self.video_widgets = [w for w in self.video_widgets if w is not widget]

    def clear(self):
        self.video_widgets = []

    def sync_transform(self, source_widget):
        if self.is_syncing:
            return
        self._prune_dead_widgets()
        self.is_syncing = True
        try:
            src_state = source_widget.get_view_state()
            for widget in self.video_widgets:
                try:
                    if widget is not source_widget and widget.isVisible():
                        widget.apply_view_state(src_state, emit_sync=False)
                except RuntimeError:
                    pass
        finally:
            self.is_syncing = False


class VideoGraphicsView(QGraphicsView):
    clicked = pyqtSignal(float, float)
    hover_info = pyqtSignal(str)
    transform_changed = pyqtSignal()
    add_requested = pyqtSignal(float, float)
    delete_requested = pyqtSignal(float, float)
    move_requested = pyqtSignal(int, float, float)

    def __init__(self, scene, owner, parent=None):
        super().__init__(scene, parent)
        self.owner = owner
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(Qt.black))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self._panning = False
        self._pan_start = None
        self._press_pos = None
        self._drag_threshold = 3
        self._drag_mol_id = None

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.scale(factor, factor)
        self.owner._user_has_transformed = True
        self.owner._update_marker_geometry()
        self.transform_changed.emit()
        self.owner.sync_manager.sync_transform(self.owner)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        tool = self.owner.edit_tool
        if event.button() == Qt.LeftButton:
            if tool == "pan":
                self._press_pos = event.pos()
                self._pan_start = event.pos()
                self._panning = False
                return
            if tool == "add":
                self.add_requested.emit(scene_pos.x(), scene_pos.y())
                return
            if tool == "delete":
                self.delete_requested.emit(scene_pos.x(), scene_pos.y())
                return
            if tool == "move":
                mol_id = self.owner.nearest_molecule_id(scene_pos.x(), scene_pos.y())
                if mol_id is not None:
                    self._drag_mol_id = int(mol_id)
                    self.owner.preview_drag_molecule(self._drag_mol_id, scene_pos.x(), scene_pos.y())
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        wx, wy = int(scene_pos.x()), int(scene_pos.y())
        pv = self.owner.get_pixel_value(wx, wy)
        if pv is not None:
            self.hover_info.emit(f"({wx}, {wy}): {pv}")
        else:
            self.hover_info.emit("")

        if self._drag_mol_id is not None:
            self.owner.preview_drag_molecule(self._drag_mol_id, scene_pos.x(), scene_pos.y())
            return

        if self._pan_start is not None and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._pan_start
            if not self._panning and delta.manhattanLength() >= self._drag_threshold:
                self._panning = True
            if self._panning:
                self._pan_start = event.pos()
                hs = self.horizontalScrollBar()
                vs = self.verticalScrollBar()
                hs.setValue(hs.value() - delta.x())
                vs.setValue(vs.value() - delta.y())
                self.owner._user_has_transformed = True
                self.owner._update_marker_geometry()
                self.transform_changed.emit()
                self.owner.sync_manager.sync_transform(self.owner)
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if self._drag_mol_id is not None:
                self.move_requested.emit(self._drag_mol_id, scene_pos.x(), scene_pos.y())
                self._drag_mol_id = None
                return
            if not self._panning and self._press_pos is not None:
                self.clicked.emit(scene_pos.x(), scene_pos.y())
            self._panning = False
            self._pan_start = None
            self._press_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.owner.edit_tool in {"pan", "zoom"} and event.button() == Qt.LeftButton:
            self.owner.fit_to_window()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.owner.on_view_resized()


class VideoWidget(QWidget):
    clicked = pyqtSignal(int, int)
    transform_changed = pyqtSignal()
    add_molecule_requested = pyqtSignal(float, float)
    delete_molecule_requested = pyqtSignal(float, float)
    move_molecule_requested = pyqtSignal(int, float, float)

    def __init__(self, channel_name, channel_idx, multi_cache, sync_manager, lut_panel=None, registration_params=None, parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.channel_idx = channel_idx
        self.multi_cache = multi_cache
        self.sync_manager = sync_manager
        self.registration_params = registration_params
        self.current_frame_idx = 0
        self.raw_image_16bit = None
        self._image_h = 0
        self._image_w = 0
        self._lut_panel_external = lut_panel is not None
        self.lut_panel = lut_panel if lut_panel is not None else LUTControlPanel(channel_name)
        if not self._lut_panel_external:
            self.lut_panel.settings_changed.connect(self.on_lut_changed)

        self.molecules = []
        self.selected_molecule = None
        self.edit_tool = "pan"
        self._fit_initialized = False
        self._user_has_transformed = False
        self._pixmap_cache = OrderedDict()
        self._pixmap_cache_max = 40
        self._pixmap_cache_lock = Lock()
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._mol_items = {}

        self._marker_world_radius = 4.5
        self._marker_min_screen_radius = 4.0
        self._marker_pen_normal = QPen(QColor(255, 255, 0), 1)
        self._marker_pen_selected = QPen(QColor(255, 64, 64), 1)
        self._marker_pen_normal.setCosmetic(True)
        self._marker_pen_selected.setCosmetic(True)

        self.gfx_view = VideoGraphicsView(self._scene, self)
        self.gfx_view.clicked.connect(self._on_view_clicked)
        self.gfx_view.hover_info.connect(self._on_hover_info)
        self.gfx_view.transform_changed.connect(self.transform_changed.emit)
        self.gfx_view.add_requested.connect(self.add_molecule_requested.emit)
        self.gfx_view.delete_requested.connect(self.delete_molecule_requested.emit)
        self.gfx_view.move_requested.connect(self.move_molecule_requested.emit)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._update_timer = DebounceTimer(delay_ms=16, parent=self)
        self.setup_ui()
        self.sync_manager.register(self)

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        self.title_label = QLabel(f"{self.channel_name} nm")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12pt; }")
        layout.addWidget(self.title_label)
        layout.addWidget(self.gfx_view, stretch=1)
        self.hover_label = QLabel("")
        self.hover_label.setStyleSheet("QLabel { color: yellow; font-family: monospace; }")
        layout.addWidget(self.hover_label)
        if not self._lut_panel_external:
            layout.addWidget(self.lut_panel)
        ctrl = QHBoxLayout()
        reset_btn = QPushButton("100%")
        reset_btn.clicked.connect(self.reset_view)
        ctrl.addWidget(reset_btn)
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.fit_to_window)
        ctrl.addWidget(fit_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)
        self.setLayout(layout)

    def _on_hover_info(self, text):
        self.hover_label.setText(text)

    def _on_view_clicked(self, sx, sy):
        self.clicked.emit(int(sx), int(sy))

    def set_registration_params(self, registration_params):
        self.registration_params = registration_params
        self.raw_image_16bit = None
        self._clear_pixmap_cache()
        self._fit_initialized = False
        self._user_has_transformed = False
        self.schedule_update()

    def set_edit_tool(self, tool: str):
        self.edit_tool = str(tool or "pan")

    def set_frame(self, frame_idx):
        if frame_idx != self.current_frame_idx:
            self.current_frame_idx = frame_idx
            self.raw_image_16bit = None
            self.schedule_update()

    def on_lut_changed(self):
        self._clear_pixmap_cache()
        self.schedule_update()

    def schedule_update(self):
        self._update_timer.trigger(self.update_display)

    def _pixmap_cache_key(self, frame_idx):
        lut_hash = self.lut_panel.get_settings().get_hash()
        reg_id = id(self.registration_params)
        return (frame_idx, self.channel_idx, lut_hash, reg_id)

    def _clear_pixmap_cache(self):
        with self._pixmap_cache_lock:
            self._pixmap_cache.clear()

    def _get_or_render_pixmap(self, frame_idx):
        key = self._pixmap_cache_key(frame_idx)
        with self._pixmap_cache_lock:
            if key in self._pixmap_cache:
                self._pixmap_cache.move_to_end(key)
                return self._pixmap_cache[key]
        if self.multi_cache is None:
            return None
        lut_settings = self.lut_panel.get_settings()
        try:
            self.raw_image_16bit = self.multi_cache.get_registered_display_raw_image(frame_idx, self.channel_idx, self.channel_name, self.registration_params)
            rgb = self.multi_cache.get_registered_display_rgb_image(frame_idx, self.channel_idx, self.channel_name, lut_settings, self.registration_params)
        except Exception as e:
            logger.warning("Failed to render frame %s: %s", frame_idx, e)
            return None
        if rgb is None:
            return None
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        self._image_h, self._image_w = h, w
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        with self._pixmap_cache_lock:
            self._pixmap_cache[key] = pixmap
            while len(self._pixmap_cache) > self._pixmap_cache_max:
                self._pixmap_cache.popitem(last=False)
        return pixmap

    def update_display(self):
        if self.multi_cache is None:
            return
        pixmap = self._get_or_render_pixmap(self.current_frame_idx)
        if pixmap is None:
            return
        self._pixmap_item.setPixmap(pixmap)
        if not self._fit_initialized:
            img_rect = self._pixmap_item.boundingRect()
            if img_rect.width() > 0 and img_rect.height() > 0:
                self._scene.setSceneRect(-_SCENE_MARGIN, -_SCENE_MARGIN, img_rect.width() + 2 * _SCENE_MARGIN, img_rect.height() + 2 * _SCENE_MARGIN)
                self.fit_to_window()
        self._update_marker_geometry()

    def fit_to_window(self):
        rect = self._pixmap_item.boundingRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        self.gfx_view.fitInView(rect, Qt.KeepAspectRatio)
        self._fit_initialized = True
        self._user_has_transformed = False
        self._update_marker_geometry()
        self.transform_changed.emit()
        self.sync_manager.sync_transform(self)

    def reset_view(self):
        self.gfx_view.setTransform(QTransform())
        self._user_has_transformed = True
        self._update_marker_geometry()
        self.transform_changed.emit()
        self.sync_manager.sync_transform(self)

    def on_view_resized(self):
        if not self._user_has_transformed and self._fit_initialized:
            rect = self._pixmap_item.boundingRect()
            if rect.width() > 0 and rect.height() > 0:
                self.gfx_view.fitInView(rect, Qt.KeepAspectRatio)
                self._update_marker_geometry()

    def get_pixel_value(self, world_x, world_y):
        if self.raw_image_16bit is None:
            return None
        h, w = self.raw_image_16bit.shape
        if 0 <= world_x < w and 0 <= world_y < h:
            return int(self.raw_image_16bit[world_y, world_x])
        return None

    def set_molecules(self, molecules):
        self.molecules = [(int(m[0]), float(m[1]), float(m[2])) for m in list(molecules or [])]
        self._rebuild_molecule_items()

    def set_selected_molecule(self, mol_id):
        self.selected_molecule = mol_id
        self._update_molecule_selection()

    def _current_scene_radius(self) -> float:
        scale = abs(self.gfx_view.transform().m11())
        scale = max(scale, 1e-6)
        return max(float(self._marker_world_radius), float(self._marker_min_screen_radius) / scale)

    def _rebuild_molecule_items(self):
        for item in self._mol_items.values():
            self._scene.removeItem(item)
        self._mol_items.clear()
        for mol_id, mx, my in self.molecules:
            item = QGraphicsEllipseItem()
            item.setBrush(QBrush(Qt.NoBrush))
            self._scene.addItem(item)
            self._mol_items[mol_id] = item
            item.setPos(float(mx), float(my))
        self._update_marker_geometry()
        self._update_molecule_selection()

    def _update_marker_geometry(self):
        radius = self._current_scene_radius()
        for mol_id, item in self._mol_items.items():
            item.setRect(-radius, -radius, 2 * radius, 2 * radius)
            item.setPen(self._marker_pen_selected if mol_id == self.selected_molecule else self._marker_pen_normal)

    def _update_molecule_selection(self):
        self._update_marker_geometry()

    def nearest_molecule_id(self, world_x: float, world_y: float, radius_px: float = 8.0) -> int | None:
        if not self.molecules:
            return None
        scale = abs(self.gfx_view.transform().m11())
        scale = max(scale, 1e-6)
        radius_scene = max(radius_px / scale, self._marker_world_radius)
        best = None
        best_dist2 = None
        for mol_id, mx, my in self.molecules:
            dx = float(mx) - float(world_x)
            dy = float(my) - float(world_y)
            dist2 = dx * dx + dy * dy
            if dist2 <= radius_scene * radius_scene and (best_dist2 is None or dist2 < best_dist2):
                best_dist2 = dist2
                best = int(mol_id)
        return best

    def preview_drag_molecule(self, mol_id: int, world_x: float, world_y: float):
        item = self._mol_items.get(int(mol_id))
        if item is not None:
            item.setPos(float(world_x), float(world_y))

    def world_to_screen(self, world_x, world_y):
        vp = self.gfx_view.mapFromScene(QPointF(world_x, world_y))
        return vp.x(), vp.y()

    def screen_to_world(self, screen_x, screen_y):
        sp = self.gfx_view.mapToScene(QPoint(int(screen_x), int(screen_y)))
        return int(sp.x()), int(sp.y())

    def get_view_state(self) -> dict:
        transform = self.gfx_view.transform()
        return {
            "fit_mode": bool(self._fit_initialized and not self._user_has_transformed),
            "m11": float(transform.m11()),
            "m12": float(transform.m12()),
            "m21": float(transform.m21()),
            "m22": float(transform.m22()),
            "dx": float(transform.dx()),
            "dy": float(transform.dy()),
            "h_scroll": int(self.gfx_view.horizontalScrollBar().value()),
            "v_scroll": int(self.gfx_view.verticalScrollBar().value()),
        }

    def apply_view_state(self, state: dict | None, *, emit_sync: bool = True):
        if not state:
            return
        transform = QTransform(
            float(state.get("m11", 1.0)),
            float(state.get("m12", 0.0)),
            float(state.get("m21", 0.0)),
            float(state.get("m22", 1.0)),
            float(state.get("dx", 0.0)),
            float(state.get("dy", 0.0)),
        )
        self.gfx_view.setTransform(transform)
        self.gfx_view.horizontalScrollBar().setValue(int(state.get("h_scroll", 0)))
        self.gfx_view.verticalScrollBar().setValue(int(state.get("v_scroll", 0)))
        self._fit_initialized = True
        self._user_has_transformed = not bool(state.get("fit_mode", False))
        self._update_marker_geometry()
        if emit_sync:
            self.transform_changed.emit()

    def closeEvent(self, event):
        try:
            self.sync_manager.unregister(self)
        except Exception:
            pass
        super().closeEvent(event)
