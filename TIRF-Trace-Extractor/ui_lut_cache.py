"""
ui_lut_cache.py - LUT system and cache module v13.0

Updates:
- print() -> logging
- Improved MultiLevelCache thread safety
"""

import logging
import numpy as np
import hashlib
import json as _json_module
from collections import OrderedDict
from threading import Lock
from matplotlib import colormaps

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from registration import RegistrationDisplayApplier, RegistrationConverter

logger = logging.getLogger(__name__)


def _stable_hash(obj):
    s = _json_module.dumps(obj, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.md5(s.encode('utf-8')).hexdigest()


class LUTSettings:
    def __init__(self):
        self.display_min = 0
        self.display_max = 65535
        self.brightness = 0
        self.contrast = 1.0
        self.gamma = 1.0
        self.colormap = "Gray"

    def get_hash(self):
        return hash((self.display_min, self.display_max, self.brightness, self.contrast, self.gamma, self.colormap))

    def copy(self):
        s = LUTSettings()
        s.display_min, s.display_max = self.display_min, self.display_max
        s.brightness, s.contrast, s.gamma = self.brightness, self.contrast, self.gamma
        s.colormap = self.colormap
        return s

    def to_dict(self):
        return {"display_min": self.display_min, "display_max": self.display_max,
                "brightness": self.brightness, "contrast": self.contrast,
                "gamma": self.gamma, "colormap": self.colormap}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.display_min = d.get("display_min", 0)
        s.display_max = d.get("display_max", 65535)
        s.brightness = d.get("brightness", 0)
        s.contrast = d.get("contrast", 1.0)
        s.gamma = d.get("gamma", 1.0)
        s.colormap = d.get("colormap", "Gray")
        return s


class LUTGenerator:
    def __init__(self):
        self.lut_cache = {}
        self.cache_lock = Lock()
        self.colormaps = {
            "Gray": self._gen_gray(), "Green": self._gen_green(),
            "Red": self._gen_red(), "Blue": self._gen_blue(),
            "Hot": self._gen_mpl("hot"), "Rainbow": self._gen_mpl("jet"),
        }
        self._base_axis = np.arange(65536, dtype=np.float32)

    @staticmethod
    def _gen_gray():
        lut = np.zeros((256, 3), dtype=np.uint8)
        x = np.arange(256, dtype=np.uint8)
        lut[:, 0] = lut[:, 1] = lut[:, 2] = x
        return lut

    @staticmethod
    def _gen_green():
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:, 1] = np.arange(256, dtype=np.uint8)
        return lut

    @staticmethod
    def _gen_red():
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:, 0] = np.arange(256, dtype=np.uint8)
        return lut

    @staticmethod
    def _gen_blue():
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:, 2] = np.arange(256, dtype=np.uint8)
        return lut

    @staticmethod
    def _gen_mpl(name):
        colormap = colormaps.get_cmap(name)
        rgba = colormap(np.linspace(0, 1, 256))
        return np.ascontiguousarray((rgba[:, :3] * 255).astype(np.uint8))

    def generate_lut(self, settings):
        settings_hash = settings.get_hash()
        with self.cache_lock:
            cached = self.lut_cache.get(settings_hash)
            if cached is not None:
                return cached

        norm = (self._base_axis - float(settings.display_min)) / max(1e-10, float(settings.display_max - settings.display_min))
        norm = np.clip(norm, 0.0, 1.0)
        if abs(settings.gamma - 1.0) > 1e-12:
            norm = np.power(norm, settings.gamma, dtype=np.float32)
        if abs(settings.contrast - 1.0) > 1e-12:
            norm = (norm - 0.5) * settings.contrast + 0.5
        if settings.brightness != 0:
            norm = norm + (settings.brightness / 255.0)
        lut_1d = np.ascontiguousarray(np.clip(norm * 255.0, 0, 255).astype(np.uint8))

        with self.cache_lock:
            self.lut_cache[settings_hash] = lut_1d
            if len(self.lut_cache) > 100:
                del self.lut_cache[next(iter(self.lut_cache))]
        return lut_1d

    def apply_lut(self, image_16bit, settings):
        lut_1d = self.generate_lut(settings)
        image_8bit = lut_1d[image_16bit]
        if settings.colormap == "Gray":
            return np.ascontiguousarray(np.repeat(image_8bit[..., None], 3, axis=-1))
        return np.ascontiguousarray(self.colormaps[settings.colormap][image_8bit])

    def clear_cache(self):
        with self.cache_lock:
            self.lut_cache.clear()


class LUTControlPanel(QGroupBox):
    settings_changed = pyqtSignal()
    auto_requested = pyqtSignal()

    def __init__(self, channel_name, parent=None):
        super().__init__(f"{channel_name} LUT Controls", parent)
        self.settings = LUTSettings()
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout()
        row = 0

        layout.addWidget(QLabel("Display Min:"), row, 0)
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 65535)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self.on_min_changed)
        layout.addWidget(self.min_slider, row, 1)
        self.min_label = QLabel("0")
        layout.addWidget(self.min_label, row, 2)
        row += 1

        layout.addWidget(QLabel("Display Max:"), row, 0)
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 65535)
        self.max_slider.setValue(65535)
        self.max_slider.valueChanged.connect(self.on_max_changed)
        layout.addWidget(self.max_slider, row, 1)
        self.max_label = QLabel("65535")
        layout.addWidget(self.max_label, row, 2)
        row += 1

        layout.addWidget(QLabel("Brightness:"), row, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-255, 255)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        layout.addWidget(self.brightness_slider, row, 1)
        self.brightness_label = QLabel("0")
        layout.addWidget(self.brightness_label, row, 2)
        row += 1

        layout.addWidget(QLabel("Contrast:"), row, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        layout.addWidget(self.contrast_slider, row, 1)
        self.contrast_label = QLabel("1.00")
        layout.addWidget(self.contrast_label, row, 2)
        row += 1

        layout.addWidget(QLabel("Gamma:"), row, 0)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(50, 200)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.on_gamma_changed)
        layout.addWidget(self.gamma_slider, row, 1)
        self.gamma_label = QLabel("1.00")
        layout.addWidget(self.gamma_label, row, 2)
        row += 1

        layout.addWidget(QLabel("Colormap:"), row, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["Gray", "Green", "Red", "Blue", "Hot", "Rainbow"])
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        layout.addWidget(self.colormap_combo, row, 1, 1, 2)
        row += 1

        button_layout = QHBoxLayout()
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.clicked.connect(self.on_auto_clicked)
        button_layout.addWidget(self.auto_btn)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout, row, 0, 1, 3)
        self.setLayout(layout)

    def on_min_changed(self, value):
        self.settings.display_min = value
        self.min_label.setText(str(value))
        if value > self.settings.display_max:
            self.max_slider.setValue(value)
        self.settings_changed.emit()

    def on_max_changed(self, value):
        self.settings.display_max = value
        self.max_label.setText(str(value))
        if value < self.settings.display_min:
            self.min_slider.setValue(value)
        self.settings_changed.emit()

    def on_brightness_changed(self, value):
        self.settings.brightness = value
        self.brightness_label.setText(str(value))
        self.settings_changed.emit()

    def on_contrast_changed(self, value):
        self.settings.contrast = value / 100.0
        self.contrast_label.setText(f"{self.settings.contrast:.2f}")
        self.settings_changed.emit()

    def on_gamma_changed(self, value):
        self.settings.gamma = value / 100.0
        self.gamma_label.setText(f"{self.settings.gamma:.2f}")
        self.settings_changed.emit()

    def on_colormap_changed(self, colormap):
        self.settings.colormap = colormap
        self.settings_changed.emit()

    def on_auto_clicked(self):
        self.auto_requested.emit()

    def on_reset_clicked(self):
        self.min_slider.setValue(0)
        self.max_slider.setValue(65535)
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.gamma_slider.setValue(100)
        self.colormap_combo.setCurrentText("Gray")

    def set_auto_range(self, min_val, max_val):
        self.min_slider.setValue(int(min_val))
        self.max_slider.setValue(int(max_val))

    def get_settings(self):
        return self.settings.copy()

    def apply_settings(self, settings):
        self.min_slider.setValue(settings.display_min)
        self.max_slider.setValue(settings.display_max)
        self.brightness_slider.setValue(settings.brightness)
        self.contrast_slider.setValue(int(settings.contrast * 100))
        self.gamma_slider.setValue(int(settings.gamma * 100))
        self.colormap_combo.setCurrentText(settings.colormap)


def _estimate_cache_bytes(value):
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            if not isinstance(item, np.ndarray):
                return None
            total += int(item.nbytes)
        return total
    return None


class ByteBudgetCache:
    def __init__(self, max_bytes):
        self.cache = OrderedDict()
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self.lock = Lock()
        self.closed = False

    def get(self, key):
        with self.lock:
            if self.closed or key not in self.cache:
                return None
            value, size_bytes = self.cache.pop(key)
            self.cache[key] = (value, size_bytes)
            return value

    def put(self, key, value):
        size_bytes = _estimate_cache_bytes(value)
        if size_bytes is None or size_bytes <= 0:
            return
        with self.lock:
            if self.closed:
                return
            if key in self.cache:
                _, old_size = self.cache.pop(key)
                self.current_bytes -= old_size
            if size_bytes > self.max_bytes:
                self.cache.clear()
                self.current_bytes = 0
                return
            while self.cache and self.current_bytes + size_bytes > self.max_bytes:
                _old_key, (_old_value, old_size) = self.cache.popitem(last=False)
                self.current_bytes -= old_size
            self.cache[key] = (value, size_bytes)
            self.current_bytes += size_bytes

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_bytes = 0

    def close(self):
        with self.lock:
            self.closed = True
            self.cache.clear()
            self.current_bytes = 0


class LRUCache(ByteBudgetCache):
    def __init__(self, capacity):
        super().__init__(max_bytes=max(1, int(capacity)) * 1024 * 1024)


class MultiLevelCache:
    def __init__(self, file_reader, cache_budget_mb=2048):
        self.file_reader = file_reader
        self.cache_budget_mb = int(cache_budget_mb)
        total_bytes = max(1, int(self.cache_budget_mb * 1024 * 1024))
        raw_bytes = int(total_bytes * 0.60)
        rgb_bytes = int(total_bytes * 0.20)
        reg_bytes = total_bytes - raw_bytes - rgb_bytes
        self.raw_cache = ByteBudgetCache(raw_bytes)
        self.rgb_cache = ByteBudgetCache(rgb_bytes)
        self.reg_rgb_cache = ByteBudgetCache(reg_bytes)
        self.lut_generator = LUTGenerator()
        self.io_lock = Lock()
        self.drift_cache = {}
        self.drift_lock = Lock()
        self._shutdown = False

        metadata = file_reader.get_metadata()
        self.metadata = {
            "num_channels": metadata["num_channels"],
            "raw_num_channels": metadata.get("raw_num_channels", metadata["num_channels"]),
            "num_frames": metadata["num_frames"],
            "height": metadata["height"],
            "width": metadata["width"],
            "exposure_ms": metadata.get("exposure_ms", metadata.get("time_ms", 100.0)),
            "time_ms": metadata.get("time_ms", metadata.get("exposure_ms", 100.0)),
            "exposure_source": metadata.get("exposure_source", ""),
            "channel_names": metadata["channel_names"],
            "raw_channel_names": metadata.get("raw_channel_names", metadata["channel_names"]),
            "channel_index_map": metadata.get("channel_index_map", list(range(metadata["num_channels"]))),
            "channel_profile": metadata.get("channel_profile", "standard"),
        }
        logger.info(f"[MultiLevelCache] Initialized {self.metadata['num_channels']}ch "
                     f"{self.metadata['num_frames']}frames {self.metadata['height']}x{self.metadata['width']}")

    def get_raw_frame(self, frame_idx):
        if self._shutdown:
            return None
        cached = self.raw_cache.get(frame_idx)
        if cached is not None:
            return cached
        with self.io_lock:
            if self._shutdown:
                return None
            cached2 = self.raw_cache.get(frame_idx)
            if cached2 is not None:
                return cached2
            frame = self.file_reader.get_frame(frame_idx)
            frame_uint16 = []
            for channel in frame:
                ch = np.asarray(channel)
                if ch.dtype != np.uint16 or not ch.dtype.isnative:
                    ch = np.clip(ch, 0, 65535).astype(np.uint16)
                else:
                    ch = ch.copy()
                frame_uint16.append(ch)
            self.raw_cache.put(frame_idx, frame_uint16)
            return frame_uint16

    def get_channel_block(self, start_frame, stop_frame, channel_idx):
        if self._shutdown:
            return None
        start = int(start_frame)
        stop = int(stop_frame)
        channel_idx = int(channel_idx)
        num_frames = int(self.metadata.get("num_frames", 0))
        num_channels = int(self.metadata.get("num_channels", 0))
        if start < 0 or stop < start or stop > num_frames:
            raise IndexError(f"Frame block [{start}, {stop}) is out of range for {num_frames} frames")
        if channel_idx < 0 or channel_idx >= num_channels:
            raise IndexError(f"Channel {channel_idx} out of range")
        reader_block = getattr(self.file_reader, "get_channel_block", None)
        if not callable(reader_block):
            raise AttributeError("file reader does not support channel block reads")
        with self.io_lock:
            if self._shutdown:
                return None
            stack = reader_block(start, stop, channel_idx)
        arr = np.asarray(stack)
        if arr.ndim != 3:
            raise ValueError(f"Channel block must be 3D, got shape {arr.shape}")
        if arr.dtype == np.uint16 and arr.dtype.isnative:
            return np.ascontiguousarray(arr.copy())
        return np.ascontiguousarray(np.clip(arr, 0, 65535).astype(np.uint16))

    def get_rgb_image(self, frame_idx, channel_idx, lut_settings):
        cache_key = (frame_idx, channel_idx, lut_settings.get_hash())
        cached = self.rgb_cache.get(cache_key)
        if cached is not None:
            return cached
        raw_frame = self.get_raw_frame(frame_idx)
        if raw_frame is None:
            return None
        if channel_idx >= len(raw_frame):
            raise IndexError(f"Channel {channel_idx} out of range")
        channel_data = raw_frame[channel_idx]
        if channel_data.dtype != np.uint16:
            channel_data = np.clip(channel_data, 0, 65535).astype(np.uint16)
        rgb_image = self.lut_generator.apply_lut(channel_data, lut_settings)
        self.rgb_cache.put(cache_key, rgb_image)
        return rgb_image

    def get_registered_display_rgb_image(self, frame_idx, channel_idx, channel_name, lut_settings, registration_params):
        reg_model = None
        if registration_params is not None:
            reg_model = RegistrationConverter.from_legacy_or_new(registration_params)
        reg_hash = "0" if reg_model is None else _stable_hash(reg_model.to_dict())
        cache_key = ("reg_rgb", frame_idx, channel_idx, channel_name, lut_settings.get_hash(), reg_hash)
        cached = self.reg_rgb_cache.get(cache_key)
        if cached is not None:
            return cached
        raw_frame = self.get_raw_frame(frame_idx)
        if raw_frame is None:
            return None
        channel_data = raw_frame[channel_idx]
        h, w = int(self.metadata["height"]), int(self.metadata["width"])
        if reg_model is not None:
            channel_data = RegistrationDisplayApplier.warp_image_to_reference(channel_data, channel_name, reg_model, (h, w))
        if channel_data.dtype != np.uint16:
            channel_data = np.clip(channel_data, 0, 65535).astype(np.uint16)
        rgb_image = self.lut_generator.apply_lut(channel_data, lut_settings)
        self.reg_rgb_cache.put(cache_key, rgb_image)
        return rgb_image

    def get_registered_display_raw_image(self, frame_idx, channel_idx, channel_name, registration_params):
        reg_model = None
        if registration_params is not None:
            reg_model = RegistrationConverter.from_legacy_or_new(registration_params)
        reg_hash = "0" if reg_model is None else _stable_hash(reg_model.to_dict())
        cache_key = ("reg_raw", frame_idx, channel_idx, channel_name, reg_hash)
        cached = self.reg_rgb_cache.get(cache_key)
        if cached is not None:
            return cached
        raw_frame = self.get_raw_frame(frame_idx)
        if raw_frame is None:
            return None
        channel_data = raw_frame[channel_idx]
        h, w = int(self.metadata["height"]), int(self.metadata["width"])
        if reg_model is not None:
            channel_data = RegistrationDisplayApplier.warp_image_to_reference(channel_data, channel_name, reg_model, (h, w))
        if channel_data.dtype != np.uint16:
            channel_data = np.clip(channel_data, 0, 65535).astype(np.uint16)
        self.reg_rgb_cache.put(cache_key, channel_data)
        return channel_data

    def get_drift_cache(self, key):
        with self.drift_lock:
            return self.drift_cache.get(key)

    def set_drift_cache(self, key, value):
        with self.drift_lock:
            self.drift_cache[key] = value

    def clear_drift_cache(self):
        with self.drift_lock:
            self.drift_cache.clear()

    def trigger_preload(self, center_frame, preload_range=5):
        if self._shutdown:
            return
        num_frames = int(self.metadata.get("num_frames", 0))
        for f in range(max(0, center_frame), min(num_frames, center_frame + preload_range + 1)):
            self.raw_cache.get(f)
            if self.raw_cache.get(f) is None:
                try:
                    self.get_raw_frame(f)
                except Exception:
                    pass

    def clear_caches(self):
        self.raw_cache.clear()
        self.rgb_cache.clear()
        self.reg_rgb_cache.clear()
        self.lut_generator.clear_cache()
        self.clear_drift_cache()

    def shutdown(self):
        self._shutdown = True
        self.raw_cache.close()
        self.rgb_cache.close()
        self.reg_rgb_cache.close()
        self.lut_generator.clear_cache()
        self.clear_drift_cache()


class DebounceTimer(QTimer):
    def __init__(self, delay_ms=50, parent=None):
        super().__init__(parent)
        self.delay_ms = delay_ms
        self.callback = None
        self.setSingleShot(True)
        self.timeout.connect(self._on_timeout)

    def trigger(self, callback):
        self.callback = callback
        self.start(self.delay_ms)

    def _on_timeout(self):
        if self.callback:
            cb = self.callback
            self.callback = None
            try:
                cb()
            except Exception:
                logger.exception("Debounced callback failed")
