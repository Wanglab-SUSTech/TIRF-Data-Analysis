from __future__ import annotations

import threading
from typing import Any

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from detection import MoleculeDetector
from project_state import PlaybackState, ProjectState
from registration import RegistrationConverter, RegistrationIO, REFERENCE_CHANNEL
from runtime_services import PREVIEW_TOP_K, compute_file_fingerprint
from ui_lut_cache import DebounceTimer, LUTSettings
from view_models import DetectionPreviewViewModel, MainViewModel


def _is_cancelled_message(message: str) -> bool:
    text = str(message or "").lower()
    return "cancel" in text


class DetectionPreviewWorker(QThread):
    progress = pyqtSignal(int, int, str)
    result_ready = pyqtSignal(int, object)
    error = pyqtSignal(int, str)

    def __init__(self, frame, preview_limit: int = PREVIEW_TOP_K, parent=None):
        super().__init__(parent)
        self._frame = frame
        self._preview_limit = int(preview_limit)
        self._detector = MoleculeDetector()
        self._condition = threading.Condition()
        self._pending_request: tuple[int, dict[str, Any]] | None = None
        self._latest_request_id = 0
        self._shutdown = False

    def submit(self, params: dict[str, Any]) -> int:
        with self._condition:
            self._latest_request_id += 1
            request_id = int(self._latest_request_id)
            self._pending_request = (request_id, dict(params))
            self._condition.notify_all()
        return request_id

    def cancel_pending(self) -> None:
        with self._condition:
            self._latest_request_id += 1
            self._pending_request = None
            self._condition.notify_all()

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._latest_request_id += 1
            self._pending_request = None
            self._condition.notify_all()

    def _is_stale_request(self, request_id: int) -> bool:
        with self._condition:
            return self._shutdown or int(request_id) != int(self._latest_request_id)

    def run(self) -> None:
        while True:
            with self._condition:
                while not self._shutdown and self._pending_request is None:
                    self._condition.wait()
                if self._shutdown:
                    return
                request_id, params = self._pending_request
                self._pending_request = None

            def cancel_check() -> bool:
                return self._is_stale_request(request_id)

            def progress_cb(progress: int, message: str) -> None:
                if cancel_check():
                    raise RuntimeError("Task cancelled")
                self.progress.emit(int(request_id), int(progress), str(message))

            try:
                positions, stats = self._detector.detect_interactive(
                    self._frame,
                    progress_cb=progress_cb,
                    cancel_check=cancel_check,
                    **params,
                )
                if cancel_check():
                    continue
                self.result_ready.emit(
                    int(request_id),
                    {
                        "positions": list(positions or []),
                        "results": list(stats.get("results", []) or []),
                        "stats": dict(stats or {}),
                    },
                )
            except RuntimeError as exc:
                message = str(exc or "Task cancelled")
                if _is_cancelled_message(message) and self._is_stale_request(request_id):
                    continue
                self.error.emit(int(request_id), message)
            except Exception as exc:
                self.error.emit(int(request_id), str(exc))


class MainWindowController:
    def __init__(self):
        self.state = ProjectState()
        self.view_model = MainViewModel(self.state)

    def capture_window_state(self, window) -> ProjectState:
        state = self.state
        state.nd2_file = getattr(window, "nd2_file", None)
        state.nd2_fingerprint = dict(getattr(window, "nd2_fingerprint", {}) or {})
        if not state.nd2_fingerprint and state.nd2_file:
            state.nd2_fingerprint = compute_file_fingerprint(state.nd2_file)
        state.registration_params = getattr(window, "registration_params", None)
        state.detect_channel = int(getattr(window, "detect_channel", 0) or 0)
        state.channel_names = list(getattr(window, "channel_names", []) or [])
        state.num_channels = int(getattr(window, "num_channels", state.num_channels) or 0)
        state.num_frames = int(getattr(window, "num_frames", state.num_frames) or 0)
        state.current_frame = int(getattr(window, "current_frame", 0) or 0)
        state.exposure_s = float(getattr(window, "exposure_s", 0.1) or 0.1)
        state.exposure_source = str(getattr(window, "exposure_source", state.exposure_source) or "ND2 metadata")
        metadata_exposure_ms = getattr(window, "metadata_exposure_ms", state.metadata_exposure_ms)
        state.metadata_exposure_ms = None if metadata_exposure_ms is None else float(metadata_exposure_ms)
        state.metadata_exposure_source = str(getattr(window, "metadata_exposure_source", state.metadata_exposure_source) or "")
        state.exposure_override_enabled = bool(getattr(window, "exposure_override_enabled", False))
        override_ms = getattr(window, "exposure_override_ms", None)
        state.exposure_override_ms = None if override_ms is None else float(override_ms)

        state.molecules = [(int(m[0]), float(m[1]), float(m[2])) for m in list(getattr(window, "molecules", []) or [])]
        state.base_molecules = [(int(m[0]), float(m[1]), float(m[2])) for m in list(getattr(window, "base_molecules", state.base_molecules) or [])]
        state.molecule_features = {
            int(key): dict(value or {})
            for key, value in dict(getattr(window, "molecule_features", {}) or {}).items()
        }
        state.base_molecule_features = {
            int(key): dict(value or {})
            for key, value in dict(getattr(window, "base_molecule_features", {}) or {}).items()
        }
        state.deleted_molecules = set(int(v) for v in (getattr(window, "deleted_molecules", set()) or set()))
        state.selected_molecule = getattr(window, "selected_molecule", None)
        state.molecule_intensities = dict(getattr(window, "molecule_intensities", {}) or {})
        state.intensity_scales = dict(getattr(window, "intensity_scales", {}) or {})
        state.analysis_recipe = dict(getattr(window, "analysis_recipe", {}) or {})
        state.operation_log = [dict(entry or {}) for entry in list(getattr(window, "operation_log", []) or [])]
        state.uncertainty_filter = dict(getattr(window, "uncertainty_filter", {}) or {})
        state.active_edit_tool = str(getattr(window, "active_edit_tool", "pan") or "pan")
        state.view_transform = dict(getattr(window, "view_transform", {}) or {})
        state.results_mode = str(getattr(window, "results_mode", "normal") or "normal")
        state.drift_result = dict(getattr(window, "drift_result", {}) or {})
        state.backend_name = str(state.backend_name or "")
        state.backend_capabilities = dict(state.backend_capabilities or {})

        file_reader = getattr(window, "file_reader", None)
        if file_reader is not None:
            state.backend_name = str(getattr(file_reader, "backend_name", "") or "")
            state.backend_capabilities = dict(getattr(file_reader, "backend_capabilities", {}) or {})

        multi_cache = getattr(window, "multi_cache", None)
        if multi_cache is not None:
            metadata = dict(getattr(multi_cache, "metadata", {}) or {})
            state.height = int(metadata.get("height", state.height) or 0)
            state.width = int(metadata.get("width", state.width) or 0)
            if not state.channel_names:
                state.channel_names = list(metadata.get("channel_names", []) or [])
            if not state.num_channels:
                state.num_channels = int(metadata.get("num_channels", 0) or 0)
            if not state.num_frames:
                state.num_frames = int(metadata.get("num_frames", 0) or 0)

        lut_settings_by_channel = dict(state.lut_settings_by_channel or {})
        lut_panels = list(getattr(window, "lut_panels", []) or [])
        for idx, channel_name in enumerate(state.channel_names):
            if idx < len(lut_panels):
                lut_settings_by_channel[str(channel_name)] = lut_panels[idx].get_settings().to_dict()
        state.lut_settings_by_channel = lut_settings_by_channel

        state.playback = PlaybackState(
            is_playing=bool(getattr(window, "is_playing", False)),
            speed=float(getattr(window, "play_speed", 1.0) or 1.0),
        )

        video_widgets = list(getattr(window, "video_widgets", []) or [])
        if video_widgets and hasattr(video_widgets[0], "get_view_state"):
            state.view_transform = dict(video_widgets[0].get_view_state() or {})

        return state

    def apply_basic_labels(self, window) -> None:
        self.capture_window_state(window)
        self.view_model.state = self.state
        if hasattr(window, "data_info_label"):
            window.data_info_label.setText(self.view_model.data_info_text())
        if hasattr(window, "detect_result_label"):
            window.detect_result_label.setText(self.view_model.detection_result_text())
        if hasattr(window, "frame_label"):
            window.frame_label.setText(self.view_model.frame_label_text())

    def sync_state_from_runtime_sources(self, state: ProjectState, *, file_reader=None, multi_cache=None) -> ProjectState:
        if multi_cache is not None:
            metadata = dict(getattr(multi_cache, "metadata", {}) or {})
            state.num_channels = int(metadata.get("num_channels", state.num_channels) or state.num_channels)
            state.num_frames = int(metadata.get("num_frames", state.num_frames) or state.num_frames)
            state.height = int(metadata.get("height", state.height) or state.height)
            state.width = int(metadata.get("width", state.width) or state.width)
            state.channel_names = list(metadata.get("channel_names", state.channel_names) or state.channel_names)
            exposure_ms = metadata.get("exposure_ms", metadata.get("time_ms", state.exposure_s * 1000.0))
            state.metadata_exposure_ms = float(exposure_ms)
            state.metadata_exposure_source = str(metadata.get("exposure_source", state.metadata_exposure_source) or "")
            if state.exposure_override_enabled and state.exposure_override_ms is not None:
                state.exposure_s = float(state.exposure_override_ms) / 1000.0
                state.exposure_source = "manual_override"
            else:
                state.exposure_s = float(exposure_ms) / 1000.0
                state.exposure_source = state.metadata_exposure_source or "ND2 metadata"
        if file_reader is not None:
            state.backend_name = str(getattr(file_reader, "backend_name", "") or "")
            state.backend_capabilities = dict(getattr(file_reader, "backend_capabilities", {}) or {})
        return state

    @staticmethod
    def apply_registration_label(window) -> None:
        registration_params = getattr(window, "registration_params", None)
        if not registration_params:
            window.reg_info_label.setText("No registration loaded")
            return
        try:
            model = RegistrationConverter.from_legacy_or_new(registration_params)
            window.reg_info_label.setText(RegistrationIO.describe(model))
        except Exception:
            window.reg_info_label.setText("Registration loaded (from project)")

    def apply_project_state_to_window(self, window, state: ProjectState) -> None:
        window.registration_params = state.registration_params
        window.nd2_file = state.nd2_file
        window.nd2_fingerprint = dict(state.nd2_fingerprint or {})
        window.detect_channel = int(state.detect_channel)
        window.channel_names = list(state.channel_names)
        window.num_channels = int(state.num_channels)
        window.num_frames = int(state.num_frames)
        window.current_frame = int(state.current_frame)
        window.exposure_s = float(state.exposure_s)
        window.exposure_source = str(state.exposure_source or "ND2 metadata")
        window.metadata_exposure_ms = None if state.metadata_exposure_ms is None else float(state.metadata_exposure_ms)
        window.metadata_exposure_source = str(state.metadata_exposure_source or "")
        window.exposure_override_enabled = bool(state.exposure_override_enabled)
        window.exposure_override_ms = None if state.exposure_override_ms is None else float(state.exposure_override_ms)
        window.molecules = list(state.molecules)
        window.base_molecules = list(state.base_molecules)
        window.molecule_features = {int(k): dict(v or {}) for k, v in state.molecule_features.items()}
        window.base_molecule_features = {int(k): dict(v or {}) for k, v in state.base_molecule_features.items()}
        window.deleted_molecules = set(state.deleted_molecules)
        window.selected_molecule = state.selected_molecule
        window.molecule_intensities = dict(state.molecule_intensities)
        window.intensity_scales = dict(state.intensity_scales)
        window.is_playing = bool(state.playback.is_playing)
        window.play_speed = float(state.playback.speed)
        window.analysis_recipe = dict(state.analysis_recipe or {})
        window.operation_log = [dict(entry or {}) for entry in list(state.operation_log or [])]
        window.uncertainty_filter = dict(state.uncertainty_filter or {})
        window.active_edit_tool = str(state.active_edit_tool or "pan")
        window.view_transform = dict(state.view_transform or {})
        window.results_mode = str(state.results_mode or "normal")
        window.drift_result = dict(state.drift_result or {})

    def project_save_kwargs(self, window) -> dict[str, Any]:
        state = self.capture_window_state(window)
        return {
            "nd2_file": state.nd2_file,
            "nd2_fingerprint": state.nd2_fingerprint,
            "registration_params": state.registration_params,
            "detect_channel": state.detect_channel,
            "channel_names": state.channel_names,
            "molecules": state.molecules,
            "base_molecules": state.base_molecules or state.molecules,
            "molecule_features": state.molecule_features,
            "base_molecule_features": state.base_molecule_features or state.molecule_features,
            "deleted_molecules": state.deleted_molecules,
            "molecule_intensities": state.molecule_intensities,
            "intensity_scales": state.intensity_scales,
            "lut_settings_by_channel": state.lut_settings_by_channel,
            "exposure_s": state.exposure_s,
            "exposure_source": state.exposure_source,
            "metadata_exposure_ms": state.metadata_exposure_ms,
            "metadata_exposure_source": state.metadata_exposure_source,
            "exposure_override_enabled": state.exposure_override_enabled,
            "exposure_override_ms": state.exposure_override_ms,
            "current_frame": state.current_frame,
            "num_frames": state.num_frames,
            "num_channels": state.num_channels,
            "height": state.height,
            "width": state.width,
            "backend_name": state.backend_name,
            "backend_capabilities": state.backend_capabilities,
            "selected_molecule": state.selected_molecule,
            "playback": {
                "is_playing": state.playback.is_playing,
                "speed": state.playback.speed,
            },
            "analysis_recipe": state.analysis_recipe,
            "operation_log": state.operation_log,
            "uncertainty_filter": state.uncertainty_filter,
            "active_edit_tool": state.active_edit_tool,
            "view_transform": state.view_transform,
            "results_mode": state.results_mode,
            "drift_result": state.drift_result,
        }

    def apply_project_manifest(self, manifest: dict[str, Any]) -> ProjectState:
        state = self.state
        state.nd2_file = manifest.get("nd2_file") or None
        state.nd2_fingerprint = dict(manifest.get("nd2_fingerprint", {}) or {})
        state.registration_params = manifest.get("registration_params")
        state.detect_channel = int(manifest.get("detect_channel", 0) or 0)
        state.channel_names = list(manifest.get("channel_names", []) or [])
        state.num_channels = int(manifest.get("num_channels", len(state.channel_names)) or len(state.channel_names))
        state.num_frames = int(manifest.get("num_frames", 0) or 0)
        state.height = int(manifest.get("height", 0) or 0)
        state.width = int(manifest.get("width", 0) or 0)
        state.current_frame = int(manifest.get("current_frame", 0) or 0)
        state.exposure_s = float(manifest.get("exposure_s", 0.1) or 0.1)
        state.exposure_source = str(manifest.get("exposure_source", "ND2 metadata") or "ND2 metadata")
        metadata_exposure_ms = manifest.get("metadata_exposure_ms")
        state.metadata_exposure_ms = None if metadata_exposure_ms is None else float(metadata_exposure_ms)
        state.metadata_exposure_source = str(manifest.get("metadata_exposure_source", "") or "")
        state.exposure_override_enabled = bool(manifest.get("exposure_override_enabled", False))
        override_ms = manifest.get("exposure_override_ms")
        state.exposure_override_ms = None if override_ms is None else float(override_ms)
        state.backend_name = str(manifest.get("backend_name", "") or "")
        state.backend_capabilities = dict(manifest.get("backend_capabilities", {}) or {})
        state.molecules = [(int(m[0]), float(m[1]), float(m[2])) for m in manifest.get("molecules", []) or []]
        state.base_molecules = [(int(m[0]), float(m[1]), float(m[2])) for m in manifest.get("base_molecules", state.molecules) or []]
        state.molecule_features = {int(k): dict(v or {}) for k, v in dict(manifest.get("molecule_features", {}) or {}).items()}
        state.base_molecule_features = {int(k): dict(v or {}) for k, v in dict(manifest.get("base_molecule_features", state.molecule_features) or {}).items()}
        state.deleted_molecules = set(int(v) for v in (manifest.get("deleted_molecules", []) or []))
        state.selected_molecule = manifest.get("selected_molecule")
        state.molecule_intensities = dict(manifest.get("molecule_intensities", {}) or {})
        state.intensity_scales = {str(k): float(v) for k, v in dict(manifest.get("intensity_scales", {}) or {}).items()}
        state.lut_settings_by_channel = dict(manifest.get("lut_settings_by_channel", {}) or {})
        playback = dict(manifest.get("playback", {}) or {})
        state.playback = PlaybackState(
            is_playing=bool(playback.get("is_playing", False)),
            speed=float(playback.get("speed", 1.0) or 1.0),
        )
        state.analysis_recipe = dict(manifest.get("analysis_recipe", {}) or {})
        state.operation_log = [dict(entry or {}) for entry in list(manifest.get("operation_log", []) or [])]
        state.uncertainty_filter = dict(manifest.get("uncertainty_filter", {}) or {})
        state.active_edit_tool = str(manifest.get("active_edit_tool", "pan") or "pan")
        state.view_transform = dict(manifest.get("view_transform", {}) or {})
        state.results_mode = str(manifest.get("results_mode", "normal") or "normal")
        state.drift_result = dict(manifest.get("drift_result", {}) or {})
        return state

    @staticmethod
    def active_molecules(state: ProjectState) -> list[tuple[int, float, float]]:
        return state.active_molecules()

    def restore_channel_combo(self, window, *, fallback_channels: list[str] | tuple[str, ...]) -> None:
        channels = list(getattr(window, "channel_names", []) or [])
        items = channels or list(fallback_channels)
        window.channel_combo.clear()
        window.channel_combo.addItems(items)
        if channels:
            detect_channel = int(getattr(window, "detect_channel", 0) or 0)
            detect_channel = max(0, min(detect_channel, len(channels) - 1))
            window.channel_combo.setCurrentIndex(detect_channel)
        elif REFERENCE_CHANNEL in items:
            window.channel_combo.setCurrentText(REFERENCE_CHANNEL)

    def restore_lut_panels(self, window, state: ProjectState) -> bool:
        if not getattr(window, "lut_panels", None):
            return False
        restored = False
        lut_by_channel = dict(getattr(state, "lut_settings_by_channel", {}) or {})
        for idx, panel in enumerate(window.lut_panels):
            channel_name = window.channel_names[idx] if idx < len(window.channel_names) else None
            lut_dict = lut_by_channel.get(channel_name) if channel_name else None
            if not lut_dict:
                continue
            panel.apply_settings(LUTSettings.from_dict(lut_dict))
            restored = True
        return restored

    @staticmethod
    def restore_intensity_scale_controls(window, state: ProjectState) -> None:
        for channel_name, spin in dict(getattr(window, "intensity_scale_spinboxes", {}) or {}).items():
            if channel_name not in state.intensity_scales:
                continue
            spin.blockSignals(True)
            spin.setValue(float(state.intensity_scales[channel_name]))
            spin.blockSignals(False)

    def restore_molecule_list(self, window, state: ProjectState, *, fallback_to_first: bool = True, preferred_row: int | None = None) -> int | None:
        self.view_model.state = state
        window.molecule_list.clear()
        for _mol_id, label in self.view_model.molecule_list_items():
            window.molecule_list.addItem(label)

        selected_row = self.selected_active_row(state)
        if selected_row is not None and 0 <= selected_row < window.molecule_list.count():
            window.molecule_list.setCurrentRow(selected_row)
            return selected_row

        plot_widget = getattr(window, "plot_widget", None)
        if plot_widget is not None:
            plot_widget.clear_plot()

        if preferred_row is not None and 0 <= preferred_row < window.molecule_list.count():
            window.selected_molecule = None
            window.molecule_list.setCurrentRow(preferred_row)
            return preferred_row

        window.selected_molecule = None
        if fallback_to_first and window.molecule_list.count() > 0:
            window.molecule_list.setCurrentRow(0)
            return 0
        return None

    def apply_detection_results(self, window, records: list[dict[str, Any]], params: dict[str, Any] | None = None) -> ProjectState:
        ordered = sorted(list(records or []), key=lambda item: (float(item.get("y", 0.0)), float(item.get("x", 0.0))))
        molecules = []
        features = {}
        for idx, record in enumerate(ordered, start=1):
            features[idx] = {**dict(record or {}), "x": float(record.get("x", record.get("pos", [0.0, 0.0])[0])), "y": float(record.get("y", record.get("pos", [0.0, 0.0])[1])), "pos": [float(record.get("x", record.get("pos", [0.0, 0.0])[0])), float(record.get("y", record.get("pos", [0.0, 0.0])[1]))]}
            molecules.append((idx, float(features[idx]["x"]), float(features[idx]["y"])))

        window.molecules = list(molecules)
        window.base_molecules = list(molecules)
        window.molecule_features = dict(features)
        window.base_molecule_features = {mol_id: dict(feature or {}) for mol_id, feature in features.items()}
        window.deleted_molecules = set()
        window.selected_molecule = None
        window.molecule_intensities = {}
        window.operation_log = []
        window.analysis_recipe = dict(getattr(window, "analysis_recipe", {}) or {})
        if params:
            window.analysis_recipe["detection"] = dict(params)
        state = self.capture_window_state(window)
        self.restore_molecule_list(window, state, preferred_row=0)
        return state

    def nearest_active_row_for_point(self, state: ProjectState, world_x: int | float, world_y: int | float, *, radius: float = 8.0) -> int | None:
        best_row = None
        best_dist2 = None
        radius_sq = float(radius) * float(radius)
        for row, (_mol_id, x, y) in enumerate(self.active_molecules(state)):
            dx = float(x) - float(world_x)
            dy = float(y) - float(world_y)
            dist2 = dx * dx + dy * dy
            if dist2 <= radius_sq and (best_dist2 is None or dist2 < best_dist2):
                best_dist2 = dist2
                best_row = row
        return best_row

    @staticmethod
    def _next_molecule_id(state: ProjectState) -> int:
        ids = [int(m[0]) for m in list(state.molecules or [])]
        return (max(ids) + 1) if ids else 1

    def add_manual_molecule(self, window, record: dict[str, Any]) -> int:
        state = self.capture_window_state(window)
        mol_id = self._next_molecule_id(state)
        x = float(record.get("x", record.get("pos", [0.0, 0.0])[0]))
        y = float(record.get("y", record.get("pos", [0.0, 0.0])[1]))
        state.molecules.append((mol_id, x, y))
        state.molecule_features[mol_id] = {**dict(record or {}), "x": x, "y": y, "pos": [x, y], "manual": True}
        state.operation_log.append({"op": "add", "mol_id": mol_id, "x": x, "y": y})
        self.apply_project_state_to_window(window, state)
        return mol_id

    def move_molecule(self, window, mol_id: int, record: dict[str, Any]) -> None:
        state = self.capture_window_state(window)
        x = float(record.get("x", record.get("pos", [0.0, 0.0])[0]))
        y = float(record.get("y", record.get("pos", [0.0, 0.0])[1]))
        state.molecules = [
            (int(mid), x if int(mid) == int(mol_id) else float(mx), y if int(mid) == int(mol_id) else float(my))
            for mid, mx, my in state.molecules
        ]
        state.molecule_features[int(mol_id)] = {**dict(record or {}), "x": x, "y": y, "pos": [x, y], "manual": True}
        state.operation_log.append({"op": "move", "mol_id": int(mol_id), "x": x, "y": y})
        self.apply_project_state_to_window(window, state)

    def delete_active_molecule_at_row(self, window, row: int) -> tuple[int | None, bool]:
        state = self.capture_window_state(window)
        active = self.active_molecules(state)
        if row < 0 or row >= len(active):
            return None, False

        mol_id = int(active[row][0])
        state.deleted_molecules.add(mol_id)
        state.operation_log.append({"op": "delete", "mol_id": mol_id})
        selection_cleared = state.selected_molecule == mol_id
        if selection_cleared:
            state.selected_molecule = None

        self.apply_project_state_to_window(window, state)
        remaining = state.active_molecules()
        next_row = min(int(row), len(remaining) - 1) if remaining else None
        self.restore_molecule_list(window, state, fallback_to_first=False, preferred_row=next_row)
        return next_row, selection_cleared

    def selected_active_row(self, state: ProjectState) -> int | None:
        if state.selected_molecule is None:
            return None
        for row, molecule in enumerate(self.active_molecules(state)):
            if int(molecule[0]) == int(state.selected_molecule):
                return row
        return None


class DetectionPreviewController(QObject):
    def __init__(self, dialog, first_frame, preview_limit: int = PREVIEW_TOP_K):
        super().__init__(dialog)
        self.dialog = dialog
        self.first_frame = first_frame
        self.preview_limit = int(preview_limit)
        self.preview_worker = DetectionPreviewWorker(first_frame, preview_limit=self.preview_limit, parent=dialog)
        self.preview_worker.progress.connect(self._handle_preview_progress)
        self.preview_worker.result_ready.connect(self._handle_preview_result)
        self.preview_worker.error.connect(self._handle_preview_error)
        self.preview_worker.start()
        self.debounce = DebounceTimer(delay_ms=200, parent=dialog)
        self._latest_params: dict[str, Any] = {}
        self._active_request_id: int | None = None
        self._active_params: dict[str, Any] | None = None
        self._last_completed_request_id: int | None = None
        self._last_completed_params: dict[str, Any] | None = None
        self._pending_accept = False
        self._current_positions: list[tuple[float, float]] = []
        self._current_results: list[dict[str, Any]] = []
        self._current_stats: dict[str, Any] = {}
        self._closed = False

    @staticmethod
    def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
        params = dict(params or {})
        return {
            "sigma": round(float(params.get("sigma", 3.5)), 6),
            "min_snr": round(float(params.get("min_snr", 4.0)), 6),
            "min_r2": round(float(params.get("min_r2", 0.0)), 6),
            "min_distance": None if params.get("min_distance") is None else int(params.get("min_distance")),
            "auto_bg": bool(params.get("auto_bg", params.get("auto_background", True))),
            "bg_radius": None if params.get("bg_radius") is None else int(params.get("bg_radius")),
        }

    @classmethod
    def _params_match(cls, left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
        if left is None or right is None:
            return False
        return cls._normalize_params(left) == cls._normalize_params(right)

    def schedule_preview(self, params: dict[str, Any]) -> None:
        if self._closed:
            return
        self._latest_params = self._normalize_params(params)
        if self._pending_accept:
            self.dialog.set_status("Waiting for input to settle before confirm...")
        else:
            self.dialog.set_status("Waiting for input to settle...")
        self.debounce.trigger(lambda: self.start_preview(dict(self._latest_params)))

    def start_preview(self, params: dict[str, Any]) -> None:
        if self._closed:
            return
        self._latest_params = self._normalize_params(params)
        # Once a fresh request is submitted, prior successful results are no longer
        # guaranteed to match the current UI state.
        self._last_completed_request_id = None
        self._last_completed_params = None
        request_id = self.preview_worker.submit(dict(self._latest_params))
        self._active_request_id = int(request_id)
        self._active_params = dict(self._latest_params)
        self.dialog.set_busy(True)
        self.dialog.set_progress(None)
        self.dialog.set_confirm_enabled(not self._pending_accept)
        self.dialog.set_cancel_enabled(True)
        if self._pending_accept:
            self.dialog.set_status("Waiting for latest detection before confirm...")
        else:
            self.dialog.set_status("Running detection...")

    def confirm_current_or_latest(self, params: dict[str, Any]) -> None:
        if self._closed:
            return
        normalized = self._normalize_params(params)
        self._latest_params = dict(normalized)
        self._pending_accept = True
        try:
            self.debounce.stop()
        except Exception:
            pass
        if self._active_request_id is None and self._params_match(normalized, self._last_completed_params):
            self._pending_accept = False
            self.dialog.complete_accept()
            return
        self.dialog.set_confirm_enabled(False)
        self.dialog.set_cancel_enabled(True)
        self.dialog.set_busy(True)
        self.dialog.set_progress(None)
        self.dialog.set_status("Waiting for latest detection before confirm...")
        if self._active_request_id is not None and self._params_match(normalized, self._active_params):
            return
        self.start_preview(normalized)

    def cancel_preview(self) -> None:
        if self._active_request_id is not None:
            self.preview_worker.cancel_pending()
            self._active_request_id = None
            self._active_params = None
        self._pending_accept = False

    def shutdown(self) -> None:
        self.shutdown_with_timeout(timeout_ms=2000)

    def shutdown_with_timeout(self, timeout_ms: int | None = 2000) -> bool:
        self._closed = True
        try:
            self.debounce.stop()
        except Exception:
            pass
        self.cancel_preview()
        try:
            self.preview_worker.shutdown()
            if timeout_ms is None:
                return bool(self.preview_worker.wait())
            return bool(self.preview_worker.wait(int(timeout_ms)))
        except Exception:
            return True
        return True

    def _handle_preview_progress(self, request_id: int, progress: int, message: str) -> None:
        if self._closed or request_id != self._active_request_id:
            return
        self.dialog.set_busy(True)
        self.dialog.set_progress(progress)
        self.dialog.set_status(message or "Running detection...")

    def _handle_preview_error(self, request_id: int, message: str) -> None:
        if self._closed or request_id != self._active_request_id:
            return
        self._active_request_id = None
        self._active_params = None
        self._last_completed_request_id = None
        self._last_completed_params = None
        self._pending_accept = False
        self._current_positions = []
        self._current_results = []
        self._current_stats = {}
        self.dialog.set_detected_positions([])
        if hasattr(self.dialog, "set_detected_results"):
            self.dialog.set_detected_results([])
        self.dialog.set_result_count(0)
        self._render_stats()
        self.dialog.set_busy(False)
        self.dialog.set_confirm_enabled(True)
        self.dialog.set_cancel_enabled(True)
        message = str(message or "Detection failed")
        if _is_cancelled_message(message):
            self.dialog.set_status("Detection cancelled")
            self.dialog.set_progress(0)
            return
        self.dialog.set_progress(0)
        self.dialog.set_status("Detection failed")
        QMessageBox.critical(self.dialog, "Error", message)

    def _handle_preview_result(self, request_id: int, payload: dict[str, Any]) -> None:
        if self._closed or request_id != self._active_request_id:
            return
        self._last_completed_request_id = int(request_id)
        self._last_completed_params = dict(self._active_params or self._latest_params)
        self._active_request_id = None
        self._active_params = None
        self._current_positions = [(float(pos[0]), float(pos[1])) for pos in payload.get("positions", []) or []]
        self._current_results = [dict(item or {}) for item in list(payload.get("results", []) or [])]
        self._current_stats = dict(payload.get("stats", {}) or {})
        self.dialog.set_detected_positions(self._current_positions)
        if hasattr(self.dialog, "set_detected_results"):
            self.dialog.set_detected_results(self._current_results)
        self.dialog.set_busy(False)
        self.dialog.set_progress(100)
        self.dialog.set_status("Detection complete")
        self.dialog.set_result_count(len(self._current_positions))
        self.dialog.set_confirm_enabled(not self._pending_accept)
        self.dialog.set_cancel_enabled(True)
        self._render_stats()
        self._maybe_complete_accept()

    def _maybe_complete_accept(self) -> None:
        if not self._pending_accept or self._closed:
            return
        current_params = self._normalize_params(self.dialog.get_params())
        if self._active_request_id is None and self._params_match(current_params, self._last_completed_params):
            self._pending_accept = False
            self.dialog.complete_accept()
            return
        self.dialog.set_confirm_enabled(False)
        self.dialog.set_busy(True)
        self.dialog.set_progress(None)
        self.dialog.set_status("Waiting for latest detection before confirm...")

    def _render_stats(self) -> None:
        vm = DetectionPreviewViewModel(
            positions=self._current_positions,
            stats=self._current_stats,
        )
        self.dialog.set_stats_text(vm.stats_text())
