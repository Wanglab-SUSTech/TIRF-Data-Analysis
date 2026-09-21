from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlaybackState:
    is_playing: bool = False
    speed: float = 1.0


@dataclass
class ProjectState:
    nd2_file: str | None = None
    nd2_fingerprint: dict[str, Any] = field(default_factory=dict)
    registration_params: dict[str, Any] | None = None
    detect_channel: int = 0
    channel_names: list[str] = field(default_factory=list)
    num_channels: int = 0
    num_frames: int = 0
    height: int = 0
    width: int = 0
    current_frame: int = 0
    exposure_s: float = 0.1
    exposure_source: str = "ND2 metadata"
    metadata_exposure_ms: float | None = None
    metadata_exposure_source: str = ""
    exposure_override_enabled: bool = False
    exposure_override_ms: float | None = None
    backend_name: str = ""
    backend_capabilities: dict[str, Any] = field(default_factory=dict)

    molecules: list[tuple[int, float, float]] = field(default_factory=list)
    base_molecules: list[tuple[int, float, float]] = field(default_factory=list)
    molecule_features: dict[int, dict[str, Any]] = field(default_factory=dict)
    base_molecule_features: dict[int, dict[str, Any]] = field(default_factory=dict)
    deleted_molecules: set[int] = field(default_factory=set)
    selected_molecule: int | None = None

    molecule_intensities: dict[int, dict[str, Any]] = field(default_factory=dict)
    intensity_scales: dict[str, float] = field(default_factory=dict)
    lut_settings_by_channel: dict[str, dict[str, Any]] = field(default_factory=dict)
    playback: PlaybackState = field(default_factory=PlaybackState)

    analysis_recipe: dict[str, Any] = field(default_factory=dict)
    operation_log: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_filter: dict[str, Any] = field(default_factory=dict)
    active_edit_tool: str = "pan"
    view_transform: dict[str, Any] = field(default_factory=dict)
    results_mode: str = "normal"
    drift_result: dict[str, Any] = field(default_factory=dict)

    def active_molecules(self) -> list[tuple[int, float, float]]:
        filtered = [mol for mol in self.molecules if int(mol[0]) not in self.deleted_molecules]
        filt = dict(self.uncertainty_filter or {})
        if not filt.get("enabled"):
            return filtered
        max_scalar = filt.get("max_scalar")
        if max_scalar is None:
            return filtered
        out = []
        for mol_id, x, y in filtered:
            feature = dict(self.molecule_features.get(int(mol_id), {}) or {})
            uncertainty = feature.get("loc_uncertainty_scalar")
            if uncertainty is None or feature.get("uncertainty_valid") is False:
                out.append((mol_id, x, y))
                continue
            try:
                uncertainty_value = float(uncertainty)
                if math.isnan(uncertainty_value) or uncertainty_value <= float(max_scalar):
                    out.append((mol_id, x, y))
            except Exception:
                out.append((mol_id, x, y))
        return out

    def frame_label_text(self) -> str:
        if self.num_frames <= 0:
            return "0/0"
        return f"{self.current_frame}/{self.num_frames - 1}"

    def molecule_feature(self, mol_id: int) -> dict[str, Any]:
        return dict(self.molecule_features.get(int(mol_id), {}) or {})

    def uncertainty_enabled(self) -> bool:
        filt = dict(self.uncertainty_filter or {})
        return bool(filt.get("enabled"))

    def reset_data(self) -> None:
        self.nd2_file = None
        self.nd2_fingerprint = {}
        self.registration_params = None
        self.detect_channel = 0
        self.channel_names = []
        self.num_channels = 0
        self.num_frames = 0
        self.height = 0
        self.width = 0
        self.current_frame = 0
        self.exposure_s = 0.1
        self.exposure_source = "ND2 metadata"
        self.metadata_exposure_ms = None
        self.metadata_exposure_source = ""
        self.exposure_override_enabled = False
        self.exposure_override_ms = None
        self.backend_name = ""
        self.backend_capabilities = {}
        self.molecules = []
        self.base_molecules = []
        self.molecule_features = {}
        self.base_molecule_features = {}
        self.deleted_molecules = set()
        self.selected_molecule = None
        self.molecule_intensities = {}
        self.intensity_scales = {}
        self.lut_settings_by_channel = {}
        self.playback = PlaybackState()
        self.analysis_recipe = {}
        self.operation_log = []
        self.uncertainty_filter = {}
        self.active_edit_tool = "pan"
        self.view_transform = {}
        self.results_mode = "normal"
        self.drift_result = {}
