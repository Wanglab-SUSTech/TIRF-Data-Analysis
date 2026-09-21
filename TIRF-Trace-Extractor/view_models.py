from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from project_state import ProjectState


@dataclass
class MainViewModel:
    state: ProjectState

    def data_info_text(self) -> str:
        if not self.state.nd2_file:
            return "No ND2 data loaded"

        lines = [
            f"File: {os.path.basename(self.state.nd2_file)}",
            f"Channels: {self.state.num_channels}",
            f"Channel names: {', '.join(self.state.channel_names)}",
            f"Frames: {self.state.num_frames}",
        ]
        if self.state.height and self.state.width:
            lines.append(f"Size: {self.state.height}x{self.state.width}")
        exposure_ms = float(self.state.exposure_s) * 1000.0
        if self.state.exposure_override_enabled:
            lines.append(f"Exposure: {exposure_ms:.4f} ms (manual override)")
            if self.state.metadata_exposure_ms is not None:
                lines.append(
                    f"ND2 metadata exposure: {float(self.state.metadata_exposure_ms):.4f} ms "
                    f"({self.state.metadata_exposure_source or 'ND2 metadata'})"
                )
        else:
            lines.append(
                f"Exposure: {exposure_ms:.4f} ms "
                f"({self.state.exposure_source or self.state.metadata_exposure_source or 'ND2 metadata'})"
            )
        if self.state.results_mode != "normal":
            lines.append(f"Mode: {self.state.results_mode}")
        if self.state.backend_name:
            lines.append(f"ND2 backend: {self.state.backend_name}")

        backend_capabilities = dict(self.state.backend_capabilities or {})
        available_entries = [
            name
            for name in ("getitem", "to_dask", "asarray", "xarray")
            if bool(backend_capabilities.get(name))
        ]
        if available_entries:
            lines.append(f"ND2 entry points: {', '.join(available_entries)}")

        selected_capabilities = dict(backend_capabilities.get("selected_capabilities", {}) or {})
        if selected_capabilities:
            if "lazy" in selected_capabilities:
                lines.append(
                    f"Lazy loading: {'yes' if selected_capabilities.get('lazy') else 'no'}"
                )
            axes = list(selected_capabilities.get("axes", []) or [])
            if axes:
                lines.append(f"Backend axes: {'/'.join(str(axis) for axis in axes)}")
        return "\n".join(lines)

    def detection_result_text(self) -> str:
        active = self.state.active_molecules()
        parts = [f"Molecules: {len(active)}"]
        filt = dict(self.state.uncertainty_filter or {})
        if filt.get("enabled"):
            max_unc = filt.get("max_scalar")
            if max_unc is not None:
                parts.append(f"uncertainty <= {float(max_unc):.3f} px")
        return " | ".join(parts)

    def frame_label_text(self) -> str:
        return self.state.frame_label_text()

    def time_array(self) -> np.ndarray:
        return np.arange(self.state.num_frames, dtype=np.float32) * float(self.state.exposure_s)

    def molecule_list_items(self) -> list[tuple[int, str]]:
        items: list[tuple[int, str]] = []
        for mol_id, x, y in self.state.active_molecules():
            feature = self.state.molecule_feature(int(mol_id))
            unc = feature.get("loc_uncertainty_scalar")
            if unc is None or (isinstance(unc, float) and np.isnan(unc)):
                suffix = ""
            else:
                suffix = f" | u={float(unc):.3f}px"
            items.append(
                (int(mol_id), f"Molecule {int(mol_id)} ({float(x):.1f}, {float(y):.1f}){suffix}")
            )
        return items


@dataclass
class DetectionPreviewViewModel:
    positions: list[tuple[float, float]]
    stats: dict[str, Any]

    @staticmethod
    def _fmt_stats(title: str, stats: dict[str, Any]) -> str:
        if not stats:
            return f"{title}\nNo statistics available"

        lines = [
            title,
            f"Candidates: {stats.get('dog_candidates', 0)}",
            f"Detected molecules: {stats.get('final_molecules', 0)}",
            f"Average SNR: {stats.get('avg_snr', 0.0):.2f}",
            f"Average R2: {stats.get('avg_r2', 0.0):.3f}",
            f"Average sigma: {stats.get('avg_sigma', 0.0):.2f} px",
        ]
        return "\n".join(lines)

    def stats_text(self) -> str:
        return self._fmt_stats("Detection", self.stats)
