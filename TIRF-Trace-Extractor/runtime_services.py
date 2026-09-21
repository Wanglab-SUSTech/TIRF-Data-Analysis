from __future__ import annotations

import glob
import io
import json
import logging
import multiprocessing as mp
import os
import queue
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

import numpy as np

from channel_profiles import physical_channel_name
from detection import MoleculeDetector
from file_io import ND2Reader
from gpu_backend import get_cupy_module, normalize_compute_backend
from processing import DriftCalculator, compute_molecule_intensities
from registration import REFERENCE_CHANNEL, RegistrationEstimator, RegistrationIO
from ui_lut_cache import MultiLevelCache

logger = logging.getLogger(__name__)

DEFAULT_CACHE_BUDGET_MB = 2048
PREVIEW_TOP_K = 250
PROJECT_FORMAT_VERSION = "analysis_bundle_v2"


def _read_channel_block_or_none(
    multi_cache: MultiLevelCache,
    channel_idx: int,
    start: int,
    stop: int,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray | None:
    if cancel_check and cancel_check():
        raise RuntimeError("Task cancelled")
    reader = getattr(multi_cache, "get_channel_block", None)
    if not callable(reader):
        return None
    try:
        stack = reader(int(start), int(stop), int(channel_idx))
    except RuntimeError as exc:
        if "Task cancelled" in str(exc):
            raise
        logger.debug("Channel block read unavailable; falling back to per-frame reads: %s", exc)
        return None
    except Exception as exc:
        logger.debug("Channel block read unavailable; falling back to per-frame reads: %s", exc)
        return None
    if cancel_check and cancel_check():
        raise RuntimeError("Task cancelled")
    if stack is None:
        return None
    arr = np.asarray(stack)
    if arr.ndim != 3 or arr.shape[0] <= 0:
        logger.debug("Channel block read returned invalid shape %s; falling back to per-frame reads", arr.shape)
        return None
    return arr


def max_projection_frame_range(
    multi_cache: MultiLevelCache,
    channel_idx: int,
    start_frame: int = 0,
    n: int = 10,
    cancel_check: Callable[[], bool] | None = None,
    compute_backend: str = "auto",
) -> np.ndarray:
    num_frames = int(multi_cache.metadata.get("num_frames", 0))
    if num_frames <= 0:
        raise ValueError("num_frames <= 0")
    start = max(0, min(int(start_frame), num_frames - 1))
    n_use = max(1, min(int(n), num_frames - start))
    stop = start + n_use

    backend = normalize_compute_backend(compute_backend)
    block_stack = _read_channel_block_or_none(
        multi_cache,
        channel_idx,
        start,
        stop,
        cancel_check=cancel_check,
    )
    if block_stack is not None:
        if backend != "cpu" and n_use >= 2:
            try:
                cp = get_cupy_module()
                if cp is not None:
                    stack_gpu = cp.asarray(block_stack)
                    return cp.asnumpy(cp.max(stack_gpu, axis=0)).astype(np.uint16, copy=False)
            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning("[GPU] CuPy max projection failed; falling back to CPU block max: %s", exc)
        return np.max(np.asarray(block_stack), axis=0).astype(np.uint16, copy=False)

    if backend != "cpu" and n_use >= 2:
        try:
            cp = get_cupy_module()
            if cp is not None:
                frames = []
                for frame_idx in range(start, stop):
                    if cancel_check and cancel_check():
                        raise RuntimeError("Task cancelled")
                    raw = multi_cache.get_raw_frame(frame_idx)
                    if raw is None:
                        continue
                    frames.append(np.asarray(raw[channel_idx], dtype=np.uint16))
                if frames:
                    stack_gpu = cp.asarray(np.stack(frames, axis=0))
                    return cp.asnumpy(cp.max(stack_gpu, axis=0)).astype(np.uint16, copy=False)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[GPU] CuPy max projection failed; falling back to CPU: %s", exc)

    proj = None
    for frame_idx in range(start, stop):
        if cancel_check and cancel_check():
            raise RuntimeError("Task cancelled")
        raw = multi_cache.get_raw_frame(frame_idx)
        if raw is None:
            continue
        frame = raw[channel_idx]
        proj = frame.copy() if proj is None else np.maximum(proj, frame)
    if proj is None:
        raise ValueError("Unable to read any frames for projection")
    return proj.astype(np.uint16)


def max_projection_first_n_frames(
    multi_cache: MultiLevelCache,
    channel_idx: int,
    n: int = 10,
    cancel_check: Callable[[], bool] | None = None,
    compute_backend: str = "auto",
) -> np.ndarray:
    return max_projection_frame_range(
        multi_cache,
        channel_idx,
        start_frame=0,
        n=n,
        cancel_check=cancel_check,
        compute_backend=compute_backend,
    )


def _projection_search_starts(
    num_frames: int,
    *,
    start_frame: int,
    n: int,
    auto_search: bool,
    step_frames: int,
    max_windows: int | None = None,
) -> list[int]:
    if num_frames <= 0:
        return []
    window = max(1, min(int(n), int(num_frames)))
    primary = max(0, min(int(start_frame), int(num_frames) - 1))
    starts = [primary]
    if not auto_search:
        return starts

    step = max(window, int(step_frames or 0), 1)
    for candidate in range(0, int(num_frames), step):
        candidate = max(0, min(candidate, int(num_frames) - 1))
        if candidate not in starts:
            starts.append(candidate)
            if max_windows is not None and len(starts) >= int(max_windows):
                break
    return starts


def _detect_batch_molecules(
    cache: MultiLevelCache,
    detect_channel_idx: int,
    params: dict[str, Any],
    *,
    start_frame: int,
    n: int,
    auto_search: bool,
    step_frames: int,
    max_windows: int | None = None,
    cancel_check: Callable[[], bool],
    compute_backend: str = "auto",
) -> tuple[list[tuple[float, float]], dict[str, Any], list[dict[str, Any]]]:
    num_frames = int(cache.metadata.get("num_frames", 0))
    detector = MoleculeDetector()
    attempts: list[dict[str, Any]] = []
    last_stats: dict[str, Any] = {}

    for start in _projection_search_starts(
        num_frames,
        start_frame=start_frame,
        n=n,
        auto_search=auto_search,
        step_frames=step_frames,
        max_windows=max_windows,
    ):
        if cancel_check():
            raise RuntimeError("Task cancelled")
        proj_img = max_projection_frame_range(
            cache,
            detect_channel_idx,
            start_frame=start,
            n=n,
            cancel_check=cancel_check,
            compute_backend=compute_backend,
        )
        positions, stats = detector.detect_interactive(
            proj_img,
            progress_cb=None,
            cancel_check=cancel_check,
            **params,
        )
        positions = sorted(positions, key=lambda p: (p[1], p[0]))
        last_stats = dict(stats or {})
        attempts.append(
            {
                "start_frame": int(start),
                "stop_frame": int(min(start + max(1, int(n)), num_frames)),
                "molecules": int(len(positions)),
                "candidates": int(last_stats.get("dog_candidates", 0) or 0),
                "fitted": int(last_stats.get("fitted_candidates", 0) or 0),
                "refined": int(last_stats.get("refined_candidates", 0) or 0),
                "mode": str(last_stats.get("mode", "interactive") or "interactive"),
            }
        )
        if positions:
            return positions, last_stats, attempts

    return [], last_stats, attempts


def _format_csv_value(val: Any) -> str:
    try:
        f = float(val)
    except Exception:
        return ""
    if np.isnan(f):
        return ""
    return f"{f:.4f}"


def _write_one_csv(path, mol_list, molecule_intensities, time_arr, channel_names):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        header = ["Time_sec", "ROI_ID"] + [f"Net_{ch}" for ch in channel_names]
        f.write(",".join(header) + "\n")
        for mol_id, _, _ in mol_list:
            if mol_id not in molecule_intensities:
                continue
            for frame_idx in range(len(time_arr)):
                row = [f"{time_arr[frame_idx]:.4f}", str(mol_id)]
                for ch in channel_names:
                    row.append(_format_csv_value(molecule_intensities[mol_id][ch][frame_idx]))
                f.write(",".join(row) + "\n")


def write_intensities_csv_split(
    base_csv_path: str,
    *,
    molecules,
    deleted_molecules,
    molecule_intensities: dict,
    time_arr: np.ndarray,
    channel_names,
    max_rows: int = 1_000_000,
):
    if time_arr is None or len(time_arr) == 0:
        raise ValueError("time array is empty; export is not possible")

    active_molecules = [
        mol for mol in molecules
        if mol[0] not in deleted_molecules and mol[0] in molecule_intensities
    ]
    if not active_molecules:
        raise ValueError("No molecule data is available for export")

    rows_per_molecule = max(1, len(time_arr))
    mols_per_file = max(1, int(max_rows // rows_per_molecule))
    written = []
    stem, ext = os.path.splitext(base_csv_path)
    ext = ext or ".csv"

    for idx in range(0, len(active_molecules), mols_per_file):
        chunk = active_molecules[idx:idx + mols_per_file]
        out_path = base_csv_path if idx == 0 and len(active_molecules) <= mols_per_file else f"{stem}_{len(written) + 1}{ext}"
        _write_one_csv(out_path, chunk, molecule_intensities, time_arr, channel_names)
        written.append(out_path)

    return written


def _strict_filter_complete_molecules(
    molecules,
    molecule_intensities: dict,
    channel_names,
    expected_frames: int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, float, float]], dict, list[dict[str, Any]]]:
    kept = []
    filtered_intensities = {}
    dropped: list[dict[str, Any]] = []
    diag_by_mol = dict((diagnostics or {}).get("molecules", {}) or {})

    for mol_id, x, y in list(molecules or []):
        mol_id = int(mol_id)
        per_channel = dict(molecule_intensities.get(mol_id, {}) or {})
        invalid_by_channel = {}

        for ch in channel_names:
            values = per_channel.get(ch)
            arr = np.asarray(values if values is not None else [], dtype=np.float32)
            invalid = int(np.sum(~np.isfinite(arr)))
            if expected_frames is not None and arr.size != int(expected_frames):
                invalid += max(1, abs(int(expected_frames) - int(arr.size)))
            if invalid > 0:
                invalid_by_channel[str(ch)] = invalid

        if invalid_by_channel:
            mol_diag = dict(diag_by_mol.get(mol_id, {}) or {})
            channel_details = {}
            for ch in invalid_by_channel:
                details = dict((mol_diag.get("invalid_by_channel", {}) or {}).get(str(ch), {}) or {})
                if details:
                    channel_details[str(ch)] = {
                        "count": int(details.get("count", invalid_by_channel[ch]) or 0),
                        "reasons": dict(details.get("reasons", {}) or {}),
                        "frames": list(details.get("frames", []) or []),
                    }
            dropped.append(
                {
                    "mol_id": mol_id,
                    "x": float(x),
                    "y": float(y),
                    "invalid_by_channel": invalid_by_channel,
                    "reason": "contains NaN/Inf intensity values",
                    "diagnostics": {
                        "input_x": mol_diag.get("input_x"),
                        "input_y": mol_diag.get("input_y"),
                        "reference_x": mol_diag.get("reference_x"),
                        "reference_y": mol_diag.get("reference_y"),
                        "min_boundary_distance_px": mol_diag.get("min_boundary_distance_px"),
                        "invalid_by_channel": channel_details,
                    },
                }
            )
            continue

        kept.append((mol_id, float(x), float(y)))
        filtered_intensities[mol_id] = per_channel

    return kept, filtered_intensities, dropped


def _format_drop_summary(dropped: list[dict[str, Any]], limit: int = 5) -> str:
    if not dropped:
        return "none"
    parts = []
    for item in dropped[:limit]:
        channels = ", ".join(
            f"{ch}:{count}"
            for ch, count in sorted(dict(item.get("invalid_by_channel", {})).items())
        )
        parts.append(f"ROI {item.get('mol_id')} ({channels})")
    if len(dropped) > limit:
        parts.append(f"... {len(dropped) - limit} more")
    return "; ".join(parts)


def _format_optional_float(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.{int(digits)}f}"


def _qc_warning_if_suspicious(
    detected_count: int,
    exported_count: int,
    dropped_rois: list[dict[str, Any]],
    diagnostics: dict[str, Any] | None,
) -> str:
    detected_count = int(detected_count or 0)
    if detected_count <= 0 or not dropped_rois:
        return ""
    drop_fraction = (detected_count - int(exported_count or 0)) / max(1, detected_count)
    if drop_fraction < 0.10 or len(dropped_rois) < 5:
        return ""

    diag = dict(diagnostics or {})
    drift = dict(diag.get("drift", {}) or {})
    max_drift = drift.get("max_abs_drift_px")
    nonfinite_frames = list(drift.get("nonfinite_frames", []) or [])
    min_edges = []
    for item in dropped_rois:
        mol_diag = dict(item.get("diagnostics", {}) or {})
        edge = mol_diag.get("min_boundary_distance_px")
        try:
            edge_f = float(edge)
        except Exception:
            continue
        if np.isfinite(edge_f):
            min_edges.append(edge_f)

    safe_edges = bool(min_edges) and min(min_edges) >= 2.0
    try:
        safe_drift = float(max_drift) <= 2.0
    except Exception:
        safe_drift = False
    if safe_edges and safe_drift and not nonfinite_frames:
        return (
            "WARNING: high strict-QC dropout despite finite low drift and safe transformed "
            "boundary distances. This points to an extraction/coordinate bug or unexpected "
            "frame/channel data path, not normal edge clipping."
        )
    return ""


def _valid_exposure_override_ms(value: Any) -> float:
    try:
        exposure = float(value)
    except Exception as exc:
        raise ValueError("Manual exposure override must be a number in ms") from exc
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("Manual exposure override must be finite and > 0 ms")
    return exposure


def _resolve_effective_exposure_record(
    metadata: dict[str, Any],
    *,
    override_enabled: bool = False,
    override_ms: Any = None,
) -> dict[str, Any]:
    metadata_exposure_ms = float(metadata.get("exposure_ms", metadata.get("time_ms", 100.0)))
    metadata_exposure_source = str(metadata.get("exposure_source", "ND2 metadata") or "ND2 metadata")
    out = {
        "metadata_exposure_ms": metadata_exposure_ms,
        "metadata_exposure_source": metadata_exposure_source,
        "exposure_override_enabled": bool(override_enabled),
    }
    if override_enabled:
        manual_ms = _valid_exposure_override_ms(override_ms)
        out.update(
            {
                "exposure_ms": manual_ms,
                "exposure_source": "manual_override",
                "exposure_override_ms": manual_ms,
            }
        )
    else:
        out.update(
            {
                "exposure_ms": metadata_exposure_ms,
                "exposure_source": metadata_exposure_source,
                "exposure_override_ms": None,
            }
        )
    return out


def _batch_metadata_record(cache, *, exposure_override_enabled: bool = False, exposure_override_ms: Any = None) -> dict[str, Any]:
    metadata = dict(getattr(cache, "metadata", {}) or {})
    return _resolve_effective_exposure_record(
        metadata,
        override_enabled=exposure_override_enabled,
        override_ms=exposure_override_ms,
    )


def _new_batch_timings() -> dict[str, float]:
    return {
        "open_cache_sec": 0.0,
        "detection_sec": 0.0,
        "intensity_sec": 0.0,
        "strict_qc_sec": 0.0,
        "csv_write_sec": 0.0,
        "total_sec": 0.0,
    }


def _finish_batch_record(
    record: dict[str, Any],
    timings: dict[str, float],
    total_start: float,
) -> dict[str, Any]:
    timings["total_sec"] = max(0.0, time.perf_counter() - total_start)
    record["timings_sec"] = {key: round(float(value), 6) for key, value in timings.items()}
    return record


def _format_timing_line(timings: dict[str, Any]) -> str:
    if not timings:
        return ""
    return (
        f"total={_format_optional_float(timings.get('total_sec'), digits=3)}s, "
        f"open/cache={_format_optional_float(timings.get('open_cache_sec'), digits=3)}s, "
        f"detection={_format_optional_float(timings.get('detection_sec'), digits=3)}s, "
        f"intensity={_format_optional_float(timings.get('intensity_sec'), digits=3)}s, "
        f"QC={_format_optional_float(timings.get('strict_qc_sec'), digits=3)}s, "
        f"CSV={_format_optional_float(timings.get('csv_write_sec'), digits=3)}s"
    )


def _format_intensity_breakdown_line(diag: dict[str, Any]) -> str:
    if not diag:
        return ""
    timings = dict(diag.get("timings_sec", {}) or {})
    if not timings and not diag.get("gpu_or_cpu_backend"):
        return ""
    return (
        f"backend={diag.get('gpu_or_cpu_backend', 'n/a')}, "
        f"requested={diag.get('requested_compute_backend', 'n/a')}, "
        f"drift={_format_optional_float(timings.get('drift_sec'), digits=3)}s, "
        f"frame_read={_format_optional_float(timings.get('frame_read_sec'), digits=3)}s, "
        f"patch_extract={_format_optional_float(timings.get('patch_extract_sec'), digits=3)}s, "
        f"background={_format_optional_float(timings.get('background_sec'), digits=3)}s, "
        f"diagnostics={_format_optional_float(timings.get('diagnostics_sec'), digits=3)}s"
    )


def _format_drift_breakdown_line(drift: dict[str, Any]) -> str:
    timings = dict(drift.get("timings_sec", {}) or {})
    if not timings:
        return ""
    return (
        f"select={_format_optional_float(timings.get('select_sec'), digits=3)}s, "
        f"track={_format_optional_float(timings.get('track_sec'), digits=3)}s, "
        f"aggregate={_format_optional_float(timings.get('aggregate_sec'), digits=3)}s, "
        f"interpolate={_format_optional_float(timings.get('interpolate_sec'), digits=3)}s, "
        f"smooth={_format_optional_float(timings.get('smooth_sec'), digits=3)}s"
    )


def _format_shape(value: Any) -> str:
    try:
        items = list(value)
    except Exception:
        return "n/a"
    if len(items) < 2:
        return "n/a"
    try:
        return f"{int(items[0])}x{int(items[1])}"
    except Exception:
        return "n/a"


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _summarize_batch_records(
    records: list[dict[str, Any]],
    *,
    wall_time_sec: float | None = None,
    parallel_workers: int | None = None,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {"succeeded": 0, "skipped": 0, "failed": 0}
    stage_totals = _new_batch_timings()
    intensity_breakdown_totals = {
        "drift_sec": 0.0,
        "frame_read_sec": 0.0,
        "patch_extract_sec": 0.0,
        "background_sec": 0.0,
        "diagnostics_sec": 0.0,
    }
    drift_breakdown_totals = {
        "select_sec": 0.0,
        "track_sec": 0.0,
        "aggregate_sec": 0.0,
        "interpolate_sec": 0.0,
        "smooth_sec": 0.0,
    }
    drift_io_totals = {
        "drift_track_frame_read_sec": 0.0,
        "block_read_calls": 0,
        "block_read_used_files": 0,
        "sparse_requested_files": 0,
        "sparse_used_files": 0,
        "sparse_fallback_files": 0,
    }
    intensity_block_io_totals = {
        "block_read_calls": 0,
        "block_read_failures": 0,
        "raw_frame_read_calls": 0,
        "block_read_used_files": 0,
    }
    backend_counts: dict[str, int] = {}
    timed_records = 0
    succeeded_times = []
    for rec in records:
        status = str(rec.get("status", "unknown"))
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        timings = dict(rec.get("timings_sec", {}) or {})
        if timings:
            timed_records += 1
            for key in stage_totals:
                try:
                    stage_totals[key] += float(timings.get(key, 0.0) or 0.0)
                except Exception:
                    pass
            if status == "succeeded":
                try:
                    succeeded_times.append(float(timings.get("total_sec", 0.0) or 0.0))
                except Exception:
                    pass
        diag = dict(rec.get("intensity_diagnostics", {}) or {})
        backend = str(diag.get("gpu_or_cpu_backend", "") or "")
        if backend:
            backend_counts[backend] = int(backend_counts.get(backend, 0)) + 1
        breakdown = dict(diag.get("timings_sec", {}) or {})
        for key in intensity_breakdown_totals:
            try:
                intensity_breakdown_totals[key] += float(breakdown.get(key, 0.0) or 0.0)
            except Exception:
                pass
        drift_breakdown = dict(dict(diag.get("drift", {}) or {}).get("timings_sec", {}) or {})
        for key in drift_breakdown_totals:
            try:
                drift_breakdown_totals[key] += float(drift_breakdown.get(key, 0.0) or 0.0)
            except Exception:
                pass
        drift_diag = dict(diag.get("drift", {}) or {})
        try:
            drift_io_totals["drift_track_frame_read_sec"] += float(
                drift_diag.get("drift_track_frame_read_sec", 0.0) or 0.0
            )
        except Exception:
            pass
        try:
            drift_io_totals["block_read_calls"] += int(drift_diag.get("block_read_calls", 0) or 0)
        except Exception:
            pass
        if bool(drift_diag.get("block_read_used", False)):
            drift_io_totals["block_read_used_files"] += 1
        if str(drift_diag.get("drift_tracking_mode_requested", "full") or "full") == "sparse":
            drift_io_totals["sparse_requested_files"] += 1
        if str(drift_diag.get("drift_tracking_mode", "full") or "full") == "sparse":
            drift_io_totals["sparse_used_files"] += 1
        if bool(drift_diag.get("sparse_fallback_used", False)):
            drift_io_totals["sparse_fallback_files"] += 1
        block_io = dict(diag.get("channel_block_io", {}) or {})
        if block_io:
            for key in ("block_read_calls", "block_read_failures", "raw_frame_read_calls"):
                try:
                    intensity_block_io_totals[key] += int(block_io.get(key, 0) or 0)
                except Exception:
                    pass
            if bool(block_io.get("block_read_used", False)):
                intensity_block_io_totals["block_read_used_files"] += 1
    return {
        "total_files": len(records),
        "status_counts": status_counts,
        "wall_time_sec": wall_time_sec,
        "parallel_workers": parallel_workers,
        "timed_records": timed_records,
        "stage_totals_sec": {key: round(float(value), 6) for key, value in stage_totals.items()},
        "intensity_breakdown_totals_sec": {
            key: round(float(value), 6)
            for key, value in intensity_breakdown_totals.items()
        },
        "drift_breakdown_totals_sec": {
            key: round(float(value), 6)
            for key, value in drift_breakdown_totals.items()
        },
        "drift_io_totals": {
            "drift_track_frame_read_sec": round(float(drift_io_totals["drift_track_frame_read_sec"]), 6),
            "block_read_calls": int(drift_io_totals["block_read_calls"]),
            "block_read_used_files": int(drift_io_totals["block_read_used_files"]),
            "sparse_requested_files": int(drift_io_totals["sparse_requested_files"]),
            "sparse_used_files": int(drift_io_totals["sparse_used_files"]),
            "sparse_fallback_files": int(drift_io_totals["sparse_fallback_files"]),
        },
        "intensity_channel_block_io_totals": {
            key: int(value)
            for key, value in intensity_block_io_totals.items()
        },
        "intensity_backend_counts": backend_counts,
        "avg_succeeded_total_sec": (
            round(float(np.mean(succeeded_times)), 6) if succeeded_times else None
        ),
    }


def _write_batch_report(
    folder: str,
    records: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"batch_report_{timestamp}.txt")
    summary = dict(summary or _summarize_batch_records(records))
    status_counts = dict(summary.get("status_counts", {}) or {})
    stage_totals = dict(summary.get("stage_totals_sec", {}) or {})
    intensity_totals = dict(summary.get("intensity_breakdown_totals_sec", {}) or {})
    drift_totals = dict(summary.get("drift_breakdown_totals_sec", {}) or {})
    drift_io_totals = dict(summary.get("drift_io_totals", {}) or {})
    intensity_block_io_totals = dict(summary.get("intensity_channel_block_io_totals", {}) or {})
    backend_counts = dict(summary.get("intensity_backend_counts", {}) or {})
    backend_text = ", ".join(
        f"{backend}={count}" for backend, count in sorted(backend_counts.items())
    ) or "n/a"
    lines = [
        "Batch Detection Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Aggregate summary:",
        f"  Total files: {summary.get('total_files', len(records))}",
        (
            "  Status counts: "
            f"succeeded={status_counts.get('succeeded', 0)}, "
            f"skipped={status_counts.get('skipped', 0)}, "
            f"failed={status_counts.get('failed', 0)}"
        ),
        (
            "  Wall time: "
            f"{_format_optional_float(summary.get('wall_time_sec'), digits=3)} s"
            if summary.get("wall_time_sec") is not None
            else "  Wall time: n/a"
        ),
        f"  Parallel ND2 workers: {summary.get('parallel_workers', 'n/a')}",
        (
            "  Average succeeded file time: "
            f"{_format_optional_float(summary.get('avg_succeeded_total_sec'), digits=3)} s"
            if summary.get("avg_succeeded_total_sec") is not None
            else "  Average succeeded file time: n/a"
        ),
        (
            "  Stage totals: "
            f"open/cache={_format_optional_float(stage_totals.get('open_cache_sec'), digits=3)}s, "
            f"detection={_format_optional_float(stage_totals.get('detection_sec'), digits=3)}s, "
            f"intensity={_format_optional_float(stage_totals.get('intensity_sec'), digits=3)}s, "
            f"QC={_format_optional_float(stage_totals.get('strict_qc_sec'), digits=3)}s, "
            f"CSV={_format_optional_float(stage_totals.get('csv_write_sec'), digits=3)}s"
        ),
        (
            "  Intensity breakdown totals: "
            f"backend_counts={backend_text}, "
            f"drift={_format_optional_float(intensity_totals.get('drift_sec'), digits=3)}s, "
            f"frame_read={_format_optional_float(intensity_totals.get('frame_read_sec'), digits=3)}s, "
            f"patch_extract={_format_optional_float(intensity_totals.get('patch_extract_sec'), digits=3)}s, "
            f"background={_format_optional_float(intensity_totals.get('background_sec'), digits=3)}s, "
            f"diagnostics={_format_optional_float(intensity_totals.get('diagnostics_sec'), digits=3)}s"
        ),
        "",
    ]
    if drift_totals and any(float(v or 0.0) > 0 for v in drift_totals.values()):
        lines.insert(
            -1,
            "  Drift breakdown totals: "
            f"select={_format_optional_float(drift_totals.get('select_sec'), digits=3)}s, "
            f"track={_format_optional_float(drift_totals.get('track_sec'), digits=3)}s, "
            f"aggregate={_format_optional_float(drift_totals.get('aggregate_sec'), digits=3)}s, "
            f"interpolate={_format_optional_float(drift_totals.get('interpolate_sec'), digits=3)}s, "
            f"smooth={_format_optional_float(drift_totals.get('smooth_sec'), digits=3)}s"
        )
    if drift_io_totals:
        lines.insert(
            -1,
            "  Drift I/O totals: "
            f"drift_track_frame_read={_format_optional_float(drift_io_totals.get('drift_track_frame_read_sec'), digits=3)}s, "
            f"block_read_used_files={int(drift_io_totals.get('block_read_used_files', 0) or 0)}, "
            f"block_read_calls={int(drift_io_totals.get('block_read_calls', 0) or 0)}, "
            f"sparse_requested_files={int(drift_io_totals.get('sparse_requested_files', 0) or 0)}, "
            f"sparse_used_files={int(drift_io_totals.get('sparse_used_files', 0) or 0)}, "
            f"sparse_fallback_files={int(drift_io_totals.get('sparse_fallback_files', 0) or 0)}"
        )
    if intensity_block_io_totals and any(int(v or 0) > 0 for v in intensity_block_io_totals.values()):
        lines.insert(
            -1,
            "  Intensity channel-block I/O totals: "
            f"block_read_used_files={int(intensity_block_io_totals.get('block_read_used_files', 0) or 0)}, "
            f"block_read_calls={int(intensity_block_io_totals.get('block_read_calls', 0) or 0)}, "
            f"block_read_failures={int(intensity_block_io_totals.get('block_read_failures', 0) or 0)}, "
            f"raw_frame_read_calls={int(intensity_block_io_totals.get('raw_frame_read_calls', 0) or 0)}"
        )
    for rec in records:
        lines.append(f"[{str(rec.get('status', 'unknown')).upper()}] {rec.get('file', '')}")
        lines.append(f"Reason: {rec.get('reason', '')}")
        timing_line = _format_timing_line(dict(rec.get("timings_sec", {}) or {}))
        if timing_line:
            lines.append(f"Timings: {timing_line}")
        if "exposure_ms" in rec:
            source = rec.get("exposure_source") or "ND2 metadata"
            lines.append(
                f"Exposure: {_format_optional_float(rec.get('exposure_ms'), digits=4)} ms "
                f"(source: {source})"
            )
            if rec.get("exposure_override_enabled") and "metadata_exposure_ms" in rec:
                lines.append(
                    "Original ND2 metadata exposure: "
                    f"{_format_optional_float(rec.get('metadata_exposure_ms'), digits=4)} ms "
                    f"(source: {rec.get('metadata_exposure_source') or 'ND2 metadata'})"
                )
        if rec.get("detection_window"):
            lines.append(f"Detection window: {rec['detection_window']}")
        if rec.get("detection_attempts"):
            lines.append("Detection attempts:")
            for attempt in rec["detection_attempts"][:12]:
                lines.append(
                    "  "
                    f"{attempt.get('start_frame')}:{attempt.get('stop_frame')} "
                    f"mode={attempt.get('mode', 'interactive')} "
                    f"candidates={attempt.get('candidates', 0)} "
                    f"fitted={attempt.get('fitted', 0)} "
                    f"molecules={attempt.get('molecules', 0)}"
                )
            if len(rec["detection_attempts"]) > 12:
                lines.append(f"  ... {len(rec['detection_attempts']) - 12} more")
        if rec.get("detection_stats"):
            stats = dict(rec.get("detection_stats", {}) or {})
            lines.append(
                "Detection stats: "
                f"mode={stats.get('mode', 'interactive')}, "
                f"dog_candidates={stats.get('dog_candidates', 0)}, "
                f"fitted_candidates={stats.get('fitted_candidates', 0)}, "
                f"final_molecules={stats.get('final_molecules', 0)}, "
                f"avg_snr={_format_optional_float(stats.get('avg_snr'))}, "
                f"avg_r2={_format_optional_float(stats.get('avg_r2'))}"
            )
        if "detected_rois" in rec:
            lines.append(f"Detected ROIs: {rec.get('detected_rois')}")
        if "detected_before_qc" in rec:
            lines.append(f"Detected before strict QC: {rec.get('detected_before_qc')}")
        if "exported_rois" in rec:
            lines.append(f"Exported complete ROIs: {rec.get('exported_rois')}")
        if rec.get("intensity_diagnostics"):
            diag = dict(rec.get("intensity_diagnostics", {}) or {})
            breakdown_line = _format_intensity_breakdown_line(diag)
            if breakdown_line:
                lines.append(f"Intensity breakdown: {breakdown_line}")
            drift = dict(diag.get("drift", {}) or {})
            nonfinite = list(drift.get("nonfinite_frames", []) or [])
            lines.append(
                "Intensity diagnostics: "
                f"boundary_margin={diag.get('boundary_margin_px', 'n/a')} px, "
                f"max_abs_drift={_format_optional_float(drift.get('max_abs_drift_px'))} px, "
                f"nonfinite_drift_frames={len(nonfinite)}"
            )
            lines.append(
                "Drift I/O diagnostics: "
                f"num_frames={drift.get('num_frames', 'n/a')}, "
                f"image_shape={_format_shape(drift.get('image_shape', []))}, "
                f"num_channels={drift.get('num_channels', 'n/a')}, "
                f"drift_track_frame_read={_format_optional_float(drift.get('drift_track_frame_read_sec'), digits=3)}s, "
                f"block_read_used={_format_bool(drift.get('block_read_used', False))}, "
                f"block_size={drift.get('block_size', 'n/a')}, "
                f"block_read_calls={drift.get('block_read_calls', 'n/a')}"
            )
            lines.append(
                "Drift tracking mode: "
                f"requested={drift.get('drift_tracking_mode_requested', 'full')}, "
                f"used={drift.get('drift_tracking_mode', 'full')}, "
                f"track_interval_sec={_format_optional_float(drift.get('track_interval_sec'), digits=3)}, "
                f"track_interval_frames={drift.get('track_interval_frames', 'n/a')}, "
                f"sampled_frames={drift.get('sampled_frames', 'n/a')}, "
                f"exposure_fallback={_format_bool(drift.get('track_interval_exposure_fallback', False))}, "
                f"sparse_fallback_used={_format_bool(drift.get('sparse_fallback_used', False))}"
            )
            if drift.get("sparse_fallback_reason"):
                lines.append(f"  Sparse fallback reason: {drift.get('sparse_fallback_reason')}")
            block_io = dict(diag.get("channel_block_io", {}) or {})
            if block_io:
                lines.append(
                    "Intensity channel-block I/O: "
                    f"block_read_used={_format_bool(block_io.get('block_read_used', False))}, "
                    f"block_size={block_io.get('block_size', 'n/a')}, "
                    f"block_read_calls={block_io.get('block_read_calls', 'n/a')}, "
                    f"block_read_failures={block_io.get('block_read_failures', 'n/a')}, "
                    f"raw_frame_read_calls={block_io.get('raw_frame_read_calls', 'n/a')}"
                )
            drift_breakdown_line = _format_drift_breakdown_line(drift)
            if drift_breakdown_line:
                lines.append(f"Drift breakdown: {drift_breakdown_line}")
            if nonfinite:
                lines.append(f"  Nonfinite drift frame preview: {nonfinite[:20]}")
        if rec.get("qc_warning"):
            lines.append(str(rec["qc_warning"]))
        if rec.get("dropped_rois"):
            lines.append(f"Dropped ROIs: {len(rec['dropped_rois'])}")
            for item in rec["dropped_rois"][:20]:
                channels = ", ".join(
                    f"{ch}={count}"
                    for ch, count in sorted(dict(item.get("invalid_by_channel", {})).items())
                )
                lines.append(
                    f"  ROI {item.get('mol_id')}: {channels} invalid frames; "
                    f"x={item.get('x'):.2f}, y={item.get('y'):.2f}"
                )
                mol_diag = dict(item.get("diagnostics", {}) or {})
                if mol_diag:
                    lines.append(
                        "    diagnostics: "
                        f"input=({_format_optional_float(mol_diag.get('input_x'))}, "
                        f"{_format_optional_float(mol_diag.get('input_y'))}), "
                        f"reference=({_format_optional_float(mol_diag.get('reference_x'))}, "
                        f"{_format_optional_float(mol_diag.get('reference_y'))}), "
                        f"min_boundary_distance={_format_optional_float(mol_diag.get('min_boundary_distance_px'))} px"
                    )
                    for ch, details in sorted(dict(mol_diag.get("invalid_by_channel", {}) or {}).items()):
                        reasons = dict(details.get("reasons", {}) or {})
                        reason_text = ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items())) or "unknown"
                        lines.append(f"    {ch}: reasons {reason_text}")
                        for frame in list(details.get("frames", []) or [])[:8]:
                            drift = frame.get("drift", [None, None])
                            lines.append(
                                "      "
                                f"frame={frame.get('frame')} reason={frame.get('reason')} "
                                f"x_raw={_format_optional_float(frame.get('x_raw'))} "
                                f"y_raw={_format_optional_float(frame.get('y_raw'))} "
                                f"edge={_format_optional_float(frame.get('edge_distance_px'))} "
                                f"drift=({_format_optional_float(drift[0] if len(drift) > 0 else None)}, "
                                f"{_format_optional_float(drift[1] if len(drift) > 1 else None)}) "
                                f"out_of_bounds={frame.get('transformed_out_of_bounds', False)}"
                            )
            if len(rec["dropped_rois"]) > 20:
                lines.append(f"  ... {len(rec['dropped_rois']) - 20} more")
        if rec.get("outputs"):
            lines.append("Outputs:")
            for out in rec["outputs"]:
                lines.append(f"  {out}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def _safe_json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _json_safe_numeric_dict(value: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(value, dict):
        return result
    for key, raw in value.items():
        try:
            numeric = float(raw)
        except Exception:
            continue
        if np.isfinite(numeric):
            result[str(key)] = numeric
    return result


def compute_file_fingerprint(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        stat = os.stat(path)
    except OSError:
        return {"path": path, "exists": False}
    return {
        "path": path,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def fingerprint_matches(expected: dict[str, Any] | None, actual_path: str | None) -> bool:
    expected = dict(expected or {})
    actual = compute_file_fingerprint(actual_path)
    if not expected or not actual.get("exists"):
        return False
    for key in ("size", "mtime_ns"):
        if key in expected and key in actual and int(expected[key]) != int(actual[key]):
            return False
    return True


def _molecule_arrays(
    molecules: list[tuple[int, float, float]] | None,
    molecule_features: dict[int, dict[str, Any]] | None,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    molecules = list(molecules or [])
    feature_lookup = dict(molecule_features or {})
    ids = np.asarray([int(m[0]) for m in molecules], dtype=np.int32)
    x = np.asarray([float(m[1]) for m in molecules], dtype=np.float32) if molecules else np.zeros((0,), dtype=np.float32)
    y = np.asarray([float(m[2]) for m in molecules], dtype=np.float32) if molecules else np.zeros((0,), dtype=np.float32)

    arrays = {
        f"{prefix}_ids": ids,
        f"{prefix}_x": x,
        f"{prefix}_y": y,
    }
    numeric_fields = [
        "snr",
        "sigma",
        "sigma_x",
        "sigma_y",
        "photons",
        "background",
        "peak_over_background",
        "loc_uncertainty_x",
        "loc_uncertainty_y",
        "loc_cov_xy",
        "loc_uncertainty_scalar",
        "fit_score",
        "r2",
    ]
    string_fields = ["model_type", "fit_status", "fit_message"]
    bool_fields = ["crowded_flag", "uncertainty_valid"]
    int_fields = ["emitter_count"]

    for field_name in numeric_fields:
        values = np.full(ids.shape, np.nan, dtype=np.float32)
        for idx, mol_id in enumerate(ids):
            feature = dict(feature_lookup.get(int(mol_id), {}) or {})
            value = feature.get(field_name)
            if value is None:
                continue
            try:
                values[idx] = float(value)
            except Exception:
                continue
        arrays[f"{prefix}_{field_name}"] = values

    for field_name in string_fields:
        values = np.asarray(
            [str(dict(feature_lookup.get(int(mol_id), {}) or {}).get(field_name, "")) for mol_id in ids],
            dtype=np.dtype("U64"),
        )
        arrays[f"{prefix}_{field_name}"] = values

    for field_name in bool_fields:
        values = np.asarray(
            [bool(dict(feature_lookup.get(int(mol_id), {}) or {}).get(field_name, False)) for mol_id in ids],
            dtype=np.uint8,
        )
        arrays[f"{prefix}_{field_name}"] = values

    for field_name in int_fields:
        values = np.asarray(
            [int(dict(feature_lookup.get(int(mol_id), {}) or {}).get(field_name, 0) or 0) for mol_id in ids],
            dtype=np.int16,
        )
        arrays[f"{prefix}_{field_name}"] = values

    return arrays


def _restore_molecule_features(data: dict[str, np.ndarray], prefix: str) -> tuple[list[tuple[int, float, float]], dict[int, dict[str, Any]]]:
    ids = np.asarray(data.get(f"{prefix}_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    xs = np.asarray(data.get(f"{prefix}_x", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    ys = np.asarray(data.get(f"{prefix}_y", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    molecules = [(int(mol_id), float(x), float(y)) for mol_id, x, y in zip(ids, xs, ys)]
    features: dict[int, dict[str, Any]] = {}
    for idx, mol_id in enumerate(ids):
        feature: dict[str, Any] = {}
        for field_name in (
            "snr",
            "sigma",
            "sigma_x",
            "sigma_y",
            "photons",
            "background",
            "peak_over_background",
            "loc_uncertainty_x",
            "loc_uncertainty_y",
            "loc_cov_xy",
            "loc_uncertainty_scalar",
            "fit_score",
            "r2",
        ):
            key = f"{prefix}_{field_name}"
            if key not in data:
                continue
            value = float(np.asarray(data[key])[idx])
            if not np.isnan(value):
                feature[field_name] = value
        for field_name in ("model_type", "fit_status", "fit_message"):
            key = f"{prefix}_{field_name}"
            if key in data:
                feature[field_name] = str(np.asarray(data[key])[idx])
        for field_name in ("crowded_flag", "uncertainty_valid"):
            key = f"{prefix}_{field_name}"
            if key in data:
                feature[field_name] = bool(np.asarray(data[key])[idx])
        key = f"{prefix}_emitter_count"
        if key in data:
            feature["emitter_count"] = int(np.asarray(data[key])[idx])
        feature["x"] = float(xs[idx])
        feature["y"] = float(ys[idx])
        feature["pos"] = [float(xs[idx]), float(ys[idx])]
        features[int(mol_id)] = feature
    return molecules, features


def _build_bundle_payload(
    *,
    nd2_file,
    nd2_fingerprint,
    registration_params,
    detect_channel,
    channel_names,
    molecules,
    base_molecules,
    molecule_features,
    base_molecule_features,
    deleted_molecules,
    intensity_scales,
    lut_settings_by_channel,
    exposure_s,
    exposure_source="ND2 metadata",
    metadata_exposure_ms=None,
    metadata_exposure_source="",
    exposure_override_enabled=False,
    exposure_override_ms=None,
    current_frame,
    num_frames,
    num_channels=0,
    height=0,
    width=0,
    backend_name="",
    backend_capabilities=None,
    selected_molecule=None,
    playback=None,
    analysis_recipe=None,
    operation_log=None,
    uncertainty_filter=None,
    active_edit_tool="pan",
    view_transform=None,
    results_mode="normal",
    drift_result=None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, np.ndarray]]:
    manifest = {
        "version": PROJECT_FORMAT_VERSION,
        "saved_at_epoch_s": time.time(),
        "compatible_app": "single_molecule_nd2",
        "source_nd2_path": nd2_file or "",
        "source_nd2_fingerprint": dict(nd2_fingerprint or compute_file_fingerprint(nd2_file)),
        "backend_name": str(backend_name or ""),
        "backend_capabilities": dict(backend_capabilities or {}),
    }
    project_state = {
        "nd2_file": nd2_file or "",
        "nd2_fingerprint": dict(nd2_fingerprint or compute_file_fingerprint(nd2_file)),
        "registration_params": registration_params,
        "detect_channel": int(detect_channel),
        "channel_names": list(channel_names or []),
        "num_channels": int(num_channels or len(channel_names or [])),
        "height": int(height or 0),
        "width": int(width or 0),
        "exposure_s": float(exposure_s),
        "exposure_source": str(exposure_source or "ND2 metadata"),
        "metadata_exposure_ms": None if metadata_exposure_ms is None else float(metadata_exposure_ms),
        "metadata_exposure_source": str(metadata_exposure_source or ""),
        "exposure_override_enabled": bool(exposure_override_enabled),
        "exposure_override_ms": None if exposure_override_ms is None else float(exposure_override_ms),
        "current_frame": int(current_frame),
        "num_frames": int(num_frames),
        "selected_molecule": None if selected_molecule is None else int(selected_molecule),
        "deleted_molecules": [int(m) for m in list(deleted_molecules or [])],
        "intensity_scales": dict(intensity_scales or {}),
        "lut_settings_by_channel": dict(lut_settings_by_channel or {}),
        "playback": dict(playback or {}),
        "uncertainty_filter": dict(uncertainty_filter or {}),
        "active_edit_tool": str(active_edit_tool or "pan"),
        "view_transform": dict(view_transform or {}),
        "results_mode": str(results_mode or "normal"),
    }
    analysis_recipe_payload = dict(analysis_recipe or {})
    operation_log_text = "\n".join(
        json.dumps(dict(entry), ensure_ascii=False)
        for entry in list(operation_log or [])
    )

    arrays = {
        "deleted_molecule_ids": np.asarray(sorted(int(m) for m in list(deleted_molecules or [])), dtype=np.int32),
    }
    arrays.update(_molecule_arrays(molecules, molecule_features, prefix="current"))
    arrays.update(_molecule_arrays(base_molecules, base_molecule_features, prefix="base"))

    drift_result = dict(drift_result or {})
    drift_timings = _json_safe_numeric_dict(drift_result.get("timings_sec", {}))
    if drift_timings:
        project_state["drift_timings_sec"] = drift_timings
    for key, value in drift_result.items():
        if key == "timings_sec":
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.dtype == object or arr.dtype.kind not in "biufc":
            continue
        arrays[f"drift_{key}"] = arr

    return manifest, project_state, analysis_recipe_payload, operation_log_text, arrays


def save_project_bundle(
    path: str,
    *,
    nd2_file,
    registration_params,
    detect_channel,
    channel_names,
    molecules,
    deleted_molecules,
    molecule_intensities,
    intensity_scales,
    lut_settings_by_channel,
    exposure_s,
    exposure_source="ND2 metadata",
    metadata_exposure_ms=None,
    metadata_exposure_source="",
    exposure_override_enabled=False,
    exposure_override_ms=None,
    current_frame,
    num_frames,
    num_channels=0,
    height=0,
    width=0,
    backend_name="",
    backend_capabilities=None,
    selected_molecule=None,
    playback=None,
    nd2_fingerprint=None,
    analysis_recipe=None,
    operation_log=None,
    uncertainty_filter=None,
    active_edit_tool="pan",
    view_transform=None,
    results_mode="normal",
    molecule_features=None,
    base_molecules=None,
    base_molecule_features=None,
    drift_result=None,
) -> None:
    manifest, project_state, analysis_recipe_payload, operation_log_text, arrays = _build_bundle_payload(
        nd2_file=nd2_file,
        nd2_fingerprint=nd2_fingerprint,
        registration_params=registration_params,
        detect_channel=detect_channel,
        channel_names=channel_names,
        molecules=molecules,
        base_molecules=base_molecules or molecules,
        molecule_features=molecule_features or {},
        base_molecule_features=base_molecule_features or molecule_features or {},
        deleted_molecules=deleted_molecules,
        intensity_scales=intensity_scales,
        lut_settings_by_channel=lut_settings_by_channel,
        exposure_s=exposure_s,
        exposure_source=exposure_source,
        metadata_exposure_ms=metadata_exposure_ms,
        metadata_exposure_source=metadata_exposure_source,
        exposure_override_enabled=exposure_override_enabled,
        exposure_override_ms=exposure_override_ms,
        current_frame=current_frame,
        num_frames=num_frames,
        num_channels=num_channels,
        height=height,
        width=width,
        backend_name=backend_name,
        backend_capabilities=backend_capabilities,
        selected_molecule=selected_molecule,
        playback=playback,
        analysis_recipe=analysis_recipe,
        operation_log=operation_log,
        uncertainty_filter=uncertainty_filter,
        active_edit_tool=active_edit_tool,
        view_transform=view_transform,
        results_mode=results_mode,
        drift_result=drift_result,
    )

    intensity_arrays = {}
    for mol_id, ch_dict in (molecule_intensities or {}).items():
        for ch_name, arr in ch_dict.items():
            intensity_arrays[f"mol_{mol_id}__{ch_name}"] = np.asarray(arr, dtype=np.float32)

    tmp_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_zip = tempfile.mkstemp(prefix=".smproj_", suffix=".tmp", dir=tmp_dir)
    os.close(fd)
    tmp_npz = None
    tmp_intensity_npz = None
    try:
        fd_npz, tmp_npz = tempfile.mkstemp(prefix=".smproj_", suffix=".npz", dir=tmp_dir)
        os.close(fd_npz)
        np.savez_compressed(tmp_npz, **arrays)

        if intensity_arrays:
            fd_int, tmp_intensity_npz = tempfile.mkstemp(prefix=".smproj_", suffix=".npz", dir=tmp_dir)
            os.close(fd_int)
            np.savez_compressed(tmp_intensity_npz, **intensity_arrays)

        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", _safe_json_dump(manifest))
            zf.writestr("project_state.json", _safe_json_dump(project_state))
            zf.writestr("analysis_recipe.json", _safe_json_dump(analysis_recipe_payload))
            zf.writestr("operation_log.jsonl", operation_log_text)
            zf.write(tmp_npz, arcname="analysis_arrays.npz")
            if tmp_intensity_npz is not None:
                zf.write(tmp_intensity_npz, arcname="intensities.npz")
        os.replace(tmp_zip, path)
    finally:
        for tmp_path in (tmp_zip, tmp_npz, tmp_intensity_npz):
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)


def load_project_bundle(path: str) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        raise RuntimeError("Legacy project files are no longer supported. Please export again as the new .smproj format")

    with zipfile.ZipFile(path, "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise RuntimeError("Project bundle is missing manifest.json")
        required = {"project_state.json", "analysis_recipe.json", "operation_log.jsonl", "analysis_arrays.npz"}
        missing = [name for name in required if name not in zf.namelist()]
        if missing:
            raise RuntimeError(f"Project bundle is missing required entries: {', '.join(missing)}")

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("version") != PROJECT_FORMAT_VERSION:
            raise RuntimeError(f"Unsupported project version: {manifest.get('version')}")

        project_state = json.loads(zf.read("project_state.json").decode("utf-8"))
        analysis_recipe = json.loads(zf.read("analysis_recipe.json").decode("utf-8"))
        operation_log = []
        operation_text = zf.read("operation_log.jsonl").decode("utf-8")
        for line in operation_text.splitlines():
            line = line.strip()
            if not line:
                continue
            operation_log.append(json.loads(line))

        analysis_arrays: dict[str, np.ndarray] = {}
        with np.load(io.BytesIO(zf.read("analysis_arrays.npz")), allow_pickle=False) as data:
            for key in data.files:
                analysis_arrays[key] = data[key]

        molecules, molecule_features = _restore_molecule_features(analysis_arrays, "current")
        base_molecules, base_molecule_features = _restore_molecule_features(analysis_arrays, "base")
        deleted = set(int(v) for v in np.asarray(analysis_arrays.get("deleted_molecule_ids", np.zeros((0,), dtype=np.int32))).tolist())

        molecule_intensities = {}
        if "intensities.npz" in zf.namelist():
            npz_bytes = io.BytesIO(zf.read("intensities.npz"))
            with np.load(npz_bytes) as data:
                for key in data.files:
                    if not key.startswith("mol_") or "__" not in key:
                        continue
                    mol_part, ch_name = key.split("__", 1)
                    mol_id = int(mol_part.replace("mol_", "", 1))
                    molecule_intensities.setdefault(mol_id, {})[ch_name] = data[key].astype(np.float32)

        drift_result = {}
        for key, value in analysis_arrays.items():
            if key.startswith("drift_"):
                drift_result[key.replace("drift_", "", 1)] = np.asarray(value)
        drift_timings = _json_safe_numeric_dict(project_state.get("drift_timings_sec", {}))
        if drift_timings:
            drift_result["timings_sec"] = drift_timings

        merged = {
            **project_state,
            "version": manifest.get("version"),
            "manifest": manifest,
            "analysis_recipe": analysis_recipe,
            "operation_log": operation_log,
            "molecules": molecules,
            "base_molecules": base_molecules,
            "molecule_features": molecule_features,
            "base_molecule_features": base_molecule_features,
            "deleted_molecules": list(deleted),
            "molecule_intensities": molecule_intensities,
            "backend_name": str(manifest.get("backend_name", "") or project_state.get("backend_name", "")),
            "backend_capabilities": dict(manifest.get("backend_capabilities", {}) or project_state.get("backend_capabilities", {})),
            "nd2_fingerprint": dict(project_state.get("nd2_fingerprint", {}) or manifest.get("source_nd2_fingerprint", {})),
            "drift_result": drift_result,
        }
        return merged


def _send_event(event_queue: mp.Queue, event: dict[str, Any]) -> None:
    event_queue.put(event)


def _make_progress_callback(event_queue: mp.Queue) -> Callable[[int, str], None]:
    def _progress(progress: int, message: str) -> None:
        _send_event(
            event_queue,
            {"type": "progress", "progress": int(progress), "message": str(message)},
        )

    return _progress


def _make_cancel_check(cancel_event: mp.synchronize.Event) -> Callable[[], bool]:
    return cancel_event.is_set


def task_projection(event_queue, cancel_event, *, nd2_path, exposure_ms=None, channel_idx, n=10, cache_budget_mb=DEFAULT_CACHE_BUDGET_MB, compute_backend="auto"):
    reader = None
    cache = None
    try:
        reader = ND2Reader(nd2_path)
        cache = MultiLevelCache(reader, cache_budget_mb=cache_budget_mb)
        proj = max_projection_first_n_frames(
            cache,
            int(channel_idx),
            n=int(n),
            cancel_check=_make_cancel_check(cancel_event),
            compute_backend=compute_backend,
        )
        _send_event(event_queue, {"type": "result", "payload": proj})
    finally:
        if cache is not None:
            cache.shutdown()
        if reader is not None:
            reader.close()


def task_affine_registration(event_queue, cancel_event, *, bead_file, exposure_ms=None, cache_budget_mb=DEFAULT_CACHE_BUDGET_MB):
    reader = None
    cache = None
    try:
        reader = ND2Reader(bead_file)
        cache = MultiLevelCache(reader, cache_budget_mb=cache_budget_mb)
        channel_names = list(cache.metadata["channel_names"])
        proj_by_name = {}
        for idx, logical_ch in enumerate(channel_names):
            if cancel_event.is_set():
                raise RuntimeError("Task cancelled")
            ch = physical_channel_name(logical_ch)
            if ch in {"488", "532", "638"} and ch not in proj_by_name:
                proj_by_name[ch] = max_projection_first_n_frames(
                    cache,
                    idx,
                    n=10,
                    cancel_check=_make_cancel_check(cancel_event),
                ).astype(np.float32)
        if REFERENCE_CHANNEL not in proj_by_name:
            raise RuntimeError(f"Bead calibration file is missing channel {REFERENCE_CHANNEL}")
        estimator = RegistrationEstimator()
        model = estimator.estimate_from_bead_images(proj_by_name)
        _send_event(event_queue, {"type": "result", "payload": model.to_dict()})
    finally:
        if cache is not None:
            cache.shutdown()
        if reader is not None:
            reader.close()


def task_detection(event_queue, cancel_event, *, frame, params, final=False, preview_limit=PREVIEW_TOP_K):
    detector = MoleculeDetector()
    progress_cb = _make_progress_callback(event_queue)
    cancel_check = _make_cancel_check(cancel_event)
    if final:
        positions, stats = detector.detect_final(
            frame,
            preview_limit=preview_limit,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            **params,
        )
    else:
        positions, stats = detector.detect_preview(
            frame,
            preview_limit=preview_limit,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            **params,
        )
    _send_event(
        event_queue,
        {
            "type": "result",
            "payload": {
                "positions": positions,
                "results": list(stats.get("results", []) or []),
                "stats": stats,
            },
        },
    )


def task_process_intensities(
    event_queue,
    cancel_event,
    *,
    nd2_path,
    exposure_ms=None,
    molecules,
    registration_params,
    detect_channel,
    channel_names,
    intensity_scales,
    cache_budget_mb=DEFAULT_CACHE_BUDGET_MB,
    compute_backend="auto",
    drift_tracking_mode="full",
    sparse_drift_interval_sec=1.0,
):
    reader = None
    cache = None
    try:
        reader = ND2Reader(nd2_path)
        cache = MultiLevelCache(reader, cache_budget_mb=cache_budget_mb)
        result, drift_result = compute_molecule_intensities(
            cache,
            molecules,
            registration_params,
            detect_channel,
            channel_names,
            intensity_scales=intensity_scales,
            progress_cb=_make_progress_callback(event_queue),
            stop_checker=_make_cancel_check(cancel_event),
            compute_backend=compute_backend,
            return_drift_result=True,
            drift_tracking_mode=drift_tracking_mode,
            sparse_drift_interval_sec=sparse_drift_interval_sec,
        )
        _send_event(
            event_queue,
            {
                "type": "result",
                "payload": {
                    "molecule_intensities": result,
                    "drift_result": drift_result or {},
                },
            },
        )
    finally:
        if cache is not None:
            cache.shutdown()
        if reader is not None:
            reader.close()


def _csv_already_exists_for_stem(folder: str, stem: str) -> bool:
    base = os.path.join(folder, f"{stem}.csv")
    if os.path.exists(base):
        return True
    pattern = os.path.join(folder, f"{stem}_[0-9]*.csv")
    return len(glob.glob(pattern)) > 0


def _is_cancel_exception(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "Task cancelled" in str(exc)


def _process_one_batch_file(
    *,
    file_index: int,
    total_files: int,
    nd2_path: str,
    folder: str,
    detect_channel_name: str,
    params: dict[str, Any],
    registration_params,
    intensity_scales: dict[str, float],
    proj_n: int,
    proj_start: int,
    auto_search: bool,
    search_step: int,
    max_search_windows: int,
    cache_budget_mb: int,
    compute_backend: str,
    drift_tracking_mode: str = "full",
    sparse_drift_interval_sec: float = 1.0,
    exposure_override_enabled: bool = False,
    exposure_override_ms: Any = None,
    cancel_check: Callable[[], bool],
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    del total_files
    total_start = time.perf_counter()
    timings = _new_batch_timings()
    stem = os.path.splitext(os.path.basename(nd2_path))[0]
    base_csv = os.path.join(folder, f"{stem}.csv")
    reader = None
    cache = None
    metadata_record: dict[str, Any] = {}

    def emit(progress: int, message: str) -> None:
        if progress_cb is not None:
            progress_cb(int(file_index), int(progress), str(message))

    try:
        if cancel_check():
            raise RuntimeError("Task cancelled")

        if _csv_already_exists_for_stem(folder, stem):
            return _finish_batch_record(
                {
                    "file_index": int(file_index),
                    "file": os.path.basename(nd2_path),
                    "status": "skipped",
                    "reason": "CSV output already exists for this file stem",
                },
                timings,
                total_start,
            )

        emit(0, f"Reading: {stem}.nd2")
        stage_start = time.perf_counter()
        reader = ND2Reader(nd2_path)
        cache = MultiLevelCache(reader, cache_budget_mb=cache_budget_mb)
        timings["open_cache_sec"] += time.perf_counter() - stage_start
        metadata_record = _batch_metadata_record(
            cache,
            exposure_override_enabled=exposure_override_enabled,
            exposure_override_ms=exposure_override_ms,
        )

        channel_names = list(cache.metadata["channel_names"])
        if detect_channel_name not in channel_names:
            return _finish_batch_record(
                {
                    "file_index": int(file_index),
                    "file": os.path.basename(nd2_path),
                    "status": "skipped",
                    "reason": (
                        f"Detection channel {detect_channel_name} is missing; "
                        f"available channels: {', '.join(channel_names)}"
                    ),
                    **metadata_record,
                },
                timings,
                total_start,
            )

        detect_channel_idx = channel_names.index(detect_channel_name)
        stage_start = time.perf_counter()
        positions, detection_stats, attempts = _detect_batch_molecules(
            cache,
            detect_channel_idx,
            params,
            start_frame=proj_start,
            n=proj_n,
            auto_search=auto_search,
            step_frames=search_step,
            max_windows=max_search_windows,
            cancel_check=cancel_check,
            compute_backend=compute_backend,
        )
        timings["detection_sec"] += time.perf_counter() - stage_start

        if not positions:
            tried = ", ".join(
                f"{a['start_frame']}:{a['stop_frame']} ({a['molecules']})"
                for a in attempts[:8]
            )
            if len(attempts) > 8:
                tried += ", ..."
            detection_window = (
                f"{attempts[0]['start_frame']}:{attempts[0]['stop_frame']}"
                if attempts else f"{proj_start}:{proj_start + proj_n}"
            )
            reason = "No molecules were detected by the interactive detector in the required first-frame detection window"
            if tried:
                reason = f"{reason}; windows tried: {tried}"
            return _finish_batch_record(
                {
                    "file_index": int(file_index),
                    "file": os.path.basename(nd2_path),
                    "status": "failed",
                    "reason": reason,
                    "detection_window": detection_window,
                    "detected_rois": 0,
                    "detected_before_qc": 0,
                    "detection_attempts": attempts,
                    "detection_stats": detection_stats,
                    **metadata_record,
                },
                timings,
                total_start,
            )

        detection_window = attempts[-1] if attempts else None
        if detection_window is not None and int(detection_window["start_frame"]) != proj_start:
            emit(
                20,
                (
                    f"{stem}: auto-selected detection window "
                    f"{detection_window['start_frame']}:{detection_window['stop_frame']} "
                    f"({len(positions)} molecules)"
                ),
            )
        molecules = [(i + 1, p[0], p[1]) for i, p in enumerate(positions)]

        def _progress(progress: int, message: str) -> None:
            emit(progress, f"{stem}: {message}")

        stage_start = time.perf_counter()
        intensities, intensity_diagnostics = compute_molecule_intensities(
            cache,
            molecules,
            registration_params,
            detect_channel_idx,
            channel_names,
            intensity_scales=intensity_scales,
            progress_cb=_progress,
            stop_checker=cancel_check,
            return_diagnostics=True,
            compute_backend=compute_backend,
            drift_tracking_mode=drift_tracking_mode,
            sparse_drift_interval_sec=sparse_drift_interval_sec,
        )
        timings["intensity_sec"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        num_frames = cache.metadata["num_frames"]
        complete_molecules, complete_intensities, dropped_rois = _strict_filter_complete_molecules(
            molecules,
            intensities,
            channel_names,
            expected_frames=int(num_frames),
            diagnostics=intensity_diagnostics,
        )
        qc_warning = _qc_warning_if_suspicious(
            len(molecules),
            len(complete_molecules),
            dropped_rois,
            intensity_diagnostics,
        )
        timings["strict_qc_sec"] += time.perf_counter() - stage_start

        common_record = {
            "file_index": int(file_index),
            "file": os.path.basename(nd2_path),
            "detection_window": (
                f"{detection_window['start_frame']}:{detection_window['stop_frame']}"
                if detection_window else f"{proj_start}:{proj_start + proj_n}"
            ),
            "detected_rois": len(molecules),
            "detected_before_qc": len(molecules),
            "dropped_rois": dropped_rois,
            "detection_attempts": attempts,
            "detection_stats": detection_stats,
            "intensity_diagnostics": intensity_diagnostics,
            "qc_warning": qc_warning,
            **metadata_record,
        }

        if not complete_molecules:
            return _finish_batch_record(
                {
                    **common_record,
                    "status": "failed",
                    "reason": "All detected ROIs were removed by strict NaN/Inf quality control",
                    "exported_rois": 0,
                },
                timings,
                total_start,
            )

        stage_start = time.perf_counter()
        exposure_s = float(metadata_record.get("exposure_ms", cache.metadata.get("exposure_ms", cache.metadata.get("time_ms", 100.0)))) / 1000.0
        time_arr = np.arange(num_frames) * exposure_s
        written = write_intensities_csv_split(
            base_csv,
            molecules=complete_molecules,
            deleted_molecules=set(),
            molecule_intensities=complete_intensities,
            time_arr=time_arr,
            channel_names=channel_names,
        )
        timings["csv_write_sec"] += time.perf_counter() - stage_start

        return _finish_batch_record(
            {
                **common_record,
                "status": "succeeded",
                "reason": "CSV exported with complete ROI traces only",
                "exported_rois": len(complete_molecules),
                "outputs": written,
            },
            timings,
            total_start,
        )
    except Exception as exc:
        if cancel_check() or _is_cancel_exception(exc):
            raise
        failed_record = {
            "file_index": int(file_index),
            "file": os.path.basename(nd2_path),
            "status": "failed",
            "reason": f"Unhandled processing error: {exc}",
        }
        failed_record.update(metadata_record)
        return _finish_batch_record(failed_record, timings, total_start)
    finally:
        if cache is not None:
            cache.shutdown()
        if reader is not None:
            reader.close()


def _batch_record_completion_message(record: dict[str, Any]) -> str:
    stem = os.path.splitext(str(record.get("file", "")))[0]
    status = str(record.get("status", "unknown"))
    timings = dict(record.get("timings_sec", {}) or {})
    elapsed = _format_optional_float(timings.get("total_sec"), digits=2)
    suffix = f" in {elapsed}s" if elapsed != "n/a" else ""
    if status == "succeeded":
        dropped_rois = list(record.get("dropped_rois", []) or [])
        drop_msg = f", dropped {len(dropped_rois)} incomplete ROI(s)" if dropped_rois else ""
        return (
            f"[Done] {stem} "
            f"({record.get('exported_rois', 0)}/{record.get('detected_rois', 0)} "
            f"complete molecules{drop_msg}){suffix}"
        )
    if status == "skipped":
        return f"[Skipped] {stem}: {record.get('reason', '')}{suffix}"
    return f"[Failed] {stem}: {record.get('reason', '')}{suffix}"


def _task_batch_serial_legacy(event_queue, cancel_event, *, config):
    folder = config["folder"]
    detect_channel_name = config["detect_channel_name"]
    params = dict(config["params"])
    registration_params = config["registration_params"]
    intensity_scales = dict(config["intensity_scales"])
    proj_n = int(config.get("detect_projection_frames", 10))
    proj_start = int(config.get("detect_projection_start_frame", 0) or 0)
    auto_search = bool(config.get("auto_search_detection_window", False))
    search_step = int(config.get("auto_search_step_frames", 50) or 50)
    max_search_windows = int(config.get("auto_search_max_windows", 12) or 12)
    cache_budget_mb = int(config.get("cache_budget_mb", DEFAULT_CACHE_BUDGET_MB))
    compute_backend = str(config.get("compute_backend", "auto") or "auto")
    drift_tracking_mode = str(config.get("drift_tracking_mode", "full") or "full")
    sparse_drift_interval_sec = float(config.get("sparse_drift_interval_sec", 1.0) or 1.0)

    nd2_files = sorted(glob.glob(os.path.join(folder, "*.nd2")))
    if not nd2_files:
        _send_event(event_queue, {"type": "result", "payload": "No .nd2 files were found in the folder"})
        return

    total = len(nd2_files)
    processed = 0
    skipped = 0
    failed_count = 0
    records: list[dict[str, Any]] = []

    for idx, nd2_path in enumerate(nd2_files, start=1):
        if cancel_event.is_set():
            raise RuntimeError("Task cancelled")

        stem = os.path.splitext(os.path.basename(nd2_path))[0]
        base_csv = os.path.join(folder, f"{stem}.csv")
        if _csv_already_exists_for_stem(folder, stem):
            skipped += 1
            records.append(
                {
                    "file": os.path.basename(nd2_path),
                    "status": "skipped",
                    "reason": "CSV output already exists for this file stem",
                }
            )
            _send_event(event_queue, {"type": "progress", "progress": int(idx / total * 100), "message": f"[Skipped] CSV already exists: {stem}"})
            continue

        reader = None
        cache = None
        metadata_record: dict[str, Any] = {}
        try:
            _send_event(event_queue, {"type": "progress", "progress": int((idx - 1) / total * 100), "message": f"Reading: {stem}.nd2"})
            reader = ND2Reader(nd2_path)
            cache = MultiLevelCache(reader, cache_budget_mb=cache_budget_mb)
            metadata_record = _batch_metadata_record(cache)
            channel_names = list(cache.metadata["channel_names"])
            if detect_channel_name not in channel_names:
                skipped += 1
                records.append(
                    {
                        "file": os.path.basename(nd2_path),
                        "status": "skipped",
                        "reason": f"Detection channel {detect_channel_name} is missing; available channels: {', '.join(channel_names)}",
                        **metadata_record,
                    }
                )
                _send_event(event_queue, {"type": "progress", "progress": int(idx / total * 100), "message": f"[Skipped] Missing channel {detect_channel_name}: {stem}"})
                continue

            detect_channel_idx = channel_names.index(detect_channel_name)
            positions, detection_stats, attempts = _detect_batch_molecules(
                cache,
                detect_channel_idx,
                params,
                start_frame=proj_start,
                n=proj_n,
                auto_search=auto_search,
                step_frames=search_step,
                max_windows=max_search_windows,
                cancel_check=_make_cancel_check(cancel_event),
                compute_backend=compute_backend,
            )
            if not positions:
                failed_count += 1
                tried = ", ".join(
                    f"{a['start_frame']}:{a['stop_frame']} ({a['molecules']})"
                    for a in attempts[:8]
                )
                if len(attempts) > 8:
                    tried += ", ..."
                detection_window = (
                    f"{attempts[0]['start_frame']}:{attempts[0]['stop_frame']}"
                    if attempts else f"{proj_start}:{proj_start + proj_n}"
                )
                records.append(
                    {
                        "file": os.path.basename(nd2_path),
                        "status": "failed",
                        "reason": "No molecules were detected by the interactive detector in the required first-frame detection window",
                        "detection_window": detection_window,
                        "detected_rois": 0,
                        "detected_before_qc": 0,
                        "detection_attempts": attempts,
                        "detection_stats": detection_stats,
                        **metadata_record,
                    }
                )
                _send_event(
                    event_queue,
                    {
                        "type": "progress",
                        "progress": int(idx / total * 100),
                        "message": (
                            f"[Failed] No molecules detected in {stem}; "
                            f"detection window {tried or detection_window}"
                        ),
                    },
                )
                continue

            detection_window = attempts[-1] if attempts else None
            if detection_window is not None and int(detection_window["start_frame"]) != proj_start:
                _send_event(
                    event_queue,
                    {
                        "type": "progress",
                        "progress": int((idx - 1) / total * 100),
                        "message": (
                            f"{stem}: auto-selected detection window "
                            f"{detection_window['start_frame']}:{detection_window['stop_frame']} "
                            f"({len(positions)} molecules)"
                        ),
                    },
                )
            molecules = [(i + 1, p[0], p[1]) for i, p in enumerate(positions)]

            def _progress(progress: int, message: str) -> None:
                base = (idx - 1) / total * 100
                span = 100 / total
                _send_event(
                    event_queue,
                    {
                        "type": "progress",
                        "progress": int(base + (progress / 100.0) * span),
                        "message": f"{stem}: {message}",
                    },
                )

            intensities, intensity_diagnostics = compute_molecule_intensities(
                cache,
                molecules,
                registration_params,
                detect_channel_idx,
                channel_names,
                intensity_scales=intensity_scales,
                progress_cb=_progress,
                stop_checker=_make_cancel_check(cancel_event),
                return_diagnostics=True,
                compute_backend=compute_backend,
                drift_tracking_mode=drift_tracking_mode,
                sparse_drift_interval_sec=sparse_drift_interval_sec,
            )
            num_frames = cache.metadata["num_frames"]
            complete_molecules, complete_intensities, dropped_rois = _strict_filter_complete_molecules(
                molecules,
                intensities,
                channel_names,
                expected_frames=int(num_frames),
                diagnostics=intensity_diagnostics,
            )
            qc_warning = _qc_warning_if_suspicious(
                len(molecules),
                len(complete_molecules),
                dropped_rois,
                intensity_diagnostics,
            )
            if not complete_molecules:
                failed_count += 1
                records.append(
                    {
                        "file": os.path.basename(nd2_path),
                        "status": "failed",
                        "reason": "All detected ROIs were removed by strict NaN/Inf quality control",
                        "detection_window": (
                            f"{detection_window['start_frame']}:{detection_window['stop_frame']}"
                            if detection_window else f"{proj_start}:{proj_start + proj_n}"
                        ),
                        "detected_rois": len(molecules),
                        "detected_before_qc": len(molecules),
                        "exported_rois": 0,
                        "dropped_rois": dropped_rois,
                        "detection_attempts": attempts,
                        "detection_stats": detection_stats,
                        "intensity_diagnostics": intensity_diagnostics,
                        "qc_warning": qc_warning,
                        **metadata_record,
                    }
                )
                _send_event(
                    event_queue,
                    {
                        "type": "progress",
                        "progress": int(idx / total * 100),
                        "message": (
                            f"[Failed] {stem}: all {len(molecules)} detected ROIs "
                            f"were dropped by strict NaN QC ({_format_drop_summary(dropped_rois)})"
                        ),
                    },
                )
                continue
            exposure_s = float(cache.metadata.get("exposure_ms", cache.metadata.get("time_ms", 100.0))) / 1000.0
            time_arr = np.arange(num_frames) * exposure_s
            written = write_intensities_csv_split(
                base_csv,
                molecules=complete_molecules,
                deleted_molecules=set(),
                molecule_intensities=complete_intensities,
                time_arr=time_arr,
                channel_names=channel_names,
            )
            processed += 1
            records.append(
                {
                    "file": os.path.basename(nd2_path),
                    "status": "succeeded",
                    "reason": "CSV exported with complete ROI traces only",
                    "detection_window": (
                        f"{detection_window['start_frame']}:{detection_window['stop_frame']}"
                        if detection_window else f"{proj_start}:{proj_start + proj_n}"
                    ),
                    "detected_rois": len(molecules),
                    "detected_before_qc": len(molecules),
                    "exported_rois": len(complete_molecules),
                    "dropped_rois": dropped_rois,
                    "detection_attempts": attempts,
                    "detection_stats": detection_stats,
                    "intensity_diagnostics": intensity_diagnostics,
                    "qc_warning": qc_warning,
                    "outputs": written,
                    **metadata_record,
                }
            )
            drop_msg = f", dropped {len(dropped_rois)} incomplete ROI(s)" if dropped_rois else ""
            _send_event(event_queue, {"type": "progress", "progress": int(idx / total * 100), "message": f"[Done] {stem} ({len(complete_molecules)}/{len(molecules)} complete molecules{drop_msg})"})
        except Exception as e:
            if cancel_event.is_set() or "Task cancelled" in str(e):
                raise
            failed_count += 1
            failed_record = {
                "file": os.path.basename(nd2_path),
                "status": "failed",
                "reason": f"Unhandled processing error: {e}",
            }
            failed_record.update(metadata_record)
            records.append(failed_record)
            _send_event(event_queue, {"type": "progress", "progress": int(idx / total * 100), "message": f"[Failed] {stem}: {e}"})
        finally:
            if cache is not None:
                cache.shutdown()
            if reader is not None:
                reader.close()

    try:
        report_path = _write_batch_report(folder, records)
    except Exception as exc:
        logger.exception("Failed to write batch report")
        report_path = f"failed to write report: {exc}"

    _send_event(
        event_queue,
        {
            "type": "result",
            "payload": (
                f"Batch complete: succeeded={processed}, skipped={skipped}, "
                f"failed={failed_count}\nReport: {report_path}"
            ),
        },
    )


def task_batch(event_queue, cancel_event, *, config):
    folder = config["folder"]
    detect_channel_name = config["detect_channel_name"]
    params = dict(config["params"])
    registration_params = config["registration_params"]
    intensity_scales = dict(config["intensity_scales"])
    proj_n = int(config.get("detect_projection_frames", 10))
    proj_start = int(config.get("detect_projection_start_frame", 0) or 0)
    auto_search = bool(config.get("auto_search_detection_window", False))
    search_step = int(config.get("auto_search_step_frames", 50) or 50)
    max_search_windows = int(config.get("auto_search_max_windows", 12) or 12)
    cache_budget_mb = int(config.get("cache_budget_mb", DEFAULT_CACHE_BUDGET_MB))
    compute_backend = str(config.get("compute_backend", "auto") or "auto")
    drift_tracking_mode = str(config.get("drift_tracking_mode", "full") or "full")
    sparse_drift_interval_sec = float(config.get("sparse_drift_interval_sec", 1.0) or 1.0)
    requested_workers = int(config.get("parallel_workers", 2) or 1)
    exposure_override_enabled = bool(config.get("exposure_override_enabled", False))
    exposure_override_ms = config.get("exposure_override_ms")
    if exposure_override_enabled:
        exposure_override_ms = _valid_exposure_override_ms(exposure_override_ms)

    nd2_files = sorted(glob.glob(os.path.join(folder, "*.nd2")))
    if not nd2_files:
        _send_event(event_queue, {"type": "result", "payload": "No .nd2 files were found in the folder"})
        return

    total = len(nd2_files)
    parallel_workers = max(1, min(4, requested_workers, total))
    worker_cache_budget_mb = max(512, int(cache_budget_mb // max(1, parallel_workers)))
    records: list[dict[str, Any]] = []
    batch_start = time.perf_counter()
    progress_state = {"completed": 0, "last": 0}
    progress_lock = Lock()

    _send_event(
        event_queue,
        {
            "type": "progress",
            "progress": 0,
            "message": (
                f"Starting batch: {total} ND2 file(s), "
                f"parallel_workers={parallel_workers}, "
                f"worker_cache_budget={worker_cache_budget_mb} MB"
            ),
        },
    )

    def cancel_check() -> bool:
        return bool(cancel_event.is_set())

    def emit_file_progress(file_index: int, inner_progress: int, message: str) -> None:
        inner = max(0, min(99, int(inner_progress)))
        with progress_lock:
            completed = int(progress_state["completed"])
            if parallel_workers == 1:
                raw_progress = (((int(file_index) - 1) + inner / 100.0) / total) * 100.0
            else:
                raw_progress = ((completed + inner / 100.0) / total) * 100.0
            progress = max(int(progress_state["last"]), int(raw_progress))
            progress = min(progress, 99)
            progress_state["last"] = progress
        _send_event(
            event_queue,
            {"type": "progress", "progress": progress, "message": message},
        )

    def mark_record_complete(record: dict[str, Any]) -> None:
        with progress_lock:
            progress_state["completed"] = int(progress_state["completed"]) + 1
            progress = max(
                int(progress_state["last"]),
                int(progress_state["completed"] / total * 100),
            )
            progress_state["last"] = progress
        _send_event(
            event_queue,
            {
                "type": "progress",
                "progress": min(100, progress),
                "message": _batch_record_completion_message(record),
            },
        )

    def run_one(idx: int, nd2_path: str) -> dict[str, Any]:
        return _process_one_batch_file(
            file_index=idx,
            total_files=total,
            nd2_path=nd2_path,
            folder=folder,
            detect_channel_name=detect_channel_name,
            params=params,
            registration_params=registration_params,
            intensity_scales=intensity_scales,
            proj_n=proj_n,
            proj_start=proj_start,
            auto_search=auto_search,
            search_step=search_step,
            max_search_windows=max_search_windows,
            cache_budget_mb=worker_cache_budget_mb,
            compute_backend=compute_backend,
            drift_tracking_mode=drift_tracking_mode,
            sparse_drift_interval_sec=sparse_drift_interval_sec,
            exposure_override_enabled=exposure_override_enabled,
            exposure_override_ms=exposure_override_ms,
            cancel_check=cancel_check,
            progress_cb=emit_file_progress,
        )

    if parallel_workers == 1:
        for idx, nd2_path in enumerate(nd2_files, start=1):
            if cancel_event.is_set():
                raise RuntimeError("Task cancelled")
            record = run_one(idx, nd2_path)
            records.append(record)
            mark_record_complete(record)
    else:
        futures = []
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_file: dict[Any, tuple[int, str]] = {}
            for idx, nd2_path in enumerate(nd2_files, start=1):
                if cancel_event.is_set():
                    raise RuntimeError("Task cancelled")
                future = executor.submit(run_one, idx, nd2_path)
                futures.append(future)
                future_to_file[future] = (idx, nd2_path)
            try:
                for future in as_completed(futures):
                    if cancel_event.is_set():
                        raise RuntimeError("Task cancelled")
                    idx, nd2_path = future_to_file[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        if cancel_event.is_set() or _is_cancel_exception(exc):
                            raise
                        record = {
                            "file_index": int(idx),
                            "file": os.path.basename(nd2_path),
                            "status": "failed",
                            "reason": f"Unhandled parallel worker error: {exc}",
                            "timings_sec": _new_batch_timings(),
                        }
                    records.append(record)
                    mark_record_complete(record)
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    records.sort(key=lambda rec: int(rec.get("file_index", 0) or 0))
    summary = _summarize_batch_records(
        records,
        wall_time_sec=max(0.0, time.perf_counter() - batch_start),
        parallel_workers=parallel_workers,
    )
    status_counts = dict(summary.get("status_counts", {}) or {})
    processed = int(status_counts.get("succeeded", 0) or 0)
    skipped = int(status_counts.get("skipped", 0) or 0)
    failed_count = int(status_counts.get("failed", 0) or 0)

    try:
        report_path = _write_batch_report(folder, records, summary=summary)
    except Exception as exc:
        logger.exception("Failed to write batch report")
        report_path = f"failed to write report: {exc}"

    _send_event(
        event_queue,
        {
            "type": "result",
            "payload": (
                f"Batch complete: succeeded={processed}, skipped={skipped}, "
                f"failed={failed_count}, wall_time="
                f"{_format_optional_float(summary.get('wall_time_sec'), digits=2)}s, "
                f"parallel_workers={parallel_workers}\nReport: {report_path}"
            ),
        },
    )


def task_test_loop(event_queue, cancel_event, *, steps=10, delay_s=0.01, cooperative=True):
    total_steps = max(1, int(steps))
    for index in range(total_steps):
        if cooperative and cancel_event.is_set():
            raise RuntimeError("Task cancelled")
        time.sleep(float(delay_s))
        _send_event(
            event_queue,
            {
                "type": "progress",
                "progress": int((index + 1) / total_steps * 100),
                "message": f"step {index + 1}",
            },
        )
    _send_event(event_queue, {"type": "result", "payload": {"steps": total_steps}})


TASK_REGISTRY = {
    "projection": task_projection,
    "affine_registration": task_affine_registration,
    "detection_preview": task_detection,
    "detection_final": task_detection,
    "process_intensities": task_process_intensities,
    "batch": task_batch,
    "test_loop": task_test_loop,
}


def _task_entry(task_id: str, kind: str, event_queue, cancel_event, kwargs):
    try:
        fn = TASK_REGISTRY[kind]
        fn(event_queue, cancel_event, **kwargs)
    except Exception as e:
        logger.exception("Task %s failed", kind)
        event_queue.put({"type": "error", "message": str(e)})
    finally:
        event_queue.put({"type": "finished"})


@dataclass
class CancellableTaskHandle:
    task_id: str
    kind: str
    process: mp.Process
    event_queue: Any
    cancel_event: Any
    cancel_requested_at: float | None = None
    grace_seconds: float = 0.75
    terminate_deadline: float | None = None
    terminate_requested_at: float | None = None
    kill_deadline: float | None = None

    def request_cancel(self) -> None:
        if self.cancel_requested_at is None:
            self.cancel_requested_at = time.time()
            self.cancel_event.set()
            self.terminate_deadline = self.cancel_requested_at + self.grace_seconds


@dataclass
class TaskEvent:
    task_id: str
    kind: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


class ProcessTaskService:
    def __init__(self, mp_context: str = "spawn"):
        self.ctx = mp.get_context(mp_context)
        self.active: dict[str, CancellableTaskHandle] = {}

    def start_task(self, kind: str, **kwargs) -> CancellableTaskHandle:
        if kind not in TASK_REGISTRY:
            raise KeyError(f"Unknown task kind: {kind}")
        task_id = uuid.uuid4().hex
        event_queue = self.ctx.Queue()
        cancel_event = self.ctx.Event()
        process = self.ctx.Process(
            target=_task_entry,
            args=(task_id, kind, event_queue, cancel_event, kwargs),
            daemon=True,
        )
        handle = CancellableTaskHandle(
            task_id=task_id,
            kind=kind,
            process=process,
            event_queue=event_queue,
            cancel_event=cancel_event,
        )
        process.start()
        self.active[task_id] = handle
        return handle

    def cancel_task(self, task_id: str) -> None:
        handle = self.active.get(task_id)
        if handle is not None:
            handle.request_cancel()

    def cancel_all(self) -> None:
        for task_id in list(self.active.keys()):
            self.cancel_task(task_id)

    def _cleanup_handle(self, task_id: str) -> None:
        handle = self.active.pop(task_id, None)
        if handle is None:
            return
        try:
            if handle.process.is_alive():
                handle.process.join(timeout=0.05)
        except Exception:
            pass
        try:
            handle.event_queue.close()
        except Exception:
            pass

    def drain_events(self) -> list[TaskEvent]:
        now = time.time()
        drained: list[TaskEvent] = []
        finished_ids: set[str] = set()

        for task_id, handle in list(self.active.items()):
            if (
                handle.terminate_deadline is not None
                and now >= handle.terminate_deadline
                and handle.terminate_requested_at is None
            ):
                if handle.process.is_alive():
                    handle.process.terminate()
                    handle.terminate_requested_at = now
                    handle.kill_deadline = now + 0.5
                handle.terminate_deadline = None
            if (
                handle.kill_deadline is not None
                and now >= handle.kill_deadline
                and handle.process.is_alive()
            ):
                handle.process.kill()
                handle.kill_deadline = None

            while True:
                try:
                    event = handle.event_queue.get_nowait()
                except queue.Empty:
                    break

                event_type = event.get("type", "")
                if event_type == "finished":
                    finished_ids.add(task_id)
                    continue
                drained.append(
                    TaskEvent(
                        task_id=task_id,
                        kind=handle.kind,
                        event_type=event_type,
                        payload={"task_id": task_id, **event},
                    )
                )

            if not handle.process.is_alive():
                finished_ids.add(task_id)

        for task_id in finished_ids:
            self._cleanup_handle(task_id)

        return drained

    def shutdown(self) -> None:
        self.cancel_all()
        deadline = time.time() + 2.0
        while self.active and time.time() < deadline:
            self.drain_events()
            time.sleep(0.05)
        for task_id in list(self.active.keys()):
            handle = self.active[task_id]
            try:
                if handle.process.is_alive():
                    handle.process.kill()
            except Exception:
                pass
            self._cleanup_handle(task_id)
