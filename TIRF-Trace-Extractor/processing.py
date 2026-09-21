"""
processing.py - data processing module v13.0

Updates:
- print() -> logging
- Added a thread lock for DriftCalculator._last_drift_result
- Improved signal safety
"""

import logging
import hashlib
import time as _time
import warnings
import numpy as np
from scipy.signal import savgol_filter
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2
from threading import Lock

from channel_profiles import physical_channel_name
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    pass
from matplotlib.figure import Figure

from gpu_backend import get_cupy_module, normalize_compute_backend
from registration import (
    RegistrationConverter,
    RegistrationCoordinateMapper,
    REFERENCE_CHANNEL,
    validate_quantitative_registration,
)

logger = logging.getLogger(__name__)

GPU_MIN_WORK_ITEMS = 20_000


def _raise_if_cancelled(stop_checker):
    if stop_checker and stop_checker():
        raise RuntimeError("Task cancelled")


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    mask = ~(np.isnan(values) | np.isnan(weights))
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    weights = np.maximum(weights, 0.0)
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return float(np.median(values))
    idx_sort = np.argsort(values)
    sorted_v, sorted_w = values[idx_sort], weights[idx_sort]
    cumsum = np.cumsum(sorted_w)
    i = min(int(np.searchsorted(cumsum, 0.5 * w_sum)), len(sorted_v) - 1)
    return float(sorted_v[i])


def _nan_aware_temporal_median(data_3d, window):
    num_frames = data_3d.shape[0]
    result = np.empty_like(data_3d)
    half = window // 2
    for i in range(num_frames):
        start = max(0, i - half)
        end = min(num_frames, i + half + 1)
        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result[i] = np.nanmedian(data_3d[start:end], axis=0)
    return result


def _map_reference_arrays_to_channel(x_ref, y_ref, channel_name, reg_model, *, strict):
    channel_name = str(channel_name)
    if reg_model is None or physical_channel_name(channel_name) == REFERENCE_CHANNEL:
        return x_ref.astype(np.float64, copy=False), y_ref.astype(np.float64, copy=False)
    Minv = reg_model.get_inverse_matrix(channel_name)
    if Minv is None:
        if strict:
            raise ValueError(f"Registration inverse transform is missing for target channel {channel_name}")
        return x_ref.astype(np.float64, copy=False), y_ref.astype(np.float64, copy=False)
    M = np.asarray(Minv, dtype=np.float64)
    xs = M[0, 0] * x_ref + M[0, 1] * y_ref + M[0, 2]
    ys = M[1, 0] * x_ref + M[1, 1] * y_ref + M[1, 2]
    return xs.astype(np.float64, copy=False), ys.astype(np.float64, copy=False)


def _should_try_cupy_backend(backend, *, num_frames, num_mols, num_ch, collect_diagnostics):
    if backend == "cpu":
        return False
    if backend == "cupy":
        return True
    return get_cupy_module() is not None


def _choose_channel_block_size(
    metadata,
    *,
    cache_budget_mb=None,
    num_channels=1,
    num_mols=0,
    patch_size=0,
    dtype_bytes=2,
    min_frames=20,
    max_frames=500,
):
    try:
        height = int(dict(metadata or {}).get("height", 0) or 0)
        width = int(dict(metadata or {}).get("width", 0) or 0)
    except Exception:
        height, width = 0, 0
    try:
        budget_mb = float(cache_budget_mb)
    except Exception:
        budget_mb = 512.0
    if not np.isfinite(budget_mb) or budget_mb <= 0:
        budget_mb = 512.0
    if height <= 0 or width <= 0:
        return max(1, int(min_frames))

    target_bytes = int(min(1024.0, max(128.0, budget_mb * 0.25)) * 1024 * 1024)
    channels = max(1, int(num_channels or 1))
    frame_bytes = max(1, int(height * width * max(1, int(dtype_bytes)) * channels))
    by_frame_stack = max(1, target_bytes // frame_bytes)

    by_patch_work = int(max_frames)
    if num_mols and patch_size:
        patch_bytes_per_frame = int(max(1, int(num_mols) * int(patch_size) * int(patch_size) * 4))
        by_patch_work = max(1, target_bytes // patch_bytes_per_frame)

    candidate = max(1, min(int(max_frames), int(by_frame_stack), int(by_patch_work)))
    if candidate >= 500:
        return 500
    if candidate >= 200:
        return 200
    if candidate >= 100:
        return 100
    if candidate >= 50:
        return 50
    return max(1, min(int(min_frames), candidate))


def _extract_signal_bg_cupy(
    multi_cache,
    *,
    num_frames,
    num_mols,
    num_ch,
    mol_x0,
    mol_y0,
    drift_offsets,
    channel_names,
    reg_model,
    require_complete_registration,
    mask_inner,
    mask_outer,
    bg_stat,
    batch_size,
    patch_size,
    boundary_margin,
    stop_checker,
):
    cp = get_cupy_module()
    if cp is None:
        return None
    timings = {
        "frame_read_sec": 0.0,
        "patch_extract_sec": 0.0,
        "background_sec": 0.0,
    }

    offsets_np = (np.arange(int(patch_size), dtype=np.float32) - float(int(patch_size) // 2))
    inner_mask_np = np.asarray(mask_inner, dtype=bool)
    outer_mask_np = np.asarray(mask_outer, dtype=bool)
    if not np.any(inner_mask_np) or not np.any(outer_mask_np):
        raise RuntimeError("empty signal or background mask")

    offsets_x = cp.asarray(offsets_np).reshape((1, 1, 1, int(patch_size)))
    offsets_y = cp.asarray(offsets_np).reshape((1, 1, int(patch_size), 1))
    inner_mask_gpu = cp.asarray(inner_mask_np).reshape((1, 1, int(patch_size), int(patch_size)))
    outer_mask_gpu = cp.asarray(outer_mask_np)

    signal_sum = np.full((num_frames, num_mols, num_ch), np.nan, dtype=np.float32)
    bg_est = np.full((num_frames, num_mols, num_ch), np.nan, dtype=np.float32)
    requested_batch_size = max(1, int(batch_size))
    batch_size = max(
        requested_batch_size,
        _choose_channel_block_size(
            getattr(multi_cache, "metadata", {}),
            cache_budget_mb=getattr(multi_cache, "cache_budget_mb", None),
            num_channels=1,
            num_mols=num_mols,
            patch_size=patch_size,
            dtype_bytes=2,
            min_frames=requested_batch_size,
            max_frames=500,
        ),
    )
    block_io = {
        "block_read_used": False,
        "block_size": int(batch_size),
        "block_read_calls": 0,
        "block_read_failures": 0,
        "raw_frame_read_calls": 0,
    }
    block_read_disabled = False

    for start in range(0, int(num_frames), batch_size):
        _raise_if_cancelled(stop_checker)
        end = min(start + batch_size, int(num_frames))
        block_reader = getattr(multi_cache, "get_channel_block", None)
        raw_frame_cache = None
        batch_len = int(end - start)

        drift_b = np.asarray(drift_offsets[start:end], dtype=np.float64)
        if drift_b.shape[0] != batch_len:
            raise RuntimeError("invalid drift offsets")
        finite_drift = np.all(np.isfinite(drift_b), axis=1)
        drift_safe = np.where(np.isfinite(drift_b), drift_b, 0.0)
        x_ref = mol_x0.reshape((1, num_mols)) + drift_safe[:, 0].reshape((batch_len, 1))
        y_ref = mol_y0.reshape((1, num_mols)) + drift_safe[:, 1].reshape((batch_len, 1))

        for ch_idx, ch_name in enumerate(channel_names):
            channel_stack = None
            if callable(block_reader) and not block_read_disabled:
                t_read = _time.perf_counter()
                try:
                    _raise_if_cancelled(stop_checker)
                    block = block_reader(start, end, ch_idx)
                    block_io["block_read_calls"] += 1
                    if block is None:
                        raise RuntimeError("missing channel block data")
                    arr = np.asarray(block)
                    if arr.ndim != 3 or arr.shape[0] != batch_len:
                        raise RuntimeError(f"invalid channel block shape {arr.shape}")
                    channel_stack = arr.astype(np.float32, copy=False)
                    timings["frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
                    block_io["block_read_used"] = True
                except Exception as exc:
                    timings["frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
                    if isinstance(exc, RuntimeError) and "Task cancelled" in str(exc):
                        raise
                    block_io["block_read_failures"] += 1
                    block_read_disabled = True
                    channel_stack = None
                    logger.debug("[GPU] Channel block load failed; falling back to per-frame load: %s", exc)

            if channel_stack is None:
                if raw_frame_cache is None:
                    t_read = _time.perf_counter()
                    raw_frame_cache = []
                    for frame_idx in range(start, end):
                        raw = multi_cache.get_raw_frame(frame_idx)
                        block_io["raw_frame_read_calls"] += 1
                        if raw is None or len(raw) < num_ch:
                            raise RuntimeError("missing frame/channel data")
                        raw_frame_cache.append(raw)
                    timings["frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
                t_read = _time.perf_counter()
                channel_frames = []
                for raw in raw_frame_cache:
                    arr = np.asarray(raw[ch_idx])
                    if arr.ndim != 2:
                        raise RuntimeError("invalid channel frame")
                    channel_frames.append(arr.astype(np.float32, copy=False))
                channel_stack = np.stack(channel_frames, axis=0)
                timings["frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)

            if channel_stack.ndim != 3 or channel_stack.shape[0] != batch_len:
                raise RuntimeError("invalid GPU channel stack")
            _, height, width = channel_stack.shape
            xs_f, ys_f = _map_reference_arrays_to_channel(
                x_ref,
                y_ref,
                ch_name,
                reg_model,
                strict=require_complete_registration,
            )
            finite_coord = np.isfinite(xs_f) & np.isfinite(ys_f)
            valid = (
                finite_drift.reshape((batch_len, 1))
                & finite_coord
                & (xs_f >= boundary_margin)
                & (xs_f < width - boundary_margin)
                & (ys_f >= boundary_margin)
                & (ys_f < height - boundary_margin)
            )
            xs_safe = np.where(valid, xs_f, float(boundary_margin)).astype(np.float32, copy=False)
            ys_safe = np.where(valid, ys_f, float(boundary_margin)).astype(np.float32, copy=False)

            t_patch = _time.perf_counter()
            frames_gpu = cp.asarray(channel_stack, dtype=cp.float32)
            xs_gpu = cp.asarray(xs_safe).reshape((batch_len, num_mols, 1, 1))
            ys_gpu = cp.asarray(ys_safe).reshape((batch_len, num_mols, 1, 1))
            x_grid = xs_gpu + offsets_x
            y_grid = ys_gpu + offsets_y
            x0 = cp.floor(x_grid).astype(cp.int32)
            y0 = cp.floor(y_grid).astype(cp.int32)
            x1 = cp.clip(x0 + 1, 0, width - 1)
            y1 = cp.clip(y0 + 1, 0, height - 1)
            x0 = cp.clip(x0, 0, width - 1)
            y0 = cp.clip(y0, 0, height - 1)
            wx = x_grid - x0.astype(cp.float32)
            wy = y_grid - y0.astype(cp.float32)
            batch_idx = cp.arange(batch_len, dtype=cp.int32).reshape((batch_len, 1, 1, 1))

            v00 = frames_gpu[batch_idx, y0, x0]
            v01 = frames_gpu[batch_idx, y0, x1]
            v10 = frames_gpu[batch_idx, y1, x0]
            v11 = frames_gpu[batch_idx, y1, x1]
            patch = (
                v00 * (1.0 - wx) * (1.0 - wy)
                + v01 * wx * (1.0 - wy)
                + v10 * (1.0 - wx) * wy
                + v11 * wx * wy
            )
            timings["patch_extract_sec"] += max(0.0, _time.perf_counter() - t_patch)

            t_bg = _time.perf_counter()
            sig = cp.sum(cp.where(inner_mask_gpu, patch, cp.asarray(0.0, dtype=cp.float32)), axis=(2, 3))
            bg_pixels = patch[:, :, outer_mask_gpu]
            if str(bg_stat).lower() == "mean":
                bg = cp.mean(bg_pixels, axis=2)
            else:
                bg = cp.median(bg_pixels, axis=2)
            valid_gpu = cp.asarray(valid)
            nan_gpu = cp.asarray(np.nan, dtype=cp.float32)
            sig = cp.where(valid_gpu, sig, nan_gpu)
            bg = cp.where(valid_gpu, bg, nan_gpu)
            signal_sum[start:end, :, ch_idx] = cp.asnumpy(sig).astype(np.float32, copy=False)
            bg_est[start:end, :, ch_idx] = cp.asnumpy(bg).astype(np.float32, copy=False)
            timings["background_sec"] += max(0.0, _time.perf_counter() - t_bg)
            del channel_stack, frames_gpu, xs_gpu, ys_gpu, x_grid, y_grid, patch, sig, bg

    timings["channel_block_io"] = block_io
    return signal_sum, bg_est, timings


class IntensityProcessor:
    @staticmethod
    def precompute_masks(inner_radius=3, outer_radius=8):
        size = outer_radius * 2 + 1
        y_grid, x_grid = np.ogrid[-outer_radius:outer_radius + 1, -outer_radius:outer_radius + 1]
        dist = np.sqrt(x_grid ** 2 + y_grid ** 2)
        return dist <= inner_radius, (dist > inner_radius) & (dist <= outer_radius)

    @staticmethod
    def extract_signal_bg_from_patch(roi_patch, mask_inner, mask_outer, bg_stat="median"):
        signal_pixels = roi_patch[mask_inner]
        background_pixels = roi_patch[mask_outer]
        if signal_pixels.size == 0 or background_pixels.size == 0:
            return 0.0, 0.0, False
        signal_sum = float(np.sum(signal_pixels))
        bg_value = float(np.mean(background_pixels)) if bg_stat == "mean" else float(np.median(background_pixels))
        return signal_sum, bg_value, True

    @staticmethod
    def mask_indices(mask_inner, mask_outer):
        return (
            np.flatnonzero(np.ravel(mask_inner)),
            np.flatnonzero(np.ravel(mask_outer)),
        )

    @staticmethod
    def extract_signal_bg_from_flat_patch(roi_patch, inner_idx, outer_idx, bg_stat="median"):
        flat = np.ravel(roi_patch)
        signal_pixels = flat[inner_idx]
        background_pixels = flat[outer_idx]
        if signal_pixels.size == 0 or background_pixels.size == 0:
            return 0.0, 0.0, False
        signal_sum = float(np.sum(signal_pixels))
        bg_value = float(np.mean(background_pixels)) if bg_stat == "mean" else float(np.median(background_pixels))
        return signal_sum, bg_value, True


class DriftCalculator:
    ROI_HALF = 5
    INNER_RADIUS_TRACK = 3
    MIN_ISOLATION = 16
    MAX_FIDUCIALS = 150
    BRIGHTNESS_PERCENTILE = 30
    SNR_THRESHOLD_TRACK = 2.0
    MAX_CENTROID_DEVIATION = 3.0
    PERSISTENCE_CHECK_FRAMES = 20
    PERSISTENCE_FRACTION = 0.7
    MIN_VALID_FIDUCIALS = 3
    SAVGOL_WINDOW = 31
    SAVGOL_POLYORDER = 3
    SPARSE_VALID_FRAME_FRACTION = 0.80
    SPARSE_MAX_INVALID_GAP_SEC = 3.0
    DEFAULT_SPARSE_INTERVAL_FRAMES = 20
    TIMING_KEYS = (
        "select_sec",
        "track_sec",
        "aggregate_sec",
        "interpolate_sec",
        "smooth_sec",
    )

    _last_drift_result = None
    _drift_result_lock = Lock()

    @classmethod
    def set_last_drift_result(cls, result):
        with cls._drift_result_lock:
            cls._last_drift_result = result

    @classmethod
    def get_last_drift_result(cls):
        with cls._drift_result_lock:
            r = cls._last_drift_result
            if r is None:
                return None
            return dict(r)

    @staticmethod
    def _empty_timings():
        return {key: 0.0 for key in DriftCalculator.TIMING_KEYS}

    @staticmethod
    def _empty_tracking_io(metadata=None, *, block_size=0):
        metadata = dict(metadata or {})
        height = int(metadata.get("height", 0) or 0)
        width = int(metadata.get("width", 0) or 0)
        return {
            "num_frames": int(metadata.get("num_frames", 0) or 0),
            "image_shape": [height, width],
            "num_channels": int(metadata.get("num_channels", 0) or 0),
            "drift_track_frame_read_sec": 0.0,
            "block_read_used": False,
            "block_size": int(block_size),
            "block_read_calls": 0,
            "drift_tracking_mode": "full",
            "drift_tracking_mode_requested": "full",
            "track_interval_sec": 0.0,
            "track_interval_frames": 1,
            "track_interval_exposure_fallback": False,
            "sampled_frames": int(metadata.get("num_frames", 0) or 0),
            "sampled_frame_indices": [],
            "sparse_fallback_used": False,
            "sparse_fallback_reason": "",
            "sparse_valid_sample_fraction": 1.0,
            "sparse_max_invalid_gap_sec": 0.0,
        }

    @staticmethod
    def _zero_drift_result(num_frames, drift_offsets=None, metadata=None):
        n = max(int(num_frames), 1)
        if drift_offsets is None:
            offsets = np.zeros((n, 2), dtype=np.float32)
        else:
            offsets = np.asarray(drift_offsets, dtype=np.float32)
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                offsets = np.zeros((n, 2), dtype=np.float32)
        n = int(offsets.shape[0])
        return {
            'drift_offsets': offsets.copy(),
            'drift_raw': offsets.copy(),
            'drift_interp': offsets.copy(),
            'quality': np.zeros((n,), dtype=np.float32),
            'n_valid_per_frame': np.zeros((n,), dtype=int),
            'mad_values': np.zeros((n,), dtype=np.float32),
            'num_frames': n,
            'num_fiducials': 0,
            'fiducial_ids': [],
            'timings_sec': DriftCalculator._empty_timings(),
            **DriftCalculator._empty_tracking_io({"num_frames": n, **dict(metadata or {})}),
        }

    @staticmethod
    def _return_drift_result(drift_offsets, drift_result, return_result=False):
        DriftCalculator.set_last_drift_result(drift_result)
        if return_result:
            return drift_offsets, drift_result
        return drift_offsets

    @staticmethod
    def _normalize_tracking_mode(mode):
        mode = str(mode or "full").strip().lower()
        return "sparse" if mode == "sparse" else "full"

    @staticmethod
    def _resolve_sparse_interval_frames(metadata, interval_sec):
        metadata = dict(metadata or {})
        exposure_ms = metadata.get("exposure_ms", metadata.get("time_ms"))
        exposure_fallback = False
        try:
            exposure_ms = float(exposure_ms)
        except Exception:
            exposure_ms = float("nan")
        try:
            interval_sec = float(interval_sec)
        except Exception:
            interval_sec = 1.0
        if not np.isfinite(interval_sec) or interval_sec <= 0:
            interval_sec = 1.0
        if np.isfinite(exposure_ms) and exposure_ms > 0:
            frames = max(1, int(round(interval_sec * 1000.0 / exposure_ms)))
        else:
            frames = DriftCalculator.DEFAULT_SPARSE_INTERVAL_FRAMES
            exposure_fallback = True
        return frames, interval_sec, exposure_fallback

    @staticmethod
    def _sampled_frame_indices(num_frames, interval_frames):
        num_frames = int(num_frames)
        if num_frames <= 0:
            return np.zeros((0,), dtype=np.int32)
        interval_frames = max(1, int(interval_frames))
        indices = list(range(0, num_frames, interval_frames))
        last = num_frames - 1
        if not indices or indices[-1] != last:
            indices.append(last)
        return np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int32)

    @staticmethod
    def _sparse_fallback_reason(drift_raw, n_valid_arr, sampled_indices, interval_sec):
        sampled_indices = np.asarray(sampled_indices, dtype=np.int32).reshape(-1)
        if sampled_indices.size == 0:
            return "no sampled frames", 0.0, 0.0
        finite = np.all(np.isfinite(np.asarray(drift_raw)[sampled_indices]), axis=1)
        enough = np.asarray(n_valid_arr)[sampled_indices] >= DriftCalculator.MIN_VALID_FIDUCIALS
        valid = finite & enough
        valid_fraction = float(np.mean(valid)) if valid.size else 0.0

        max_run = 0
        current = 0
        for ok in valid:
            if bool(ok):
                current = 0
            else:
                current += 1
                max_run = max(max_run, current)
        try:
            interval_sec = float(interval_sec)
        except Exception:
            interval_sec = 1.0
        if not np.isfinite(interval_sec) or interval_sec <= 0:
            interval_sec = 1.0
        max_gap_sec = float(max_run * interval_sec)

        if valid_fraction < DriftCalculator.SPARSE_VALID_FRAME_FRACTION:
            return (
                f"valid sampled frame fraction {valid_fraction:.3f} < "
                f"{DriftCalculator.SPARSE_VALID_FRAME_FRACTION:.2f}",
                valid_fraction,
                max_gap_sec,
            )
        if max_gap_sec > DriftCalculator.SPARSE_MAX_INVALID_GAP_SEC:
            return (
                f"consecutive invalid sampled gap {max_gap_sec:.3f}s > "
                f"{DriftCalculator.SPARSE_MAX_INVALID_GAP_SEC:.1f}s",
                valid_fraction,
                max_gap_sec,
            )
        return "", valid_fraction, max_gap_sec

    @staticmethod
    def _sample_patches_bilinear(frame, xs, ys, roi_size):
        xs = np.asarray(xs, dtype=np.float32).reshape(-1)
        ys = np.asarray(ys, dtype=np.float32).reshape(-1)
        n = xs.size
        roi_size = int(roi_size)
        if n == 0:
            return np.empty((0, roi_size, roi_size), dtype=np.float32)

        frame_f = np.asarray(frame, dtype=np.float32)
        height, width = frame_f.shape
        half = roi_size // 2
        offsets = np.arange(roi_size, dtype=np.float32) - float(half)
        x_grid = xs.reshape((n, 1, 1)) + offsets.reshape((1, 1, roi_size))
        y_grid = ys.reshape((n, 1, 1)) + offsets.reshape((1, roi_size, 1))

        x0 = np.floor(x_grid).astype(np.intp)
        y0 = np.floor(y_grid).astype(np.intp)
        wx = x_grid - x0.astype(np.float32)
        wy = y_grid - y0.astype(np.float32)

        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)
        x0 = np.clip(x0, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)

        v00 = frame_f[y0, x0]
        v01 = frame_f[y0, x1]
        v10 = frame_f[y1, x0]
        v11 = frame_f[y1, x1]
        return (
            v00 * (1.0 - wx) * (1.0 - wy)
            + v01 * wx * (1.0 - wy)
            + v10 * (1.0 - wx) * wy
            + v11 * wx * wy
        ).astype(np.float32, copy=False)

    @staticmethod
    def _select_fiducials(multi_cache, detect_channel, molecules, stop_checker=None):
        _raise_if_cancelled(stop_checker)
        if not molecules or len(molecules) < 3:
            return []

        roi_half = DriftCalculator.ROI_HALF
        roi_size = 2 * roi_half + 1
        inner_r = DriftCalculator.INNER_RADIUS_TRACK
        min_iso = DriftCalculator.MIN_ISOLATION
        max_fid = DriftCalculator.MAX_FIDUCIALS
        bright_pct = DriftCalculator.BRIGHTNESS_PERCENTILE
        snr_thresh = DriftCalculator.SNR_THRESHOLD_TRACK
        persist_frames = DriftCalculator.PERSISTENCE_CHECK_FRAMES
        persist_frac = DriftCalculator.PERSISTENCE_FRACTION

        yg, xg = np.mgrid[0:roi_size, 0:roi_size]
        dist_g = np.sqrt((xg - roi_half) ** 2 + (yg - roi_half) ** 2)
        inner_mask = dist_g <= inner_r
        outer_mask = (dist_g > inner_r) & (dist_g <= roi_half)

        frame0 = multi_cache.get_raw_frame(0)[detect_channel]
        if frame0.dtype != np.float32:
            frame0 = frame0.astype(np.float32)
        h, w = frame0.shape
        boundary = roi_half + 2

        mol_ids, mol_xy = [], []
        for mol_id, mx, my in molecules:
            if mx < boundary or mx >= w - boundary or my < boundary or my >= h - boundary:
                continue
            mol_ids.append(mol_id)
            mol_xy.append((float(mx), float(my)))

        if not mol_xy:
            return []

        mol_ids = np.array(mol_ids)
        mol_xy = np.array(mol_xy, dtype=np.float64)
        patches = DriftCalculator._sample_patches_bilinear(frame0, mol_xy[:, 0], mol_xy[:, 1], roi_size)
        bg_pix = patches[:, outer_mask]
        bg = np.median(bg_pix, axis=1)
        bg_std = np.maximum(np.std(bg_pix, axis=1), 0.5)
        peak = np.max(patches[:, inner_mask], axis=1) - bg
        mol_snr = peak / bg_std
        mol_snr = np.array(mol_snr, dtype=np.float64)

        bright_ok = mol_snr >= np.percentile(mol_snr, bright_pct)
        order = np.argsort(-mol_snr)
        selected_idx, selected_pos = [], []

        for idx in order:
            if not bright_ok[idx]:
                continue
            pos = mol_xy[idx]
            if selected_pos:
                dists = np.sqrt(np.sum((np.array(selected_pos) - pos) ** 2, axis=1))
                if np.min(dists) < min_iso:
                    continue
            selected_idx.append(idx)
            selected_pos.append(pos)
            if len(selected_idx) >= max_fid:
                break

        if not selected_idx:
            order_bright = np.argsort(-mol_snr)
            selected_idx = list(order_bright[:min(max_fid, len(order_bright))])

        num_frames = int(multi_cache.metadata["num_frames"])
        n_check = min(persist_frames, num_frames)
        detect_count = np.zeros(len(selected_idx), dtype=int)

        selected_xy = mol_xy[selected_idx]
        for fi in range(n_check):
            _raise_if_cancelled(stop_checker)
            frame = multi_cache.get_raw_frame(fi)[detect_channel]
            if frame.dtype != np.float32:
                frame = frame.astype(np.float32)
            patches = DriftCalculator._sample_patches_bilinear(frame, selected_xy[:, 0], selected_xy[:, 1], roi_size)
            bg_pix = patches[:, outer_mask]
            bg = np.median(bg_pix, axis=1)
            bg_std = np.maximum(np.std(bg_pix, axis=1), 0.5)
            peak = np.max(patches[:, inner_mask], axis=1) - bg
            detect_count += ((peak / bg_std) >= snr_thresh).astype(int)

        persist_ok = detect_count >= (n_check * persist_frac)
        final_indices = [selected_idx[si] for si in range(len(selected_idx)) if persist_ok[si]]

        if len(final_indices) < DriftCalculator.MIN_VALID_FIDUCIALS:
            order_persist = np.argsort(-detect_count)
            n_take = min(max(DriftCalculator.MIN_VALID_FIDUCIALS, 10), len(selected_idx))
            final_indices = [selected_idx[si] for si in order_persist[:n_take]]

        return [(int(mol_ids[idx]), float(mol_xy[idx, 0]), float(mol_xy[idx, 1])) for idx in final_indices]

    @staticmethod
    def _iter_tracking_channel_frames(
        multi_cache,
        detect_channel,
        start_frame,
        stop_frame,
        block_size,
        track_io,
        stop_checker=None,
        frame_indices=None,
    ):
        block_reader = getattr(multi_cache, "get_channel_block", None)
        use_block = callable(block_reader)
        stop_frame = int(stop_frame)
        block_size = max(1, int(block_size))

        if frame_indices is not None:
            for frame_idx in np.asarray(frame_indices, dtype=np.int32).reshape(-1):
                _raise_if_cancelled(stop_checker)
                frame_idx = int(frame_idx)
                if frame_idx < int(start_frame) or frame_idx >= stop_frame:
                    continue
                if use_block:
                    t_read = _time.perf_counter()
                    try:
                        block = block_reader(frame_idx, frame_idx + 1, detect_channel)
                        arr = np.asarray(block)
                        if arr.ndim != 3 or arr.shape[0] != 1:
                            raise RuntimeError(f"invalid drift channel sparse block shape {arr.shape}")
                        track_io["drift_track_frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
                        track_io["block_read_used"] = True
                        track_io["block_read_calls"] = int(track_io.get("block_read_calls", 0) or 0) + 1
                        yield frame_idx, arr[0]
                        continue
                    except Exception as exc:
                        track_io["drift_track_frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
                        if isinstance(exc, RuntimeError) and "Task cancelled" in str(exc):
                            raise
                        logger.debug("[Drift correction] Sparse channel block read failed; falling back to per-frame reads: %s", exc)
                        use_block = False

                t_read = _time.perf_counter()
                raw = multi_cache.get_raw_frame(frame_idx)
                track_io["drift_track_frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
                if raw is None or int(detect_channel) >= len(raw):
                    raise RuntimeError(f"missing drift tracking frame/channel at frame {frame_idx}")
                yield frame_idx, raw[int(detect_channel)]
            return

        frame_idx = int(start_frame)
        while frame_idx < stop_frame:
            _raise_if_cancelled(stop_checker)
            if use_block:
                block_start = frame_idx
                block_stop = min(block_start + block_size, stop_frame)
                t_read = _time.perf_counter()
                try:
                    block = block_reader(block_start, block_stop, detect_channel)
                    arr = np.asarray(block)
                    if arr.ndim != 3 or arr.shape[0] != (block_stop - block_start):
                        raise RuntimeError(f"invalid drift channel block shape {arr.shape}")
                    track_io["drift_track_frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
                    track_io["block_read_used"] = True
                    track_io["block_read_calls"] = int(track_io.get("block_read_calls", 0) or 0) + 1
                    for offset in range(arr.shape[0]):
                        _raise_if_cancelled(stop_checker)
                        yield block_start + offset, arr[offset]
                    frame_idx = block_stop
                    continue
                except Exception as exc:
                    track_io["drift_track_frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
                    if isinstance(exc, RuntimeError) and "Task cancelled" in str(exc):
                        raise
                    logger.debug("[Drift correction] Channel block tracking read failed; falling back to per-frame reads: %s", exc)
                    use_block = False

            t_read = _time.perf_counter()
            raw = multi_cache.get_raw_frame(frame_idx)
            track_io["drift_track_frame_read_sec"] += max(0.0, _time.perf_counter() - t_read)
            if raw is None or int(detect_channel) >= len(raw):
                raise RuntimeError(f"missing drift tracking frame/channel at frame {frame_idx}")
            yield frame_idx, raw[int(detect_channel)]
            frame_idx += 1

    @staticmethod
    def _track_fiducials(
        multi_cache,
        detect_channel,
        init_x_raw,
        init_y_raw,
        progress_cb=None,
        stop_checker=None,
        frame_indices=None,
        tracking_mode="full",
        track_interval_sec=0.0,
        track_interval_frames=1,
        exposure_fallback=False,
    ):
        roi_half = DriftCalculator.ROI_HALF
        roi_size = 2 * roi_half + 1
        inner_r = DriftCalculator.INNER_RADIUS_TRACK
        snr_thresh = DriftCalculator.SNR_THRESHOLD_TRACK
        max_dev = DriftCalculator.MAX_CENTROID_DEVIATION
        min_valid_fid = DriftCalculator.MIN_VALID_FIDUCIALS
        boundary = roi_half + 2

        num_frames = int(multi_cache.metadata["num_frames"])
        h, w = int(multi_cache.metadata["height"]), int(multi_cache.metadata["width"])
        num_fid = len(init_x_raw)
        block_size = _choose_channel_block_size(
            getattr(multi_cache, "metadata", {}),
            cache_budget_mb=getattr(multi_cache, "cache_budget_mb", None),
            num_channels=1,
            dtype_bytes=2,
            min_frames=20,
            max_frames=500,
        )
        track_io = DriftCalculator._empty_tracking_io(
            getattr(multi_cache, "metadata", {}),
            block_size=block_size,
        )
        tracking_mode = DriftCalculator._normalize_tracking_mode(tracking_mode)
        if frame_indices is None:
            frame_indices_arr = None
            sampled_frames = int(num_frames)
            sampled_frame_indices = []
        else:
            frame_indices_arr = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            sampled_frames = int(frame_indices_arr.size)
            sampled_frame_indices = frame_indices_arr.astype(int).tolist()
        track_io.update(
            {
                "drift_tracking_mode": tracking_mode,
                "drift_tracking_mode_requested": tracking_mode,
                "track_interval_sec": float(track_interval_sec or 0.0),
                "track_interval_frames": int(max(1, track_interval_frames)),
                "track_interval_exposure_fallback": bool(exposure_fallback),
                "sampled_frames": int(sampled_frames),
                "sampled_frame_indices": sampled_frame_indices,
            }
        )

        yg, xg = np.mgrid[0:roi_size, 0:roi_size]
        dist_g = np.sqrt((xg - roi_half) ** 2 + (yg - roi_half) ** 2)
        inner_mask = dist_g <= inner_r
        outer_mask = (dist_g > inner_r) & (dist_g <= roi_half)

        frame_iter = DriftCalculator._iter_tracking_channel_frames(
            multi_cache,
            detect_channel,
            0,
            num_frames,
            block_size,
            track_io,
            stop_checker=stop_checker,
            frame_indices=frame_indices_arr,
        )
        try:
            frame0_idx, frame0 = next(frame_iter)
        except StopIteration:
            raise RuntimeError("missing drift tracking frames")
        if int(frame0_idx) != 0:
            raise RuntimeError(f"first drift tracking frame is {frame0_idx}, expected 0")
        if frame0.dtype != np.float32:
            frame0 = frame0.astype(np.float32)

        refined_x, refined_y = np.copy(init_x_raw), np.copy(init_y_raw)
        valid0 = (
            (init_x_raw >= boundary)
            & (init_x_raw < w - boundary)
            & (init_y_raw >= boundary)
            & (init_y_raw < h - boundary)
        )
        valid0_idx = np.nonzero(valid0)[0]
        if valid0_idx.size:
            patches = DriftCalculator._sample_patches_bilinear(
                frame0,
                init_x_raw[valid0_idx],
                init_y_raw[valid0_idx],
                roi_size,
            ).astype(np.float64, copy=False)
            bg = np.median(patches[:, outer_mask], axis=1)
            signal = np.maximum(patches - bg.reshape((-1, 1, 1)), 0.0)
            total = np.sum(signal, axis=(1, 2))
            good_total = total > 0
            dx = np.zeros(valid0_idx.size, dtype=np.float64)
            dy = np.zeros(valid0_idx.size, dtype=np.float64)
            dx[good_total] = np.sum(signal[good_total] * xg, axis=(1, 2)) / total[good_total] - roi_half
            dy[good_total] = np.sum(signal[good_total] * yg, axis=(1, 2)) / total[good_total] - roi_half
            refined_ok = good_total & (np.abs(dx) < max_dev) & (np.abs(dy) < max_dev)
            refined_idx = valid0_idx[refined_ok]
            refined_x[refined_idx] = init_x_raw[refined_idx] + dx[refined_ok]
            refined_y[refined_idx] = init_y_raw[refined_idx] + dy[refined_ok]

        positions = np.zeros((num_frames, num_fid, 2), dtype=np.float64)
        snr_vals = np.zeros((num_frames, num_fid), dtype=np.float64)
        valid = np.zeros((num_frames, num_fid), dtype=bool)

        positions[0, :, 0], positions[0, :, 1] = refined_x, refined_y
        valid[0, :] = True
        snr_vals[0, :] = 99.0

        running_drift_x, running_drift_y = 0.0, 0.0
        report_interval = max(1, num_frames // 50)

        for fi, frame in frame_iter:
            _raise_if_cancelled(stop_checker)
            if frame.dtype != np.float32:
                frame = frame.astype(np.float32)

            exp_x = refined_x + running_drift_x
            exp_y = refined_y + running_drift_y
            positions[fi, :, 0] = exp_x
            positions[fi, :, 1] = exp_y

            in_bounds = (
                (exp_x >= boundary)
                & (exp_x < w - boundary)
                & (exp_y >= boundary)
                & (exp_y < h - boundary)
            )
            in_bounds_idx = np.nonzero(in_bounds)[0]
            if in_bounds_idx.size:
                patches = DriftCalculator._sample_patches_bilinear(
                    frame,
                    exp_x[in_bounds_idx],
                    exp_y[in_bounds_idx],
                    roi_size,
                ).astype(np.float64, copy=False)
                bg_pix = patches[:, outer_mask]
                bg = np.median(bg_pix, axis=1)
                bg_std = np.maximum(np.std(bg_pix, axis=1), 0.5)
                snr = (np.max(patches[:, inner_mask], axis=1) - bg) / bg_std
                snr_vals[fi, in_bounds_idx] = snr

                signal = np.maximum(patches - bg.reshape((-1, 1, 1)), 0.0)
                total = np.sum(signal, axis=(1, 2))
                good_total = total > 0
                dx = np.zeros(in_bounds_idx.size, dtype=np.float64)
                dy = np.zeros(in_bounds_idx.size, dtype=np.float64)
                dx[good_total] = np.sum(signal[good_total] * xg, axis=(1, 2)) / total[good_total] - roi_half
                dy[good_total] = np.sum(signal[good_total] * yg, axis=(1, 2)) / total[good_total] - roi_half
                ok = (
                    (snr >= snr_thresh)
                    & good_total
                    & (np.abs(dx) <= max_dev)
                    & (np.abs(dy) <= max_dev)
                )
                ok_idx = in_bounds_idx[ok]
                if ok_idx.size:
                    mx = exp_x[ok_idx] + dx[ok]
                    my = exp_y[ok_idx] + dy[ok]
                    positions[fi, ok_idx, 0] = mx
                    positions[fi, ok_idx, 1] = my
                    valid[fi, ok_idx] = True
                    disps_x = mx - refined_x[ok_idx]
                    disps_y = my - refined_y[ok_idx]
                    weights = snr[ok]
                    if ok_idx.size >= min_valid_fid:
                        running_drift_x = _weighted_median(disps_x, weights)
                        running_drift_y = _weighted_median(disps_y, weights)

            if progress_cb and (fi % report_interval == 0 or fi == num_frames - 1):
                p = 10 + int(fi / num_frames * 80)
                progress_cb(min(p, 90), f"Tracking frame {fi}/{num_frames}  valid fiducials: {int(np.sum(valid[fi]))}/{num_fid}")

        track_io["drift_track_frame_read_sec"] = round(float(track_io.get("drift_track_frame_read_sec", 0.0) or 0.0), 6)
        return positions, snr_vals, valid, refined_x, refined_y, track_io

    @staticmethod
    def _aggregate_drift(positions, refined_x, refined_y, snr_vals, valid, stop_checker=None):
        num_frames = positions.shape[0]
        min_fid = DriftCalculator.MIN_VALID_FIDUCIALS
        drift = np.full((num_frames, 2), np.nan, dtype=np.float64)
        quality = np.zeros(num_frames, dtype=np.float64)
        n_valid = np.zeros(num_frames, dtype=int)
        mad_values = np.zeros(num_frames, dtype=np.float64)

        drift[0] = [0.0, 0.0]
        quality[0] = 1.0
        n_valid[0] = int(np.sum(valid[0]))

        for fi in range(1, num_frames):
            _raise_if_cancelled(stop_checker)
            v = valid[fi]
            nv = int(np.sum(v))
            n_valid[fi] = nv
            if nv < min_fid:
                continue
            disp_x = positions[fi, v, 0] - refined_x[v]
            disp_y = positions[fi, v, 1] - refined_y[v]
            w = snr_vals[fi, v]
            dx = _weighted_median(disp_x, w)
            dy = _weighted_median(disp_y, w)
            drift[fi] = [dx, dy]
            mad = np.sqrt(float(np.median(np.abs(disp_x - dx))) ** 2 + float(np.median(np.abs(disp_y - dy))) ** 2)
            mad_values[fi] = mad
            quality[fi] = float(nv) / (1.0 + mad * 10.0)

        return drift, quality, n_valid, mad_values

    @staticmethod
    def _interpolate_nan_frames(drift, stop_checker=None):
        _raise_if_cancelled(stop_checker)
        nan_mask = np.any(np.isnan(drift), axis=1)
        if not np.any(nan_mask):
            return drift.copy()
        good_idx = np.where(~nan_mask)[0]
        bad_idx = np.where(nan_mask)[0]
        drift_out = drift.copy()
        if len(good_idx) == 0:
            drift_out[:] = 0.0
            return drift_out
        if len(good_idx) == 1:
            drift_out[bad_idx] = drift_out[good_idx[0]]
            return drift_out
        for dim in range(2):
            drift_out[bad_idx, dim] = np.interp(bad_idx.astype(np.float64), good_idx.astype(np.float64), drift_out[good_idx, dim])
        logger.info(f"  Interpolated correction: {len(bad_idx)}/{drift.shape[0]} frames")
        return drift_out

    @staticmethod
    def _savgol_smooth(drift, window_length=31, polyorder=3, stop_checker=None):
        _raise_if_cancelled(stop_checker)
        n = drift.shape[0]
        wl = int(window_length)
        if wl % 2 == 0:
            wl -= 1
        if wl > n:
            wl = n if n % 2 == 1 else max(1, n - 1)
        if wl < polyorder + 2:
            return drift.copy()
        smoothed = np.zeros_like(drift)
        smoothed[:, 0] = savgol_filter(drift[:, 0], wl, polyorder, mode='nearest')
        smoothed[:, 1] = savgol_filter(drift[:, 1], wl, polyorder, mode='nearest')
        return smoothed

    @classmethod
    def generate_drift_figure(cls):
        result = cls.get_last_drift_result()
        if result is None:
            return None

        drift = np.asarray(result['drift_offsets'])
        drift_raw = np.asarray(result['drift_raw'])
        n_valid_arr = np.asarray(result['n_valid_per_frame'])
        mad_arr = np.asarray(result['mad_values'])
        num_frames = int(result['num_frames'])
        num_fid = int(result['num_fiducials'])

        frames = np.arange(num_frames)
        dx, dy = drift[:, 0], drift[:, 1]
        total = np.sqrt(dx ** 2 + dy ** 2)
        raw_dx, raw_dy = drift_raw[:, 0], drift_raw[:, 1]
        raw_ok = ~np.isnan(raw_dx)

        fig = Figure(figsize=(15, 10))
        fig.suptitle(f'Drift trajectory (fiducial tracking)  |  {num_frames} frames  |  {num_fid} fiducials  |  max drift = {np.nanmax(total):.3f} px', fontsize=13)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(frames[raw_ok], raw_dx[raw_ok], color='steelblue', alpha=0.3, linewidth=0.5, label='Raw')
        ax1.plot(frames, dx, color='royalblue', linewidth=1.2, label='SG smoothed')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Drift X (px)')
        ax1.set_title('X-axis drift')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 2, 2)
        raw_ok_y = ~np.isnan(raw_dy)
        ax2.plot(frames[raw_ok_y], raw_dy[raw_ok_y], color='salmon', alpha=0.3, linewidth=0.5, label='Raw')
        ax2.plot(frames, dy, color='crimson', linewidth=1.2, label='SG smoothed')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Drift Y (px)')
        ax2.set_title('Y-axis drift')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.fill_between(frames, 0, n_valid_arr, color='seagreen', alpha=0.3, label='Valid fiducials')
        ax3.plot(frames, n_valid_arr, color='seagreen', linewidth=0.8)
        ax3.axhline(y=num_fid, color='gray', linestyle='--', alpha=0.5, label=f'Total fiducials ({num_fid})')
        ax3.axhline(y=DriftCalculator.MIN_VALID_FIDUCIALS, color='red', linestyle=':', alpha=0.5, label='Minimum threshold')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Valid fiducials', color='seagreen')
        ax3.set_ylim(bottom=0)
        ax3_r = ax3.twinx()
        ax3_r.plot(frames, mad_arr, color='orange', alpha=0.5, linewidth=0.5, label='MAD')
        ax3_r.set_ylabel('MAD (px)', color='orange')
        ax3.set_title('Tracking quality')
        ax3.legend(fontsize=7, loc='upper left')
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(dx, dy, 'k-', linewidth=0.5, alpha=0.5)
        ax4.plot(dx[0], dy[0], 'go', markersize=10, zorder=5, label='Start')
        ax4.plot(dx[-1], dy[-1], 'rs', markersize=10, zorder=5, label='End')
        ax4.set_xlabel('Drift X (px)')
        ax4.set_ylabel('Drift Y (px)')
        ax4.set_title('XY drift trajectory')
        ax4.set_aspect('equal')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    @staticmethod
    def calculate_drift_parallel(
        multi_cache,
        detect_channel,
        molecules=None,
        inner_radius=3,
        progress_cb=None,
        stop_checker=None,
        return_result=False,
        drift_tracking_mode="full",
        sparse_drift_interval_sec=1.0,
    ):
        _raise_if_cancelled(stop_checker)
        num_frames = int(multi_cache.metadata["num_frames"])
        metadata = dict(getattr(multi_cache, "metadata", {}) or {})
        requested_tracking_mode = DriftCalculator._normalize_tracking_mode(drift_tracking_mode)
        interval_frames, interval_sec, exposure_fallback = DriftCalculator._resolve_sparse_interval_frames(
            metadata,
            sparse_drift_interval_sec,
        )
        effective_tracking_mode = (
            "sparse"
            if requested_tracking_mode == "sparse" and interval_frames > 1 and num_frames > interval_frames
            else "full"
        )
        if num_frames <= 1:
            drift_offsets = np.zeros((max(num_frames, 1), 2), dtype=np.float32)
            result = DriftCalculator._zero_drift_result(num_frames, drift_offsets, metadata)
            result["drift_tracking_mode_requested"] = requested_tracking_mode
            result["drift_tracking_mode"] = "full"
            return DriftCalculator._return_drift_result(drift_offsets, result, return_result=return_result)
        if molecules is None or len(molecules) < 3:
            logger.info("[Drift correction] Molecule count < 3; skipping drift correction")
            drift_offsets = np.zeros((num_frames, 2), dtype=np.float32)
            result = DriftCalculator._zero_drift_result(num_frames, drift_offsets, metadata)
            result["drift_tracking_mode_requested"] = requested_tracking_mode
            result["drift_tracking_mode"] = "full"
            return DriftCalculator._return_drift_result(drift_offsets, result, return_result=return_result)

        pos_data = np.array([(m[1], m[2]) for m in molecules], dtype=np.float64)
        mol_hash = hashlib.md5(pos_data.tobytes()).hexdigest()[:12]
        cache_key = (
            "drift_v6_fiducial",
            int(detect_channel),
            mol_hash,
            effective_tracking_mode,
            int(interval_frames if effective_tracking_mode == "sparse" else 1),
        )

        cached = multi_cache.get_drift_cache(cache_key)
        if cached is not None:
            logger.info("[Drift correction] Using cached drift data")
            if isinstance(cached, dict):
                cached_result = dict(cached)
                drift_offsets = np.asarray(cached_result.get("drift_offsets"), dtype=np.float32)
                cached_timings = dict(cached_result.get("timings_sec", {}) or {})
                empty_timings = DriftCalculator._empty_timings()
                for key, value in empty_timings.items():
                    cached_timings.setdefault(key, value)
                cached_result["timings_sec"] = cached_timings
                cached_tracking_io = DriftCalculator._empty_tracking_io(
                    metadata,
                    block_size=int(cached_result.get("block_size", 0) or 0),
                )
                for key, value in cached_tracking_io.items():
                    cached_result.setdefault(key, value)
                return DriftCalculator._return_drift_result(
                    drift_offsets,
                    cached_result,
                    return_result=return_result,
                )
            drift_offsets = np.asarray(cached, dtype=np.float32)
            return DriftCalculator._return_drift_result(
                drift_offsets,
                DriftCalculator._zero_drift_result(num_frames, drift_offsets, metadata),
                return_result=return_result,
            )

        t0 = _time.time()
        total_t0 = _time.perf_counter()
        drift_timings = DriftCalculator._empty_timings()
        logger.info(f"[Drift correction v5] Fiducial drift estimation  frames={num_frames}, molecules={len(molecules)}")

        if progress_cb:
            progress_cb(2, "Drift: selecting fiducials...")

        phase_t0 = _time.perf_counter()
        fiducials = DriftCalculator._select_fiducials(
            multi_cache,
            detect_channel,
            molecules,
            stop_checker=stop_checker,
        )
        drift_timings["select_sec"] += max(0.0, _time.perf_counter() - phase_t0)
        num_fid = len(fiducials)

        if num_fid < DriftCalculator.MIN_VALID_FIDUCIALS:
            logger.warning(f"  Only {num_fid} fiducials were selected; not enough for tracking")
            drift_offsets = np.zeros((num_frames, 2), dtype=np.float32)
            drift_timings["total_sec"] = max(0.0, _time.perf_counter() - total_t0)
            drift_result = DriftCalculator._zero_drift_result(
                num_frames,
                drift_offsets,
                metadata,
            )
            drift_result["num_fiducials"] = int(num_fid)
            drift_result["fiducial_ids"] = [f[0] for f in fiducials]
            drift_result["timings_sec"] = {
                key: round(float(value), 6)
                for key, value in drift_timings.items()
            }
            return DriftCalculator._return_drift_result(
                drift_offsets,
                drift_result,
                return_result=return_result,
            )

        logger.info(f"  Phase 1: selected {num_fid} fiducials")
        init_x = np.array([f[1] for f in fiducials], dtype=np.float64)
        init_y = np.array([f[2] for f in fiducials], dtype=np.float64)

        if progress_cb:
            if effective_tracking_mode == "sparse":
                progress_cb(5, f"Drift: sparse tracking every {interval_frames} frames...")
            else:
                progress_cb(5, "Drift: tracking fiducials...")

        phase_t0 = _time.perf_counter()
        sampled_indices = (
            DriftCalculator._sampled_frame_indices(num_frames, interval_frames)
            if effective_tracking_mode == "sparse"
            else None
        )
        positions, snr_vals, valid_mask, refined_x, refined_y, track_io = DriftCalculator._track_fiducials(
            multi_cache,
            detect_channel,
            init_x,
            init_y,
            progress_cb=progress_cb,
            stop_checker=stop_checker,
            frame_indices=sampled_indices,
            tracking_mode=effective_tracking_mode,
            track_interval_sec=interval_sec if effective_tracking_mode == "sparse" else 0.0,
            track_interval_frames=interval_frames if effective_tracking_mode == "sparse" else 1,
            exposure_fallback=exposure_fallback if effective_tracking_mode == "sparse" else False,
        )
        drift_timings["track_sec"] += max(0.0, _time.perf_counter() - phase_t0)

        if progress_cb:
            progress_cb(92, "Drift: aggregating drift...")

        phase_t0 = _time.perf_counter()
        drift_raw, quality, n_valid_arr, mad_arr = DriftCalculator._aggregate_drift(
            positions,
            refined_x,
            refined_y,
            snr_vals,
            valid_mask,
            stop_checker=stop_checker,
        )
        drift_timings["aggregate_sec"] += max(0.0, _time.perf_counter() - phase_t0)

        if effective_tracking_mode == "sparse":
            reason, valid_fraction, max_gap_sec = DriftCalculator._sparse_fallback_reason(
                drift_raw,
                n_valid_arr,
                sampled_indices,
                interval_sec,
            )
            track_io["sparse_valid_sample_fraction"] = round(float(valid_fraction), 6)
            track_io["sparse_max_invalid_gap_sec"] = round(float(max_gap_sec), 6)
            if reason:
                logger.warning("[Drift correction] Sparse drift fallback to full tracking: %s", reason)
                if progress_cb:
                    progress_cb(7, f"Drift: sparse fallback to full tracking ({reason})")
                sparse_summary = dict(track_io or {})
                phase_t0 = _time.perf_counter()
                positions, snr_vals, valid_mask, refined_x, refined_y, track_io = DriftCalculator._track_fiducials(
                    multi_cache,
                    detect_channel,
                    init_x,
                    init_y,
                    progress_cb=progress_cb,
                    stop_checker=stop_checker,
                    tracking_mode="full",
                    track_interval_sec=0.0,
                    track_interval_frames=1,
                    exposure_fallback=False,
                )
                drift_timings["track_sec"] += max(0.0, _time.perf_counter() - phase_t0)
                phase_t0 = _time.perf_counter()
                drift_raw, quality, n_valid_arr, mad_arr = DriftCalculator._aggregate_drift(
                    positions,
                    refined_x,
                    refined_y,
                    snr_vals,
                    valid_mask,
                    stop_checker=stop_checker,
                )
                drift_timings["aggregate_sec"] += max(0.0, _time.perf_counter() - phase_t0)
                track_io["drift_tracking_mode_requested"] = requested_tracking_mode
                track_io["sparse_fallback_used"] = True
                track_io["sparse_fallback_reason"] = reason
                track_io["track_interval_sec"] = float(interval_sec)
                track_io["track_interval_frames"] = int(interval_frames)
                track_io["track_interval_exposure_fallback"] = bool(exposure_fallback)
                track_io["sampled_frames"] = int(sparse_summary.get("sampled_frames", 0) or 0)
                track_io["sampled_frame_indices"] = list(sparse_summary.get("sampled_frame_indices", []) or [])
                track_io["sparse_valid_sample_fraction"] = round(float(valid_fraction), 6)
                track_io["sparse_max_invalid_gap_sec"] = round(float(max_gap_sec), 6)

        frac_valid = np.mean(valid_mask[1:]) if num_frames > 1 else 1.0
        logger.info(f"    Average valid tracking rate: {frac_valid * 100:.1f}%")

        n_nan = int(np.sum(np.any(np.isnan(drift_raw), axis=1)))
        if n_nan > 0:
            logger.info(f"  Phase 4a: interpolating {n_nan} invalid frames...")
        phase_t0 = _time.perf_counter()
        drift_interp = DriftCalculator._interpolate_nan_frames(drift_raw, stop_checker=stop_checker)
        drift_timings["interpolate_sec"] += max(0.0, _time.perf_counter() - phase_t0)

        sg_win, sg_ord = DriftCalculator.SAVGOL_WINDOW, DriftCalculator.SAVGOL_POLYORDER
        phase_t0 = _time.perf_counter()
        drift_smooth = DriftCalculator._savgol_smooth(
            drift_interp,
            window_length=sg_win,
            polyorder=sg_ord,
            stop_checker=stop_checker,
        )
        drift_smooth -= drift_smooth[0:1, :]
        drift_offsets = drift_smooth.astype(np.float32)
        drift_timings["smooth_sec"] += max(0.0, _time.perf_counter() - phase_t0)
        drift_timings["total_sec"] = max(0.0, _time.perf_counter() - total_t0)

        elapsed = _time.time() - t0
        max_drift = float(np.max(np.sqrt(np.sum(drift_offsets ** 2, axis=1))))
        logger.info(f"  Drift estimation complete ({elapsed:.1f}s) max drift={max_drift:.4f}px")

        drift_result = {
            'drift_offsets': drift_offsets.copy(),
            'drift_raw': drift_raw.astype(np.float32),
            'drift_interp': drift_interp.astype(np.float32),
            'quality': quality.astype(np.float32),
            'n_valid_per_frame': n_valid_arr.copy(),
            'mad_values': mad_arr.astype(np.float32),
            'num_frames': num_frames,
            'num_fiducials': num_fid,
            'fiducial_ids': [f[0] for f in fiducials],
            'timings_sec': {key: round(float(value), 6) for key, value in drift_timings.items()},
            **dict(track_io or {}),
        }
        DriftCalculator.set_last_drift_result(drift_result)

        if progress_cb:
            progress_cb(95, "Drift estimation complete")
        multi_cache.set_drift_cache(cache_key, drift_result)
        return DriftCalculator._return_drift_result(
            drift_offsets,
            drift_result,
            return_result=return_result,
        )


def compute_molecule_intensities(
    multi_cache, molecules, registration_params, detect_channel, channel_names,
    intensity_scales=None, *, bg_stat="median", temporal_bg_window=5,
    temporal_bg_enabled=True, inner_radius=3, outer_radius=8,
    batch_size=20, max_workers=None, progress_cb=None, stop_checker=None,
    return_diagnostics=False, require_complete_registration=True,
    compute_backend="auto",
    return_drift_result=False,
    drift_tracking_mode="full",
    sparse_drift_interval_sec=1.0,
):
    intensity_scales = intensity_scales or {}
    requested_compute_backend = compute_backend
    compute_backend = normalize_compute_backend(compute_backend)
    num_frames = multi_cache.metadata["num_frames"]
    num_mols = len(molecules)
    num_ch = len(channel_names)
    intensity_timings = {
        "drift_sec": 0.0,
        "frame_read_sec": 0.0,
        "patch_extract_sec": 0.0,
        "background_sec": 0.0,
        "diagnostics_sec": 0.0,
    }
    if num_mols == 0 or num_frames <= 0 or num_ch == 0:
        empty = {}
        empty_drift_result = {}
        if return_diagnostics:
            diagnostics = {
                "num_frames": int(num_frames),
                "num_molecules": int(num_mols),
                "num_channels": int(num_ch),
                "molecules": {},
                "timings_sec": {key: round(float(value), 6) for key, value in intensity_timings.items()},
                "gpu_or_cpu_backend": "none",
                "requested_compute_backend": str(requested_compute_backend),
            }
            if return_drift_result:
                return empty, diagnostics, empty_drift_result
            return empty, diagnostics
        if return_drift_result:
            return empty, empty_drift_result
        return empty

    reg_model = None
    if require_complete_registration:
        reg_model = validate_quantitative_registration(
            registration_params,
            channel_names,
            detect_channel=detect_channel,
        )
    elif registration_params is not None:
        reg_model = RegistrationConverter.from_legacy_or_new(registration_params)

    detect_channel_name = channel_names[int(detect_channel)]
    mask_inner, mask_outer = IntensityProcessor.precompute_masks(inner_radius=inner_radius, outer_radius=outer_radius)
    inner_idx, outer_idx = IntensityProcessor.mask_indices(mask_inner, mask_outer)
    n_signal = int(np.sum(mask_inner))
    patch_size = outer_radius * 2 + 1

    if progress_cb:
        progress_cb(1, "Computing drift...")

    DriftCalculator.set_last_drift_result(None)
    t0 = _time.perf_counter()
    drift_output = DriftCalculator.calculate_drift_parallel(
        multi_cache,
        detect_channel,
        molecules=molecules,
        inner_radius=inner_radius,
        progress_cb=progress_cb,
        stop_checker=stop_checker,
        return_result=True,
        drift_tracking_mode=drift_tracking_mode,
        sparse_drift_interval_sec=sparse_drift_interval_sec,
    )
    intensity_timings["drift_sec"] += max(0.0, _time.perf_counter() - t0)
    if isinstance(drift_output, tuple) and len(drift_output) == 2:
        drift_offsets, drift_result_for_diagnostics = drift_output
        drift_result_for_diagnostics = dict(drift_result_for_diagnostics or {})
    else:
        drift_offsets = drift_output
        drift_result_for_diagnostics = DriftCalculator.get_last_drift_result() or {}
    drift_stage_timings = dict(drift_result_for_diagnostics.get("timings_sec", {}) or {})

    _raise_if_cancelled(stop_checker)

    mol_ids = [m[0] for m in molecules]
    mol_x0 = np.array([m[1] for m in molecules], dtype=np.float64)
    mol_y0 = np.array([m[2] for m in molecules], dtype=np.float64)
    mol_x_input = mol_x0.copy()
    mol_y_input = mol_y0.copy()

    need_coord_transform = (
        physical_channel_name(detect_channel_name) != REFERENCE_CHANNEL
        and reg_model is not None
        and reg_model.has_channel(detect_channel_name)
    )

    if need_coord_transform:
        logger.info(f"[Registration] Converting detection channel coordinates {detect_channel_name} -> {REFERENCE_CHANNEL}")
        mol_x0_ref, mol_y0_ref = np.empty_like(mol_x0), np.empty_like(mol_y0)
        for mi in range(num_mols):
            rx, ry = reg_model.map_point_to_reference(
                detect_channel_name,
                float(mol_x0[mi]),
                float(mol_y0[mi]),
                strict=require_complete_registration,
            )
            mol_x0_ref[mi], mol_y0_ref[mi] = rx, ry
        mol_x0, mol_y0 = mol_x0_ref, mol_y0_ref

        M = reg_model.get_matrix(detect_channel_name)
        if M is not None:
            A = M[:2, :2].astype(np.float64)
            drift_ref = np.empty_like(drift_offsets, dtype=np.float64)
            for i in range(num_frames):
                drift_ref[i] = A @ drift_offsets[i].astype(np.float64)
            drift_offsets = drift_ref.astype(np.float32)

    mol_x0, mol_y0 = mol_x0.astype(np.float64), mol_y0.astype(np.float64)

    if progress_cb:
        progress_cb(10, "Extracting intensities...")

    signal_sum = np.full((num_frames, num_mols, num_ch), np.nan, dtype=np.float32)
    bg_est = np.full((num_frames, num_mols, num_ch), np.nan, dtype=np.float32)

    scale_vec = np.ones((num_ch,), dtype=np.float32)
    for j, ch_name in enumerate(channel_names):
        scale_vec[j] = float(intensity_scales.get(ch_name, 1.0))

    if max_workers is None:
        max_workers = min(8, mp.cpu_count())
    boundary_margin = outer_radius + 1
    collect_diagnostics = bool(return_diagnostics)

    diagnostics = None
    intensity_block_io = {}
    t0 = _time.perf_counter()
    if collect_diagnostics:
        drift_arr = np.asarray(drift_offsets, dtype=np.float64)
        drift_norm = (
            np.sqrt(np.sum(drift_arr * drift_arr, axis=1))
            if drift_arr.ndim == 2 and drift_arr.shape[1] >= 2
            else np.asarray([], dtype=np.float64)
        )
        nonfinite_drift = (
            np.where(~np.all(np.isfinite(drift_arr), axis=1))[0].astype(int).tolist()
            if drift_arr.ndim == 2
            else list(range(int(num_frames)))
        )
        raw_drift_image_shape = drift_result_for_diagnostics.get("image_shape", [])
        try:
            drift_image_shape = np.asarray(raw_drift_image_shape, dtype=np.int64).reshape(-1).tolist()
        except Exception:
            drift_image_shape = []
        if len(drift_image_shape) != 2:
            drift_image_shape = [
                int(multi_cache.metadata.get("height", 0) or 0),
                int(multi_cache.metadata.get("width", 0) or 0),
            ]
        diagnostics = {
            "num_frames": int(num_frames),
            "num_molecules": int(num_mols),
            "num_channels": int(num_ch),
            "channel_names": list(channel_names),
            "boundary_margin_px": int(boundary_margin),
            "patch_size_px": int(patch_size),
            "drift": {
                "max_abs_drift_px": float(np.nanmax(drift_norm)) if drift_norm.size else 0.0,
                "nonfinite_frames": nonfinite_drift[:200],
                "timings_sec": {
                    key: round(float(value), 6)
                    for key, value in drift_stage_timings.items()
                },
                "num_frames": int(drift_result_for_diagnostics.get("num_frames", num_frames) or num_frames),
                "image_shape": [int(drift_image_shape[0]), int(drift_image_shape[1])],
                "num_channels": int(
                    drift_result_for_diagnostics.get(
                        "num_channels",
                        multi_cache.metadata.get("num_channels", num_ch),
                    )
                    or 0
                ),
                "drift_track_frame_read_sec": round(
                    float(drift_result_for_diagnostics.get("drift_track_frame_read_sec", 0.0) or 0.0),
                    6,
                ),
                "block_read_used": bool(drift_result_for_diagnostics.get("block_read_used", False)),
                "block_size": int(drift_result_for_diagnostics.get("block_size", 0) or 0),
                "block_read_calls": int(drift_result_for_diagnostics.get("block_read_calls", 0) or 0),
                "drift_tracking_mode": str(drift_result_for_diagnostics.get("drift_tracking_mode", "full") or "full"),
                "drift_tracking_mode_requested": str(
                    drift_result_for_diagnostics.get("drift_tracking_mode_requested", "full") or "full"
                ),
                "track_interval_sec": round(
                    float(drift_result_for_diagnostics.get("track_interval_sec", 0.0) or 0.0),
                    6,
                ),
                "track_interval_frames": int(drift_result_for_diagnostics.get("track_interval_frames", 1) or 1),
                "track_interval_exposure_fallback": bool(
                    drift_result_for_diagnostics.get("track_interval_exposure_fallback", False)
                ),
                "sampled_frames": int(drift_result_for_diagnostics.get("sampled_frames", num_frames) or 0),
                "sparse_fallback_used": bool(drift_result_for_diagnostics.get("sparse_fallback_used", False)),
                "sparse_fallback_reason": str(drift_result_for_diagnostics.get("sparse_fallback_reason", "") or ""),
                "sparse_valid_sample_fraction": round(
                    float(drift_result_for_diagnostics.get("sparse_valid_sample_fraction", 1.0) or 0.0),
                    6,
                ),
                "sparse_max_invalid_gap_sec": round(
                    float(drift_result_for_diagnostics.get("sparse_max_invalid_gap_sec", 0.0) or 0.0),
                    6,
                ),
            },
            "molecules": {
                int(mol_id): {
                    "input_x": float(mol_x_input[mi]),
                    "input_y": float(mol_y_input[mi]),
                    "reference_x": float(mol_x0[mi]),
                    "reference_y": float(mol_y0[mi]),
                    "min_boundary_distance_px": None,
                    "invalid_by_channel": {},
                }
                for mi, mol_id in enumerate(mol_ids)
            },
        }
    intensity_timings["diagnostics_sec"] += max(0.0, _time.perf_counter() - t0)

    used_cupy = False
    if _should_try_cupy_backend(
        compute_backend,
        num_frames=num_frames,
        num_mols=num_mols,
        num_ch=num_ch,
        collect_diagnostics=collect_diagnostics,
    ):
        try:
            if progress_cb:
                progress_cb(10, "Extracting intensities on CuPy GPU...")
            t0 = _time.perf_counter()
            gpu_arrays = _extract_signal_bg_cupy(
                multi_cache,
                num_frames=int(num_frames),
                num_mols=int(num_mols),
                num_ch=int(num_ch),
                mol_x0=mol_x0,
                mol_y0=mol_y0,
                drift_offsets=drift_offsets,
                channel_names=channel_names,
                reg_model=reg_model,
                require_complete_registration=require_complete_registration,
                mask_inner=mask_inner,
                mask_outer=mask_outer,
                bg_stat=bg_stat,
                batch_size=batch_size,
                patch_size=patch_size,
                boundary_margin=boundary_margin,
                stop_checker=stop_checker,
            )
            gpu_elapsed = max(0.0, _time.perf_counter() - t0)
            if gpu_arrays is not None:
                signal_sum, bg_est, gpu_timings = gpu_arrays
                intensity_block_io = dict(dict(gpu_timings or {}).get("channel_block_io", {}) or {})
                accounted = 0.0
                for key in ("frame_read_sec", "patch_extract_sec", "background_sec"):
                    value = float(dict(gpu_timings or {}).get(key, 0.0) or 0.0)
                    intensity_timings[key] += value
                    accounted += value
                if accounted <= 0.0:
                    intensity_timings["patch_extract_sec"] += gpu_elapsed
                used_cupy = True
                logger.info("[GPU] CuPy intensity extraction completed")
            else:
                intensity_timings["patch_extract_sec"] += gpu_elapsed
                logger.info("[GPU] CuPy backend unavailable; using CPU intensity extraction")
        except Exception as exc:
            logger.warning("[GPU] CuPy intensity extraction failed; falling back to CPU: %s", exc)

    if collect_diagnostics and diagnostics is not None and intensity_block_io:
        diagnostics["channel_block_io"] = {
            "block_read_used": bool(intensity_block_io.get("block_read_used", False)),
            "block_size": int(intensity_block_io.get("block_size", 0) or 0),
            "block_read_calls": int(intensity_block_io.get("block_read_calls", 0) or 0),
            "block_read_failures": int(intensity_block_io.get("block_read_failures", 0) or 0),
            "raw_frame_read_calls": int(intensity_block_io.get("raw_frame_read_calls", 0) or 0),
        }

    def _merge_channel_detail(dest, src):
        dest["count"] = int(dest.get("count", 0) or 0) + int(src.get("count", 0) or 0)
        dest_reasons = dict(dest.get("reasons", {}) or {})
        for reason, count in dict(src.get("reasons", {}) or {}).items():
            dest_reasons[str(reason)] = int(dest_reasons.get(str(reason), 0) or 0) + int(count or 0)
        dest["reasons"] = dest_reasons
        dest_frames = list(dest.get("frames", []) or [])
        if len(dest_frames) < 20:
            dest_frames.extend(list(src.get("frames", []) or [])[: max(0, 20 - len(dest_frames))])
        dest["frames"] = dest_frames

    def _merge_diagnostics(local_diag):
        if not collect_diagnostics or not local_diag or diagnostics is None:
            return
        diag_t0 = _time.perf_counter()
        for mol_id, local_mol in dict(local_diag.get("molecules", {}) or {}).items():
            mol_id = int(mol_id)
            target = diagnostics["molecules"].setdefault(
                mol_id,
                {
                    "input_x": None,
                    "input_y": None,
                    "reference_x": None,
                    "reference_y": None,
                    "min_boundary_distance_px": None,
                    "invalid_by_channel": {},
                },
            )
            edge = local_mol.get("min_boundary_distance_px")
            if edge is not None:
                try:
                    edge = float(edge)
                    if np.isfinite(edge):
                        current = target.get("min_boundary_distance_px")
                        target["min_boundary_distance_px"] = edge if current is None else min(float(current), edge)
                except Exception:
                    pass
            target_channels = dict(target.get("invalid_by_channel", {}) or {})
            for ch_name, detail in dict(local_mol.get("invalid_by_channel", {}) or {}).items():
                ch_name = str(ch_name)
                merged = dict(target_channels.get(ch_name, {}) or {})
                _merge_channel_detail(merged, dict(detail or {}))
                target_channels[ch_name] = merged
            target["invalid_by_channel"] = target_channels
        intensity_timings["diagnostics_sec"] += max(0.0, _time.perf_counter() - diag_t0)

    def _ensure_global_mol_diag(mi):
        if not collect_diagnostics or diagnostics is None:
            return None
        return diagnostics["molecules"].setdefault(
            int(mol_ids[mi]),
            {
                "input_x": None,
                "input_y": None,
                "reference_x": None,
                "reference_y": None,
                "min_boundary_distance_px": None,
                "invalid_by_channel": {},
            },
        )

    def _update_global_edge(mi, edge):
        mol_diag = _ensure_global_mol_diag(mi)
        if mol_diag is None:
            return
        try:
            edge = float(edge)
        except Exception:
            return
        if not np.isfinite(edge):
            return
        current = mol_diag.get("min_boundary_distance_px")
        mol_diag["min_boundary_distance_px"] = edge if current is None else min(float(current), edge)

    def _record_global_invalid(mi, ch_name, frame_idx, reason, x_raw=None, y_raw=None, edge=None, drift=None, out_of_bounds=False):
        mol_diag = _ensure_global_mol_diag(mi)
        if mol_diag is None:
            return
        ch_detail = mol_diag.setdefault("invalid_by_channel", {}).setdefault(
            str(ch_name),
            {"count": 0, "reasons": {}, "frames": []},
        )
        ch_detail["count"] = int(ch_detail.get("count", 0) or 0) + 1
        reasons = ch_detail.setdefault("reasons", {})
        reasons[str(reason)] = int(reasons.get(str(reason), 0) or 0) + 1
        if edge is not None:
            _update_global_edge(mi, edge)
        if len(ch_detail.setdefault("frames", [])) < 20:
            ch_detail["frames"].append(
                {
                    "frame": int(frame_idx),
                    "reason": str(reason),
                    "x_raw": None if x_raw is None else float(x_raw),
                    "y_raw": None if y_raw is None else float(y_raw),
                    "edge_distance_px": None if edge is None else float(edge),
                    "drift": [None, None] if drift is None else [float(drift[0]), float(drift[1])],
                    "transformed_out_of_bounds": bool(out_of_bounds),
                }
            )

    def _collect_cupy_coordinate_diagnostics():
        if not collect_diagnostics or diagnostics is None:
            return
        diag_t0 = _time.perf_counter()
        height = int(multi_cache.metadata.get("height", 0) or 0)
        width = int(multi_cache.metadata.get("width", 0) or 0)
        if height <= 0 or width <= 0:
            intensity_timings["diagnostics_sec"] += max(0.0, _time.perf_counter() - diag_t0)
            return

        for frame_idx in range(int(num_frames)):
            dx_f, dy_f = drift_offsets[frame_idx]
            if not (np.isfinite(dx_f) and np.isfinite(dy_f)):
                for mi in range(num_mols):
                    for ch_name in channel_names:
                        _record_global_invalid(mi, ch_name, frame_idx, "nonfinite_drift", drift=(dx_f, dy_f))
                continue

            x_ref = mol_x0 + float(dx_f)
            y_ref = mol_y0 + float(dy_f)
            for ch_name in channel_names:
                try:
                    xs_f, ys_f = _map_reference_arrays_to_channel(
                        x_ref,
                        y_ref,
                        ch_name,
                        reg_model,
                        strict=require_complete_registration,
                    )
                    xs_f = np.asarray(xs_f, dtype=np.float64)
                    ys_f = np.asarray(ys_f, dtype=np.float64)
                except Exception:
                    for mi in range(num_mols):
                        _record_global_invalid(mi, ch_name, frame_idx, "coordinate_transform_failed", drift=(dx_f, dy_f))
                    continue

                edge_dist = np.minimum.reduce([
                    xs_f - float(boundary_margin),
                    float(width - boundary_margin) - xs_f,
                    ys_f - float(boundary_margin),
                    float(height - boundary_margin) - ys_f,
                ])
                for mi, edge in enumerate(edge_dist):
                    _update_global_edge(mi, edge)
                invalid_coord = ~np.isfinite(xs_f) | ~np.isfinite(ys_f)
                valid_mol = (
                    (xs_f >= boundary_margin)
                    & (xs_f < width - boundary_margin)
                    & (ys_f >= boundary_margin)
                    & (ys_f < height - boundary_margin)
                )
                for mi in np.nonzero(invalid_coord)[0]:
                    _record_global_invalid(
                        mi,
                        ch_name,
                        frame_idx,
                        "nonfinite_coordinate",
                        x_raw=xs_f[mi],
                        y_raw=ys_f[mi],
                        edge=edge_dist[mi],
                        drift=(dx_f, dy_f),
                    )
                for mi in np.nonzero(~valid_mol & ~invalid_coord)[0]:
                    _record_global_invalid(
                        mi,
                        ch_name,
                        frame_idx,
                        "transformed_out_of_bounds",
                        x_raw=xs_f[mi],
                        y_raw=ys_f[mi],
                        edge=edge_dist[mi],
                        drift=(dx_f, dy_f),
                        out_of_bounds=True,
                    )
        intensity_timings["diagnostics_sec"] += max(0.0, _time.perf_counter() - diag_t0)

    if used_cupy:
        _collect_cupy_coordinate_diagnostics()

    def process_batch(start, end):
        local_timings = {
            "frame_read_sec": 0.0,
            "patch_extract_sec": 0.0,
            "background_sec": 0.0,
            "diagnostics_sec": 0.0,
        }

        def add_local_timing(key, started):
            local_timings[key] += max(0.0, _time.perf_counter() - started)

        _raise_if_cancelled(stop_checker)
        batch_len = end - start
        sig_b = np.full((batch_len, num_mols, num_ch), np.nan, dtype=np.float32)
        bg_b = np.full((batch_len, num_mols, num_ch), np.nan, dtype=np.float32)
        diag_b = {"molecules": {}} if collect_diagnostics else None

        def ensure_mol_diag(mi):
            if not collect_diagnostics or diag_b is None:
                return None
            mol_id = int(mol_ids[mi])
            return diag_b["molecules"].setdefault(
                mol_id,
                {
                    "min_boundary_distance_px": None,
                    "invalid_by_channel": {},
                },
            )

        def update_edge(mi, edge):
            if not collect_diagnostics:
                return
            diag_t0 = _time.perf_counter()
            try:
                edge = float(edge)
            except Exception:
                add_local_timing("diagnostics_sec", diag_t0)
                return
            if not np.isfinite(edge):
                add_local_timing("diagnostics_sec", diag_t0)
                return
            mol_diag = ensure_mol_diag(mi)
            if mol_diag is None:
                add_local_timing("diagnostics_sec", diag_t0)
                return
            current = mol_diag.get("min_boundary_distance_px")
            mol_diag["min_boundary_distance_px"] = edge if current is None else min(float(current), edge)
            add_local_timing("diagnostics_sec", diag_t0)

        def record_invalid(mi, ch_name, frame_idx, reason, x_raw=None, y_raw=None, edge=None, drift=None, out_of_bounds=False):
            if not collect_diagnostics:
                return
            diag_t0 = _time.perf_counter()
            mol_diag = ensure_mol_diag(mi)
            if mol_diag is None:
                add_local_timing("diagnostics_sec", diag_t0)
                return
            ch_detail = mol_diag["invalid_by_channel"].setdefault(
                str(ch_name),
                {"count": 0, "reasons": {}, "frames": []},
            )
            ch_detail["count"] = int(ch_detail.get("count", 0) or 0) + 1
            reasons = ch_detail.setdefault("reasons", {})
            reasons[str(reason)] = int(reasons.get(str(reason), 0) or 0) + 1
            if edge is not None:
                update_edge(mi, edge)
            if len(ch_detail.setdefault("frames", [])) < 20:
                ch_detail["frames"].append(
                    {
                        "frame": int(frame_idx),
                        "reason": str(reason),
                        "x_raw": None if x_raw is None else float(x_raw),
                        "y_raw": None if y_raw is None else float(y_raw),
                        "edge_distance_px": None if edge is None else float(edge),
                        "drift": [None, None] if drift is None else [float(drift[0]), float(drift[1])],
                        "transformed_out_of_bounds": bool(out_of_bounds),
                    }
                )
            add_local_timing("diagnostics_sec", diag_t0)

        for k, frame_idx in enumerate(range(start, end)):
            _raise_if_cancelled(stop_checker)
            t_frame = _time.perf_counter()
            frame_data_list = multi_cache.get_raw_frame(frame_idx)
            add_local_timing("frame_read_sec", t_frame)
            dx_f, dy_f = drift_offsets[frame_idx]
            if frame_data_list is None:
                for mi in range(num_mols):
                    for ch_name in channel_names:
                        record_invalid(mi, ch_name, frame_idx, "frame_missing", drift=(dx_f, dy_f))
                continue
            if not (np.isfinite(dx_f) and np.isfinite(dy_f)):
                for mi in range(num_mols):
                    for ch_name in channel_names:
                        record_invalid(mi, ch_name, frame_idx, "nonfinite_drift", drift=(dx_f, dy_f))
                continue
            x_ref = mol_x0 + float(dx_f)
            y_ref = mol_y0 + float(dy_f)

            for ch_idx, ch_name in enumerate(channel_names):
                if ch_idx >= len(frame_data_list):
                    for mi in range(num_mols):
                        record_invalid(mi, ch_name, frame_idx, "missing_channel", drift=(dx_f, dy_f))
                    continue
                channel_frame = frame_data_list[ch_idx]
                if channel_frame.dtype != np.float32:
                    channel_frame = channel_frame.astype(np.float32)
                if channel_frame.ndim != 2:
                    for mi in range(num_mols):
                        record_invalid(mi, ch_name, frame_idx, "invalid_channel_frame", drift=(dx_f, dy_f))
                    continue
                hc, wc = channel_frame.shape
                try:
                    xs_f, ys_f = _map_reference_arrays_to_channel(
                        x_ref,
                        y_ref,
                        ch_name,
                        reg_model,
                        strict=require_complete_registration,
                    )
                    xs_f = np.asarray(xs_f, dtype=np.float64)
                    ys_f = np.asarray(ys_f, dtype=np.float64)
                except Exception:
                    xs_f = np.full(num_mols, np.nan, dtype=np.float64)
                    ys_f = np.full(num_mols, np.nan, dtype=np.float64)
                    for mi in range(num_mols):
                        record_invalid(mi, ch_name, frame_idx, "coordinate_transform_failed", drift=(dx_f, dy_f))
                edge_dist = np.minimum.reduce([
                    xs_f - float(boundary_margin),
                    float(wc - boundary_margin) - xs_f,
                    ys_f - float(boundary_margin),
                    float(hc - boundary_margin) - ys_f,
                ])
                for mi, edge in enumerate(edge_dist):
                    update_edge(mi, edge)
                valid_mol = (xs_f >= boundary_margin) & (xs_f < wc - boundary_margin) & (ys_f >= boundary_margin) & (ys_f < hc - boundary_margin)
                invalid_coord = ~np.isfinite(xs_f) | ~np.isfinite(ys_f)
                for mi in np.nonzero(invalid_coord)[0]:
                    record_invalid(mi, ch_name, frame_idx, "nonfinite_coordinate", x_raw=xs_f[mi], y_raw=ys_f[mi], edge=edge_dist[mi], drift=(dx_f, dy_f))
                for mi in np.nonzero(~valid_mol & ~invalid_coord)[0]:
                    record_invalid(
                        mi,
                        ch_name,
                        frame_idx,
                        "transformed_out_of_bounds",
                        x_raw=xs_f[mi],
                        y_raw=ys_f[mi],
                        edge=edge_dist[mi],
                        drift=(dx_f, dy_f),
                        out_of_bounds=True,
                    )
                for mi in np.nonzero(valid_mol)[0]:
                    t_patch = _time.perf_counter()
                    patch = cv2.getRectSubPix(channel_frame, (patch_size, patch_size), (float(xs_f[mi]), float(ys_f[mi])))
                    add_local_timing("patch_extract_sec", t_patch)
                    if patch.shape != (patch_size, patch_size):
                        record_invalid(mi, ch_name, frame_idx, "invalid_patch_shape", x_raw=xs_f[mi], y_raw=ys_f[mi], edge=edge_dist[mi], drift=(dx_f, dy_f))
                        continue
                    t_bg = _time.perf_counter()
                    s, b, ok = IntensityProcessor.extract_signal_bg_from_flat_patch(patch, inner_idx, outer_idx, bg_stat=bg_stat)
                    add_local_timing("background_sec", t_bg)
                    if ok:
                        sig_b[k, mi, ch_idx], bg_b[k, mi, ch_idx] = s, b
                    else:
                        record_invalid(mi, ch_name, frame_idx, "empty_signal_or_background", x_raw=xs_f[mi], y_raw=ys_f[mi], edge=edge_dist[mi], drift=(dx_f, dy_f))
        return start, end, sig_b, bg_b, diag_b, local_timings

    if not used_cupy:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for start in range(0, num_frames, batch_size):
                _raise_if_cancelled(stop_checker)
                futures.append(executor.submit(process_batch, start, min(start + batch_size, num_frames)))
            for fut in futures:
                _raise_if_cancelled(stop_checker)
                start, end, sig_b, bg_b, diag_b, local_timings = fut.result()
                signal_sum[start:end] = sig_b
                bg_est[start:end] = bg_b
                for key, value in dict(local_timings or {}).items():
                    if key in intensity_timings:
                        intensity_timings[key] += float(value or 0.0)
                _merge_diagnostics(diag_b)
                if progress_cb:
                    progress_cb(10 + int((end / num_frames) * 85), f"Processing frame {end}/{num_frames}...")
    elif progress_cb:
        progress_cb(95, "CuPy intensity extraction complete")

    _raise_if_cancelled(stop_checker)

    if temporal_bg_enabled and temporal_bg_window and int(temporal_bg_window) >= 3:
        win = int(temporal_bg_window)
        if win % 2 == 0:
            win += 1
        t0 = _time.perf_counter()
        bg_smooth = _nan_aware_temporal_median(bg_est, win)
        intensity_timings["background_sec"] += max(0.0, _time.perf_counter() - t0)
    else:
        t0 = _time.perf_counter()
        bg_smooth = bg_est.copy()
        intensity_timings["background_sec"] += max(0.0, _time.perf_counter() - t0)

    _raise_if_cancelled(stop_checker)

    t0 = _time.perf_counter()
    net = signal_sum - bg_smooth * float(n_signal)
    net *= scale_vec.reshape((1, 1, -1))
    intensity_timings["background_sec"] += max(0.0, _time.perf_counter() - t0)

    if collect_diagnostics and diagnostics is not None:
        t0 = _time.perf_counter()
        for mi, mol_id in enumerate(mol_ids):
            mol_diag = diagnostics["molecules"].setdefault(int(mol_id), {"invalid_by_channel": {}})
            for ch_idx, ch_name in enumerate(channel_names):
                arr = net[:, mi, ch_idx]
                invalid_frames = np.where(~np.isfinite(arr))[0]
                if invalid_frames.size == 0:
                    continue
                ch_detail = mol_diag.setdefault("invalid_by_channel", {}).setdefault(
                    str(ch_name),
                    {"count": 0, "reasons": {}, "frames": []},
                )
                existing = int(ch_detail.get("count", 0) or 0)
                if existing < int(invalid_frames.size):
                    extra = int(invalid_frames.size) - existing
                    ch_detail["count"] = int(invalid_frames.size)
                    reasons = ch_detail.setdefault("reasons", {})
                    reasons["net_nonfinite_after_background"] = int(reasons.get("net_nonfinite_after_background", 0) or 0) + extra
                    for frame_idx in invalid_frames[: max(0, 20 - len(ch_detail.setdefault("frames", [])))]:
                        ch_detail["frames"].append(
                            {
                                "frame": int(frame_idx),
                                "reason": "net_nonfinite_after_background",
                                "x_raw": None,
                                "y_raw": None,
                                "edge_distance_px": None,
                                "drift": [
                                    float(drift_offsets[int(frame_idx), 0]),
                                    float(drift_offsets[int(frame_idx), 1]),
                                ],
                                "transformed_out_of_bounds": False,
                            }
                        )
        intensity_timings["diagnostics_sec"] += max(0.0, _time.perf_counter() - t0)

    molecule_intensities = {}
    for mi, mol_id in enumerate(mol_ids):
        molecule_intensities[mol_id] = {}
        for ch_idx, ch_name in enumerate(channel_names):
            molecule_intensities[mol_id][ch_name] = net[:, mi, ch_idx].astype(np.float32)

    if progress_cb:
        progress_cb(100, "Processing complete")
    if return_diagnostics:
        if diagnostics is not None:
            diagnostics["timings_sec"] = {
                key: round(float(value), 6)
                for key, value in intensity_timings.items()
            }
            diagnostics["gpu_or_cpu_backend"] = "cupy" if used_cupy else "cpu"
            diagnostics["requested_compute_backend"] = str(requested_compute_backend)
        if return_drift_result:
            return molecule_intensities, diagnostics, drift_result_for_diagnostics
        return molecule_intensities, diagnostics
    if return_drift_result:
        return molecule_intensities, drift_result_for_diagnostics
    return molecule_intensities
