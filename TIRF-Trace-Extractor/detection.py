from __future__ import annotations

import math
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit, minimize


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check and cancel_check():
        raise RuntimeError("Task cancelled")


def _disk_footprint(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x_grid * x_grid + y_grid * y_grid) <= radius * radius


def _peak_local_max(
    image: np.ndarray,
    *,
    min_distance: int,
    threshold_abs: float,
    exclude_border: int,
) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 2), dtype=int)

    min_distance = max(1, int(min_distance))
    exclude_border = max(0, int(exclude_border))
    size = 2 * min_distance + 1
    local_max = arr == ndimage.maximum_filter(arr, size=size, mode="nearest")
    local_max &= arr >= float(threshold_abs)

    if exclude_border:
        local_max[:exclude_border, :] = False
        local_max[-exclude_border:, :] = False
        local_max[:, :exclude_border] = False
        local_max[:, -exclude_border:] = False

    coords = np.argwhere(local_max)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=int)

    values = arr[coords[:, 0], coords[:, 1]]
    order = np.argsort(-values, kind="mergesort")
    return coords[order].astype(int, copy=False)


class RollingBallBackground:
    @staticmethod
    def rolling_ball_background(image: np.ndarray, radius: int) -> np.ndarray:
        radius = int(radius)
        struct_elem = _disk_footprint(radius)
        return ndimage.grey_opening(image, footprint=struct_elem)

    @staticmethod
    def subtract_background(image: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
        background = RollingBallBackground.rolling_ball_background(image, radius)
        corrected = image.astype(np.float32) - background.astype(np.float32)
        return corrected, background.astype(np.float32)

    @staticmethod
    def estimate_adaptive_radius(
        image: np.ndarray,
        sigma: float,
        quick_detect_func: Callable[[np.ndarray], Any] | None = None,
    ) -> int:
        base_radius = int(10 * sigma)
        if quick_detect_func is not None:
            try:
                positions = quick_detect_func(image)
                if len(positions) > 50:
                    from scipy.spatial.distance import pdist

                    distances = pdist(np.asarray(positions[:200]))
                    median_distance = np.median(distances)
                    radius = int((base_radius + median_distance * 2.5) / 2)
                else:
                    radius = base_radius
            except Exception:
                radius = base_radius
        else:
            radius = base_radius
        radius = int(np.clip(radius, 30, 120))
        return radius if radius % 2 else radius + 1


def mad_std(data: np.ndarray) -> float:
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return float(1.4826 * mad)


def calculate_r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = np.sum((observed - fitted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


def calculate_snr_improved(image: np.ndarray, x: float, y: float, sigma: float) -> float:
    h, w = image.shape
    x_int, y_int = int(round(x)), int(round(y))
    signal_radius = int(1.5 * sigma)
    bg_outer_radius = signal_radius + 6

    if (
        x_int < bg_outer_radius
        or x_int >= w - bg_outer_radius
        or y_int < bg_outer_radius
        or y_int >= h - bg_outer_radius
    ):
        return 0.0

    y_grid, x_grid = np.ogrid[-bg_outer_radius : bg_outer_radius + 1, -bg_outer_radius : bg_outer_radius + 1]
    dist = np.sqrt(x_grid**2 + y_grid**2)
    signal_mask = dist <= signal_radius
    bg_mask = (dist > (signal_radius + 2)) & (dist <= bg_outer_radius)

    roi = image[
        y_int - bg_outer_radius : y_int + bg_outer_radius + 1,
        x_int - bg_outer_radius : x_int + bg_outer_radius + 1,
    ]
    sig_pixels = roi[signal_mask]
    bg_pixels = roi[bg_mask]
    if sig_pixels.size == 0 or bg_pixels.size == 0:
        return 0.0

    bg_std = float(np.std(bg_pixels))
    if bg_std <= 0:
        return 0.0

    net_signal = float(np.sum(sig_pixels) - np.mean(bg_pixels) * sig_pixels.size)
    return float(net_signal / bg_std)


def _gaussian_kernel(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.6)
    return np.exp(-(((xx - x0) ** 2) + ((yy - y0) ** 2)) / (2.0 * sigma * sigma))


def _normalize_gaussian(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(xx, yy, x0, y0, sigma)
    total = float(np.sum(kernel))
    if total <= 0:
        return np.full_like(kernel, 1.0 / kernel.size)
    return kernel / total


def _poisson_nll(observed: np.ndarray, expected: np.ndarray) -> float:
    mu = np.clip(np.asarray(expected, dtype=np.float64), 1e-6, None)
    roi = np.asarray(observed, dtype=np.float64)
    return float(np.sum(mu - roi * np.log(mu)))


def _tile_coordinate_centers(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [length // 2]
    centers = []
    pos = tile // 2
    while pos < length - tile // 2:
        centers.append(pos)
        pos += step
    if centers[-1] != length - tile // 2:
        centers.append(length - tile // 2)
    return centers


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out):
        return None
    return out


class MoleculeDetector:
    def __init__(self):
        self.detection_stats: dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._preview_prepare_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._preview_candidate_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._preview_rank_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._preview_record_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._interactive_prepare_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._interactive_candidate_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._interactive_fit_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}

    @staticmethod
    def _copy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(item or {}) for item in list(records or [])]

    def _remember_preview_cache(
        self,
        cache: dict[tuple[Any, ...], Any],
        key: tuple[Any, ...],
        value: Any,
        *,
        limit: int = 6,
    ) -> Any:
        with self._cache_lock:
            cache[key] = value
            while len(cache) > limit:
                first_key = next(iter(cache))
                cache.pop(first_key, None)
        return value

    @staticmethod
    def _preview_frame_id(image: np.ndarray) -> tuple[Any, ...]:
        arr = np.asarray(image)
        data_ptr = int(arr.__array_interface__["data"][0]) if arr.size else 0
        return (data_ptr, tuple(int(v) for v in arr.shape), str(arr.dtype))

    def _preview_prepare_cache_key(
        self,
        image: np.ndarray,
        sigma: float,
        auto_background: bool,
        bg_radius: int | None,
    ) -> tuple[Any, ...]:
        return (
            self._preview_frame_id(image),
            round(float(sigma), 4),
            bool(auto_background),
            int(bg_radius) if bg_radius is not None else None,
        )

    def _interactive_prepare_cache_key(
        self,
        image: np.ndarray,
        sigma: float,
        auto_background: bool,
        bg_radius: int | None,
    ) -> tuple[Any, ...]:
        return (
            self._preview_frame_id(image),
            round(float(sigma), 4),
            bool(auto_background),
            int(bg_radius) if bg_radius is not None else None,
        )

    @staticmethod
    def _preview_fast_cache_key(
        prepare_key: tuple[Any, ...],
        min_distance: int | None,
    ) -> tuple[Any, ...]:
        return (
            prepare_key,
            int(min_distance) if min_distance is not None else None,
        )

    @staticmethod
    def _interactive_candidate_cache_key(
        prepare_key: tuple[Any, ...],
        min_distance: int | None,
    ) -> tuple[Any, ...]:
        return (
            prepare_key,
            int(min_distance) if min_distance is not None else None,
        )

    @staticmethod
    def _interactive_fit_cache_key(
        candidate_key: tuple[Any, ...],
        x: int,
        y: int,
    ) -> tuple[Any, ...]:
        return candidate_key + (int(x), int(y))

    @staticmethod
    def _preview_rank_cache_key(
        fast_key: tuple[Any, ...],
        preview_limit: int,
    ) -> tuple[Any, ...]:
        return fast_key + (int(preview_limit),)

    def _subtract_background_preview(
        self,
        image: np.ndarray,
        radius: int,
        *,
        cancel_check: Callable[[], bool] | None = None,
        downsample_factor: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(image, dtype=np.float32)
        factor = max(1, int(downsample_factor))
        if factor <= 1 or min(image.shape) < 32:
            return RollingBallBackground.subtract_background(image, radius)

        zoom = 1.0 / float(factor)
        small = ndimage.zoom(image, (zoom, zoom), order=1)
        _raise_if_cancelled(cancel_check)
        scaled_radius = max(1, int(round(float(radius) / float(factor))))
        background_small = RollingBallBackground.rolling_ball_background(small, scaled_radius).astype(np.float32, copy=False)
        _raise_if_cancelled(cancel_check)
        background = ndimage.zoom(
            background_small,
            (
                image.shape[0] / max(1, background_small.shape[0]),
                image.shape[1] / max(1, background_small.shape[1]),
            ),
            order=1,
        )
        background = background[: image.shape[0], : image.shape[1]].astype(np.float32, copy=False)
        corrected = image.astype(np.float32, copy=False) - background
        return corrected, background

    def _prepare_detection_image_preview_cached(
        self,
        image: np.ndarray,
        sigma: float,
        auto_background: bool,
        bg_radius: int | None,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        prepare_key = self._preview_prepare_cache_key(image, sigma, auto_background, bg_radius)
        cached = self._preview_prepare_cache.get(prepare_key)
        if cached is not None:
            return (
                prepare_key,
                cached["corrected_image"],
                cached["rolling_background"],
                cached["background_map"],
                cached["noise_map"],
                dict(cached["stats"]),
            )

        corrected_image, rolling_background, background_map, noise_map, stats = self._prepare_detection_image(
            image,
            sigma,
            auto_background,
            bg_radius,
            cancel_check=cancel_check,
            preview_mode=True,
        )
        self._remember_preview_cache(
            self._preview_prepare_cache,
            prepare_key,
            {
                "corrected_image": corrected_image,
                "rolling_background": rolling_background,
                "background_map": background_map,
                "noise_map": noise_map,
                "stats": dict(stats),
            },
        )
        return prepare_key, corrected_image, rolling_background, background_map, noise_map, dict(stats)

    def _estimate_local_background_and_noise(
        self,
        image: np.ndarray,
        sigma: float,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        _raise_if_cancelled(cancel_check)
        image = np.asarray(image, dtype=np.float32)
        h, w = image.shape
        tile_size = max(24, int(round(max(6.0 * sigma, min(h, w) / 12.0))))
        tile_size = min(tile_size, min(h, w))
        overlap = max(4, tile_size // 3)
        step = max(4, tile_size - overlap)
        y_centers = _tile_coordinate_centers(h, tile_size, step)
        x_centers = _tile_coordinate_centers(w, tile_size, step)

        bg_grid = np.zeros((len(y_centers), len(x_centers)), dtype=np.float32)
        noise_grid = np.zeros_like(bg_grid)

        half = tile_size // 2
        for yi, cy in enumerate(y_centers):
            _raise_if_cancelled(cancel_check)
            y0 = max(0, cy - half)
            y1 = min(h, y0 + tile_size)
            y0 = max(0, y1 - tile_size)
            for xi, cx in enumerate(x_centers):
                x0 = max(0, cx - half)
                x1 = min(w, x0 + tile_size)
                x0 = max(0, x1 - tile_size)
                tile = image[y0:y1, x0:x1]
                bg = float(np.percentile(tile, 30))
                noise = max(mad_std(tile - bg), 1.0)
                bg_grid[yi, xi] = bg
                noise_grid[yi, xi] = noise

        zoom_y = h / max(1, bg_grid.shape[0])
        zoom_x = w / max(1, bg_grid.shape[1])
        background_map = ndimage.zoom(bg_grid, (zoom_y, zoom_x), order=1)
        noise_map = ndimage.zoom(noise_grid, (zoom_y, zoom_x), order=1)
        background_map = background_map[:h, :w].astype(np.float32, copy=False)
        noise_map = np.clip(noise_map[:h, :w], 1.0, None).astype(np.float32, copy=False)

        return background_map, noise_map, {
            "tile_size": int(tile_size),
            "tile_overlap": int(overlap),
        }

    def _prepare_detection_image(
        self,
        image: np.ndarray,
        sigma: float,
        auto_background: bool,
        bg_radius: int | None,
        *,
        cancel_check: Callable[[], bool] | None = None,
        preview_mode: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        image = np.asarray(image, dtype=np.float32)
        stats = {
            "sigma": float(sigma),
            "background_radius": None,
            "background_mode": "adaptive_tile_only",
        }
        corrected = image.copy()
        rolling_background = np.zeros_like(image, dtype=np.float32)

        if auto_background:
            if bg_radius is None:
                bg_radius = RollingBallBackground.estimate_adaptive_radius(image, sigma)
            if preview_mode:
                corrected, rolling_background = self._subtract_background_preview(
                    image,
                    bg_radius,
                    cancel_check=cancel_check,
                )
            else:
                corrected, rolling_background = RollingBallBackground.subtract_background(image, bg_radius)
            stats["background_radius"] = int(bg_radius)
            stats["background_mode"] = (
                "preview_approx_rolling_ball_plus_adaptive_tile"
                if preview_mode
                else "rolling_ball_plus_adaptive_tile"
            )
            if preview_mode:
                stats["background_downsample"] = 2

        background_map, noise_map, tile_stats = self._estimate_local_background_and_noise(
            corrected,
            sigma,
            cancel_check=cancel_check,
        )
        whitened = (corrected - background_map) / np.clip(noise_map, 1.0, None)

        return corrected, rolling_background, background_map, noise_map, {
            **stats,
            **tile_stats,
            "whitened_mean": float(np.mean(whitened)),
            "whitened_std": float(np.std(whitened)),
        }

    def _adaptive_detection(
        self,
        corrected_image: np.ndarray,
        background_map: np.ndarray,
        noise_map: np.ndarray,
        sigma: float,
        min_distance: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        whitened = (corrected_image - background_map) / np.clip(noise_map, 1.0, None)
        dog = ndimage.gaussian_filter(whitened, sigma=sigma) - ndimage.gaussian_filter(
            whitened,
            sigma=sigma * 2.3,
        )
        local_noise = ndimage.gaussian_filter(np.abs(whitened), sigma=max(1.0, sigma))
        response = dog / np.clip(local_noise, 0.75, None)
        response_peak = float(np.max(response)) if response.size else 0.0
        threshold = max(1.4, min(response_peak * 0.55, float(np.percentile(response, 99.0) * 0.35 + 0.8)))
        dist = max(3, int(round(1.5 * sigma))) if min_distance is None else int(min_distance)
        local_max = _peak_local_max(
            response,
            min_distance=dist,
            threshold_abs=threshold,
            exclude_border=max(2, int(math.ceil(3.0 * sigma))),
        )
        if len(local_max) == 0 and response_peak > 0:
            relaxed_threshold = max(1.1, response_peak * 0.35)
            local_max = _peak_local_max(
                response,
                min_distance=dist,
                threshold_abs=relaxed_threshold,
                exclude_border=max(2, int(math.ceil(3.0 * sigma))),
            )
            threshold = float(relaxed_threshold)
        if len(local_max) == 0:
            whitened_threshold = max(1.0, float(np.max(whitened)) * 0.25) if whitened.size else 1.0
            local_max = _peak_local_max(
                whitened,
                min_distance=dist,
                threshold_abs=whitened_threshold,
                exclude_border=max(2, int(math.ceil(3.0 * sigma))),
            )
            threshold = float(whitened_threshold)
        return whitened, dog, local_max, float(threshold)

    def _compute_fast_quality(
        self,
        corrected_image: np.ndarray,
        background_map: np.ndarray,
        noise_map: np.ndarray,
        whitened: np.ndarray,
        x: float,
        y: float,
        sigma: float,
    ) -> dict[str, float] | None:
        h, w = corrected_image.shape
        x_int = int(round(x))
        y_int = int(round(y))
        inner_radius = max(1, int(round(1.5 * sigma)))
        outer_radius = inner_radius + 5
        if (
            x_int < outer_radius
            or x_int >= w - outer_radius
            or y_int < outer_radius
            or y_int >= h - outer_radius
        ):
            return None

        roi = corrected_image[
            y_int - outer_radius : y_int + outer_radius + 1,
            x_int - outer_radius : x_int + outer_radius + 1,
        ]
        w_roi = whitened[
            y_int - outer_radius : y_int + outer_radius + 1,
            x_int - outer_radius : x_int + outer_radius + 1,
        ]
        y_grid, x_grid = np.ogrid[
            -outer_radius : outer_radius + 1,
            -outer_radius : outer_radius + 1,
        ]
        dist = np.sqrt(x_grid**2 + y_grid**2)
        signal_mask = dist <= inner_radius
        bg_mask = (dist > inner_radius + 1) & (dist <= outer_radius)
        isolation_mask = (dist > inner_radius + 1) & (dist <= inner_radius + 3)

        signal_pixels = roi[signal_mask]
        bg_pixels = roi[bg_mask]
        isolation_pixels = roi[isolation_mask]
        white_signal = w_roi[signal_mask]
        if signal_pixels.size == 0 or bg_pixels.size == 0:
            return None

        local_peak = float(np.max(signal_pixels))
        local_bg = float(np.median(bg_pixels))
        bg_std = float(np.std(bg_pixels))
        rough_snr = (local_peak - local_bg) / max(
            float(np.median(noise_map[max(0, y_int - 1): y_int + 2, max(0, x_int - 1): x_int + 2])),
            1e-6,
        )
        neighborhood_peak = (
            float(np.max(isolation_pixels))
            if isolation_pixels.size
            else max(local_bg, 1e-6)
        )
        isolation = (local_peak - local_bg) / max(neighborhood_peak - local_bg, 1.0)
        integrated_signal = float(np.sum(signal_pixels - local_bg))
        white_peak = float(np.max(white_signal)) if white_signal.size else 0.0
        quality = (
            max(white_peak, 0.0)
            * max(rough_snr, 0.0)
            * max(integrated_signal, 0.0)
            * max(isolation, 0.1)
        )

        return {
            "peak": local_peak,
            "background": local_bg,
            "noise": max(bg_std, 1e-6),
            "rough_snr": float(rough_snr),
            "isolation": float(isolation),
            "integrated_signal": integrated_signal,
            "white_peak": white_peak,
            "quality": float(quality),
        }

    @staticmethod
    def _annotate_crowdedness(
        candidates: list[dict[str, Any]],
        sigma: float,
    ) -> tuple[list[dict[str, Any]], int]:
        if not candidates:
            return [], 0

        crowd_radius = max(3.0, float(2.5 * sigma))
        crowded_regions = 0
        for index, candidate in enumerate(candidates):
            nearest = None
            for other_index, other in enumerate(candidates):
                if index == other_index:
                    continue
                dx = float(candidate["x"]) - float(other["x"])
                dy = float(candidate["y"]) - float(other["y"])
                dist = math.hypot(dx, dy)
                if nearest is None or dist < nearest:
                    nearest = dist
            nearest = float(nearest) if nearest is not None else float("inf")
            candidate["nearest_neighbor_dist"] = nearest
            candidate["crowded_flag"] = bool(nearest < crowd_radius)
            candidate["crowded_score"] = max(0.0, 1.0 - nearest / max(crowd_radius, 1e-6)) if np.isfinite(nearest) else 0.0
            if candidate["crowded_flag"]:
                crowded_regions += 1
        return candidates, crowded_regions

    def detect_candidates_fast(
        self,
        image: np.ndarray,
        sigma: float = 3.5,
        min_distance: int | None = None,
        auto_background: bool = True,
        bg_radius: int | None = None,
        auto_bg: bool | None = None,
        *,
        preview_mode: bool = False,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        _raise_if_cancelled(cancel_check)
        if auto_bg is not None:
            auto_background = bool(auto_bg)
        if preview_mode:
            prepare_key, corrected_image, _rolling_background, background_map, noise_map, prep_stats = (
                self._prepare_detection_image_preview_cached(
                    image,
                    sigma,
                    auto_background,
                    bg_radius,
                    cancel_check=cancel_check,
                )
            )
            fast_key = self._preview_fast_cache_key(prepare_key, min_distance)
            cached_fast = self._preview_candidate_cache.get(fast_key)
            if cached_fast is not None:
                return self._copy_records(cached_fast["candidates"]), dict(cached_fast["stats"])
        else:
            corrected_image, _rolling_background, background_map, noise_map, prep_stats = self._prepare_detection_image(
                image,
                sigma,
                auto_background,
                bg_radius,
                cancel_check=cancel_check,
            )
            fast_key = None
        whitened, _dog_image, raw_candidates, threshold = self._adaptive_detection(
            corrected_image,
            background_map,
            noise_map,
            sigma,
            min_distance,
        )
        _raise_if_cancelled(cancel_check)

        boundary = max(4, int(math.ceil(3 * sigma)))
        h, w = corrected_image.shape
        candidates: list[dict[str, Any]] = []
        for candidate in raw_candidates:
            _raise_if_cancelled(cancel_check)
            y, x = int(candidate[0]), int(candidate[1])
            if x < boundary or x >= w - boundary or y < boundary or y >= h - boundary:
                continue
            metrics = self._compute_fast_quality(
                corrected_image,
                background_map,
                noise_map,
                whitened,
                x,
                y,
                sigma,
            )
            if metrics is None:
                continue
            candidates.append(
                {
                    "x": float(x),
                    "y": float(y),
                    **metrics,
                }
            )

        candidates, crowded_regions = self._annotate_crowdedness(candidates, sigma)
        stats = {
            **prep_stats,
            "dog_threshold": float(threshold),
            "dog_candidates": int(len(raw_candidates)),
            "fast_candidates": int(len(candidates)),
            "crowded_regions": int(crowded_regions),
        }
        if preview_mode and fast_key is not None:
            self._remember_preview_cache(
                self._preview_candidate_cache,
                fast_key,
                {
                    "candidates": self._copy_records(candidates),
                    "stats": dict(stats),
                },
            )
        return candidates, stats

    def rank_candidates_fast(
        self,
        candidates: list[dict[str, Any]],
        *,
        limit: int | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        _raise_if_cancelled(cancel_check)
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item.get("quality", 0.0)),
                -float(item.get("white_peak", 0.0)),
                float(item.get("crowded_score", 0.0)),
                float(item.get("y", 0.0)),
                float(item.get("x", 0.0)),
            ),
        )
        if limit is not None:
            ranked = ranked[: int(limit)]
        return ranked

    @staticmethod
    def _build_fast_preview_record(candidate: dict[str, Any], sigma: float) -> dict[str, Any]:
        x = float(candidate.get("x", 0.0))
        y = float(candidate.get("y", 0.0))
        rough_snr = max(float(candidate.get("rough_snr", 0.0)), 0.0)
        isolation = max(float(candidate.get("isolation", 0.0)), 0.0)
        white_peak = max(float(candidate.get("white_peak", 0.0)), 0.0)
        crowded_score = max(float(candidate.get("crowded_score", 0.0)), 0.0)
        quality = max(float(candidate.get("quality", 0.0)), 0.0)
        background = max(float(candidate.get("background", 1.0)), 0.0)
        peak = max(float(candidate.get("peak", background + max(rough_snr, 1.0))), background)
        sigma_value = max(float(sigma), 0.6)
        photons = max(
            float(candidate.get("integrated_signal", 0.0)),
            max(peak - background, 0.0) * 2.0 * math.pi * sigma_value * sigma_value,
        )
        peak_over_background = max(peak - background, 0.0) / max(background, 1.0)
        r2_proxy = float(
            np.clip(
                0.08
                + 0.12 * np.log1p(rough_snr)
                + 0.10 * np.log1p(isolation)
                + 0.08 * np.log1p(white_peak)
                - 0.14 * crowded_score,
                -0.25,
                0.99,
            )
        )
        uncertainty = float(
            np.clip(
                0.85 / max(math.sqrt(max(rough_snr, 0.25)), 0.35) * (1.0 + 0.35 * crowded_score),
                0.08,
                1.5,
            )
        )
        return {
            "x": x,
            "y": y,
            "pos": [x, y],
            "snr": float(rough_snr),
            "r2": r2_proxy,
            "sigma": sigma_value,
            "photons": float(photons),
            "background": float(background),
            "peak_over_background": float(peak_over_background),
            "loc_uncertainty_scalar": uncertainty,
            "model_type": "fast_preview",
            "fit_status": "fast_preview",
            "emitter_count": 1,
            "crowded_flag": bool(candidate.get("crowded_flag")),
            "fit_score": float(quality),
            "quality": float(quality),
            "white_peak": float(white_peak),
            "rough_snr": float(rough_snr),
            "isolation": float(isolation),
        }

    def _roi_for_center(self, image: np.ndarray, x: float, y: float, sigma: float, half: int | None = None):
        roi_half = int(half or max(4, math.ceil(4.0 * sigma)))
        cx = int(round(x))
        cy = int(round(y))
        if (
            cx < roi_half
            or cx >= image.shape[1] - roi_half
            or cy < roi_half
            or cy >= image.shape[0] - roi_half
        ):
            return None
        roi = image[cy - roi_half : cy + roi_half + 1, cx - roi_half : cx + roi_half + 1].astype(np.float64)
        yy, xx = np.mgrid[0 : roi.shape[0], 0 : roi.shape[1]]
        origin_x = float(cx - roi_half)
        origin_y = float(cy - roi_half)
        return roi, xx.astype(np.float64), yy.astype(np.float64), origin_x, origin_y, roi_half

    def _fit_single_emitter_mle(self, roi: np.ndarray, xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma_init: float) -> tuple[np.ndarray, dict[str, Any]] | None:
        bg0 = max(float(np.percentile(roi, 25)), 0.0)
        photons0 = max(float(np.sum(np.clip(roi - bg0, 0.0, None))), 1.0)
        p0 = np.array([x0, y0, photons0, bg0, float(np.clip(sigma_init, 0.8, 6.0))], dtype=np.float64)
        bounds = [
            (x0 - 2.5, x0 + 2.5),
            (y0 - 2.5, y0 + 2.5),
            (1e-3, max(1e6, photons0 * 8.0)),
            (0.0, max(float(np.max(roi)), bg0 + photons0)),
            (0.7, max(6.0, sigma_init * 1.8)),
        ]

        def _objective(params: np.ndarray) -> float:
            x, y, photons, background, sigma = params
            kernel = _normalize_gaussian(xx, yy, x, y, sigma)
            model = background + photons * kernel
            return _poisson_nll(roi, model)

        result = minimize(_objective, p0, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            return None

        fitted = np.asarray(result.x, dtype=np.float64)
        x, y, photons, background, sigma = fitted
        kernel = _normalize_gaussian(xx, yy, x, y, sigma)
        model = background + photons * kernel
        return fitted, {
            "model": model,
            "nll": float(result.fun),
            "aic": float(2 * len(fitted) + 2 * result.fun),
            "success": True,
        }

    def _fit_double_emitter_mle(
        self,
        roi: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        init_points: list[tuple[float, float]],
        sigma_init: float,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        if len(init_points) < 2:
            return None

        bg0 = max(float(np.percentile(roi, 25)), 0.0)
        photons0 = max(float(np.sum(np.clip(roi - bg0, 0.0, None))) / 2.0, 1.0)
        p0 = np.array(
            [
                init_points[0][0],
                init_points[0][1],
                photons0,
                init_points[1][0],
                init_points[1][1],
                photons0,
                bg0,
                float(np.clip(sigma_init, 0.8, 6.0)),
            ],
            dtype=np.float64,
        )
        bounds = [
            (init_points[0][0] - 3.0, init_points[0][0] + 3.0),
            (init_points[0][1] - 3.0, init_points[0][1] + 3.0),
            (1e-3, max(1e6, photons0 * 8.0)),
            (init_points[1][0] - 3.0, init_points[1][0] + 3.0),
            (init_points[1][1] - 3.0, init_points[1][1] + 3.0),
            (1e-3, max(1e6, photons0 * 8.0)),
            (0.0, max(float(np.max(roi)), bg0 + photons0 * 2.0)),
            (0.7, max(6.0, sigma_init * 1.8)),
        ]

        def _objective(params: np.ndarray) -> float:
            x1, y1, p1, x2, y2, p2, background, sigma = params
            kernel1 = _normalize_gaussian(xx, yy, x1, y1, sigma)
            kernel2 = _normalize_gaussian(xx, yy, x2, y2, sigma)
            model = background + p1 * kernel1 + p2 * kernel2
            return _poisson_nll(roi, model)

        result = minimize(_objective, p0, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            return None

        fitted = np.asarray(result.x, dtype=np.float64)
        x1, y1, p1, x2, y2, p2, background, sigma = fitted
        kernel1 = _normalize_gaussian(xx, yy, x1, y1, sigma)
        kernel2 = _normalize_gaussian(xx, yy, x2, y2, sigma)
        model = background + p1 * kernel1 + p2 * kernel2
        return fitted, {
            "model": model,
            "nll": float(result.fun),
            "aic": float(2 * len(fitted) + 2 * result.fun),
            "success": True,
        }

    def _estimate_localization_uncertainty(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        x: float,
        y: float,
        photons: float,
        sigma: float,
        mu: np.ndarray,
    ) -> dict[str, Any]:
        kernel = _normalize_gaussian(xx, yy, x, y, sigma)
        sigma2 = max(float(sigma) ** 2, 1e-6)
        dmu_dx = photons * kernel * ((xx - x) / sigma2)
        dmu_dy = photons * kernel * ((yy - y) / sigma2)
        inv_mu = 1.0 / np.clip(mu, 1e-6, None)

        fim_xx = float(np.sum(inv_mu * dmu_dx * dmu_dx))
        fim_xy = float(np.sum(inv_mu * dmu_dx * dmu_dy))
        fim_yy = float(np.sum(inv_mu * dmu_dy * dmu_dy))
        fim = np.array([[fim_xx, fim_xy], [fim_xy, fim_yy]], dtype=np.float64)

        try:
            cov = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            return {
                "loc_uncertainty_x": np.nan,
                "loc_uncertainty_y": np.nan,
                "loc_cov_xy": np.nan,
                "loc_uncertainty_scalar": np.nan,
                "uncertainty_valid": False,
            }

        if np.any(~np.isfinite(cov)) or cov[0, 0] < 0 or cov[1, 1] < 0:
            return {
                "loc_uncertainty_x": np.nan,
                "loc_uncertainty_y": np.nan,
                "loc_cov_xy": np.nan,
                "loc_uncertainty_scalar": np.nan,
                "uncertainty_valid": False,
            }

        ux = math.sqrt(float(cov[0, 0]))
        uy = math.sqrt(float(cov[1, 1]))
        return {
            "loc_uncertainty_x": ux,
            "loc_uncertainty_y": uy,
            "loc_cov_xy": float(cov[0, 1]),
            "loc_uncertainty_scalar": float(math.sqrt(max(ux * uy, 0.0))),
            "uncertainty_valid": True,
        }

    def _build_result_record(
        self,
        *,
        roi: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        origin_x: float,
        origin_y: float,
        model: np.ndarray,
        x: float,
        y: float,
        photons: float,
        background: float,
        sigma: float,
        crowded_flag: bool,
        emitter_count: int,
        model_type: str,
        score_hint: float,
    ) -> dict[str, Any]:
        fitted_roi = np.asarray(model, dtype=np.float64)
        r2 = calculate_r_squared(roi.ravel(), fitted_roi.ravel())
        snr = float(photons / max(math.sqrt(max(photons + background * roi.size, 1e-6)), 1e-6))
        sigma_for_peak = max(float(sigma), 0.6)
        peak_over_background = float(
            photons / max(2.0 * math.pi * sigma_for_peak * sigma_for_peak * max(float(background), 1.0), 1e-6)
        )
        uncertainty = self._estimate_localization_uncertainty(
            xx,
            yy,
            x,
            y,
            photons,
            sigma,
            fitted_roi,
        )
        return {
            "pos": [float(origin_x + x), float(origin_y + y)],
            "x": float(origin_x + x),
            "y": float(origin_y + y),
            "photons": float(photons),
            "background": float(background),
            "sigma": float(sigma),
            "sigma_x": float(sigma),
            "sigma_y": float(sigma),
            "snr": snr,
            "r2": float(r2),
            "fit_score": float(score_hint),
            "peak_over_background": peak_over_background,
            "model_type": str(model_type),
            "crowded_flag": bool(crowded_flag),
            "emitter_count": int(emitter_count),
            "fit_status": "ok",
            "fit_message": "",
            **uncertainty,
        }

    @staticmethod
    def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        uncertainties = [
            float(item["loc_uncertainty_scalar"])
            for item in results
            if _float_or_none(item.get("loc_uncertainty_scalar")) is not None
        ]
        return {
            "final_molecules": int(len(results)),
            "avg_snr": float(np.mean([r["snr"] for r in results])) if results else 0.0,
            "avg_r2": float(np.mean([r["r2"] for r in results])) if results else 0.0,
            "avg_sigma": float(np.mean([r["sigma"] for r in results])) if results else 0.0,
            "avg_loc_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
            "deblended_emitters": int(sum(1 for item in results if item.get("model_type") == "double")),
        }

    def _post_filter_results(
        self,
        results: list[dict[str, Any]],
        *,
        sigma: float,
        min_r2: float,
        preview_mode: bool,
        min_snr: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        _raise_if_cancelled(cancel_check)
        if not results:
            return [], {
                "postfilter_in": 0,
                "postfilter_out": 0,
                "postfilter_rejected": 0,
                "postfilter_mode": "preview" if preview_mode else "final",
                "postfilter_rejection_counts": {},
                "postfilter_thresholds": {},
                **self._summarize_results([]),
            }

        sigma_upper_bound = max(6.0, float(sigma) * 1.8)
        sigma_bound_margin = max(0.15, sigma_upper_bound * 0.03)
        min_r2_floor = max(float(min_r2), 0.05 if preview_mode else 0.12)
        broad_sigma_limit = max(4.5, float(sigma) * (1.70 if preview_mode else 1.55))
        broad_sigma_r2_floor = max(min_r2_floor, 0.18 if preview_mode else 0.22)
        max_uncertainty = 0.75 if preview_mode else 0.45
        min_peak_over_background = 0.12 if preview_mode else 0.18

        rejection_counts: dict[str, int] = {}
        kept: list[dict[str, Any]] = []
        for item in results:
            _raise_if_cancelled(cancel_check)
            record = dict(item or {})
            reasons: list[str] = []

            snr_value = _float_or_none(record.get("snr"))
            sigma_value = _float_or_none(record.get("sigma"))
            r2_value = _float_or_none(record.get("r2"))
            uncertainty = _float_or_none(record.get("loc_uncertainty_scalar"))
            peak_over_background = _float_or_none(record.get("peak_over_background"))

            if peak_over_background is None:
                photons = _float_or_none(record.get("photons"))
                background = _float_or_none(record.get("background"))
                if photons is not None and background is not None and sigma_value is not None:
                    sigma_for_peak = max(float(sigma_value), 0.6)
                    peak_over_background = float(
                        photons
                        / max(
                            2.0 * math.pi * sigma_for_peak * sigma_for_peak * max(float(background), 1.0),
                            1e-6,
                        )
                    )
                    record["peak_over_background"] = peak_over_background

            if min_snr is not None and snr_value is not None and snr_value < float(min_snr):
                reasons.append("low_snr")
            if r2_value is not None and r2_value < min_r2_floor:
                reasons.append("low_r2")
            if sigma_value is not None and sigma_value >= sigma_upper_bound - sigma_bound_margin:
                reasons.append("sigma_at_upper_bound")
            if (
                sigma_value is not None
                and r2_value is not None
                and sigma_value >= broad_sigma_limit
                and r2_value < broad_sigma_r2_floor
            ):
                reasons.append("broad_sigma_low_r2")
            if uncertainty is not None and uncertainty > max_uncertainty:
                reasons.append("high_uncertainty")
            if peak_over_background is not None and peak_over_background < min_peak_over_background:
                reasons.append("low_peak_over_background")

            if reasons:
                for reason in set(reasons):
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            kept.append(record)

        stats = {
            "postfilter_in": int(len(results)),
            "postfilter_out": int(len(kept)),
            "postfilter_rejected": int(len(results) - len(kept)),
            "postfilter_mode": "preview" if preview_mode else "final",
            "postfilter_rejection_counts": dict(sorted(rejection_counts.items())),
            "postfilter_thresholds": {
                "min_r2_floor": float(min_r2_floor),
                "sigma_upper_bound": float(sigma_upper_bound),
                "sigma_bound_margin": float(sigma_bound_margin),
                "broad_sigma_limit": float(broad_sigma_limit),
                "broad_sigma_r2_floor": float(broad_sigma_r2_floor),
                "max_uncertainty": float(max_uncertainty),
                "min_peak_over_background": float(min_peak_over_background),
                "min_snr": float(min_snr) if min_snr is not None else None,
            },
            **self._summarize_results(kept),
        }
        return kept, stats

    @staticmethod
    def _candidate_groups(candidates: list[dict[str, Any]], radius: float) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        used: set[int] = set()
        for index, candidate in enumerate(candidates):
            if index in used:
                continue
            group = [candidate]
            used.add(index)
            changed = True
            while changed:
                changed = False
                for other_index, other in enumerate(candidates):
                    if other_index in used:
                        continue
                    for member in group:
                        dx = float(member["x"]) - float(other["x"])
                        dy = float(member["y"]) - float(other["y"])
                        if math.hypot(dx, dy) <= radius:
                            group.append(other)
                            used.add(other_index)
                            changed = True
                            break
            groups.append(group)
        return groups

    def _refine_candidate_group(
        self,
        image: np.ndarray,
        group: list[dict[str, Any]],
        sigma: float,
        min_snr: float,
        min_r2: float,
    ) -> list[dict[str, Any]]:
        anchor = max(group, key=lambda item: float(item.get("quality", 0.0)))
        roi_data = self._roi_for_center(image, anchor["x"], anchor["y"], sigma)
        if roi_data is None:
            return []

        roi, xx, yy, origin_x, origin_y, _roi_half = roi_data
        init_x = float(anchor["x"] - origin_x)
        init_y = float(anchor["y"] - origin_y)

        single_fit = self._fit_single_emitter_mle(roi, xx, yy, init_x, init_y, sigma)
        if single_fit is None:
            return []

        single_params, single_meta = single_fit
        sx, sy, sphotons, sbackground, ssigma = single_params
        results = [
            self._build_result_record(
                roi=roi,
                xx=xx,
                yy=yy,
                origin_x=origin_x,
                origin_y=origin_y,
                model=single_meta["model"],
                x=sx,
                y=sy,
                photons=sphotons,
                background=sbackground,
                sigma=ssigma,
                crowded_flag=bool(anchor.get("crowded_flag")),
                emitter_count=1,
                model_type="single",
                score_hint=-single_meta["nll"],
            )
        ]

        if len(group) < 2:
            candidate = results[0]
            if candidate["snr"] < min_snr or candidate["r2"] < min_r2:
                return []
            return results

        ordered = sorted(group, key=lambda item: -float(item.get("quality", 0.0)))[:2]
        init_points = [
            (float(item["x"] - origin_x), float(item["y"] - origin_y))
            for item in ordered
        ]
        double_fit = self._fit_double_emitter_mle(roi, xx, yy, init_points, sigma)
        if double_fit is None:
            filtered = [item for item in results if item["snr"] >= min_snr and item["r2"] >= min_r2]
            return filtered

        double_params, double_meta = double_fit
        use_double = double_meta["aic"] + 4.0 < single_meta["aic"]
        if not use_double:
            filtered = [item for item in results if item["snr"] >= min_snr and item["r2"] >= min_r2]
            return filtered

        x1, y1, p1, x2, y2, p2, background, dsigma = double_params
        model = double_meta["model"]
        out = [
            self._build_result_record(
                roi=roi,
                xx=xx,
                yy=yy,
                origin_x=origin_x,
                origin_y=origin_y,
                model=model,
                x=x1,
                y=y1,
                photons=p1,
                background=background,
                sigma=dsigma,
                crowded_flag=True,
                emitter_count=2,
                model_type="double",
                score_hint=-double_meta["nll"],
            ),
            self._build_result_record(
                roi=roi,
                xx=xx,
                yy=yy,
                origin_x=origin_x,
                origin_y=origin_y,
                model=model,
                x=x2,
                y=y2,
                photons=p2,
                background=background,
                sigma=dsigma,
                crowded_flag=True,
                emitter_count=2,
                model_type="double",
                score_hint=-double_meta["nll"],
            ),
        ]
        filtered = [item for item in out if item["snr"] >= min_snr and item["r2"] >= min_r2]
        return filtered

    @staticmethod
    def _remove_duplicates_scored(
        items: list[dict[str, Any]],
        min_dist: float,
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        pos = np.array([it["pos"] for it in items], dtype=np.float32)
        scores = np.array([it.get("fit_score", it.get("snr", 0.0)) for it in items], dtype=np.float32)
        order = np.argsort(-scores)
        kept: list[int] = []
        for idx in order:
            point = pos[idx]
            keep = True
            for kept_idx in kept:
                diff = point - pos[kept_idx]
                if float(diff[0] * diff[0] + diff[1] * diff[1]) < float(min_dist * min_dist):
                    keep = False
                    break
            if keep:
                kept.append(int(idx))
        kept.sort()
        return [items[i] for i in kept]

    def _prepare_interactive_image_cached(
        self,
        image: np.ndarray,
        sigma: float,
        auto_background: bool,
        bg_radius: int | None,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]:
        image = np.asarray(image, dtype=np.float32)
        prepare_key = self._interactive_prepare_cache_key(image, sigma, auto_background, bg_radius)
        cached = self._interactive_prepare_cache.get(prepare_key)
        if cached is not None:
            return prepare_key, cached["corrected_image"], dict(cached["stats"])

        corrected_image = image.astype(np.float32, copy=True)
        stats = {
            "sigma": float(sigma),
            "background_radius": None,
            "background_mode": "none",
        }
        if auto_background:
            if bg_radius is None:
                bg_radius = RollingBallBackground.estimate_adaptive_radius(image, sigma)
            corrected_image, _background = RollingBallBackground.subtract_background(image, bg_radius)
            stats["background_radius"] = int(bg_radius)
            stats["background_mode"] = "rolling_ball"
        _raise_if_cancelled(cancel_check)
        self._remember_preview_cache(
            self._interactive_prepare_cache,
            prepare_key,
            {
                "corrected_image": corrected_image,
                "stats": dict(stats),
            },
        )
        return prepare_key, corrected_image, dict(stats)

    def _interactive_dog_detection(
        self,
        image: np.ndarray,
        sigma: float,
        min_distance: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        peak = max(float(np.max(image)), 1e-10) if image.size else 1e-10
        img_norm = image / peak
        gaussian1 = ndimage.gaussian_filter(img_norm, sigma=sigma)
        gaussian2 = ndimage.gaussian_filter(img_norm, sigma=sigma * 2.5)
        dog = gaussian1 - gaussian2
        threshold = float(4.0 * mad_std(dog))
        dist = max(8, int(2.0 * sigma)) if min_distance is None else int(min_distance)
        local_max = _peak_local_max(
            dog,
            min_distance=dist,
            threshold_abs=threshold,
            exclude_border=max(1, int(3.0 * sigma)),
        )
        return dog, local_max, threshold

    def _interactive_candidates_cached(
        self,
        corrected_image: np.ndarray,
        prepare_key: tuple[Any, ...],
        sigma: float,
        min_distance: int | None,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]:
        candidate_key = self._interactive_candidate_cache_key(prepare_key, min_distance)
        cached = self._interactive_candidate_cache.get(candidate_key)
        if cached is not None:
            return candidate_key, np.asarray(cached["candidates"], dtype=np.int32), dict(cached["stats"])

        _dog, candidates, threshold = self._interactive_dog_detection(corrected_image, sigma, min_distance)
        _raise_if_cancelled(cancel_check)
        stats = {
            "dog_candidates": int(len(candidates)),
            "dog_threshold": float(threshold),
        }
        self._remember_preview_cache(
            self._interactive_candidate_cache,
            candidate_key,
            {
                "candidates": np.asarray(candidates, dtype=np.int32),
                "stats": dict(stats),
            },
        )
        return candidate_key, np.asarray(candidates, dtype=np.int32), dict(stats)

    @staticmethod
    def _fit_gaussian_2d_interactive(roi: np.ndarray, sigma: float) -> tuple[float, float, float, float, float, float, float] | None:
        def g2d(xy, amp, x0, y0, sx, sy, offset):
            x, y = xy
            return (
                amp
                * np.exp(-(((x - x0) ** 2) / (2.0 * sx**2) + ((y - y0) ** 2) / (2.0 * sy**2)))
                + offset
            ).ravel()

        size = roi.shape[0]
        x, y = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
        p0 = (
            float(np.max(roi) - np.min(roi)),
            float(size / 2.0),
            float(size / 2.0),
            float(sigma),
            float(sigma),
            float(np.min(roi)),
        )
        bounds = (
            [0.0, size / 2.0 - 3.0, size / 2.0 - 3.0, 0.5 * sigma, 0.5 * sigma, -np.inf],
            [np.inf, size / 2.0 + 3.0, size / 2.0 + 3.0, 2.0 * sigma, 2.0 * sigma, np.inf],
        )
        try:
            popt, _ = curve_fit(g2d, (x, y), roi.ravel(), p0=p0, bounds=bounds, maxfev=400)
            fitted = g2d((x, y), *popt).reshape(roi.shape)
            r2 = calculate_r_squared(roi.ravel(), fitted.ravel())
            amp = float(popt[0])
            return float(popt[2]), float(popt[1]), float(popt[3]), float(popt[4]), float(r2), amp, float(popt[5])
        except Exception:
            return None

    def _fit_interactive_candidate_raw(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        sigma: float,
        half: int,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        _raise_if_cancelled(cancel_check)
        snr = calculate_snr_improved(image, x, y, sigma)
        y_int = int(y)
        x_int = int(x)
        if (
            y_int <= half
            or y_int >= image.shape[0] - half - 1
            or x_int <= half
            or x_int >= image.shape[1] - half - 1
        ):
            return None

        roi = image[y_int - half : y_int + half + 1, x_int - half : x_int + half + 1].astype(np.float32)
        m00 = float(np.sum(roi))
        if m00 <= 0:
            return None
        m10 = float(np.sum(np.sum(roi, axis=0) * np.arange(roi.shape[1], dtype=np.float64)) / m00)
        m01 = float(np.sum(np.sum(roi, axis=1) * np.arange(roi.shape[0], dtype=np.float64)) / m00)
        center = roi.shape[0] / 2.0
        if abs(m10 - center) > 2.0 or abs(m01 - center) > 2.0:
            return None

        fit = self._fit_gaussian_2d_interactive(roi, sigma)
        _raise_if_cancelled(cancel_check)
        if fit is None:
            return None

        fy, fx, sx, sy, r2, amp, offset = fit
        sigma_mean = float((sx + sy) / 2.0)
        if (
            sigma_mean < 0.5 * sigma
            or sigma_mean > 2.0 * sigma
            or abs(sx - sy) / max(sigma_mean, 1e-6) > 0.4
        ):
            return None

        background = float(max(offset, 0.0))
        peak_over_background = float(max(amp, 0.0) / max(background, 1.0))
        fit_score = float(max(snr, 0.0) * max(r2, 1e-6) * max(amp, 1e-6))
        x_abs = float(x_int - half + fx)
        y_abs = float(y_int - half + fy)
        return {
            "x": x_abs,
            "y": y_abs,
            "pos": [x_abs, y_abs],
            "snr": float(snr),
            "r2": float(r2),
            "sigma": sigma_mean,
            "sigma_x": float(sx),
            "sigma_y": float(sy),
            "background": background,
            "peak_over_background": peak_over_background,
            "fit_score": fit_score,
            "model_type": "interactive_gaussian",
            "fit_status": "ok",
            "fit_message": "",
            "emitter_count": 1,
            "loc_uncertainty_x": float("nan"),
            "loc_uncertainty_y": float("nan"),
            "loc_cov_xy": float("nan"),
            "loc_uncertainty_scalar": float("nan"),
            "uncertainty_valid": False,
        }

    def _fit_interactive_candidate_chunk(
        self,
        image: np.ndarray,
        chunk: np.ndarray,
        sigma: float,
        half: int,
        candidate_key: tuple[Any, ...],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        sentinel = object()
        for candidate in np.asarray(chunk, dtype=np.int32):
            _raise_if_cancelled(cancel_check)
            y, x = int(candidate[0]), int(candidate[1])
            fit_key = self._interactive_fit_cache_key(candidate_key, x, y)
            cached = self._interactive_fit_cache.get(fit_key, sentinel)
            if cached is sentinel:
                record = self._fit_interactive_candidate_raw(
                    image,
                    x,
                    y,
                    sigma,
                    half,
                    cancel_check=cancel_check,
                )
                cached = None if record is None else dict(record)
                self._remember_preview_cache(
                    self._interactive_fit_cache,
                    fit_key,
                    cached,
                    limit=4096,
                )
            if cached is not None:
                out.append(dict(cached))
            _raise_if_cancelled(cancel_check)
        return out

    def _fit_interactive_candidates_cached(
        self,
        image: np.ndarray,
        candidates: np.ndarray,
        sigma: float,
        candidate_key: tuple[Any, ...],
        *,
        cancel_check: Callable[[], bool] | None = None,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = np.asarray(candidates, dtype=np.int32)
        if len(candidates) == 0:
            return []

        half = int(3.0 * sigma)
        max_workers = min(max(1, mp.cpu_count()), 8)
        chunk_size = max(4, int(np.ceil(len(candidates) / max(1, max_workers * 2))))
        chunks = [candidates[i : i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        image = np.asarray(image, dtype=np.float32)
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._fit_interactive_candidate_chunk,
                    image,
                    chunk,
                    sigma,
                    half,
                    candidate_key,
                    cancel_check=cancel_check,
                )
                for chunk in chunks
            ]
            total = len(futures)
            for index, future in enumerate(futures, start=1):
                _raise_if_cancelled(cancel_check)
                results.extend(future.result())
                if progress_cb:
                    progress = 45 + int(index / max(1, total) * 45)
                    progress_cb(min(progress, 90), f"Fitting Gaussian ROIs {index}/{total}...")
        return results

    def detect_interactive(
        self,
        image: np.ndarray,
        sigma: float = 3.5,
        min_snr: float = 4.0,
        min_r2: float = 0.0,
        min_distance: int | None = None,
        auto_background: bool = True,
        bg_radius: int | None = None,
        auto_bg: bool | None = None,
        *,
        progress_cb: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        if auto_bg is not None:
            auto_background = bool(auto_bg)
        if progress_cb:
            progress_cb(5, "Preparing image for interactive detection...")

        prepare_key, corrected_image, prepare_stats = self._prepare_interactive_image_cached(
            image,
            sigma,
            auto_background,
            bg_radius,
            cancel_check=cancel_check,
        )
        _raise_if_cancelled(cancel_check)
        if progress_cb:
            progress_cb(20, "Running DoG candidate detection...")

        candidate_key, candidates, candidate_stats = self._interactive_candidates_cached(
            corrected_image,
            prepare_key,
            sigma,
            min_distance,
            cancel_check=cancel_check,
        )
        _raise_if_cancelled(cancel_check)
        if progress_cb:
            progress_cb(35, f"Fitting {len(candidates)} Gaussian candidates...")

        fitted_results = self._fit_interactive_candidates_cached(
            np.asarray(image, dtype=np.float32),
            candidates,
            sigma,
            candidate_key,
            cancel_check=cancel_check,
            progress_cb=progress_cb,
        )
        _raise_if_cancelled(cancel_check)
        filtered = [
            dict(item)
            for item in fitted_results
            if float(item.get("snr", 0.0)) >= float(min_snr) and float(item.get("r2", 0.0)) >= float(min_r2)
        ]
        _raise_if_cancelled(cancel_check)
        if filtered:
            filtered = self._remove_duplicates_scored(
                filtered,
                float(min_distance if min_distance is not None else sigma * 2.0),
            )
        _raise_if_cancelled(cancel_check)
        if progress_cb:
            progress_cb(100, "Detection complete")

        positions = self._results_to_positions(filtered)
        stats = {
            **prepare_stats,
            **candidate_stats,
            **self._summarize_results(filtered),
            "mode": "interactive",
            "fitted_candidates": int(len(fitted_results)),
            "results": filtered,
        }
        self.detection_stats = stats
        return positions, stats

    def refine_candidates_precise(
        self,
        image: np.ndarray,
        candidates: list[dict[str, Any]],
        sigma: float = 3.5,
        min_snr: float = 4.0,
        min_r2: float = 0.0,
        min_distance: int | None = None,
        *,
        cancel_check: Callable[[], bool] | None = None,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        _raise_if_cancelled(cancel_check)
        if len(candidates) == 0:
            return [], {
                "refined_candidates": 0,
                "final_molecules": 0,
                "avg_snr": 0.0,
                "avg_sigma": 0.0,
                "avg_loc_uncertainty": 0.0,
                "deblended_emitters": 0,
            }

        groups = self._candidate_groups(
            candidates,
            radius=float(min_distance if min_distance else max(3.0, sigma * 2.0)),
        )
        refined: list[dict[str, Any]] = []
        if progress_cb:
            progress_cb(55, "Refining high-value candidates with Poisson-aware MLE...")

        def _refine(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return self._refine_candidate_group(np.asarray(image, dtype=np.float32), group, sigma, min_snr, min_r2)

        max_workers = min(max(1, mp.cpu_count()), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_refine, group) for group in groups]
            total = len(futures)
            for index, future in enumerate(futures, start=1):
                _raise_if_cancelled(cancel_check)
                refined.extend(future.result())
                if progress_cb:
                    progress = 55 + int(index / max(1, total) * 40)
                    progress_cb(min(progress, 95), f"MLE fitting group {index}/{total}...")

        if refined:
            refined = self._remove_duplicates_scored(
                refined,
                min_distance if min_distance else sigma * 1.5,
            )
        stats = {
            "precise_fit_requested": int(len(candidates)),
            "refined_candidates": int(len(refined)),
            **self._summarize_results(refined),
        }
        if progress_cb:
            progress_cb(100, "Precise fitting complete")
        return refined, stats

    def refine_manual_point(
        self,
        image: np.ndarray,
        x: float,
        y: float,
        *,
        sigma: float = 3.5,
    ) -> dict[str, Any] | None:
        candidate = {
            "x": float(x),
            "y": float(y),
            "quality": 1.0,
            "crowded_flag": False,
        }
        refined = self._refine_candidate_group(np.asarray(image, dtype=np.float32), [candidate], sigma, min_snr=0.0, min_r2=-1.0)
        return refined[0] if refined else None

    @staticmethod
    def _results_to_positions(results: list[dict[str, Any]]) -> list[list[float]]:
        positions = []
        for item in results:
            if "x" in item and "y" in item:
                positions.append([float(item["x"]), float(item["y"])])
                continue
            pos = list(item.get("pos", []) or [])
            if len(pos) >= 2:
                positions.append([float(pos[0]), float(pos[1])])
        return positions

    def detect_preview(
        self,
        image: np.ndarray,
        sigma: float = 3.5,
        min_snr: float = 4.0,
        min_r2: float = 0.0,
        min_distance: int | None = None,
        auto_background: bool = True,
        bg_radius: int | None = None,
        auto_bg: bool | None = None,
        *,
        preview_limit: int = 250,
        progress_cb: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        if progress_cb:
            progress_cb(5, "Running fast-only preview candidate screening...")
        candidates, fast_stats = self.detect_candidates_fast(
            image,
            sigma=sigma,
            min_distance=min_distance,
            auto_background=auto_background,
            bg_radius=bg_radius,
            auto_bg=auto_bg,
            preview_mode=True,
            cancel_check=cancel_check,
        )
        auto_background = bool(auto_bg) if auto_bg is not None else bool(auto_background)
        prepare_key = self._preview_prepare_cache_key(image, sigma, auto_background, bg_radius)
        fast_key = self._preview_fast_cache_key(prepare_key, min_distance)
        rank_key = self._preview_rank_cache_key(fast_key, preview_limit)
        cached_ranked = self._preview_rank_cache.get(rank_key)
        if cached_ranked is None:
            ranked = self.rank_candidates_fast(
                candidates,
                limit=preview_limit,
                cancel_check=cancel_check,
            )
            self._remember_preview_cache(
                self._preview_rank_cache,
                rank_key,
                self._copy_records(ranked),
            )
        else:
            ranked = self._copy_records(cached_ranked)
        if progress_cb:
            progress_cb(
                65,
                f"Fast-only preview kept {len(ranked)} provisional candidates; applying preview post-filter...",
            )
        cached_records = self._preview_record_cache.get(rank_key)
        if cached_records is None:
            preview_records = [self._build_fast_preview_record(candidate, sigma) for candidate in ranked]
            self._remember_preview_cache(
                self._preview_record_cache,
                rank_key,
                self._copy_records(preview_records),
            )
        else:
            preview_records = self._copy_records(cached_records)
        if progress_cb:
            progress_cb(98, "Applying fast-only preview quality post-filter...")
        filtered, postfilter_stats = self._post_filter_results(
            preview_records,
            sigma=sigma,
            min_r2=min_r2,
            preview_mode=True,
            min_snr=min_snr,
            cancel_check=cancel_check,
        )
        positions = self._results_to_positions(filtered)
        stats = {
            **fast_stats,
            **postfilter_stats,
            "mode": "preview",
            "preview_fast_only": True,
            "preview_limit": int(preview_limit),
            "ranked_candidates": int(len(ranked)),
            "results": filtered,
        }
        if progress_cb:
            progress_cb(100, "Fast-only preview complete")
        self.detection_stats = stats
        return positions, stats

    def detect_final(
        self,
        image: np.ndarray,
        sigma: float = 3.5,
        min_snr: float = 4.0,
        min_r2: float = 0.0,
        min_distance: int | None = None,
        auto_background: bool = True,
        bg_radius: int | None = None,
        auto_bg: bool | None = None,
        *,
        preview_limit: int | None = None,
        progress_cb: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        if progress_cb:
            progress_cb(5, "Running adaptive fast candidate screening...")
        candidates, fast_stats = self.detect_candidates_fast(
            image,
            sigma=sigma,
            min_distance=min_distance,
            auto_background=auto_background,
            bg_radius=bg_radius,
            auto_bg=auto_bg,
            cancel_check=cancel_check,
        )
        ranked = self.rank_candidates_fast(candidates, cancel_check=cancel_check)
        if progress_cb:
            progress_cb(35, f"Fast screening kept {len(ranked)} candidates; starting final MLE fitting...")
        refined, precise_stats = self.refine_candidates_precise(
            np.asarray(image),
            ranked,
            sigma=sigma,
            min_snr=min_snr,
            min_r2=min_r2,
            min_distance=min_distance,
            cancel_check=cancel_check,
            progress_cb=progress_cb,
        )
        if progress_cb:
            progress_cb(98, "Applying final quality post-filter...")
        filtered, postfilter_stats = self._post_filter_results(
            refined,
            sigma=sigma,
            min_r2=min_r2,
            preview_mode=False,
            min_snr=None,
            cancel_check=cancel_check,
        )
        positions = self._results_to_positions(filtered)
        stats = {
            **fast_stats,
            **precise_stats,
            **postfilter_stats,
            "mode": "final",
            "preview_limit": int(preview_limit) if preview_limit is not None else None,
            "ranked_candidates": int(len(ranked)),
            "precise_molecules_before_postfilter": int(len(refined)),
            "results": filtered,
        }
        self.detection_stats = stats
        return positions, stats

    def detect_molecules(
        self,
        image: np.ndarray,
        sigma: float = 3.5,
        min_snr: float = 4.0,
        min_r2: float = 0.0,
        min_distance: int | None = None,
        auto_background: bool = True,
        bg_radius: int | None = None,
        auto_bg: bool | None = None,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        return self.detect_final(
            image,
            sigma=sigma,
            min_snr=min_snr,
            min_r2=min_r2,
            min_distance=min_distance,
            auto_background=auto_background,
            bg_radius=bg_radius,
            auto_bg=auto_bg,
            cancel_check=cancel_check,
        )

    def get_detection_report(self) -> str:
        s = self.detection_stats
        if not s:
            return "No statistics available"
        mode = str(s.get("mode", "final") or "final")
        if mode == "interactive":
            lines = [
                "=== Detection Report ===",
                "Mode: Interactive Detection",
                f"DoG candidates: {s.get('dog_candidates', 0)}",
                f"Gaussian fits attempted: {s.get('fitted_candidates', 0)}",
                f"Final molecules: {s.get('final_molecules', 0)}",
            ]
            if "background_mode" in s:
                lines.append(f"Background mode: {s.get('background_mode', 'none')}")
            if "background_radius" in s and s.get("background_radius") is not None:
                lines.append(f"Background radius: {s.get('background_radius')}")
            if "dog_threshold" in s:
                lines.append(f"DoG threshold: {s.get('dog_threshold', 0.0):.4f}")
            lines.extend(
                [
                    f"Average SNR: {s.get('avg_snr', 0.0):.2f}",
                    f"Average R2: {s.get('avg_r2', 0.0):.3f}",
                    f"Average sigma: {s.get('avg_sigma', 0.0):.2f} px",
                ]
            )
            return "\n".join(lines)

        lines = [
            "=== Detection Report ===",
            f"Mode: {'Quick Preview (Fast-Only)' if mode == 'preview' else 'Final Detection'}",
            f"Raw candidates: {s.get('dog_candidates', 0)}",
            f"Adaptive fast pass: {s.get('fast_candidates', 0)}",
            f"Post-filter kept: {s.get('postfilter_out', s.get('final_molecules', 0))}/{s.get('postfilter_in', s.get('final_molecules', 0))}",
            f"Final molecules: {s.get('final_molecules', 0)}",
            f"Crowded regions: {s.get('crowded_regions', 0)}",
        ]
        if mode == "preview":
            lines.append(f"Preview limit: {s.get('preview_limit', 0)}")
            lines.append(f"Ranked preview candidates: {s.get('ranked_candidates', s.get('postfilter_in', 0))}")
        else:
            lines.extend(
                [
                    f"Precisely refined: {s.get('refined_candidates', 0)}",
                    f"MLE output before post-filter: {s.get('precise_molecules_before_postfilter', s.get('postfilter_in', s.get('final_molecules', 0)))}",
                    f"Deblended emitters: {s.get('deblended_emitters', 0)}",
                ]
            )
        rejection_counts = dict(s.get("postfilter_rejection_counts", {}) or {})
        if rejection_counts:
            lines.append(
                "Post-filter rejects: "
                + ", ".join(f"{reason}={count}" for reason, count in sorted(rejection_counts.items()))
            )
        lines.extend(
            [
                f"Average SNR: {s.get('avg_snr', 0.0):.2f}",
                f"Average sigma: {s.get('avg_sigma', 0.0):.2f} px",
                f"Average uncertainty: {s.get('avg_loc_uncertainty', 0.0):.3f} px",
            ]
        )
        return "\n".join(lines)
