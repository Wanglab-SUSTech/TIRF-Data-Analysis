from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np
from nd2 import ND2File
from PyQt5.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from channel_profiles import resolve_channel_profile

logger = logging.getLogger(__name__)


class ND2ReaderError(RuntimeError):
    """Base class for ND2 reader failures."""


class ND2OpenError(ND2ReaderError):
    """Opening the ND2 container failed."""


class ND2MetadataError(ND2ReaderError):
    """Metadata parsing failed."""


class ND2BackendSelectionError(ND2ReaderError):
    """No usable frame backend could be selected."""


class ND2FrameReadError(ND2ReaderError):
    """Reading a specific frame failed."""


@dataclass(frozen=True)
class ND2Metadata:
    num_frames: int
    num_channels: int
    height: int
    width: int
    channel_names: list[str]
    exposure_ms: float
    exposure_source: str
    is_merged: bool = False
    raw_num_channels: int | None = None
    raw_channel_names: list[str] = field(default_factory=list)
    channel_index_map: list[int] = field(default_factory=list)
    channel_profile: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        raw_num_channels = int(self.raw_num_channels or self.num_channels)
        raw_channel_names = (
            list(self.raw_channel_names)
            if self.raw_channel_names
            else list(self.channel_names)
        )
        channel_index_map = (
            list(self.channel_index_map)
            if self.channel_index_map
            else list(range(int(self.num_channels)))
        )
        return {
            "num_frames": int(self.num_frames),
            "num_channels": int(self.num_channels),
            "raw_num_channels": raw_num_channels,
            "height": int(self.height),
            "width": int(self.width),
            "channel_names": list(self.channel_names),
            "raw_channel_names": raw_channel_names,
            "channel_index_map": channel_index_map,
            "channel_profile": str(self.channel_profile),
            "exposure_ms": float(self.exposure_ms),
            "time_ms": float(self.exposure_ms),
            "exposure_source": str(self.exposure_source),
            "is_merged": bool(self.is_merged),
        }


def _safe_sizes(nd2_file: Any) -> dict[str, int]:
    sizes = getattr(nd2_file, "sizes", {}) or {}
    out: dict[str, int] = {}
    for key, value in sizes.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            continue
    return out


def _materialize_array(value: Any) -> np.ndarray:
    if hasattr(value, "compute"):
        value = value.compute()
    if hasattr(value, "values"):
        value = value.values
    return np.asarray(value)


def _fallback_axes(shape: tuple[int, ...], sizes: dict[str, int]) -> list[str]:
    ndim = len(shape)
    if ndim < 2:
        raise ND2MetadataError(f"Unsupported ND2 array rank: {ndim}")
    axes = [f"AX{i}" for i in range(ndim)]
    axes[-2:] = ["Y", "X"]
    if ndim >= 3:
        if "C" in sizes and "T" not in sizes:
            axes[-3] = "C"
        elif "T" in sizes and "C" not in sizes:
            axes[-3] = "T"
        elif sizes.get("C", 1) > 1:
            axes[-3] = "C"
        elif sizes.get("T", 1) > 1:
            axes[-3] = "T"
    if ndim >= 4:
        if "T" in sizes and "C" in sizes:
            axes[-4] = "T"
            axes[-3] = "C"
        elif sizes.get("T", 1) > 1 and axes[-3] != "T":
            axes[-4] = "T"
        elif sizes.get("C", 1) > 1 and axes[-3] != "C":
            axes[-4] = "C"
    return axes


def _infer_axes(
    shape: tuple[int, ...],
    *,
    preferred_axes: list[str] | tuple[str, ...] | None,
    sizes: dict[str, int],
) -> list[str]:
    axes = list(preferred_axes or [])
    if len(axes) == len(shape) and "Y" in axes and "X" in axes:
        return axes

    ordered_axes = list(sizes.keys())
    if len(ordered_axes) == len(shape) and "Y" in ordered_axes and "X" in ordered_axes:
        return ordered_axes

    return _fallback_axes(shape, sizes)


def _build_index(axes: list[str], frame_idx: int, ch_idx: int) -> tuple[Any, ...]:
    index: list[Any] = []
    for axis in axes:
        if axis == "T":
            index.append(int(frame_idx))
        elif axis == "C":
            index.append(int(ch_idx))
        elif axis in ("Y", "X"):
            index.append(slice(None))
        else:
            index.append(0)
    return tuple(index)


def _build_block_index(
    axes: list[str],
    frame_start: int,
    frame_stop: int,
    ch_idx: int,
) -> tuple[Any, ...]:
    index: list[Any] = []
    for axis in axes:
        if axis == "T":
            index.append(slice(int(frame_start), int(frame_stop)))
        elif axis == "C":
            index.append(int(ch_idx))
        elif axis in ("Y", "X"):
            index.append(slice(None))
        else:
            index.append(0)
    return tuple(index)


def _extract_2d_plane(
    value: Any,
    *,
    preferred_axes: list[str] | tuple[str, ...] | None,
    sizes: dict[str, int],
    frame_idx: int,
    ch_idx: int,
) -> np.ndarray:
    arr = _materialize_array(value)
    axes = _infer_axes(arr.shape, preferred_axes=preferred_axes, sizes=sizes)
    if len(axes) != arr.ndim:
        raise ND2FrameReadError("Axis inference mismatch")
    plane = arr[_build_index(axes, frame_idx, ch_idx)]
    plane = _materialize_array(plane)
    if plane.ndim != 2:
        raise ND2FrameReadError(f"Expected 2D plane, got shape {tuple(plane.shape)}")
    return plane


def _extract_3d_stack(
    value: Any,
    *,
    preferred_axes: list[str] | tuple[str, ...] | None,
    sizes: dict[str, int],
    frame_start: int,
    frame_stop: int,
    ch_idx: int,
) -> np.ndarray:
    arr = _materialize_array(value)
    axes = _infer_axes(arr.shape, preferred_axes=preferred_axes, sizes=sizes)
    if len(axes) != arr.ndim:
        raise ND2FrameReadError("Axis inference mismatch")

    stack = arr[_build_block_index(axes, frame_start, frame_stop, ch_idx)]
    stack = _materialize_array(stack)
    remaining_axes = [
        axis
        for axis, item in zip(axes, _build_block_index(axes, frame_start, frame_stop, ch_idx))
        if isinstance(item, slice)
    ]

    if stack.ndim == 2:
        if frame_stop - frame_start != 1:
            raise ND2FrameReadError(
                f"Expected {frame_stop - frame_start} frames, got single plane"
            )
        stack = stack[np.newaxis, ...]
        remaining_axes = ["T", "Y", "X"]

    if "Y" not in remaining_axes or "X" not in remaining_axes:
        raise ND2FrameReadError(f"Expected Y/X axes, got {remaining_axes}")

    if "T" not in remaining_axes:
        if frame_stop - frame_start != 1:
            raise ND2FrameReadError("Requested multiple frames from non-time ND2 data")
        stack = stack[np.newaxis, ...]
        remaining_axes = ["T", *remaining_axes]

    order = [remaining_axes.index("T"), remaining_axes.index("Y"), remaining_axes.index("X")]
    stack = np.moveaxis(stack, order, (0, 1, 2))
    if stack.ndim != 3:
        raise ND2FrameReadError(f"Expected 3D stack, got shape {tuple(stack.shape)}")
    expected_frames = int(frame_stop - frame_start)
    if stack.shape[0] != expected_frames:
        raise ND2FrameReadError(
            f"Expected {expected_frames} frames, got shape {tuple(stack.shape)}"
        )
    return stack


def _valid_exposure_ms(value: Any) -> float | None:
    try:
        exposure = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(exposure) or exposure <= 0:
        return None
    return exposure


def _read_exposure_time_from_metadata(nd2_file: Any, num_frames: int) -> tuple[float, str]:
    try:
        experiment = getattr(nd2_file, "experiment", None)
        if experiment:
            for loop in experiment:
                params = getattr(loop, "parameters", None)
                if params is None:
                    continue
                period_ms = None
                if isinstance(params, dict):
                    period_ms = params.get("periodMs")
                else:
                    period_ms = getattr(params, "periodMs", None)
                exposure = _valid_exposure_ms(period_ms)
                if exposure is not None:
                    return exposure, "experiment.periodMs"

        metadata = getattr(nd2_file, "metadata", None)
        channels = getattr(metadata, "channels", None)
        if channels:
            for channel in channels:
                camera = getattr(channel, "camera", None)
                exposure = _valid_exposure_ms(getattr(camera, "exposureTimeMs", None))
                if exposure is not None:
                    return exposure, "metadata.channels.camera.exposureTimeMs"

        frame_metadata = getattr(nd2_file, "frame_metadata", None)
        if callable(frame_metadata) and num_frames > 1:
            try:
                t0 = frame_metadata(0).channels[0].time.absoluteJulianDayNumber
                t1 = frame_metadata(1).channels[0].time.absoluteJulianDayNumber
                interval_ms = (t1 - t0) * 24 * 3600 * 1000
                exposure = _valid_exposure_ms(interval_ms)
                if exposure is not None and 1 < exposure < 10000:
                    return exposure, "frame_metadata.absoluteJulianDayNumber"
            except Exception:
                pass
    except Exception:
        logger.exception("Failed to parse ND2 exposure metadata")
    raise ND2MetadataError(
        "ND2 metadata does not contain a usable exposure time; refusing to use a manual or fallback 100 ms time axis"
    )


def _generate_channel_names(nd2_file: Any, num_channels: int) -> list[str]:
    try:
        metadata = getattr(nd2_file, "metadata", None)
        channels = getattr(metadata, "channels", None)
        parsed: list[str] = []
        if channels:
            for channel in channels:
                channel_info = getattr(channel, "channel", None)
                name = getattr(channel_info, "name", None)
                if name:
                    match = re.search(r"\d{3}", str(name))
                    parsed.append(match.group() if match else str(name))
                else:
                    parsed.append(f"Ch{len(parsed) + 1}")
        if len(parsed) == num_channels:
            return parsed
    except Exception:
        logger.exception("Failed to parse ND2 channel metadata")

    defaults = {
        1: ["532"],
        2: ["532", "638"],
        3: ["532", "638", "488"],
    }
    return defaults.get(num_channels, [f"Ch{i + 1}" for i in range(num_channels)])


def _infer_num_frames_from_length(nd2_file: Any, num_channels: int, fallback: int) -> int:
    try:
        total_planes = len(nd2_file)
    except TypeError:
        return fallback
    except Exception:
        logger.debug("Unable to derive plane count from ND2 object", exc_info=True)
        return fallback

    if total_planes and total_planes > max(1, num_channels):
        return max(1, int(total_planes) // max(1, int(num_channels)))
    return fallback


def _build_metadata(nd2_file: Any, user_exposure_ms: float | None) -> ND2Metadata:
    sizes = _safe_sizes(nd2_file)
    height = int(sizes.get("Y", 0))
    width = int(sizes.get("X", 0))
    raw_num_channels = max(1, int(sizes.get("C", 1)))
    channel_profile = resolve_channel_profile(raw_num_channels)
    channel_index_map = list(channel_profile["channel_index_map"])
    num_channels = len(channel_index_map)
    num_frames = max(1, int(sizes.get("T", 1)))
    if num_frames == 1:
        num_frames = _infer_num_frames_from_length(nd2_file, raw_num_channels, num_frames)
    if height <= 0 or width <= 0:
        raise ND2MetadataError("ND2 metadata missing valid Y/X dimensions")
    raw_channel_names = _generate_channel_names(nd2_file, raw_num_channels)
    channel_names = channel_profile["channel_names"] or raw_channel_names
    if user_exposure_ms is not None:
        logger.info("Ignoring manual exposure_ms=%s; ND2 metadata is required", user_exposure_ms)
    exposure_ms, exposure_source = _read_exposure_time_from_metadata(nd2_file, num_frames)
    return ND2Metadata(
        num_frames=num_frames,
        num_channels=num_channels,
        height=height,
        width=width,
        channel_names=list(channel_names),
        exposure_ms=exposure_ms,
        exposure_source=exposure_source,
        raw_num_channels=raw_num_channels,
        raw_channel_names=raw_channel_names,
        channel_index_map=channel_index_map,
        channel_profile=str(channel_profile["profile"]),
    )


class ND2FrameSource:
    backend_name = "base"

    def __init__(self, nd2_file: Any, metadata: ND2Metadata, capabilities: dict[str, Any]):
        self.nd2_file = nd2_file
        self.metadata = metadata
        self.capabilities = dict(capabilities)

    def get_frame_2d(self, frame_idx: int, ch_idx: int) -> np.ndarray:
        raise NotImplementedError

    def get_frame_block_3d(self, frame_start: int, frame_stop: int, ch_idx: int) -> np.ndarray:
        if frame_stop <= frame_start:
            return np.zeros((0, self.metadata.height, self.metadata.width), dtype=np.float32)
        frames = [
            self.get_frame_2d(frame_idx, ch_idx)
            for frame_idx in range(int(frame_start), int(frame_stop))
        ]
        return np.stack(frames, axis=0)

    def get_metadata(self) -> dict[str, Any]:
        return self.metadata.to_dict()

    def close(self) -> None:
        return None


class GetItemFrameSource(ND2FrameSource):
    backend_name = "getitem"

    def __init__(self, nd2_file: Any, metadata: ND2Metadata):
        sizes = _safe_sizes(nd2_file)
        super().__init__(
            nd2_file,
            metadata,
            capabilities={
                "entry": "getitem",
                "lazy": True,
                "materializes_full_array": False,
                "axes": list(sizes.keys()),
            },
        )
        self.sizes = sizes
        self.axes = list(sizes.keys())

    def get_frame_2d(self, frame_idx: int, ch_idx: int) -> np.ndarray:
        try:
            if self.axes:
                return _extract_2d_plane(
                    self.nd2_file[_build_index(self.axes, frame_idx, ch_idx)],
                    preferred_axes=self.axes,
                    sizes=self.sizes,
                    frame_idx=frame_idx,
                    ch_idx=ch_idx,
                )
        except Exception as exc:
            logger.debug("Tuple getitem failed for ND2 getitem backend: %s", exc)

        linear_index = frame_idx * max(1, int(self.metadata.raw_num_channels or self.metadata.num_channels)) + ch_idx
        try:
            return _extract_2d_plane(
                self.nd2_file[linear_index],
                preferred_axes=None,
                sizes={"Y": self.metadata.height, "X": self.metadata.width},
                frame_idx=0,
                ch_idx=0,
            )
        except Exception as exc:
            raise ND2FrameReadError(
                f"Backend getitem failed to read frame {frame_idx}, channel {ch_idx}: {exc}"
            ) from exc

    def get_frame_block_3d(self, frame_start: int, frame_stop: int, ch_idx: int) -> np.ndarray:
        try:
            if self.axes:
                return _extract_3d_stack(
                    self.nd2_file[_build_block_index(self.axes, frame_start, frame_stop, ch_idx)],
                    preferred_axes=self.axes,
                    sizes=self.sizes,
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                    ch_idx=0,
                )
        except Exception as exc:
            logger.debug("Tuple getitem block read failed for ND2 getitem backend: %s", exc)
        return super().get_frame_block_3d(frame_start, frame_stop, ch_idx)


class DaskFrameSource(ND2FrameSource):
    backend_name = "to_dask"

    def __init__(self, nd2_file: Any, metadata: ND2Metadata):
        sizes = _safe_sizes(nd2_file)
        array = nd2_file.to_dask()
        axes = getattr(array, "dims", None) or list(sizes.keys())
        super().__init__(
            nd2_file,
            metadata,
            capabilities={
                "entry": "to_dask",
                "lazy": True,
                "materializes_full_array": False,
                "axes": list(axes),
            },
        )
        self.array = array
        self.sizes = sizes
        self.axes = list(axes)

    def get_frame_2d(self, frame_idx: int, ch_idx: int) -> np.ndarray:
        try:
            return _extract_2d_plane(
                self.array[_build_index(self.axes, frame_idx, ch_idx)],
                preferred_axes=self.axes,
                sizes=self.sizes,
                frame_idx=0,
                ch_idx=0,
            )
        except Exception as exc:
            raise ND2FrameReadError(
                f"Backend to_dask failed to read frame {frame_idx}, channel {ch_idx}: {exc}"
            ) from exc

    def get_frame_block_3d(self, frame_start: int, frame_stop: int, ch_idx: int) -> np.ndarray:
        try:
            index = _build_block_index(self.axes, frame_start, frame_stop, ch_idx)
            remaining_axes = [
                axis
                for axis, item in zip(self.axes, index)
                if isinstance(item, slice)
            ]
            return _extract_3d_stack(
                self.array[index],
                preferred_axes=remaining_axes,
                sizes={
                    "T": int(frame_stop - frame_start),
                    "Y": self.metadata.height,
                    "X": self.metadata.width,
                },
                frame_start=0,
                frame_stop=int(frame_stop - frame_start),
                ch_idx=0,
            )
        except Exception as exc:
            raise ND2FrameReadError(
                f"Backend to_dask failed to read frames {frame_start}:{frame_stop}, channel {ch_idx}: {exc}"
            ) from exc


class AsArrayFrameSource(ND2FrameSource):
    backend_name = "asarray"

    def __init__(self, nd2_file: Any, metadata: ND2Metadata):
        sizes = _safe_sizes(nd2_file)
        array = _materialize_array(nd2_file.asarray())
        axes = _infer_axes(array.shape, preferred_axes=list(sizes.keys()), sizes=sizes)
        super().__init__(
            nd2_file,
            metadata,
            capabilities={
                "entry": "asarray",
                "lazy": False,
                "materializes_full_array": True,
                "axes": list(axes),
            },
        )
        self.array = array
        self.sizes = sizes
        self.axes = list(axes)

    def get_frame_2d(self, frame_idx: int, ch_idx: int) -> np.ndarray:
        try:
            return _extract_2d_plane(
                self.array,
                preferred_axes=self.axes,
                sizes=self.sizes,
                frame_idx=frame_idx,
                ch_idx=ch_idx,
            )
        except Exception as exc:
            raise ND2FrameReadError(
                f"Backend asarray failed to read frame {frame_idx}, channel {ch_idx}: {exc}"
            ) from exc

    def get_frame_block_3d(self, frame_start: int, frame_stop: int, ch_idx: int) -> np.ndarray:
        try:
            return _extract_3d_stack(
                self.array,
                preferred_axes=self.axes,
                sizes=self.sizes,
                frame_start=frame_start,
                frame_stop=frame_stop,
                ch_idx=ch_idx,
            )
        except Exception as exc:
            raise ND2FrameReadError(
                f"Backend asarray failed to read frames {frame_start}:{frame_stop}, channel {ch_idx}: {exc}"
            ) from exc


class XArrayFrameSource(ND2FrameSource):
    backend_name = "xarray"

    def __init__(self, nd2_file: Any, metadata: ND2Metadata):
        sizes = _safe_sizes(nd2_file)
        if hasattr(nd2_file, "to_xarray"):
            data_array = nd2_file.to_xarray()
        elif hasattr(nd2_file, "xarray"):
            data_array = nd2_file.xarray()
        else:
            raise ND2BackendSelectionError("ND2 object has no xarray entry point")
        axes = list(getattr(data_array, "dims", None) or [])
        super().__init__(
            nd2_file,
            metadata,
            capabilities={
                "entry": "xarray",
                "lazy": hasattr(data_array, "data") and hasattr(data_array.data, "chunks"),
                "materializes_full_array": False,
                "axes": list(axes),
            },
        )
        self.data_array = data_array
        self.sizes = sizes
        self.axes = axes

    def get_frame_2d(self, frame_idx: int, ch_idx: int) -> np.ndarray:
        try:
            selector = {}
            for axis in self.axes:
                if axis == "T":
                    selector[axis] = int(frame_idx)
                elif axis == "C":
                    selector[axis] = int(ch_idx)
                elif axis not in ("Y", "X"):
                    selector[axis] = 0
            plane = self.data_array.isel(**selector) if selector else self.data_array
            return _extract_2d_plane(
                plane,
                preferred_axes=[ax for ax in self.axes if ax in ("Y", "X")],
                sizes={"Y": self.metadata.height, "X": self.metadata.width},
                frame_idx=0,
                ch_idx=0,
            )
        except Exception as exc:
            raise ND2FrameReadError(
                f"Backend xarray failed to read frame {frame_idx}, channel {ch_idx}: {exc}"
            ) from exc

    def get_frame_block_3d(self, frame_start: int, frame_stop: int, ch_idx: int) -> np.ndarray:
        try:
            selector = {}
            for axis in self.axes:
                if axis == "T":
                    selector[axis] = slice(int(frame_start), int(frame_stop))
                elif axis == "C":
                    selector[axis] = int(ch_idx)
                elif axis not in ("Y", "X"):
                    selector[axis] = 0
            stack = self.data_array.isel(**selector) if selector else self.data_array
            return _extract_3d_stack(
                stack,
                preferred_axes=[ax for ax in self.axes if ax in ("T", "Y", "X")],
                sizes={"T": frame_stop - frame_start, "Y": self.metadata.height, "X": self.metadata.width},
                frame_start=0,
                frame_stop=frame_stop - frame_start,
                ch_idx=0,
            )
        except Exception as exc:
            raise ND2FrameReadError(
                f"Backend xarray failed to read frames {frame_start}:{frame_stop}, channel {ch_idx}: {exc}"
            ) from exc


class ND2Reader:
    BACKEND_TYPES = (
        ("getitem", GetItemFrameSource, lambda f: hasattr(f, "__getitem__")),
        ("to_dask", DaskFrameSource, lambda f: hasattr(f, "to_dask")),
        ("asarray", AsArrayFrameSource, lambda f: hasattr(f, "asarray")),
        (
            "xarray",
            XArrayFrameSource,
            lambda f: hasattr(f, "to_xarray") or hasattr(f, "xarray"),
        ),
    )

    def __init__(self, filepath: str, user_exposure_ms: float | None = None, user_time_ms: float | None = None):
        self.filepath = filepath
        self.nd2_file: Any | None = None
        self._read_lock = Lock()
        self._frame_source: ND2FrameSource | None = None
        self.backend_name = "uninitialized"
        self.backend_capabilities: dict[str, Any] = {}
        self._metadata: ND2Metadata | None = None

        logger.info("=" * 60)
        logger.info("Opening ND2 file with capability detection...")

        try:
            self.nd2_file = ND2File(filepath)
        except Exception as exc:
            raise ND2OpenError(f"Failed to open ND2 file: {exc}") from exc

        effective_exposure_ms = user_exposure_ms if user_exposure_ms is not None else user_time_ms
        try:
            self._metadata = _build_metadata(self.nd2_file, effective_exposure_ms)
        except ND2ReaderError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise ND2MetadataError(f"Failed to parse ND2 metadata: {exc}") from exc

        try:
            self._frame_source = self._select_backend()
        except ND2ReaderError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise ND2BackendSelectionError(f"Failed to select ND2 backend: {exc}") from exc

        metadata = self._metadata
        self.num_frames = metadata.num_frames
        self.num_channels = metadata.num_channels
        self.raw_num_channels = int(metadata.raw_num_channels or metadata.num_channels)
        self.height = metadata.height
        self.width = metadata.width
        self.channel_names = list(metadata.channel_names)
        self.raw_channel_names = list(metadata.raw_channel_names or metadata.channel_names)
        self.channel_index_map = list(metadata.channel_index_map or range(metadata.num_channels))
        self.channel_profile = str(metadata.channel_profile)
        self.exposure_ms = metadata.exposure_ms
        self.time_ms = metadata.exposure_ms
        self.exposure_source = metadata.exposure_source
        self.is_merged = metadata.is_merged

        logger.info("ND2 loaded successfully")
        logger.info("  backend: %s", self.backend_name)
        if self.channel_profile != "standard":
            logger.info(
                "  channel profile: %s raw=%s map=%s",
                self.channel_profile,
                self.raw_channel_names,
                self.channel_index_map,
            )
        logger.info("  channels: %s", self.channel_names)
        logger.info("  frames: %s", self.num_frames)
        logger.info("  shape per channel: %sx%s", self.height, self.width)
        logger.info("  exposure_ms: %.3f", self.exposure_ms)
        logger.info("  exposure_source: %s", self.exposure_source)
        logger.info("  file size: %s", self._format_file_size())

    def _select_backend(self) -> ND2FrameSource:
        assert self.nd2_file is not None
        assert self._metadata is not None
        availability = {
            name: bool(probe(self.nd2_file))
            for name, _cls, probe in self.BACKEND_TYPES
        }
        errors: list[str] = []
        for name, backend_cls, probe in self.BACKEND_TYPES:
            if not probe(self.nd2_file):
                continue
            try:
                backend = backend_cls(self.nd2_file, self._metadata)
                sample = backend.get_frame_2d(0, 0)
                if tuple(sample.shape) != (self._metadata.height, self._metadata.width):
                    raise ND2BackendSelectionError(
                        f"backend {name} returned shape {tuple(sample.shape)}, "
                        f"expected {(self._metadata.height, self._metadata.width)}"
                    )
                self.backend_name = backend.backend_name
                self.backend_capabilities = {
                    **availability,
                    "selected": backend.backend_name,
                    "selected_capabilities": dict(backend.capabilities),
                }
                return backend
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                logger.debug("Skipping ND2 backend %s", name, exc_info=True)
        detail = "; ".join(errors) if errors else "no backend entry points available"
        raise ND2BackendSelectionError(f"Failed to select a usable ND2 backend: {detail}")

    def _format_file_size(self) -> str:
        try:
            size_bytes = float(os.path.getsize(self.filepath))
        except Exception:
            return "Unknown"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def _ensure_uint16(self, array: Any) -> np.ndarray:
        arr = np.asarray(array)
        if arr.dtype == np.uint16 and arr.dtype.isnative:
            return np.ascontiguousarray(arr.copy())
        if arr.dtype == np.uint8:
            return np.ascontiguousarray((arr.astype(np.float32) / 255.0 * 65535).astype(np.uint16))
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.nanmax(arr)) if arr.size else 0.0
            if max_val <= 1.0:
                return np.ascontiguousarray(np.clip(arr * 65535.0, 0, 65535).astype(np.uint16))
            if max_val <= 255.0:
                return np.ascontiguousarray(np.clip(arr / 255.0 * 65535.0, 0, 65535).astype(np.uint16))
            return np.ascontiguousarray(np.clip(arr, 0, 65535).astype(np.uint16))
        return np.ascontiguousarray(np.clip(arr, 0, 65535).astype(np.uint16))

    def get_frame(self, frame_index: int) -> list[np.ndarray]:
        return self.get_frame_data(frame_index, allow_fallback_blank=False)

    def get_frame_data(self, frame_index: int, allow_fallback_blank: bool = False) -> list[np.ndarray]:
        if frame_index < 0 or frame_index >= self.num_frames:
            raise ND2FrameReadError(
                f"Frame index {frame_index} is out of range [0, {self.num_frames - 1}]"
            )
        if self._frame_source is None:
            raise ND2FrameReadError("ND2 reader backend is not initialized")

        channels: list[np.ndarray] = []
        with self._read_lock:
            try:
                for raw_ch_idx in self.channel_index_map:
                    plane = self._frame_source.get_frame_2d(frame_index, int(raw_ch_idx))
                    channels.append(self._ensure_uint16(plane))
            except Exception as exc:
                if allow_fallback_blank:
                    return [
                        np.zeros((self.height, self.width), dtype=np.uint16)
                        for _ in range(self.num_channels)
                    ]
                raise ND2FrameReadError(
                    f"Failed to read frame {frame_index}: {exc}"
                ) from exc
        return channels

    def get_metadata(self) -> dict[str, Any]:
        base = self._metadata.to_dict() if self._metadata is not None else {}
        base["backend_name"] = self.backend_name
        base["backend_capabilities"] = dict(self.backend_capabilities)
        return base

    def get_channel_block(self, start_frame: int, stop_frame: int, channel_index: int) -> np.ndarray:
        if start_frame < 0 or stop_frame < start_frame or stop_frame > self.num_frames:
            raise ND2FrameReadError(
                f"Frame block [{start_frame}, {stop_frame}) is out of range for {self.num_frames} frames"
            )
        if channel_index < 0 or channel_index >= self.num_channels:
            raise ND2FrameReadError(
                f"Channel index {channel_index} is out of range [0, {self.num_channels - 1}]"
            )
        raw_channel_index = int(self.channel_index_map[channel_index])
        if self._frame_source is None:
            raise ND2FrameReadError("ND2 reader backend is not initialized")
        with self._read_lock:
            try:
                stack = self._frame_source.get_frame_block_3d(start_frame, stop_frame, raw_channel_index)
            except Exception as exc:
                raise ND2FrameReadError(
                    f"Failed to read frame block {start_frame}:{stop_frame} for channel {channel_index}: {exc}"
                ) from exc
        stack = np.asarray(stack)
        if stack.dtype == np.uint16 and stack.dtype.isnative:
            return np.ascontiguousarray(stack.copy())
        converted = [self._ensure_uint16(frame) for frame in stack]
        return np.stack(converted, axis=0) if converted else np.zeros((0, self.height, self.width), dtype=np.uint16)

    def close(self) -> None:
        if self._frame_source is not None:
            try:
                self._frame_source.close()
            except Exception:
                logger.debug("Failed to close ND2 frame source", exc_info=True)
            self._frame_source = None
        if self.nd2_file is not None:
            try:
                self.nd2_file.close()
            except Exception:
                logger.debug("Failed to close ND2 container", exc_info=True)
            self.nd2_file = None


class ExposureTimeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Exposure Time")
        self.resize(300, 120)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Please enter the exposure time (ms):"))
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(1.0, 10000.0)
        self.spinbox.setValue(100.0)
        self.spinbox.setDecimals(1)
        self.spinbox.setSuffix(" ms")
        layout.addWidget(self.spinbox)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)
        self.setLayout(layout)

    def get_exposure_time(self) -> float:
        return float(self.spinbox.value())
