from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cupy_module() -> Any | None:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return None
    except Exception:
        logger.debug("CuPy is importable but no CUDA device is available", exc_info=True)
        return None
    return cp


@lru_cache(maxsize=1)
def opencv_cuda_available() -> bool:
    try:
        import cv2

        return int(cv2.cuda.getCudaEnabledDeviceCount()) > 0
    except Exception:
        return False


def cupy_available() -> bool:
    return get_cupy_module() is not None


def gpu_status() -> dict[str, bool | str]:
    cupy_ok = cupy_available()
    opencv_ok = opencv_cuda_available()
    default_backend = "CuPy" if cupy_ok else "CPU"
    return {
        "cupy_detected": cupy_ok,
        "opencv_cuda_detected": opencv_ok,
        "default_compute_backend": default_backend,
    }


def normalize_compute_backend(value: str | None) -> str:
    backend = str(value or "auto").strip().lower()
    if backend not in {"auto", "cpu", "cupy"}:
        raise ValueError(f"Unsupported compute backend: {value!r}")
    return backend
