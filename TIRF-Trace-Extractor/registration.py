"""
registration.py - affine registration module

Design principles:
- Always use channel 532 as the reference channel
- Persist transforms as source_channel_raw -> 532_reference
- Display path may warp images with warpAffine (bilinear interpolation)
- Quantitative path never interpolates the raw image; it only inverse-maps coordinates
- Remains compatible with legacy offsets JSON

Upgrade notes:
- Do not trust inverse_matrix_2x3 directly during deserialization
- Always recompute the inverse from matrix_2x3 for consistency and numerical stability
"""

import json
import datetime
import numpy as np
import cv2

from channel_profiles import physical_channel_name
from detection import MoleculeDetector


REFERENCE_CHANNEL = "532"
SUPPORTED_CHANNELS = ("488", "532", "638")


def _registration_channel_name(channel_name) -> str:
    return physical_channel_name(channel_name)


class RegistrationValidationError(ValueError):
    """Registration is missing or unsafe for quantitative extraction."""


def _to_float32_image(img):
    arr = np.asarray(img)
    if arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr.astype(np.float32))


class RegistrationModel:
    """
    Persisted convention:
    source channel raw coordinates -> 532 reference coordinates

    affine_2x3[ch] means:
        [x_ref, y_ref]^T = A * [x_src, y_src, 1]^T
    """

    def __init__(self, reference_channel=REFERENCE_CHANNEL):
        self.version = "affine_v2"
        self.reference_channel = str(reference_channel)
        self.transforms = {}  # ch_name -> dict
        self.metadata = {}

    def set_identity_for_reference(self):
        self.transforms[self.reference_channel] = {
            "matrix_2x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "inverse_matrix_2x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "type": "identity",
            "rmse": 0.0,
            "num_matches": 0,
        }

    def set_transform(self, source_channel, matrix_2x3, rmse=None, num_matches=None, transform_type="affine"):
        source_channel = _registration_channel_name(source_channel)
        M = np.asarray(matrix_2x3, dtype=np.float64).reshape(2, 3)
        M3 = np.vstack([M, [0.0, 0.0, 1.0]])
        Minv3 = np.linalg.inv(M3)
        Minv = Minv3[:2, :]

        self.transforms[source_channel] = {
            "matrix_2x3": M.tolist(),
            "inverse_matrix_2x3": Minv.tolist(),
            "type": transform_type,
            "rmse": None if rmse is None else float(rmse),
            "num_matches": None if num_matches is None else int(num_matches),
        }

    def has_channel(self, channel_name):
        return _registration_channel_name(channel_name) in self.transforms

    def get_matrix(self, source_channel):
        source_channel = _registration_channel_name(source_channel)
        if source_channel == self.reference_channel and source_channel not in self.transforms:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        item = self.transforms.get(source_channel)
        if item is None:
            return None
        return np.asarray(item["matrix_2x3"], dtype=np.float64)

    def get_inverse_matrix(self, source_channel):
        source_channel = _registration_channel_name(source_channel)
        if source_channel == self.reference_channel and source_channel not in self.transforms:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        item = self.transforms.get(source_channel)
        if item is None:
            return None
        return np.asarray(item["inverse_matrix_2x3"], dtype=np.float64)

    def map_point_to_reference(self, source_channel, x, y, *, strict=False):
        original_channel = str(source_channel)
        source_channel = _registration_channel_name(source_channel)
        if source_channel == self.reference_channel:
            return float(x), float(y)

        M = self.get_matrix(source_channel)
        if M is None:
            if strict:
                raise RegistrationValidationError(
                    f"Registration transform is missing for source channel {original_channel}"
                )
            return float(x), float(y)

        out = M @ np.array([float(x), float(y), 1.0], dtype=np.float64)
        return float(out[0]), float(out[1])

    def map_point_from_reference(self, target_channel, x_ref, y_ref, *, strict=False):
        """
        Used by the quantitative path:
        map reference coordinates (532) back to the target channel raw image coordinates.
        """
        original_channel = str(target_channel)
        target_channel = _registration_channel_name(target_channel)
        if target_channel == self.reference_channel:
            return float(x_ref), float(y_ref)

        Minv = self.get_inverse_matrix(target_channel)
        if Minv is None:
            if strict:
                raise RegistrationValidationError(
                    f"Registration inverse transform is missing for target channel {original_channel}"
                )
            return float(x_ref), float(y_ref)

        out = Minv @ np.array([float(x_ref), float(y_ref), 1.0], dtype=np.float64)
        return float(out[0]), float(out[1])

    def to_dict(self):
        return {
            "version": self.version,
            "reference_channel": self.reference_channel,
            "transforms": self.transforms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d):
        model = cls(reference_channel=d.get("reference_channel", REFERENCE_CHANNEL))
        model.version = d.get("version", "affine_v2")
        model.metadata = dict(d.get("metadata", {}))

        raw_transforms = dict(d.get("transforms", {}))
        model.transforms = {}

        for ch, item in raw_transforms.items():
            ch = str(ch)
            M = np.asarray(item.get("matrix_2x3"), dtype=np.float64).reshape(2, 3)
            ttype = item.get("type", "affine")
            rmse = item.get("rmse", None)
            n = item.get("num_matches", None)
            # Recompute the inverse matrix every time instead of trusting the JSON payload.
            model.set_transform(ch, M, rmse=rmse, num_matches=n, transform_type=ttype)

        if model.reference_channel not in model.transforms:
            model.set_identity_for_reference()

        return model


class RegistrationIO:
    @staticmethod
    def save_json(path, model: RegistrationModel):
        data = model.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path) -> RegistrationModel:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return RegistrationConverter.from_legacy_or_new(d)

    @staticmethod
    def describe(model: RegistrationModel):
        lines = []
        lines.append(f"Version: {model.version}")
        lines.append(f"Reference channel: {model.reference_channel}")

        for ch in SUPPORTED_CHANNELS:
            if not model.has_channel(ch):
                continue
            item = model.transforms[ch]
            ttype = item.get("type", "unknown")
            rmse = item.get("rmse", None)
            n = item.get("num_matches", None)
            if rmse is None:
                lines.append(f"{ch} -> {model.reference_channel}: {ttype}")
            else:
                lines.append(f"{ch} -> {model.reference_channel}: {ttype}, RMSE={rmse:.3f}px, matches={n}")
        return "\n".join(lines)


class RegistrationConverter:
    @staticmethod
    def from_legacy_or_new(d) -> RegistrationModel:
        """
        Compatibility:
        1) New affine_v2 format
        2) Legacy offsets format:
           {"offsets": {"638": [dx, dy], "488": [dx, dy]}, ...}
           This assumes a source -> reference translation:
               x_ref = x_src + dx
               y_ref = y_src + dy
        """
        if "transforms" in d:
            return RegistrationModel.from_dict(d)

        model = RegistrationModel(reference_channel=d.get("reference_channel", REFERENCE_CHANNEL))
        model.version = "legacy_offsets_upconverted"
        model.metadata = {
            "legacy_source": True,
            "date": d.get("date", ""),
            "num_channels": d.get("num_channels", None),
            "num_beads": d.get("num_beads", None),
        }
        model.set_identity_for_reference()

        offsets = d.get("offsets", {})
        for ch, off in offsets.items():
            dx = float(off[0])
            dy = float(off[1])
            M = np.array([[1.0, 0.0, dx],
                          [0.0, 1.0, dy]], dtype=np.float64)
            model.set_transform(ch, M, rmse=None, num_matches=None, transform_type="translation")

        return model


class RegistrationDisplayApplier:
    @staticmethod
    def warp_image_to_reference(image_2d, source_channel, reg_model: RegistrationModel, out_shape_hw):
        """
        Display path: warp the source-channel image into the 532 coordinate system.
        Bilinear interpolation is used only for display, overlay, and inspection.
        """
        if reg_model is None:
            return np.asarray(image_2d)

        source_channel = _registration_channel_name(source_channel)
        if source_channel == reg_model.reference_channel:
            return np.asarray(image_2d)

        M = reg_model.get_matrix(source_channel)
        if M is None:
            return np.asarray(image_2d)

        h, w = out_shape_hw
        warped = cv2.warpAffine(
            np.asarray(image_2d),
            M.astype(np.float32),
            (int(w), int(h)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return warped


class RegistrationCoordinateMapper:
    @staticmethod
    def reference_to_channel_raw(x_ref, y_ref, target_channel, reg_model: RegistrationModel, *, strict=False):
        target_key = _registration_channel_name(target_channel)
        if reg_model is None:
            if strict and target_key != REFERENCE_CHANNEL:
                raise RegistrationValidationError(
                    f"Registration is required to map reference coordinates to channel {target_channel}"
                )
            return float(x_ref), float(y_ref)
        return reg_model.map_point_from_reference(target_channel, x_ref, y_ref, strict=strict)

    @staticmethod
    def channel_raw_to_reference(x, y, source_channel, reg_model: RegistrationModel, *, strict=False):
        source_key = _registration_channel_name(source_channel)
        if reg_model is None:
            if strict and source_key != REFERENCE_CHANNEL:
                raise RegistrationValidationError(
                    f"Registration is required to map channel {source_channel} coordinates to reference"
                )
            return float(x), float(y)
        return reg_model.map_point_to_reference(source_channel, x, y, strict=strict)


def _resolve_detect_channel_name(detect_channel, channel_names):
    if detect_channel is None:
        return None
    if isinstance(detect_channel, str):
        return str(detect_channel)
    try:
        idx = int(detect_channel)
    except (TypeError, ValueError):
        return str(detect_channel)
    if idx < 0 or idx >= len(channel_names):
        raise RegistrationValidationError(
            f"Detection channel index {idx} is outside available channels: {', '.join(channel_names)}"
        )
    return str(channel_names[idx])


def _matrix_is_valid(matrix) -> bool:
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except Exception:
        return False
    return arr.shape == (2, 3) and bool(np.all(np.isfinite(arr)))


def validate_quantitative_registration(registration_params, channel_names, detect_channel=None) -> RegistrationModel | None:
    """
    Quantitative extraction must not silently reuse reference coordinates for
    shifted channels. Multi-channel outputs require complete transforms.
    """
    names = [str(ch) for ch in list(channel_names or [])]
    detect_name = _resolve_detect_channel_name(detect_channel, names)
    required = {
        _registration_channel_name(ch)
        for ch in names
        if _registration_channel_name(ch) != REFERENCE_CHANNEL
    }
    if detect_name and _registration_channel_name(detect_name) != REFERENCE_CHANNEL:
        required.add(_registration_channel_name(detect_name))

    if not required:
        return (
            RegistrationConverter.from_legacy_or_new(registration_params)
            if registration_params is not None
            else None
        )

    if registration_params is None:
        raise RegistrationValidationError(
            "Registration is required for multi-channel quantitative output. "
            f"Missing transforms for: {', '.join(sorted(required))}"
        )

    try:
        model = RegistrationConverter.from_legacy_or_new(registration_params)
    except Exception as exc:
        raise RegistrationValidationError(f"Registration parameters could not be parsed: {exc}") from exc

    if str(model.reference_channel) != REFERENCE_CHANNEL:
        raise RegistrationValidationError(
            f"Registration reference channel must be {REFERENCE_CHANNEL}, got {model.reference_channel}"
        )

    missing = []
    invalid = []
    for ch in sorted(required):
        if not model.has_channel(ch):
            missing.append(ch)
            continue
        if not _matrix_is_valid(model.get_matrix(ch)) or not _matrix_is_valid(model.get_inverse_matrix(ch)):
            invalid.append(ch)

    parts = []
    if missing:
        parts.append(f"missing transforms for: {', '.join(missing)}")
    if invalid:
        parts.append(f"invalid transforms for: {', '.join(invalid)}")
    if parts:
        raise RegistrationValidationError("Registration is incomplete for quantitative output: " + "; ".join(parts))

    return model


class RegistrationEstimator:
    """
    Automatically estimate affine transforms from bead calibration images:
    - Input: dict of calibration images keyed by channel name, e.g. {"532": img532, "638": img638, "488": img488}
    - Output: RegistrationModel
    """

    def __init__(self):
        self.detector = MoleculeDetector()

    def _detect_beads(self, image):
        positions, stats = self.detector.detect_molecules(
            image,
            sigma=3.5,
            min_snr=3.0,
            min_r2=0.0,
            min_distance=6,
            auto_background=True,
            bg_radius=None
        )
        pts = np.asarray(positions, dtype=np.float32)
        return pts, stats

    @staticmethod
    def _mutual_nearest_pairs(src_pts, ref_pts, max_distance=12.0):
        if len(src_pts) == 0 or len(ref_pts) == 0:
            return np.empty((0, 2), dtype=np.int32)

        src = np.asarray(src_pts, dtype=np.float32)
        ref = np.asarray(ref_pts, dtype=np.float32)

        d2 = np.sum((src[:, None, :] - ref[None, :, :]) ** 2, axis=2)

        src_to_ref = np.argmin(d2, axis=1)
        ref_to_src = np.argmin(d2, axis=0)

        pairs = []
        max_d2 = float(max_distance * max_distance)
        for si, ri in enumerate(src_to_ref):
            if ref_to_src[ri] == si and d2[si, ri] <= max_d2:
                pairs.append((si, ri))

        if not pairs:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(pairs, dtype=np.int32)

    @staticmethod
    def _estimate_affine_ransac(src_pts, ref_pts):
        if len(src_pts) < 3 or len(ref_pts) < 3:
            raise ValueError("At least 3 matched point pairs are required to estimate an affine transform")

        M, inliers = cv2.estimateAffine2D(
            src_pts.astype(np.float32),
            ref_pts.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=5000,
            confidence=0.995,
            refineIters=50
        )
        if M is None:
            raise RuntimeError("cv2.estimateAffine2D fitting failed")

        pred = cv2.transform(src_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        err = np.sqrt(np.sum((pred - ref_pts) ** 2, axis=1))
        if inliers is not None:
            mask = inliers.ravel().astype(bool)
            if np.any(mask):
                rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
                num_matches = int(np.sum(mask))
            else:
                rmse = float(np.sqrt(np.mean(err ** 2)))
                num_matches = int(len(src_pts))
        else:
            rmse = float(np.sqrt(np.mean(err ** 2)))
            num_matches = int(len(src_pts))

        return M, rmse, num_matches

    def estimate_from_bead_images(self, image_by_channel: dict):
        if REFERENCE_CHANNEL not in image_by_channel:
            raise ValueError("Bead calibration images are missing the 532 reference channel")

        model = RegistrationModel(reference_channel=REFERENCE_CHANNEL)
        model.set_identity_for_reference()
        model.metadata = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "bead_affine_auto",
            "reference_channel": REFERENCE_CHANNEL,
        }

        ref_img = _to_float32_image(image_by_channel[REFERENCE_CHANNEL])
        ref_pts, ref_stats = self._detect_beads(ref_img)

        model.metadata["ref_beads_detected"] = int(len(ref_pts))

        for ch in SUPPORTED_CHANNELS:
            if ch == REFERENCE_CHANNEL:
                continue
            if ch not in image_by_channel:
                continue

            src_img = _to_float32_image(image_by_channel[ch])
            src_pts, src_stats = self._detect_beads(src_img)

            pairs = self._mutual_nearest_pairs(src_pts, ref_pts, max_distance=12.0)
            if len(pairs) < 3:
                raise RuntimeError(f"Channel {ch} does not have enough matched points with 532 to estimate an affine transform")

            src_match = src_pts[pairs[:, 0]]
            ref_match = ref_pts[pairs[:, 1]]

            M, rmse, num_matches = self._estimate_affine_ransac(src_match, ref_match)
            model.set_transform(ch, M, rmse=rmse, num_matches=num_matches, transform_type="affine")

        return model
