# data_model.py
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)

ALEX_LOGICAL_CHANNELS = (
    "532ex_532",
    "532ex_638",
    "488ex_532",
    "488ex_488",
)

ALEX_ALPHA_DEFAULT = 0.34
ALEX_BETA_DEFAULT = 0.0
ALEX_GAMMA_DEFAULT = 1.0

ALEX_CY3_TOTAL_LANE = "ALEX_CY3_TOTAL_532EX"
ALEX_S_CORR_LANE = "ALEX_S_CORR"
ALEX_AF488_INDEX_LANE = "ALEX_AF488_INDEX"
ALEX_CY3_ALIVE_LANE = "ALEX_CY3_ALIVE"

ALEX_CORRECTION_COLUMN_NAMES = {
    ALEX_CY3_TOTAL_LANE: "Cy3_total_532ex",
    ALEX_CY3_ALIVE_LANE: "Cy3_alive",
    ALEX_S_CORR_LANE: "S_488ex_532_corr",
    ALEX_AF488_INDEX_LANE: "AF488_index",
}

ALEX_CHANNEL_METADATA = {
    "532ex_532": {"label": "532ex->532", "excitation": "532", "emission": "532", "physical": "532", "color": "#00d26a"},
    "532ex_638": {"label": "532ex->638", "excitation": "532", "emission": "638", "physical": "638", "color": "#ff5555"},
    "488ex_532": {"label": "488ex->532", "excitation": "488", "emission": "532", "physical": "532", "color": "#f5c542"},
    "488ex_488": {"label": "488ex->488", "excitation": "488", "emission": "488", "physical": "488", "color": "#55aaff"},
}

STANDARD_CHANNEL_METADATA = {
    "488": {"label": "488nm", "excitation": None, "emission": "488", "physical": "488", "color": "blue"},
    "532": {"label": "532nm", "excitation": None, "emission": "532", "physical": "532", "color": "green"},
    "638": {"label": "638nm", "excitation": None, "emission": "638", "physical": "638", "color": "red"},
}

ALEX_DERIVED_LANES = {
    "ALEX_E_532EX": {
        "label": "E 532ex",
        "color": "purple",
        "kind": "ratio",
        "numerator": ("532ex_638",),
        "denominator": ("532ex_532", "532ex_638"),
    },
    "ALEX_R_488EX": {
        "label": "R 488ex",
        "color": "#8a5a00",
        "kind": "ratio",
        "numerator": ("488ex_532",),
        "denominator": ("488ex_488", "488ex_532"),
    },
    "ALEX_TOTAL_532EX": {
        "label": "Total 532ex",
        "color": "#444444",
        "kind": "sum",
        "terms": ("532ex_532", "532ex_638"),
    },
    "ALEX_TOTAL_488EX": {
        "label": "Total 488ex",
        "color": "#666666",
        "kind": "sum",
        "terms": ("488ex_488", "488ex_532"),
    },
    ALEX_CY3_TOTAL_LANE: {
        "label": "Cy3 total 532ex",
        "color": "#444444",
        "kind": "alex_correction",
        "column": ALEX_CORRECTION_COLUMN_NAMES[ALEX_CY3_TOTAL_LANE],
    },
    ALEX_S_CORR_LANE: {
        "label": "S corrected",
        "color": "#d18f00",
        "kind": "alex_correction",
        "column": ALEX_CORRECTION_COLUMN_NAMES[ALEX_S_CORR_LANE],
    },
    ALEX_AF488_INDEX_LANE: {
        "label": "AF488 index",
        "color": "#1b66cc",
        "kind": "alex_correction",
        "column": ALEX_CORRECTION_COLUMN_NAMES[ALEX_AF488_INDEX_LANE],
    },
}

ALEX_CORRECTION_DISPLAY_LANES = (
    ALEX_CY3_TOTAL_LANE,
    ALEX_S_CORR_LANE,
    ALEX_AF488_INDEX_LANE,
)

ALEX_DETECTION_DERIVED_CHANNELS = (ALEX_AF488_INDEX_LANE,)


def _std1(values) -> float:
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    denom = n - 1 if n > 1 else n
    centered = x - (np.sum(x) / n)
    return float(np.sqrt(np.sum(centered ** 2) / denom)) if denom else 0.0


def _gradient1(values):
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)
    if n == 2:
        delta = x[1] - x[0]
        return np.array([delta, delta], dtype=float)

    result = np.empty_like(x, dtype=float)
    result[0] = x[1] - x[0]
    result[1:-1] = 0.5 * (x[2:] - x[:-2])
    result[-1] = x[-1] - x[-2]
    return result


def _moving_median(values, window: int):
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).median().to_numpy()


def calc_fret_lifetime_index(total, tau=9, nstd=8) -> int:
    values = np.asarray(total, dtype=float)
    n_frames = values.size
    if n_frames == 0:
        return 0
    if n_frames < 3:
        return n_frames - 1

    filtered = _moving_median(values, int(tau))
    gradient = _gradient1(filtered)
    initial_threshold = (np.sum(gradient) / n_frames) + 6 * _std1(gradient)
    clean_gradient = gradient[np.abs(gradient) <= initial_threshold]
    if clean_gradient.size == 0:
        return n_frames - 1

    drop_threshold = (np.sum(clean_gradient) / clean_gradient.size) - float(nstd) * _std1(clean_gradient)
    if drop_threshold < 0:
        candidate_positions = np.flatnonzero(gradient[:-1] <= drop_threshold)
    else:
        candidate_positions = np.flatnonzero(gradient[:-1] < drop_threshold)
    if candidate_positions.size == 0:
        return n_frames - 1

    return int(max(1, candidate_positions[-1]))


def threshold_total_alive_mask(total, tau=9, nstd=8, nbk=100, blink_nstd=4):
    values = np.asarray(total, dtype=float)
    n_frames = values.size
    alive = np.ones(n_frames, dtype=bool)
    if n_frames == 0:
        return alive

    safe_total = np.where(np.isfinite(values), values, 0.0)
    life_idx = calc_fret_lifetime_index(safe_total, tau=tau, nstd=nstd)
    life_idx = int(max(0, min(n_frames - 1, life_idx)))

    alive[life_idx:] = False

    background_start = life_idx + 5
    background_end = min(background_start + int(nbk) + 1, n_frames)
    background = safe_total[background_start:background_end]
    if background.size >= 10:
        threshold = float(blink_nstd) * _std1(background)
        dark_range = safe_total[:life_idx + 1] <= threshold
        alive_before_bleach = alive[:life_idx + 1]
        alive_before_bleach[dark_range] = False

    return alive


def compute_alex_correction_columns(
    df_subset: pd.DataFrame,
    signal_columns: Dict[str, str],
    *,
    alpha: float = ALEX_ALPHA_DEFAULT,
    beta: float = ALEX_BETA_DEFAULT,
    gamma: float = ALEX_GAMMA_DEFAULT,
) -> Optional[Dict[str, np.ndarray]]:
    required = {
        "488ex_488": signal_columns.get("488ex_488"),
        "488ex_532": signal_columns.get("488ex_532"),
        "532ex_532": signal_columns.get("532ex_532"),
        "532ex_638": signal_columns.get("532ex_638"),
    }
    if any(not col or col not in df_subset.columns for col in required.values()):
        return None

    a = df_subset[required["488ex_488"]].astype(float).to_numpy()
    c = df_subset[required["488ex_532"]].astype(float).to_numpy()
    d = df_subset[required["532ex_532"]].astype(float).to_numpy()
    e = df_subset[required["532ex_638"]].astype(float).to_numpy()

    cy3_total = d + e
    cy3_alive = threshold_total_alive_mask(cy3_total)
    s_corr = c - float(alpha) * a - float(beta) * d
    s_corr_alive = np.array(s_corr, dtype=float, copy=True)
    s_corr_alive[~cy3_alive] = 0.0
    af488_index = a + float(gamma) * s_corr_alive

    return {
        ALEX_CY3_TOTAL_LANE: cy3_total,
        ALEX_CY3_ALIVE_LANE: cy3_alive,
        ALEX_S_CORR_LANE: s_corr,
        ALEX_AF488_INDEX_LANE: af488_index,
    }


@dataclass
class ChannelDescriptor:
    key: str
    column: str
    label: str
    excitation: Optional[str] = None
    emission: Optional[str] = None
    physical_channel: Optional[str] = None
    color: Optional[str] = None
    role: str = "raw"


@dataclass
class LoadedData:
    df: pd.DataFrame
    filepath: str
    separator: str
    is_three_color: bool
    id_list: List
    original_signal_columns: Dict[str, str]  # e.g. {"488": "Net_488", "532": "Intensity_532"}
    profile: str = "standard"
    channel_order: Tuple[str, ...] = ()
    channel_descriptors: Dict[str, ChannelDescriptor] = None


class DataModel:
    TIME_CANDIDATES = ['Time_sec', 'Time_s', 'Time']
    ID_CANDIDATES = ['ROI_ID', 'ID']
    FRET_TAU = 9
    FRET_NSTD = 8
    FRET_NBK = 100
    FRET_BLINK_NSTD = 4

    def __init__(self):
        self.loaded: Optional[LoadedData] = None

    def load_csv(self, filepath: str) -> LoadedData:
        logger.info("Loading file: %s", filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            separator = '\t' if '\t' in first_line else ','

        df = pd.read_csv(filepath, sep=separator)
        df.columns = df.columns.str.strip()

        logger.info("Detected columns: %s", df.columns.tolist())

        time_col = self._find_first_existing(df.columns.tolist(), self.TIME_CANDIDATES)
        if not time_col:
            raise ValueError("Time column not found (Time_sec / Time_s / Time)")
        if time_col != 'Time_sec':
            df.rename(columns={time_col: 'Time_sec'}, inplace=True)

        id_col = self._find_first_existing(df.columns.tolist(), self.ID_CANDIDATES)
        if not id_col:
            raise ValueError("ID column not found (ROI_ID / ID)")
        if id_col != 'ROI_ID':
            df.rename(columns={id_col: 'ROI_ID'}, inplace=True)

        signal_cols, profile, channel_order, descriptors = self._detect_signal_columns(df.columns.tolist())
        if profile.startswith("standard") and not signal_cols.get('532') and not signal_cols.get('638'):
            raise ValueError(
                "Intensity columns not found "
                "(Net_532/Net_638, Intensity_532/Intensity_638, or ALEX Net_*ex_* columns)"
            )

        is_three_color = signal_cols.get('488') is not None

        if df['Time_sec'].isna().any():
            raise ValueError("Time column contains NaN values. Clean the data first.")
        if df['ROI_ID'].isna().any():
            raise ValueError("ID column contains NaN values. Clean the data first.")

        df = df.sort_values(by=['ROI_ID', 'Time_sec']).reset_index(drop=True)

        id_list = sorted(df['ROI_ID'].unique().tolist())

        loaded = LoadedData(
            df=df,
            filepath=filepath,
            separator=separator,
            is_three_color=is_three_color,
            id_list=id_list,
            original_signal_columns=signal_cols,
            profile=profile,
            channel_order=tuple(channel_order),
            channel_descriptors=descriptors,
        )
        self.loaded = loaded

        logger.info(
            "File loaded successfully. sep=%s, profile=%s, three_color=%s, ids=%d",
            "TAB" if separator == '\t' else ",",
            profile,
            is_three_color,
            len(id_list)
        )

        return loaded
    def _find_first_existing(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        return None

    def _signal_column_for_channel(self, columns: List[str], channel: str) -> Optional[str]:
        for candidate in (f'Net_{channel}', f'Intensity_{channel}', channel):
            if candidate in columns:
                return candidate
        return None

    def _detect_signal_columns(
        self, columns: List[str]
    ) -> Tuple[Dict[str, Optional[str]], str, List[str], Dict[str, ChannelDescriptor]]:
        alex_columns = {
            ch: self._signal_column_for_channel(columns, ch)
            for ch in ALEX_LOGICAL_CHANNELS
        }
        has_alex_columns = any(col is not None for col in alex_columns.values())
        if has_alex_columns:
            missing = [ch for ch, col in alex_columns.items() if col is None]
            if missing:
                missing_text = ", ".join(f"Net_{ch}" for ch in missing)
                raise ValueError(f"ALEX intensity columns are incomplete. Missing: {missing_text}")

            descriptors = {}
            for ch in ALEX_LOGICAL_CHANNELS:
                meta = ALEX_CHANNEL_METADATA[ch]
                descriptors[ch] = ChannelDescriptor(
                    key=ch,
                    column=alex_columns[ch],
                    label=meta["label"],
                    excitation=meta["excitation"],
                    emission=meta["emission"],
                    physical_channel=meta["physical"],
                    color=meta["color"],
                )
            return alex_columns, "alex_4ch", list(ALEX_LOGICAL_CHANNELS), descriptors

        result = {'488': None, '532': None, '638': None}
        for ch in ['488', '532', '638']:
            result[ch] = self._signal_column_for_channel(columns, ch)

        channel_order = [ch for ch in ['488', '532', '638'] if result.get(ch) is not None]
        descriptors = {}
        for ch in channel_order:
            meta = STANDARD_CHANNEL_METADATA[ch]
            descriptors[ch] = ChannelDescriptor(
                key=ch,
                column=result[ch],
                label=meta["label"],
                excitation=meta["excitation"],
                emission=meta["emission"],
                physical_channel=meta["physical"],
                color=meta["color"],
            )

        profile = "standard_3ch" if result.get('488') is not None else "standard_2ch"
        return result, profile, channel_order, descriptors

    def get_signal_column(self, channel: str) -> Optional[str]:
        if not self.loaded:
            return None
        return self.loaded.original_signal_columns.get(channel)

    def get_channel_descriptor(self, channel: str) -> Optional[ChannelDescriptor]:
        if not self.loaded or not self.loaded.channel_descriptors:
            return None
        return self.loaded.channel_descriptors.get(channel)

    def get_channel_label(self, channel: str) -> str:
        descriptor = self.get_channel_descriptor(channel)
        if descriptor:
            return descriptor.label
        spec = ALEX_DERIVED_LANES.get(channel)
        if spec:
            return spec.get("label", str(channel))
        return str(channel)

    def get_channel_color(self, channel: str) -> str:
        descriptor = self.get_channel_descriptor(channel)
        if descriptor and descriptor.color:
            return descriptor.color
        spec = ALEX_DERIVED_LANES.get(channel)
        if spec and spec.get("color"):
            return spec["color"]
        return STANDARD_CHANNEL_METADATA.get(channel, {}).get("color", "black")

    def is_alex_profile(self) -> bool:
        return bool(self.loaded and self.loaded.profile == "alex_4ch")

    @staticmethod
    def _std1(values) -> float:
        x = np.asarray(values, dtype=float)
        n = x.size
        if n == 0:
            return 0.0
        denom = n - 1 if n > 1 else n
        centered = x - (np.sum(x) / n)
        return float(np.sqrt(np.sum(centered ** 2) / denom)) if denom else 0.0

    @staticmethod
    def _gradient1(values):
        x = np.asarray(values, dtype=float)
        n = x.size
        if n == 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.array([0.0], dtype=float)
        if n == 2:
            delta = x[1] - x[0]
            return np.array([delta, delta], dtype=float)

        result = np.empty_like(x, dtype=float)
        result[0] = x[1] - x[0]
        result[1:-1] = 0.5 * (x[2:] - x[:-2])
        result[-1] = x[-1] - x[-2]
        return result

    @staticmethod
    def _moving_median(values, window: int):
        x = np.asarray(values, dtype=float)
        if x.size == 0:
            return x
        return pd.Series(x).rolling(window=window, center=True, min_periods=1).median().to_numpy()

    def _calc_fret_lifetime_index(self, total) -> int:
        values = np.asarray(total, dtype=float)
        n_frames = values.size
        if n_frames == 0:
            return 0
        if n_frames < 3:
            return n_frames - 1

        filtered = self._moving_median(values, self.FRET_TAU)
        gradient = self._gradient1(filtered)
        initial_threshold = (np.sum(gradient) / n_frames) + 6 * self._std1(gradient)
        clean_gradient = gradient[np.abs(gradient) <= initial_threshold]
        if clean_gradient.size == 0:
            return n_frames - 1

        drop_threshold = (np.sum(clean_gradient) / clean_gradient.size) - self.FRET_NSTD * self._std1(clean_gradient)
        if drop_threshold < 0:
            candidate_positions = np.flatnonzero(gradient[:-1] <= drop_threshold)
        else:
            candidate_positions = np.flatnonzero(gradient[:-1] < drop_threshold)
        if candidate_positions.size == 0:
            return n_frames - 1

        return int(max(1, candidate_positions[-1]))

    def _threshold_total_alive_mask(self, total):
        values = np.asarray(total, dtype=float)
        n_frames = values.size
        alive = np.ones(n_frames, dtype=bool)
        if n_frames == 0:
            return alive

        safe_total = np.where(np.isfinite(values), values, 0.0)
        life_idx = self._calc_fret_lifetime_index(safe_total)
        life_idx = int(max(0, min(n_frames - 1, life_idx)))

        alive[life_idx:] = False

        background_start = life_idx + 5
        background_end = min(background_start + self.FRET_NBK + 1, n_frames)
        background = safe_total[background_start:background_end]
        if background.size >= 10:
            threshold = self.FRET_BLINK_NSTD * self._std1(background)
            dark_range = safe_total[:life_idx + 1] <= threshold
            alive_before_bleach = alive[:life_idx + 1]
            alive_before_bleach[dark_range] = False

        return alive

    def compute_fret(self, df_subset: pd.DataFrame):
        if self.is_alex_profile():
            donor_col = self.get_signal_column('532ex_532')
            acceptor_col = self.get_signal_column('532ex_638')
        else:
            donor_col = self.get_signal_column('532')
            acceptor_col = self.get_signal_column('638')
        if (
            not donor_col or not acceptor_col
            or donor_col not in df_subset.columns
            or acceptor_col not in df_subset.columns
        ):
            return None

        donor = df_subset[donor_col].astype(float).to_numpy()
        acceptor = df_subset[acceptor_col].astype(float).to_numpy()
        total = donor + acceptor

        fret = np.zeros_like(total, dtype=float)
        valid = total != 0
        np.divide(acceptor, total, out=fret, where=valid)
        fret[~np.isfinite(fret)] = 0.0
        fret[~self._threshold_total_alive_mask(total)] = 0.0
        return fret

    def compute_alex_correction_columns(
        self,
        df_subset: pd.DataFrame,
        alpha: float = ALEX_ALPHA_DEFAULT,
        beta: float = ALEX_BETA_DEFAULT,
        gamma: float = ALEX_GAMMA_DEFAULT,
    ):
        if not self.loaded or not self.is_alex_profile():
            return None
        return compute_alex_correction_columns(
            df_subset,
            self.loaded.original_signal_columns,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    def compute_derived_signal(
        self,
        lane: str,
        df_subset: pd.DataFrame,
        alpha: float = ALEX_ALPHA_DEFAULT,
        beta: float = ALEX_BETA_DEFAULT,
        gamma: float = ALEX_GAMMA_DEFAULT,
    ):
        if lane in {"FRET", "ALEX_E_532EX"}:
            return self.compute_fret(df_subset)
        spec = ALEX_DERIVED_LANES.get(lane)
        if not spec:
            return None

        def values_for(channel_key):
            col = self.get_signal_column(channel_key)
            if not col or col not in df_subset.columns:
                return None
            return df_subset[col].astype(float).to_numpy()

        if spec["kind"] == "sum":
            arrays = [values_for(ch) for ch in spec["terms"]]
            if any(arr is None for arr in arrays):
                return None
            total = np.zeros_like(arrays[0], dtype=float)
            for arr in arrays:
                total = total + arr
            return total

        if spec["kind"] == "ratio":
            numerator_arrays = [values_for(ch) for ch in spec["numerator"]]
            denominator_arrays = [values_for(ch) for ch in spec["denominator"]]
            if any(arr is None for arr in numerator_arrays + denominator_arrays):
                return None
            numerator = np.zeros_like(numerator_arrays[0], dtype=float)
            denominator = np.zeros_like(denominator_arrays[0], dtype=float)
            for arr in numerator_arrays:
                numerator = numerator + arr
            for arr in denominator_arrays:
                denominator = denominator + arr
            ratio = np.zeros_like(denominator, dtype=float)
            np.divide(numerator, denominator, out=ratio, where=denominator != 0)
            ratio[~np.isfinite(ratio)] = 0.0
            return ratio

        if spec["kind"] == "alex_correction":
            correction = self.compute_alex_correction_columns(
                df_subset,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            if correction is None:
                return None
            return correction.get(lane)

        return None

    def get_signal_values(
        self,
        channel: str,
        df_subset: pd.DataFrame,
        alpha: float = ALEX_ALPHA_DEFAULT,
        beta: float = ALEX_BETA_DEFAULT,
        gamma: float = ALEX_GAMMA_DEFAULT,
    ):
        col = self.get_signal_column(channel)
        if col and col in df_subset.columns:
            return df_subset[col].astype(float).to_numpy()
        return self.compute_derived_signal(
            channel,
            df_subset,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    def get_detection_channels(self) -> List[str]:
        channels = list(self.get_available_channels())
        if self.is_alex_profile():
            channels.extend(ALEX_DETECTION_DERIVED_CHANNELS)
        return channels

    def estimate_alex_beta(self, alpha: float = ALEX_ALPHA_DEFAULT, quantile: float = 0.25):
        if not self.loaded or not self.is_alex_profile():
            return None

        col_a = self.get_signal_column("488ex_488")
        col_c = self.get_signal_column("488ex_532")
        col_d = self.get_signal_column("532ex_532")
        col_e = self.get_signal_column("532ex_638")
        if not col_a or not col_c or not col_d or not col_e:
            return None

        ratio_arrays = []
        count = 0
        for _roi_id, df_id in self.loaded.df.groupby("ROI_ID", sort=False):
            a = df_id[col_a].to_numpy(dtype=float, copy=False)
            c = df_id[col_c].to_numpy(dtype=float, copy=False)
            d = df_id[col_d].to_numpy(dtype=float, copy=False)
            e = df_id[col_e].to_numpy(dtype=float, copy=False)

            alive = threshold_total_alive_mask(d + e)
            valid = alive & np.isfinite(a) & np.isfinite(c) & np.isfinite(d) & (np.abs(d) > 1e-9)
            if not np.any(valid):
                continue

            ratios = (c[valid] - float(alpha) * a[valid]) / d[valid]
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                ratio_arrays.append(ratios)
                count += int(ratios.size)

        if not ratio_arrays:
            return None
        values = ratio_arrays[0] if len(ratio_arrays) == 1 else np.concatenate(ratio_arrays)
        beta = float(np.quantile(values, float(quantile)))
        return {
            "beta": beta,
            "count": count,
            "quantile": float(quantile),
        }

    def get_roi_df(self, roi_id):
        if not self.loaded:
            return None
        return self.loaded.df[self.loaded.df['ROI_ID'] == roi_id].copy()

    def get_available_channels(self) -> List[str]:
        if not self.loaded:
            return []
        if self.loaded.channel_order:
            return [ch for ch in self.loaded.channel_order if self.loaded.original_signal_columns.get(ch) is not None]
        return [ch for ch, col in self.loaded.original_signal_columns.items() if col is not None]

    def get_raw_signal_columns(self) -> List[str]:
        if not self.loaded:
            return []
        columns = []
        for ch in self.get_available_channels():
            col = self.loaded.original_signal_columns.get(ch)
            if col:
                columns.append(col)
        return columns

    def get_profile_label(self) -> str:
        if not self.loaded:
            return "No data"
        labels = {
            "alex_4ch": "ALEX four-channel",
            "standard_3ch": "Three-color",
            "standard_2ch": "Two-color",
            "standard": "Standard",
        }
        return labels.get(self.loaded.profile, self.loaded.profile)

    def signal_signature(self) -> Dict[str, str]:
        if not self.loaded:
            return {}
        return {
            "profile": self.loaded.profile,
            "channels": list(self.get_available_channels()),
            "columns": [self.loaded.original_signal_columns[ch] for ch in self.get_available_channels()],
        }

    def export_original_columns_subset(self, df_subset: pd.DataFrame) -> pd.DataFrame:
        if not self.loaded:
            return df_subset.copy()

        cols = ['Time_sec', 'ROI_ID']
        for col in self.get_raw_signal_columns():
            if col in df_subset.columns:
                cols.append(col)
        return df_subset[cols].copy()

    @property
    def df(self):
        return self.loaded.df if self.loaded else None

    @property
    def filepath(self):
        return self.loaded.filepath if self.loaded else None

    @property
    def is_three_color(self):
        return self.loaded.is_three_color if self.loaded else False

    @property
    def profile(self):
        return self.loaded.profile if self.loaded else ""

    @property
    def id_list(self):
        return self.loaded.id_list if self.loaded else []
