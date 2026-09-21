# detection.py
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class DetectionParams:
    inj_time: float = 0.0
    min_dwell: float = 0.5
    delta_mad: float = 3.0
    pen_mad: float = 3.0  # currently unused
    merge_gap: float = 0.2
    hysteresis_ratio: float = 0.85

    mode: str = "single"  # "single" or "dual"
    primary_channel: str = "532"
    auxiliary_channel: Optional[str] = None

    aux_delta_mad: float = 3.0
    overlap_ratio_threshold: float = 0.0

    alex_alpha: float = 0.34
    alex_beta: float = 0.0
    alex_gamma: float = 1.0


@dataclass
class EventRecord:
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    dwell_time: float
    right_censored: bool = False

    event_key: Tuple[int, int] = field(default_factory=tuple)

    classification: str = "candidate"  # "candidate" or "dual_supported"
    aux_overlap_ratio: float = 0.0
    aux_supported: bool = False

    primary_peak: Optional[float] = None
    primary_mean: Optional[float] = None
    aux_peak: Optional[float] = None
    aux_mean: Optional[float] = None

    source: str = "auto"               # auto / manual_added / manual_modified
    manual_status: str = ""            # "" / manual_added / manual_bound / manual_boundary_modified
    created_by: str = "auto"           # auto / user

    def to_dict(self):
        return {
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'dwell_time': self.dwell_time,
            'right_censored': self.right_censored,
            'event_key': self.event_key,
            'classification': self.classification,
            'aux_overlap_ratio': self.aux_overlap_ratio,
            'aux_supported': self.aux_supported,
            'primary_peak': self.primary_peak,
            'primary_mean': self.primary_mean,
            'aux_peak': self.aux_peak,
            'aux_mean': self.aux_mean,
            'source': self.source,
            'manual_status': self.manual_status,
            'created_by': self.created_by,
        }


class EventDetector:
    def __init__(self, data_model):
        self.data_model = data_model

    def detect_events_for_roi(self, roi_id, params: DetectionParams, override: Optional[dict] = None) -> List[EventRecord]:
        df_id = self.data_model.get_roi_df(roi_id)
        if df_id is None or df_id.empty:
            return []

        effective = self._apply_override(params, override)

        primary_signal = self._get_channel_signal(df_id, effective.primary_channel, effective)
        if primary_signal is None or len(primary_signal) == 0:
            return []

        time = df_id['Time_sec'].values
        candidate_events = self._detect_candidate_events(time, primary_signal, effective)

        if effective.mode == "dual":
            aux_channel = effective.auxiliary_channel
            if not aux_channel:
                return candidate_events

            aux_signal = self._get_channel_signal(df_id, aux_channel, effective)
            if aux_signal is None or len(aux_signal) == 0:
                logger.warning("Auxiliary channel %s unavailable for ROI %s", aux_channel, roi_id)
                return candidate_events

            validated_events = self._validate_with_auxiliary(
                time=time,
                aux_signal=aux_signal,
                events=candidate_events,
                params=effective
            )
            return validated_events

        return candidate_events

    def _apply_override(self, params: DetectionParams, override: Optional[dict]) -> DetectionParams:
        effective = DetectionParams(**params.__dict__)

        if not override:
            effective._override_threshold = None
            return effective

        if 'threshold' in override and override['threshold'] is not None:
            effective._override_threshold = override['threshold']
        else:
            effective._override_threshold = None

        if 'min_dwell' in override and override['min_dwell'] is not None:
            effective.min_dwell = override['min_dwell']

        if 'merge_gap' in override and override['merge_gap'] is not None:
            effective.merge_gap = override['merge_gap']

        return effective

    def _get_channel_signal(self, df_id, channel: str, params: Optional[DetectionParams] = None):
        if not channel:
            return None
        alpha = getattr(params, "alex_alpha", 0.34) if params is not None else 0.34
        beta = getattr(params, "alex_beta", 0.0) if params is not None else 0.0
        gamma = getattr(params, "alex_gamma", 1.0) if params is not None else 1.0
        signal = self.data_model.get_signal_values(
            channel,
            df_id,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        return signal

    def _safe_dwell_time(self, time: np.ndarray, start_idx: int, end_idx: int) -> float:
        if time is None or len(time) == 0:
            return 0.0
        start_idx = int(max(0, min(len(time) - 1, start_idx)))
        end_idx = int(max(0, min(len(time) - 1, end_idx)))
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        try:
            dwell = float(time[end_idx] - time[start_idx])
            if dwell < 0:
                return 0.0
            return dwell
        except Exception:
            return 0.0

    def _threshold_from_signal(self, signal: np.ndarray, params: DetectionParams):
        baseline = np.median(signal)
        mad = np.median(np.abs(signal - baseline))

        override = getattr(params, "_override_threshold", None)
        if override is not None:
            if override.get('mode') == 'deltaMAD':
                high = baseline + float(override.get('value', params.delta_mad)) * mad
            else:
                high = float(override.get('value', baseline + params.delta_mad * mad))
        else:
            high = baseline + params.delta_mad * mad

        low = baseline + params.hysteresis_ratio * (high - baseline)
        return baseline, mad, high, low

    def _aux_threshold_from_signal(self, signal: np.ndarray, params: DetectionParams):
        baseline = np.median(signal)
        mad = np.median(np.abs(signal - baseline))
        high = baseline + params.aux_delta_mad * mad
        return baseline, mad, high

    def _backtrack_start_to_low_threshold(self, signal: np.ndarray, trigger_idx: int, threshold_low: float) -> int:
        """
        After the high threshold is triggered, backtrack left to the low-threshold boundary.
        Return the leftmost index inside the event, where the previous point is below
        threshold_low or the signal has already reached the boundary.
        """
        start_idx = int(trigger_idx)
        while start_idx > 0 and signal[start_idx - 1] >= threshold_low:
            start_idx -= 1
        return int(start_idx)

    def _compute_primary_metrics(self, signal: np.ndarray, start_idx: int, end_idx: int):
        window = signal[start_idx:end_idx + 1]
        if window.size == 0:
            return None, None
        return float(np.max(window)), float(np.mean(window))

    def _apply_aux_metrics_and_classification(self, event: EventRecord, aux_signal: np.ndarray, params: DetectionParams):
        _, _, aux_threshold = self._aux_threshold_from_signal(aux_signal, params)

        aux_window = aux_signal[event.start_idx:event.end_idx + 1]
        if aux_window.size == 0:
            event.classification = "candidate"
            event.aux_supported = False
            event.aux_overlap_ratio = 0.0
            event.aux_peak = None
            event.aux_mean = None
            return event

        supported_points = aux_window > aux_threshold
        overlap_ratio = float(np.sum(supported_points) / len(aux_window))

        event.aux_overlap_ratio = overlap_ratio
        event.aux_peak = float(np.max(aux_window))
        event.aux_mean = float(np.mean(aux_window))

        if np.any(supported_points) and overlap_ratio >= params.overlap_ratio_threshold:
            event.classification = "dual_supported"
            event.aux_supported = True
        else:
            event.classification = "candidate"
            event.aux_supported = False

        return event

    def _detect_candidate_events(self, time, signal, params: DetectionParams) -> List[EventRecord]:
        baseline, mad, threshold_high, threshold_low = self._threshold_from_signal(signal, params)

        logger.debug(
            "Primary detection baseline=%.4f mad=%.4f high=%.4f low=%.4f",
            baseline, mad, threshold_high, threshold_low
        )

        state = 0
        start_idx = None
        raw_events = []

        for i, val in enumerate(signal):
            if state == 0:
                if val > threshold_high:
                    state = 1
                    start_idx = self._backtrack_start_to_low_threshold(signal, i, threshold_low)
            else:
                if val < threshold_low:
                    state = 0
                    if start_idx is not None:
                        end_idx = max(start_idx, i - 1)
                        raw_events.append((start_idx, end_idx, False))
                        start_idx = None

        if state == 1 and start_idx is not None:
            raw_events.append((start_idx, len(time) - 1, True))

        filtered = []
        for start_idx, end_idx, right_censored in raw_events:
            dwell = self._safe_dwell_time(time, start_idx, end_idx)
            if dwell >= params.min_dwell:
                ev = EventRecord(
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                    start_time=float(time[start_idx]),
                    end_time=float(time[end_idx]),
                    dwell_time=float(dwell),
                    right_censored=bool(right_censored),
                    event_key=(int(start_idx), int(end_idx)),
                    classification="candidate",
                    aux_supported=False,
                    source="auto",
                    manual_status="",
                    created_by="auto"
                )
                ev.primary_peak, ev.primary_mean = self._compute_primary_metrics(signal, start_idx, end_idx)
                filtered.append(ev)

        merged = self._merge_events(filtered, time, signal, params)
        return merged

    def _merge_events(self, events: List[EventRecord], time, primary_signal, params: DetectionParams) -> List[EventRecord]:
        if len(events) <= 1:
            return events

        merged = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            gap = ev.start_time - prev.end_time
            if gap < params.merge_gap:
                prev.end_idx = int(max(prev.end_idx, ev.end_idx))
                prev.end_time = float(time[prev.end_idx])
                prev.dwell_time = self._safe_dwell_time(time, prev.start_idx, prev.end_idx)
                prev.event_key = (int(prev.start_idx), int(prev.end_idx))
                prev.right_censored = bool(prev.right_censored or ev.right_censored)
                prev.primary_peak, prev.primary_mean = self._compute_primary_metrics(
                    primary_signal, prev.start_idx, prev.end_idx
                )
            else:
                merged.append(ev)

        return merged

    def _validate_with_auxiliary(self, time, aux_signal, events: List[EventRecord], params: DetectionParams) -> List[EventRecord]:
        for ev in events:
            self._apply_aux_metrics_and_classification(ev, aux_signal, params)
        return events

    def recompute_event_metrics_for_roi(
        self,
        roi_id,
        params: DetectionParams,
        start_idx: int,
        end_idx: int,
        source: str = "manual_added",
        manual_status: str = "manual_added",
        preserve_right_censored: bool = False
    ) -> Optional[EventRecord]:
        df_id = self.data_model.get_roi_df(roi_id)
        if df_id is None or df_id.empty:
            return None

        start_idx = int(max(0, start_idx))
        end_idx = int(min(len(df_id) - 1, end_idx))
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        time = df_id['Time_sec'].values
        effective = self._apply_override(params, None)

        primary_signal = self._get_channel_signal(df_id, effective.primary_channel, effective)
        if primary_signal is None or len(primary_signal) == 0:
            return None

        ev = EventRecord(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=float(time[start_idx]),
            end_time=float(time[end_idx]),
            dwell_time=self._safe_dwell_time(time, start_idx, end_idx),
            right_censored=bool(preserve_right_censored and end_idx == len(time) - 1),
            event_key=(int(start_idx), int(end_idx)),
            classification="candidate",
            aux_overlap_ratio=0.0,
            aux_supported=False,
            source=source,
            manual_status=manual_status,
            created_by="user"
        )

        ev.primary_peak, ev.primary_mean = self._compute_primary_metrics(primary_signal, start_idx, end_idx)

        if effective.mode == "dual" and effective.auxiliary_channel:
            aux_signal = self._get_channel_signal(df_id, effective.auxiliary_channel, effective)
            if aux_signal is not None and len(aux_signal) > 0:
                self._apply_aux_metrics_and_classification(ev, aux_signal, effective)
        else:
            ev.classification = "candidate"
            ev.aux_supported = False
            ev.aux_overlap_ratio = 0.0

        return ev

    def force_mark_event_as_dual_supported(self, event: EventRecord) -> EventRecord:
        event.classification = "dual_supported"
        event.aux_supported = True
        event.manual_status = "manual_bound"
        if event.source == "auto":
            event.source = "manual_modified"
        event.created_by = "user"
        return event

    def infer_event_bounds_from_peak_click(
        self,
        roi_id,
        params: DetectionParams,
        clicked_idx: int,
        override: Optional[dict] = None,
        min_points: int = 3
    ) -> Optional[Tuple[int, int]]:
        """
        Backtrack based on the primary-channel threshold:
        1. Snap from the clicked_idx vicinity to a local peak.
        2. Then backtrack left and right from the peak until the signal drops below
           the primary-channel low threshold_low.
        3. If the interval is too short, expand it to at least min_points sampled points.
        """
        df_id = self.data_model.get_roi_df(roi_id)
        if df_id is None or df_id.empty:
            return None

        effective = self._apply_override(params, override)
        primary_signal = self._get_channel_signal(df_id, effective.primary_channel, effective)
        if primary_signal is None or len(primary_signal) == 0:
            return None

        n = len(primary_signal)
        clicked_idx = int(max(0, min(n - 1, clicked_idx)))

        _, _, _, threshold_low = self._threshold_from_signal(primary_signal, effective)

        peak_idx = self._snap_to_local_peak(primary_signal, clicked_idx)

        start_idx = peak_idx
        while start_idx > 0 and primary_signal[start_idx - 1] >= threshold_low:
            start_idx -= 1

        end_idx = peak_idx
        while end_idx < n - 1 and primary_signal[end_idx + 1] >= threshold_low:
            end_idx += 1

        start_idx, end_idx = self._ensure_min_width(start_idx, end_idx, n, min_points=max(2, int(min_points)))

        return int(start_idx), int(end_idx)

    def _snap_to_local_peak(self, signal: np.ndarray, clicked_idx: int, search_radius: int = 3) -> int:
        n = len(signal)
        if n == 0:
            return 0

        left = max(0, clicked_idx - search_radius)
        right = min(n - 1, clicked_idx + search_radius)

        local = signal[left:right + 1]
        if local.size == 0:
            return int(clicked_idx)

        local_peak_offset = int(np.argmax(local))
        return int(left + local_peak_offset)

    def _ensure_min_width(self, start_idx: int, end_idx: int, n: int, min_points: int = 3) -> Tuple[int, int]:
        if min_points <= 1:
            return int(start_idx), int(end_idx)

        current_points = end_idx - start_idx + 1
        if current_points >= min_points:
            return int(start_idx), int(end_idx)

        need = min_points - current_points
        expand_left = need // 2
        expand_right = need - expand_left

        start_idx = max(0, start_idx - expand_left)
        end_idx = min(n - 1, end_idx + expand_right)

        current_points = end_idx - start_idx + 1
        if current_points < min_points:
            remain = min_points - current_points

            more_left = min(start_idx, remain)
            start_idx -= more_left
            remain -= more_left

            if remain > 0:
                end_idx = min(n - 1, end_idx + remain)

        return int(start_idx), int(end_idx)

    def find_overlapping_event(self, roi_id, start_idx: int, end_idx: int, include_hidden: bool = False):
        from state_manager import AppState
        return None

    def get_channel_thresholds(self, roi_id, params: DetectionParams, override: Optional[dict] = None) -> Dict[str, dict]:
        df_id = self.data_model.get_roi_df(roi_id)
        if df_id is None or df_id.empty:
            return {}

        effective = self._apply_override(params, override)
        result = {}

        primary_signal = self._get_channel_signal(df_id, effective.primary_channel, effective)
        if primary_signal is not None and len(primary_signal) > 0:
            baseline, mad, high, low = self._threshold_from_signal(primary_signal, effective)
            result['primary'] = {
                'channel': effective.primary_channel,
                'baseline': baseline,
                'mad': mad,
                'high': high,
                'low': low
            }

        if effective.mode == 'dual' and effective.auxiliary_channel:
            aux_signal = self._get_channel_signal(df_id, effective.auxiliary_channel, effective)
            if aux_signal is not None and len(aux_signal) > 0:
                baseline, mad, high = self._aux_threshold_from_signal(aux_signal, effective)
                result['auxiliary'] = {
                    'channel': effective.auxiliary_channel,
                    'baseline': baseline,
                    'mad': mad,
                    'high': high
                }

        return result
