"""筛选引擎模块"""
import numpy as np
from signal_processor import SignalProcessor


class FilterEngine:
    """筛选引擎 - 负责数据筛选逻辑（使用校正后信号）"""

    def __init__(self, processed_data, is_three_color, signal_processor: SignalProcessor):
        self.processed_data = processed_data
        self.is_three_color = is_three_color
        self.signal_processor = signal_processor
        self.roi_ids = sorted(processed_data['ROI_ID'].unique())

    @staticmethod
    def _normalize_signal(signal):
        arr = np.array(signal, dtype=float)
        return np.nan_to_num(arr, nan=-np.inf)

    def _build_mask(self, signal, threshold):
        normalized = self._normalize_signal(signal)
        return normalized >= threshold

    @staticmethod
    def _segments_from_mask(mask, min_len):
        if mask.size == 0:
            return []
        padded = np.concatenate(([0], mask.astype(int), [0]))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        segments = []
        for start, end in zip(starts, ends):
            length = end - start + 1
            if length >= min_len:
                segments.append((int(start), int(end)))
        return segments

    def _has_min_run(self, mask, min_len):
        if min_len <= 1:
            return bool(mask.size and np.any(mask))
        segments = self._segments_from_mask(mask, min_len)
        return len(segments) > 0

    def check_continuous_frames(self, signal, threshold, min_frames):
        """使用向量化布尔掩码检测是否存在满足条件的连续帧。"""
        if min_frames <= 0:
            return False
        mask = self._build_mask(signal, threshold)
        return self._has_min_run(mask, min_frames)

    def get_continuous_segments(self, signal, threshold, min_frames):
        """返回满足阈值与连续帧数的片段区间列表（闭区间）"""
        if min_frames <= 0:
            return []
        mask = self._build_mask(signal, threshold)
        return self._segments_from_mask(mask, min_frames)

    def _segments_have_overlap(self, segments_dict, overlap_frames):
        """判断多通道片段是否存在共同重叠区间"""
        channel_names = list(segments_dict.keys())
        seg_lists = [segments_dict[name] for name in channel_names]
        indices = [0] * len(seg_lists)

        while True:
            current_segments = []
            for list_idx, seg_list in enumerate(seg_lists):
                ptr = indices[list_idx]
                if ptr >= len(seg_list):
                    return False
                current_segments.append(seg_list[ptr])

            max_start = max(seg[0] for seg in current_segments)
            min_end = min(seg[1] for seg in current_segments)
            overlap_len = min_end - max_start + 1

            if overlap_len >= overlap_frames:
                return True

            earliest_end = current_segments[0][1]
            earliest_idx = 0
            for i, seg in enumerate(current_segments):
                if seg[1] < earliest_end:
                    earliest_end = seg[1]
                    earliest_idx = i
            indices[earliest_idx] += 1

    def _get_corrected_signal(self, roi_id, channel: str) -> np.ndarray:
        """获取校正后的信号"""
        roi_data = self.processed_data[self.processed_data['ROI_ID'] == roi_id]
        signals = self.signal_processor.process_roi(roi_id, roi_data)

        if channel == '488':
            return signals.get('bg_corrected_488', np.array([]))
        elif channel == '532':
            return signals.get('corrected_532', np.array([]))
        elif channel == '638':
            return signals.get('corrected_638', np.array([]))
        else:
            raise ValueError(f"未知通道: {channel}")

    def _passes_overlap_filter(self, roi_id, params, overlap_cfg):
        """检查 ROI 是否通过时间重叠筛选（使用校正后信号）"""
        if not overlap_cfg or not overlap_cfg.get('enabled'):
            return True

        selected_channels = overlap_cfg.get('channels', {})
        if not selected_channels:
            return False

        overlap_frames = overlap_cfg.get('overlap_frames')
        if not overlap_frames or overlap_frames <= 0:
            return False

        segments_dict = {}
        for channel, channel_params in selected_channels.items():
            signal = self._get_corrected_signal(roi_id, channel)
            if signal is None or len(signal) == 0:
                return False
            segments = self.get_continuous_segments(
                signal,
                channel_params['threshold'],
                channel_params['frames']
            )
            if not segments:
                return False
            segments_dict[channel] = segments

        return self._segments_have_overlap(segments_dict, overlap_frames)

    def filter_single_roi(self, roi_id, params):
        """筛选单个 ROI（使用校正后信号）"""
        roi_data = self.processed_data[self.processed_data['ROI_ID'] == roi_id]
        signals = self.signal_processor.process_roi(roi_id, roi_data)

        # 使用校正后的532信号
        corrected_532 = signals.get('corrected_532')
        if corrected_532 is None:
            return False

        check_532 = self.check_continuous_frames(
            corrected_532,
            params['threshold_532'],
            params['frames_532']
        )

        # 使用校正后的638信号
        corrected_638 = signals.get('corrected_638')
        if corrected_638 is None:
            return False

        check_638 = self.check_continuous_frames(
            corrected_638,
            params['threshold_638'],
            params['frames_638']
        )

        basic_pass = check_532 and check_638

        if self.is_three_color:
            bg_corrected_488 = signals.get('bg_corrected_488')
            if bg_corrected_488 is None:
                return False

            check_488 = self.check_continuous_frames(
                bg_corrected_488,
                params['threshold_488'],
                params['frames_488']
            )
            basic_pass = check_488 and check_532 and check_638

        if not basic_pass:
            return False

        overlap_cfg = params.get('overlap_filter')
        if overlap_cfg and overlap_cfg.get('enabled'):
            return self._passes_overlap_filter(roi_id, params, overlap_cfg)

        return True

    def filter_all_rois(self, params):
        """筛选所有 ROI"""
        passed_rois = []
        for roi_id in self.roi_ids:
            if self.filter_single_roi(roi_id, params):
                passed_rois.append(roi_id)
        return passed_rois
