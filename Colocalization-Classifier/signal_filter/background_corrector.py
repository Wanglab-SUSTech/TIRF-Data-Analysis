"""背景校正模块"""
import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Optional


class BackgroundCorrector:
    """背景校正器 - 对每个ROI的每个通道进行背景校正"""
    
    # 常量定义
    DEFAULT_SKIP_LOWEST = 30
    DEFAULT_TAKE_COUNT = 20
    MIN_SIGNAL_LENGTH = 10
    VALID_CHANNELS = {'488', '532', '638'}

    def __init__(self, skip_lowest: int = DEFAULT_SKIP_LOWEST, 
                 take_count: int = DEFAULT_TAKE_COUNT):
        """
        初始化背景校正器
        
        Args:
            skip_lowest: 跳过最低的帧数（避免极端噪声）
            take_count: 取接下来的帧数用于计算背景
            
        Raises:
            ValueError: 参数不合法时抛出
        """
        if skip_lowest < 0:
            raise ValueError(f"skip_lowest 必须为非负数，当前值: {skip_lowest}")
        if take_count <= 0:
            raise ValueError(f"take_count 必须为正数，当前值: {take_count}")
            
        self.skip_lowest = skip_lowest
        self.take_count = take_count
        # 手动背景值存储: {roi_id: {'488': value, '532': value, '638': value}}
        self.manual_backgrounds: Dict[int, Dict[str, float]] = {}

    def estimate_background(self, signal: npt.NDArray[np.float64]) -> float:
        """
        自动估算背景：跳过最低的skip_lowest帧，取接下来take_count帧的中位数
        
        Args:
            signal: 输入信号数组
            
        Returns:
            估算的背景值
            
        Raises:
            ValueError: 信号数据不合法时抛出
        """
        if signal is None:
            raise ValueError("信号数据不能为 None")
        
        signal = np.asarray(signal, dtype=float)
        
        if signal.size == 0:
            raise ValueError("信号数据不能为空")
        
        if signal.size < self.MIN_SIGNAL_LENGTH:
            raise ValueError(
                f"信号长度 ({signal.size}) 小于最小要求 ({self.MIN_SIGNAL_LENGTH})"
            )
        
        n = len(signal)
        sorted_signal = np.sort(signal)

        skip = self.skip_lowest
        take = self.take_count

        # 数据不够时，按比例调整
        if n < skip + take:
            skip = max(0, n // 3)
            take = max(1, n // 5)
            if skip + take > n:
                skip = 0
                take = max(1, n // 2)

        selected = sorted_signal[skip:skip + take]
        return float(np.median(selected))

    def get_background(self, roi_id: int, channel: str, 
                      signal: npt.NDArray[np.float64]) -> float:
        """
        获取背景值（优先使用手动值）
        
        Args:
            roi_id: ROI ID
            channel: 通道名称 ('488', '532', '638')
            signal: 信号数组
            
        Returns:
            背景值
            
        Raises:
            ValueError: 通道名称不合法时抛出
        """
        self._validate_channel(channel)
        
        if roi_id in self.manual_backgrounds:
            if channel in self.manual_backgrounds[roi_id]:
                return self.manual_backgrounds[roi_id][channel]
        
        return self.estimate_background(signal)

    def set_manual_background(self, roi_id: int, channel: str, value: float):
        """
        设置手动背景值
        
        Args:
            roi_id: ROI ID
            channel: 通道名称
            value: 背景值
            
        Raises:
            ValueError: 参数不合法时抛出
        """
        self._validate_channel(channel)
        
        if not isinstance(value, (int, float)):
            raise ValueError(f"背景值必须为数字，当前类型: {type(value)}")
        
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"背景值不能为 NaN 或 Inf，当前值: {value}")
        
        if roi_id not in self.manual_backgrounds:
            self.manual_backgrounds[roi_id] = {}
        self.manual_backgrounds[roi_id][channel] = float(value)

    def clear_manual_background(self, roi_id: int, channel: Optional[str] = None):
        """
        清除手动背景值
        
        Args:
            roi_id: ROI ID
            channel: 通道名称，None 表示清除该 ROI 的所有通道
        """
        if channel is not None:
            self._validate_channel(channel)
        
        if roi_id in self.manual_backgrounds:
            if channel is None:
                del self.manual_backgrounds[roi_id]
            elif channel in self.manual_backgrounds[roi_id]:
                del self.manual_backgrounds[roi_id][channel]
                if not self.manual_backgrounds[roi_id]:
                    del self.manual_backgrounds[roi_id]

    def has_manual_background(self, roi_id: int, channel: Optional[str] = None) -> bool:
        """
        检查是否有手动背景值
        
        Args:
            roi_id: ROI ID
            channel: 通道名称，None 表示检查该 ROI 是否有任何手动背景
            
        Returns:
            是否存在手动背景值
        """
        if roi_id not in self.manual_backgrounds:
            return False
        if channel is None:
            return bool(self.manual_backgrounds[roi_id])
        return channel in self.manual_backgrounds[roi_id]

    def get_background_info(self, roi_id: int, signals: Dict[str, npt.NDArray], 
                           is_three_color: bool) -> Dict[str, Tuple[float, bool]]:
        """
        获取背景信息
        
        Args:
            roi_id: ROI ID
            signals: 信号字典 {通道: 信号数组}
            is_three_color: 是否为三色模式
            
        Returns:
            背景信息字典 {通道: (背景值, 是否手动)}
        """
        result = {}
        channels = ['488', '532', '638'] if is_three_color else ['532', '638']
        
        for channel in channels:
            if channel in signals and signals[channel] is not None:
                try:
                    bg_value = self.get_background(roi_id, channel, signals[channel])
                    is_manual = self.has_manual_background(roi_id, channel)
                    result[channel] = (bg_value, is_manual)
                except Exception as e:
                    print(f"获取 ROI {roi_id} 通道 {channel} 背景信息失败: {e}")
                    continue
        
        return result

    def _validate_channel(self, channel: str):
        """验证通道名称"""
        if channel not in self.VALID_CHANNELS:
            raise ValueError(
                f"无效的通道名称: {channel}，有效值为: {self.VALID_CHANNELS}"
            )

    def update_params(self, skip_lowest: int, take_count: int):
        """
        更新背景估算参数
        
        Args:
            skip_lowest: 跳过最低的帧数
            take_count: 取接下来的帧数
            
        Raises:
            ValueError: 参数不合法时抛出
        """
        if skip_lowest < 0:
            raise ValueError(f"skip_lowest 必须为非负数，当前值: {skip_lowest}")
        if take_count <= 0:
            raise ValueError(f"take_count 必须为正数，当前值: {take_count}")
        
        self.skip_lowest = skip_lowest
        self.take_count = take_count
