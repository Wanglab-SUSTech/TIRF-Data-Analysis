"""信号处理模块"""
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Dict, Optional
from background_corrector import BackgroundCorrector


class SignalProcessor:
    """信号处理器 - 负责背景校正、串色校正、泄漏校正"""
    
    # 常量定义
    DEFAULT_CROSSTALK_COEF = 0.34
    DEFAULT_LEAKING_COEF = 0.07
    REQUIRED_COLUMNS = {'Time_sec', 'Net_532', 'Net_638'}
    REQUIRED_COLUMNS_3COLOR = {'Time_sec', 'Net_488', 'Net_532', 'Net_638'}

    def __init__(self, bg_corrector: BackgroundCorrector, 
                 crosstalk_coef: float = DEFAULT_CROSSTALK_COEF,
                 leaking_coef: float = DEFAULT_LEAKING_COEF, 
                 is_three_color: bool = False,
                 bg_correction_enabled: bool = False):
        """
        初始化信号处理器
        
        Args:
            bg_corrector: 背景校正器实例
            crosstalk_coef: 串色校正系数 (488→532)
            leaking_coef: 泄漏校正系数 (532→638)
            is_three_color: 是否为三色模式
            bg_correction_enabled: 是否启用背景校正（默认关闭）
            
        Raises:
            ValueError: 参数不合法时抛出
        """
        if not isinstance(bg_corrector, BackgroundCorrector):
            raise ValueError("bg_corrector 必须是 BackgroundCorrector 实例")
        
        self._validate_coefficient(crosstalk_coef, "crosstalk_coef")
        self._validate_coefficient(leaking_coef, "leaking_coef")
        
        self.bg_corrector = bg_corrector
        self.crosstalk_coef = crosstalk_coef
        self.leaking_coef = leaking_coef
        self.is_three_color = is_three_color
        self.bg_correction_enabled = bg_correction_enabled

    def process_roi(self, roi_id: int, roi_data: pd.DataFrame) -> Dict[str, Optional[npt.NDArray]]:
        """
        处理单个ROI的所有信号校正
        
        校正顺序：
        1. 背景校正 (所有通道，可选)
        2. 串色校正 (488→532, 仅三色模式)
        3. 泄漏校正 (532→638)
        
        Args:
            roi_id: ROI ID
            roi_data: ROI 数据 DataFrame
            
        Returns:
            包含以下键的字典:
            - time: 时间轴
            - raw_488/532/638: 原始信号
            - bg_488/532/638: 背景值
            - bg_corrected_488/532/638: 背景校正后信号
            - corrected_532: 串色校正后532（三色模式）或背景校正后532（双色模式）
            - corrected_638: 泄漏校正后638
            
        Raises:
            ValueError: 数据不合法时抛出
        """
        try:
            # 验证数据
            self._validate_roi_data(roi_data)
            
            result = {}

            # 时间轴
            result['time'] = roi_data['Time_sec'].values

            # 获取原始信号
            raw_532 = roi_data['Net_532'].values.astype(float)
            raw_638 = roi_data['Net_638'].values.astype(float)
            result['raw_532'] = raw_532
            result['raw_638'] = raw_638

            # 背景校正 - 532
            if self.bg_correction_enabled:
                bg_532 = self.bg_corrector.get_background(roi_id, '532', raw_532)
                result['bg_532'] = bg_532
                bg_corrected_532 = raw_532 - bg_532
            else:
                bg_532 = 0.0
                result['bg_532'] = bg_532
                bg_corrected_532 = raw_532.copy()
            result['bg_corrected_532'] = bg_corrected_532

            # 背景校正 - 638
            if self.bg_correction_enabled:
                bg_638 = self.bg_corrector.get_background(roi_id, '638', raw_638)
                result['bg_638'] = bg_638
                bg_corrected_638 = raw_638 - bg_638
            else:
                bg_638 = 0.0
                result['bg_638'] = bg_638
                bg_corrected_638 = raw_638.copy()
            result['bg_corrected_638'] = bg_corrected_638

            # 三色模式处理
            if self.is_three_color and 'Net_488' in roi_data.columns:
                raw_488 = roi_data['Net_488'].values.astype(float)
                result['raw_488'] = raw_488

                # 背景校正 - 488
                if self.bg_correction_enabled:
                    bg_488 = self.bg_corrector.get_background(roi_id, '488', raw_488)
                    result['bg_488'] = bg_488
                    bg_corrected_488 = raw_488 - bg_488
                else:
                    bg_488 = 0.0
                    result['bg_488'] = bg_488
                    bg_corrected_488 = raw_488.copy()
                result['bg_corrected_488'] = bg_corrected_488

                # 串色校正 (488→532)
                corrected_532 = bg_corrected_532 - self.crosstalk_coef * bg_corrected_488
                result['corrected_532'] = corrected_532
            else:
                result['raw_488'] = None
                result['bg_488'] = None
                result['bg_corrected_488'] = None
                # 双色模式，corrected_532 = bg_corrected_532
                corrected_532 = bg_corrected_532
                result['corrected_532'] = corrected_532

            # 泄漏校正 (532→638)
            corrected_638 = bg_corrected_638 - self.leaking_coef * corrected_532
            result['corrected_638'] = corrected_638

            return result
            
        except Exception as e:
            raise RuntimeError(f"处理 ROI {roi_id} 时出错: {e}") from e

    def update_coefficients(self, crosstalk_coef: Optional[float] = None,
                           leaking_coef: Optional[float] = None):
        """
        更新校正系数
        
        Args:
            crosstalk_coef: 新的串色校正系数
            leaking_coef: 新的泄漏校正系数
            
        Raises:
            ValueError: 系数不合法时抛出
        """
        if crosstalk_coef is not None:
            self._validate_coefficient(crosstalk_coef, "crosstalk_coef")
            self.crosstalk_coef = crosstalk_coef
        
        if leaking_coef is not None:
            self._validate_coefficient(leaking_coef, "leaking_coef")
            self.leaking_coef = leaking_coef

    def _validate_roi_data(self, roi_data: pd.DataFrame):
        """验证 ROI 数据"""
        if roi_data is None:
            raise ValueError("ROI 数据不能为 None")
        
        if roi_data.empty:
            raise ValueError("ROI 数据不能为空")
        
        required_cols = self.REQUIRED_COLUMNS_3COLOR if self.is_three_color else self.REQUIRED_COLUMNS
        missing_cols = required_cols - set(roi_data.columns)
        
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

    @staticmethod
    def _validate_coefficient(value: float, name: str):
        """验证校正系数"""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} 必须为数字，当前类型: {type(value)}")
        
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"{name} 不能为 NaN 或 Inf，当前值: {value}")
        
        if not 0 <= value <= 1:
            raise ValueError(f"{name} 必须在 [0, 1] 范围内，当前值: {value}")
