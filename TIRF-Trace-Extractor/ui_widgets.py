"""
ui_widgets.py - unified UI component entry point (backward compatible)
"""

from ui_lut_cache import (
    LUTSettings,
    LUTGenerator,
    LUTControlPanel,
    ByteBudgetCache,
    LRUCache,
    MultiLevelCache,
    DebounceTimer
)

from ui_video_display import (
    SyncZoomManager,
    VideoWidget
)

from ui_detection_plot import (
    IntensityPlotWidget,
    DetectionPreviewDialog,
    QtLogHandler,
    LogPanel
)

__all__ = [
    'LUTSettings',
    'LUTGenerator',
    'LUTControlPanel',
    'ByteBudgetCache',
    'LRUCache',
    'MultiLevelCache',
    'DebounceTimer',
    'SyncZoomManager',
    'VideoWidget',
    'IntensityPlotWidget',
    'DetectionPreviewDialog',
    'QtLogHandler',
    'LogPanel',
]
