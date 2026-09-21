"""绘图管理模块"""
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import savgol_filter


class PlotManager:
    """绘图管理器"""
    
    def __init__(self, parent, gui_ref):
        self.parent = parent
        self.gui = gui_ref
        self.figure = None
        self.axes = []
        self.canvas = None
        self.toolbar = None
        
    def create_plot_area(self):
        """创建绘图区域"""
        plot_container = tk.Frame(self.parent, bg='white')
        plot_container.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(10, 9))
        self.axes = [
            self.figure.add_subplot(3, 1, 1),
            self.figure.add_subplot(3, 1, 2),
            self.figure.add_subplot(3, 1, 3)
        ]
        self.figure.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.figure, plot_container)
        
        # 添加导航工具栏（用于缩放等操作）
        toolbar_frame = tk.Frame(plot_container)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 绑定鼠标事件以获取当前视图范围
        self.canvas.mpl_connect('draw_event', self._on_draw)
    
    def _on_draw(self, event):
        """当图形绘制时更新当前x轴范围"""
        if self.axes and len(self.axes) > 0:
            self.gui.current_xlim = self.axes[0].get_xlim()
    
    def apply_smooth(self, signal):
        """应用平滑"""
        if not self.gui.show_smoothed.get():
            return signal

        window = self.gui.smooth_window.get()
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1

        if len(signal) < window:
            return signal

        return savgol_filter(signal, window, 2)
    
    def update_plots(self):
        """更新图表"""
        current_rois = self.gui._get_current_rois()
        bg_enabled = self.gui.bg_correction_enabled.get()

        for ax in self.axes:
            ax.clear()

        for idx, roi_id in enumerate(current_rois):
            ax = self.axes[idx]

            # 获取校正后的信号
            signals = self.gui._get_signals_cached(roi_id)
            time = signals['time']

            if self.gui.is_three_color:
                # 488通道（背景校正后）
                if self.gui.show_488.get():
                    signal_488 = signals.get('bg_corrected_488')
                    if signal_488 is not None:
                        signal_488 = self.apply_smooth(signal_488)
                        ax.plot(time, signal_488, 'b-', label='488nm', linewidth=1.5)

                # 532通道
                if self.gui.show_532.get():
                    # 当背景校正关闭时，只显示"校正"选项（实际是基于原始信号的串色校正）
                    if not bg_enabled or self.gui.show_corrected_532.get():
                        signal_532 = signals.get('corrected_532')
                        label_532 = '532nm (校正)'
                    else:
                        signal_532 = signals.get('bg_corrected_532')
                        label_532 = '532nm (背景校正)'

                    if signal_532 is not None:
                        signal_532 = self.apply_smooth(signal_532)
                        ax.plot(time, signal_532, 'g-', label=label_532, linewidth=1.5)

                # 638通道
                if self.gui.show_638.get():
                    # 当背景校正关闭时，只显示"校正"选项（实际是基于原始信号的泄漏校正）
                    if not bg_enabled or self.gui.show_corrected_638.get():
                        signal_638 = signals.get('corrected_638')
                        label_638 = '638nm (校正)'
                    else:
                        signal_638 = signals.get('bg_corrected_638')
                        label_638 = '638nm (背景校正)'

                    if signal_638 is not None:
                        signal_638 = self.apply_smooth(signal_638)
                        ax.plot(time, signal_638, 'r-', label=label_638, linewidth=1.5)
            else:
                # 双色模式
                # 532通道
                if self.gui.show_532.get():
                    # 当背景校正关闭时，只显示"校正"选项（实际等同于原始信号）
                    if not bg_enabled or self.gui.show_corrected_532.get():
                        signal_532 = signals.get('corrected_532')
                        label_532 = '532nm (校正)'
                    else:
                        signal_532 = signals.get('bg_corrected_532')
                        label_532 = '532nm (背景校正)'

                    if signal_532 is not None:
                        signal_532 = self.apply_smooth(signal_532)
                        ax.plot(time, signal_532, 'g-', label=label_532, linewidth=1.5)

                # 638通道
                if self.gui.show_638.get():
                    # 当背景校正关闭时，只显示"校正"选项（实际是基于原始信号的泄漏校正）
                    if not bg_enabled or self.gui.show_corrected_638.get():
                        signal_638 = signals.get('corrected_638')
                        label_638 = '638nm (校正)'
                    else:
                        signal_638 = signals.get('bg_corrected_638')
                        label_638 = '638nm (背景校正)'

                    if signal_638 is not None:
                        signal_638 = self.apply_smooth(signal_638)
                        ax.plot(time, signal_638, 'r-', label=label_638, linewidth=1.5)

            ax.set_title(f'ROI #{roi_id}', fontsize=11, fontweight='bold')
            ax.set_xlabel('时间 (s)', fontsize=9)
            ax.set_ylabel('强度', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

            # 筛选结果高亮
            if self.gui.is_filtered and roi_id not in self.gui.passed_rois:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)

        for idx in range(len(current_rois), len(self.axes)):
            self.axes[idx].set_visible(False)

        for idx in range(len(current_rois)):
            self.axes[idx].set_visible(True)

        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

        # 更新背景信息
        self.gui._update_bg_info_display()
