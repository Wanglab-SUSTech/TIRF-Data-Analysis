"""主GUI模块 - 整合所有组件"""
import os
import platform
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import pandas as pd

from config_manager import ConfigManager
from background_corrector import BackgroundCorrector
from signal_processor import SignalProcessor
from filter_engine import FilterEngine
from ui_components import UIHelper
from control_panels import ControlPanels
from plot_manager import PlotManager


class SignalFilterGUI:
    """信号筛选图形界面 - 主类"""

    def __init__(self, parent, loader, default_directory=None):
        self.window = parent
        self.loader = loader
        self.processed_data = loader.processed_data

        # 设置默认目录
        if default_directory and os.path.isdir(default_directory):
            self.default_directory = default_directory
        elif hasattr(loader, 'output_path') and loader.output_path:
            self.default_directory = os.path.dirname(os.path.abspath(loader.output_path))
        else:
            self.default_directory = os.path.expanduser("~")

        self.system = platform.system()
        self.is_three_color = 'Net_488' in self.processed_data.columns

        # ROI 管理
        self.roi_ids = sorted(self.processed_data['ROI_ID'].unique())
        self.total_rois = len(self.roi_ids)
        self.rois_per_page = 3
        self.current_page = 0
        self.total_pages = (self.total_rois + self.rois_per_page - 1) // self.rois_per_page

        # 筛选状态
        self.is_filtered = False
        self.passed_rois = []

        # 校正系数
        self.crosstalk_coef = 0.34
        self.leaking_coef = 0.07

        # 初始化变量
        self._init_variables()

        # 通道显示名称
        self.channel_display_names = {
            '488': '488 nm',
            '532': '532 nm',
            '638': '638 nm'
        }

        # 初始化核心组件
        self.bg_corrector = BackgroundCorrector(skip_lowest=30, take_count=20)
        self.signal_processor = SignalProcessor(
            self.bg_corrector,
            self.crosstalk_coef,
            self.leaking_coef,
            self.is_three_color,
            bg_correction_enabled=False  # 默认关闭背景校正
        )
        self.filter_engine = FilterEngine(
            self.processed_data,
            self.is_three_color,
            self.signal_processor
        )
        self.config_manager = ConfigManager()
        
        # UI组件
        self.ui_helper = UIHelper()
        self.control_panels = None
        self.plot_manager = None

        # 配置和缓存
        self.loaded_config = {}
        self.signals_cache = {}
        self.current_xlim = None

        # 创建UI
        self.window.title("共定位信号筛选")
        self.window.geometry("1500x900")
        self._create_ui()
        
        # 加载配置
        self._load_initial_config()
        
        # 绑定快捷键
        self._bind_keyboard_shortcuts()
        
        # 设置关闭事件
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 更新初始图表
        self._update_plots()

    def _init_variables(self):
        """初始化tkinter变量"""
        # 背景校正开关（默认关闭）
        self.bg_correction_enabled = tk.BooleanVar(value=False)
        
        # 平滑设置
        self.show_smoothed = tk.BooleanVar(value=False)
        self.smooth_window = tk.IntVar(value=5)

        # 通道显示设置
        self.show_532 = tk.BooleanVar(value=True)
        self.show_638 = tk.BooleanVar(value=True)
        self.show_corrected_532 = tk.BooleanVar(value=True)
        self.show_corrected_638 = tk.BooleanVar(value=True)

        if self.is_three_color:
            self.show_488 = tk.BooleanVar(value=True)

        # 时间重叠筛选
        self.overlap_enabled = tk.BooleanVar(value=False)
        self.overlap_channel_vars = {}
        self.overlap_channel_entries = {}
        self.overlap_frames_entry = None

    def _create_ui(self):
        """创建UI布局"""
        main_container = tk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 左侧绘图区
        left_frame = tk.Frame(main_container, bg='white')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 右侧控制区
        right_frame = tk.Frame(main_container, width=380, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # 创建绘图区域
        self.plot_manager = PlotManager(left_frame, self)
        self.plot_manager.create_plot_area()

        # 创建控制面板
        self._create_control_area(right_frame)

    def _create_control_area(self, parent):
        """创建控制区域"""
        canvas = tk.Canvas(parent, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.ui_helper.bind_mousewheel(canvas, self.system)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 创建所有控制面板
        self.control_panels = ControlPanels(scrollable_frame, self)
        self.control_panels.create_all_panels()

    def _bind_keyboard_shortcuts(self):
        """绑定键盘快捷键"""
        def _handle_left(event):
            if isinstance(event.widget, (tk.Entry, tk.ttk.Entry, tk.Spinbox)):
                return
            self._prev_page()

        def _handle_right(event):
            if isinstance(event.widget, (tk.Entry, tk.ttk.Entry, tk.Spinbox)):
                return
            self._next_page()

        self.window.bind_all("<Left>", _handle_left)
        self.window.bind_all("<Right>", _handle_right)

    def _on_close(self):
        """窗口关闭事件"""
        try:
            self._save_current_config()
        finally:
            try:
                self.window.destroy()
            finally:
                master = self.window.master
                if isinstance(master, tk.Tk):
                    try:
                        master.destroy()
                    except tk.TclError:
                        pass

    # ========================================================================
    # 缓存管理
    # ========================================================================

    def _clear_cache(self, roi_id=None):
        """清除信号缓存"""
        if roi_id is None:
            self.signals_cache.clear()
        elif roi_id in self.signals_cache:
            del self.signals_cache[roi_id]

    def _get_signals_cached(self, roi_id):
        """获取校正后的信号（使用缓存）"""
        if roi_id not in self.signals_cache:
            roi_data = self.processed_data[self.processed_data['ROI_ID'] == roi_id]
            self.signals_cache[roi_id] = self.signal_processor.process_roi(roi_id, roi_data)
        return self.signals_cache[roi_id]

    # ========================================================================
    # 导航相关
    # ========================================================================

    def _update_navigation_buttons(self):
        """更新导航按钮状态"""
        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_page < self.total_pages - 1 else tk.DISABLED)
        self.page_label.config(text=f"第 {self.current_page + 1}/{self.total_pages} 页")

    def _prev_page(self, event=None):
        """上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_navigation_buttons()
            self._update_plots()
            self._update_bg_info_display()

    def _next_page(self, event=None):
        """下一页"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._update_navigation_buttons()
            self._update_plots()
            self._update_bg_info_display()

    def _get_current_rois(self):
        """获取当前页面的ROI"""
        start_idx = self.current_page * self.rois_per_page
        end_idx = min(start_idx + self.rois_per_page, self.total_rois)
        return self.roi_ids[start_idx:end_idx]

    # ========================================================================
    # 绘图相关
    # ========================================================================

    def _update_plots(self):
        """更新图表（委托给PlotManager）"""
        self.plot_manager.update_plots()

    # ========================================================================
    # 背景校正相关
    # ========================================================================

    def _toggle_bg_correction(self):
        """切换背景校正开关"""
        self.signal_processor.bg_correction_enabled = self.bg_correction_enabled.get()
        self._clear_cache()
        self._update_plots()
        self._update_bg_info_display()

    def _apply_bg_params(self):
        """应用背景估算参数"""
        try:
            skip = int(self.bg_skip_spinbox.get())
            take = int(self.bg_take_spinbox.get())

            self.bg_corrector.skip_lowest = skip
            self.bg_corrector.take_count = take

            self._clear_cache()
            self._update_plots()
            self._update_bg_info_display()

            messagebox.showinfo("成功", f"背景参数已更新:\n跳过最低 {skip} 帧\n取接下来 {take} 帧")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数")

    def _set_manual_background(self, channel: str):
        """设置手动背景"""
        current_rois = self._get_current_rois()
        if not current_rois:
            messagebox.showwarning("警告", "当前页面没有ROI")
            return

        if self.current_xlim is None:
            messagebox.showwarning("警告", "请先放大到背景区域")
            return

        xlim = self.current_xlim
        updated_count = 0

        for roi_id in current_rois:
            roi_data = self.processed_data[self.processed_data['ROI_ID'] == roi_id]
            time = roi_data['Time_sec'].values

            # 获取原始信号
            if channel == '488':
                if 'Net_488' not in roi_data.columns:
                    continue
                raw_signal = roi_data['Net_488'].values
            elif channel == '532':
                raw_signal = roi_data['Net_532'].values
            elif channel == '638':
                raw_signal = roi_data['Net_638'].values
            else:
                continue

            # 筛选可见区域
            mask = (time >= xlim[0]) & (time <= xlim[1])
            if not np.any(mask):
                continue

            visible_data = raw_signal[mask]
            background = float(np.median(visible_data))

            # 设置手动背景
            self.bg_corrector.set_manual_background(roi_id, channel, background)
            self._clear_cache(roi_id)
            updated_count += 1

        if updated_count > 0:
            self._update_plots()
            self._update_bg_info_display()
            messagebox.showinfo("成功", f"已为 {updated_count} 个ROI设置 {channel}nm 背景值")
        else:
            messagebox.showwarning("警告", "未能设置任何背景值，请检查可见区域是否包含数据")

    def _reset_current_backgrounds(self):
        """重置当前页面ROI的背景为自动"""
        current_rois = self._get_current_rois()
        for roi_id in current_rois:
            self.bg_corrector.clear_manual_background(roi_id)
            self._clear_cache(roi_id)

        self._update_plots()
        self._update_bg_info_display()
        messagebox.showinfo("成功", "已重置当前页面ROI为自动背景")

    def _update_bg_info_display(self):
        """更新背景信息显示"""
        current_rois = self._get_current_rois()

        self.bg_info_text.config(state=tk.NORMAL)
        self.bg_info_text.delete(1.0, tk.END)

        # 如果背景校正未启用，显示提示信息
        if not self.bg_correction_enabled.get():
            self.bg_info_text.insert(tk.END, "背景校正已关闭\n\n")
            self.bg_info_text.config(state=tk.DISABLED)
            return

        for roi_id in current_rois:
            signals = self._get_signals_cached(roi_id)

            # 构建原始信号字典
            raw_signals = {
                '532': signals.get('raw_532'),
                '638': signals.get('raw_638')
            }
            if self.is_three_color:
                raw_signals['488'] = signals.get('raw_488')

            bg_info = self.bg_corrector.get_background_info(roi_id, raw_signals, self.is_three_color)

            info_parts = [f"ROI #{roi_id}:"]
            for channel in (['488', '532', '638'] if self.is_three_color else ['532', '638']):
                if channel in bg_info:
                    value, is_manual = bg_info[channel]
                    status = "(手动)" if is_manual else "(自动)"
                    info_parts.append(f"  {channel}nm: {value:.1f} {status}")

            self.bg_info_text.insert(tk.END, "\n".join(info_parts) + "\n\n")

        self.bg_info_text.config(state=tk.DISABLED)

    # ========================================================================
    # 校正系数更新
    # ========================================================================

    def _update_crosstalk(self, value):
        """更新串色校正系数"""
        self.crosstalk_coef = float(value)
        self.signal_processor.crosstalk_coef = self.crosstalk_coef

        if self.is_three_color:
            self.crosstalk_value_label.config(text=f"Crosstalk: {self.crosstalk_coef:.2f}")

        self._clear_cache()
        self._update_plots()

    def _update_leaking(self, value):
        """更新泄漏校正系数"""
        self.leaking_coef = float(value)
        self.signal_processor.leaking_coef = self.leaking_coef
        self.leaking_value_label.config(text=f"Leaking: {self.leaking_coef:.2f}")

        self._clear_cache()
        self._update_plots()

    # ========================================================================
    # 筛选相关
    # ========================================================================

    def _get_filter_params(self):
        """获取筛选参数"""
        params = {}

        try:
            params['threshold_532'] = float(self.entry_532_threshold.get())
            params['frames_532'] = int(self.entry_532_frames.get())
            params['threshold_638'] = float(self.entry_638_threshold.get())
            params['frames_638'] = int(self.entry_638_frames.get())

            if self.is_three_color:
                params['threshold_488'] = float(self.entry_488_threshold.get())
                params['frames_488'] = int(self.entry_488_frames.get())
                params['crosstalk_coef'] = self.crosstalk_coef
            else:
                params['crosstalk_coef'] = 0.0

            params['leaking_coef'] = self.leaking_coef

            overlap_cfg = {
                'enabled': self.overlap_enabled.get(),
                'channels': {},
                'overlap_frames': None
            }

            if overlap_cfg['enabled']:
                selected_channels = [
                    ch for ch, var in self.overlap_channel_vars.items() if var.get()
                ]

                if len(selected_channels) < 2:
                    raise ValueError("时间重叠筛选至少需要选择两个通道！")

                for ch in selected_channels:
                    entries = self.overlap_channel_entries[ch]
                    threshold_str = entries['threshold'].get().strip()
                    frames_str = entries['frames'].get().strip()

                    if not threshold_str or not frames_str:
                        raise ValueError(f"请为 {self.channel_display_names[ch]} 通道填写阈值与连续帧数！")

                    threshold = float(threshold_str)
                    frames = int(frames_str)

                    if threshold <= 0 or frames <= 0:
                        raise ValueError(f"{self.channel_display_names[ch]} 的阈值和连续帧必须为正数！")

                    overlap_cfg['channels'][ch] = {
                        'threshold': threshold,
                        'frames': frames
                    }

                overlap_frames_str = self.overlap_frames_entry.get().strip()
                if not overlap_frames_str:
                    raise ValueError("请填写最小重叠帧数！")

                overlap_frames = int(overlap_frames_str)
                if overlap_frames <= 0:
                    raise ValueError("最小重叠帧数必须为正整数！")

                overlap_cfg['overlap_frames'] = overlap_frames

            params['overlap_filter'] = overlap_cfg
            return params

        except ValueError as e:
            raise ValueError(str(e))

    def _execute_filter(self):
        """执行筛选"""
        try:
            params = self._get_filter_params()
            self.passed_rois = self.filter_engine.filter_all_rois(params)
            self.is_filtered = True

            pass_rate = len(self.passed_rois) / self.total_rois * 100
            overlap_cfg = params.get('overlap_filter', {})
            overlap_text = "启用" if overlap_cfg.get('enabled') else "关闭"
            status_text = (
                f"筛选状态: 已筛选 | 通过: {len(self.passed_rois)}/{self.total_rois} "
                f"({pass_rate:.1f}%) | 时间重叠: {overlap_text}"
            )
            self.filter_status_label.config(text=status_text, fg="green")

            self._save_current_config(params)
            self._update_plots()

            messagebox.showinfo(
                "筛选完成",
                f"筛选完成！\n通过: {len(self.passed_rois)}/{self.total_rois} ({pass_rate:.1f}%)"
            )

        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
        except Exception as e:
            messagebox.showerror("筛选失败", f"筛选过程出错:\n{str(e)}")

    def _reset_filter(self):
        """重置筛选"""
        self.is_filtered = False
        self.passed_rois = []

        self.entry_532_threshold.delete(0, tk.END)
        self.entry_532_frames.delete(0, tk.END)
        self.entry_638_threshold.delete(0, tk.END)
        self.entry_638_frames.delete(0, tk.END)

        if self.is_three_color:
            self.entry_488_threshold.delete(0, tk.END)
            self.entry_488_frames.delete(0, tk.END)

        self.overlap_enabled.set(False)
        for var in self.overlap_channel_vars.values():
            var.set(False)
        for entries in self.overlap_channel_entries.values():
            entries['threshold'].delete(0, tk.END)
            entries['frames'].delete(0, tk.END)
        if self.overlap_frames_entry:
            self.overlap_frames_entry.delete(0, tk.END)

        self.filter_status_label.config(text="筛选状态: 未筛选", fg="gray")
        self._save_current_config()
        self._update_plots()

    def _save_filter_results(self):
        """保存筛选结果"""
        if not self.is_filtered:
            messagebox.showwarning("警告", "请先执行筛选！")
            return

        try:
            input_path = self.loader.output_path
            input_filename = os.path.basename(input_path)
            base_name = os.path.splitext(input_filename)[0]

            output_dir = self.default_directory
            base_path = os.path.join(output_dir, base_name)

            # 准备导出数据
            export_rows = []
            for roi_id in self.passed_rois:
                signals = self._get_signals_cached(roi_id)
                time = signals['time']
                n_frames = len(time)

                for i in range(n_frames):
                    row = {
                        'ROI_ID': roi_id,
                        'Time_sec': time[i]
                    }

                    # 添加信号
                    if self.is_three_color and signals.get('bg_corrected_488') is not None:
                        row['Net_488'] = signals['bg_corrected_488'][i]

                    if signals.get('bg_corrected_532') is not None:
                        row['Net_532'] = signals['bg_corrected_532'][i]

                    if signals.get('bg_corrected_638') is not None:
                        row['Net_638'] = signals['bg_corrected_638'][i]

                    if signals.get('corrected_532') is not None:
                        row['Corrected_532'] = signals['corrected_532'][i]

                    if signals.get('corrected_638') is not None:
                        row['Corrected_638'] = signals['corrected_638'][i]

                    export_rows.append(row)

            export_df = pd.DataFrame(export_rows)
            filtered_csv_path = base_path + '_filtered.csv'
            export_df.to_csv(filtered_csv_path, index=False, sep=',')

            params = self._get_filter_params()
            self._save_current_config(params)

            # 生成报告
            report_path = base_path + '_filter_report.txt'
            self._generate_report(report_path, params)

            messagebox.showinfo(
                "保存成功",
                f"筛选结果已保存到:\n{output_dir}\n\n"
                f"数据文件: {os.path.basename(filtered_csv_path)}\n"
                f"报告文件: {os.path.basename(report_path)}"
            )

        except Exception as e:
            messagebox.showerror("保存失败", f"保存筛选结果时出错:\n{str(e)}")

    def _generate_report(self, report_path, params):
        """生成筛选报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("共定位信号筛选报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("背景校正参数:\n")
            f.write(f"- 背景校正状态: {'启用' if self.bg_correction_enabled.get() else '关闭'}\n")
            if self.bg_correction_enabled.get():
                f.write(f"- 跳过最低帧数: {self.bg_corrector.skip_lowest}\n")
                f.write(f"- 取样帧数: {self.bg_corrector.take_count}\n")
            f.write("\n")

            f.write("筛选条件:\n")
            f.write(f"- 数据类型: {'三色' if self.is_three_color else '双色'}\n")
            f.write(f"- 532nm 阈值: {params['threshold_532']}, 连续帧: {params['frames_532']}\n")
            f.write(f"- 638nm 阈值: {params['threshold_638']}, 连续帧: {params['frames_638']}\n")

            if self.is_three_color:
                f.write(f"- 488nm 阈值: {params['threshold_488']}, 连续帧: {params['frames_488']}\n")
                f.write(f"- 串色校正系数 (488→532): {params['crosstalk_coef']:.2f}\n")

            f.write(f"- 泄漏校正系数 (532→638): {params['leaking_coef']:.2f}\n")

            overlap_cfg = params.get('overlap_filter', {})
            if overlap_cfg.get('enabled'):
                f.write("\n时间重叠筛选:\n")
                f.write(f"- 最小重叠帧数: {overlap_cfg['overlap_frames']}\n")
                f.write("- 通道条件:\n")
                for ch, ch_params in overlap_cfg['channels'].items():
                    name = self.channel_display_names[ch]
                    f.write(f"  * {name}: 阈值 {ch_params['threshold']}, 连续帧 {ch_params['frames']}\n")
            else:
                f.write("\n时间重叠筛选: 未启用\n")

            f.write("\n筛选结果:\n")
            f.write(f"- 原始 ROI 总数: {self.total_rois}\n")
            f.write(f"- 通过筛选: {len(self.passed_rois)}\n")
            f.write(f"- 筛选率: {len(self.passed_rois)/self.total_rois*100:.1f}%\n")

            f.write("\n通过筛选的 ROI 列表:\n")
            f.write(", ".join(map(str, self.passed_rois)) + "\n")

            f.write("\n导出数据说明:\n")
            if self.bg_correction_enabled.get():
                f.write("- Net_488/532/638: 背景校正后的信号\n")
            else:
                f.write("- Net_488/532/638: 原始信号（背景校正已关闭）\n")
            f.write("- Corrected_532: 串色校正后的532nm信号\n")
            f.write("- Corrected_638: 泄漏校正后的638nm信号\n")

    # ========================================================================
    # 配置持久化
    # ========================================================================

    def _load_initial_config(self):
        """加载初始配置"""
        config = self.config_manager.load()
        self.loaded_config = config
        if not config:
            return

        def _set_entry(entry_widget, value):
            if entry_widget is None:
                return
            entry_widget.delete(0, tk.END)
            if value is not None:
                entry_widget.insert(0, str(value))

        entries = config.get('entries', {})
        _set_entry(self.entry_532_threshold, entries.get('threshold_532', ''))
        _set_entry(self.entry_532_frames, entries.get('frames_532', ''))
        _set_entry(self.entry_638_threshold, entries.get('threshold_638', ''))
        _set_entry(self.entry_638_frames, entries.get('frames_638', ''))

        if self.is_three_color:
            _set_entry(self.entry_488_threshold, entries.get('threshold_488', ''))
            _set_entry(self.entry_488_frames, entries.get('frames_488', ''))

        self.show_smoothed.set(bool(config.get('show_smoothed', self.show_smoothed.get())))
        self.smooth_window.set(int(config.get('smooth_window', self.smooth_window.get())))
        self.show_532.set(bool(config.get('show_532', self.show_532.get())))
        self.show_638.set(bool(config.get('show_638', self.show_638.get())))
        self.show_corrected_532.set(bool(config.get('show_corrected_532', self.show_corrected_532.get())))
        self.show_corrected_638.set(bool(config.get('show_corrected_638', self.show_corrected_638.get())))

        if self.is_three_color:
            self.show_488.set(bool(config.get('show_488', self.show_488.get())))
            self.crosstalk_coef = float(config.get('crosstalk_coef', self.crosstalk_coef))
            self.crosstalk_slider.set(self.crosstalk_coef)
            self.crosstalk_value_label.config(text=f"Crosstalk: {self.crosstalk_coef:.2f}")

        self.leaking_coef = float(config.get('leaking_coef', self.leaking_coef))
        self.leaking_slider.set(self.leaking_coef)
        self.leaking_value_label.config(text=f"Leaking: {self.leaking_coef:.2f}")

        self.signal_processor.crosstalk_coef = self.crosstalk_coef
        self.signal_processor.leaking_coef = self.leaking_coef

        # 加载背景校正开关状态
        self.bg_correction_enabled.set(bool(config.get('bg_correction_enabled', False)))
        self.signal_processor.bg_correction_enabled = self.bg_correction_enabled.get()

        bg_params = config.get('bg_params', {})
        if bg_params:
            self.bg_skip_spinbox.delete(0, tk.END)
            self.bg_skip_spinbox.insert(0, str(bg_params.get('skip_lowest', 30)))
            self.bg_take_spinbox.delete(0, tk.END)
            self.bg_take_spinbox.insert(0, str(bg_params.get('take_count', 20)))
            self.bg_corrector.skip_lowest = bg_params.get('skip_lowest', 30)
            self.bg_corrector.take_count = bg_params.get('take_count', 20)

        overlap_ui = config.get('overlap_ui', {})
        self.overlap_enabled.set(overlap_ui.get('enabled', self.overlap_enabled.get()))
        if self.overlap_frames_entry:
            _set_entry(self.overlap_frames_entry, overlap_ui.get('overlap_frames', ''))

        channel_settings = overlap_ui.get('channels', {})
        for ch, data in channel_settings.items():
            if ch not in self.overlap_channel_vars:
                continue
            self.overlap_channel_vars[ch].set(bool(data.get('selected', False)))
            entries_widgets = self.overlap_channel_entries[ch]
            _set_entry(entries_widgets['threshold'], data.get('threshold', ''))
            _set_entry(entries_widgets['frames'], data.get('frames', ''))

    def _collect_config_snapshot(self, params=None):
        """收集配置快照"""
        entries_snapshot = {
            'threshold_532': self.entry_532_threshold.get().strip(),
            'frames_532': self.entry_532_frames.get().strip(),
            'threshold_638': self.entry_638_threshold.get().strip(),
            'frames_638': self.entry_638_frames.get().strip()
        }
        if self.is_three_color:
            entries_snapshot['threshold_488'] = self.entry_488_threshold.get().strip()
            entries_snapshot['frames_488'] = self.entry_488_frames.get().strip()

        overlap_snapshot = {
            'enabled': self.overlap_enabled.get(),
            'overlap_frames': self.overlap_frames_entry.get().strip() if self.overlap_frames_entry else "",
            'channels': {}
        }
        for ch in self.overlap_channel_vars:
            overlap_snapshot['channels'][ch] = {
                'selected': self.overlap_channel_vars[ch].get(),
                'threshold': self.overlap_channel_entries[ch]['threshold'].get().strip(),
                'frames': self.overlap_channel_entries[ch]['frames'].get().strip()
            }

        snapshot = {
            'is_three_color': self.is_three_color,
            'bg_correction_enabled': self.bg_correction_enabled.get(),
            'show_smoothed': self.show_smoothed.get(),
            'smooth_window': self.smooth_window.get(),
            'show_532': self.show_532.get(),
            'show_638': self.show_638.get(),
            'show_corrected_532': self.show_corrected_532.get(),
            'show_corrected_638': self.show_corrected_638.get(),
            'entries': entries_snapshot,
            'overlap_ui': overlap_snapshot,
            'crosstalk_coef': self.crosstalk_coef,
            'leaking_coef': self.leaking_coef,
            'bg_params': {
                'skip_lowest': self.bg_corrector.skip_lowest,
                'take_count': self.bg_corrector.take_count
            }
        }

        if self.is_three_color:
            snapshot['show_488'] = self.show_488.get()

        if params is not None:
            snapshot['last_valid_params'] = params

        return snapshot

    def _save_current_config(self, params=None):
        """保存当前配置"""
        snapshot = self._collect_config_snapshot(params)
        self.config_manager.save(snapshot)
