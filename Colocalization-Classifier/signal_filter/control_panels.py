"""控制面板模块 - 包含所有右侧控制面板"""
import tkinter as tk
from tkinter import ttk, simpledialog
from ui_components import UIHelper


class ControlPanels:
    """控制面板管理器"""
    
    def __init__(self, parent, gui_ref):
        self.parent = parent
        self.gui = gui_ref  # 主GUI的引用
        self.ui_helper = UIHelper()
        
    def create_all_panels(self):
        """创建所有控制面板"""
        self.create_navigation_panel()
        self.create_background_panel()
        self.create_smooth_panel()
        self.create_channel_panel()
        self.create_correction_panel()
        self.create_overlap_panel()
        self.create_filter_panel()
    
    def create_navigation_panel(self):
        """创建导航面板"""
        panel = self.ui_helper.create_collapsible_panel(self.parent, "导航控制")
        system_font = self.ui_helper.get_system_font()

        self.gui.page_label = tk.Label(
            panel,
            text=f"第 {self.gui.current_page + 1}/{self.gui.total_pages} 页",
            font=system_font
        )
        self.gui.page_label.pack(pady=5)

        btn_frame = tk.Frame(panel)
        btn_frame.pack(pady=5)

        self.gui.prev_btn = tk.Button(btn_frame, text="◀ 上一页", 
                                       command=self.gui._prev_page, width=10)
        self.gui.prev_btn.pack(side=tk.LEFT, padx=5)

        self.gui.next_btn = tk.Button(btn_frame, text="下一页 ▶", 
                                       command=self.gui._next_page, width=10)
        self.gui.next_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(
            panel,
            text=f"共 {self.gui.total_rois} 个 ROI",
            font=(system_font[0], 9),
            fg="gray"
        ).pack(pady=5)

        tk.Label(
            panel,
            text="快捷键：← 上一页 / → 下一页",
            font=(system_font[0], 9),
            fg="#555555"
        ).pack(pady=2)

        self.gui._update_navigation_buttons()
    
    def create_background_panel(self):
        """创建背景校正面板"""
        panel = self.ui_helper.create_collapsible_panel(self.parent, "背景校正")
        system_font = self.ui_helper.get_system_font()

        # 背景校正开关
        bg_switch_frame = tk.Frame(panel)
        bg_switch_frame.pack(fill=tk.X, pady=(5, 10))
        
        tk.Checkbutton(
            bg_switch_frame,
            text="启用背景校正",
            variable=self.gui.bg_correction_enabled,
            command=self.gui._toggle_bg_correction,
            font=(system_font[0], 10, "bold")
        ).pack(side=tk.LEFT)

        tk.Label(
            panel,
            text="💡 关闭后，串色和泄漏校正将基于原始信号",
            font=(system_font[0], 8),
            fg="#FF6600"
        ).pack(anchor=tk.W, pady=(0, 10))

        # 分隔线
        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=5)

        # 自动背景估算参数
        param_label = tk.Label(panel, text="自动背景估算参数:", 
                              font=(system_font[0], 9, "bold"))
        param_label.pack(anchor=tk.W, pady=(5, 2))

        skip_frame = tk.Frame(panel)
        skip_frame.pack(fill=tk.X, pady=2)
        tk.Label(skip_frame, text="跳过最低:").pack(side=tk.LEFT)
        self.gui.bg_skip_spinbox = tk.Spinbox(skip_frame, from_=0, to=100, width=5)
        self.gui.bg_skip_spinbox.delete(0, tk.END)
        self.gui.bg_skip_spinbox.insert(0, "30")
        self.gui.bg_skip_spinbox.pack(side=tk.LEFT, padx=5)
        tk.Label(skip_frame, text="帧").pack(side=tk.LEFT)

        take_frame = tk.Frame(panel)
        take_frame.pack(fill=tk.X, pady=2)
        tk.Label(take_frame, text="取接下来:").pack(side=tk.LEFT)
        self.gui.bg_take_spinbox = tk.Spinbox(take_frame, from_=1, to=100, width=5)
        self.gui.bg_take_spinbox.delete(0, tk.END)
        self.gui.bg_take_spinbox.insert(0, "20")
        self.gui.bg_take_spinbox.pack(side=tk.LEFT, padx=5)
        tk.Label(take_frame, text="帧").pack(side=tk.LEFT)

        tk.Label(
            panel,
            text="💡 例：跳过30帧，取31-50帧的中位数",
            font=(system_font[0], 8),
            fg="#666666"
        ).pack(anchor=tk.W, pady=2)

        # 应用参数按钮
        apply_param_btn = tk.Button(
            panel,
            text="应用背景参数",
            command=self.gui._apply_bg_params,
            width=15
        )
        apply_param_btn.pack(pady=5)

        # 分隔线
        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=10)

        # 当前ROI背景信息
        bg_info_label = tk.Label(panel, text="当前页面ROI背景值:", 
                                font=(system_font[0], 9, "bold"))
        bg_info_label.pack(anchor=tk.W, pady=(5, 2))

        self.gui.bg_info_text = tk.Text(panel, height=6, width=40, state=tk.DISABLED,
                                        font=(system_font[0], 9), bg='#f8f8f8')
        self.gui.bg_info_text.pack(fill=tk.X, pady=5)

        # 手动设置背景提示
        tk.Label(
            panel,
            text="💡 放大到背景区域后点击下方按钮",
            font=(system_font[0], 8),
            fg="#0066cc"
        ).pack(anchor=tk.W, pady=2)

        # 手动设置按钮
        manual_btn_frame = tk.Frame(panel)
        manual_btn_frame.pack(fill=tk.X, pady=5)

        if self.gui.is_three_color:
            tk.Button(
                manual_btn_frame,
                text="设置488背景",
                command=lambda: self.gui._set_manual_background('488'),
                width=10
            ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            manual_btn_frame,
            text="设置532背景",
            command=lambda: self.gui._set_manual_background('532'),
            width=10
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            manual_btn_frame,
            text="设置638背景",
            command=lambda: self.gui._set_manual_background('638'),
            width=10
        ).pack(side=tk.LEFT, padx=2)

        # 重置按钮
        reset_btn_frame = tk.Frame(panel)
        reset_btn_frame.pack(fill=tk.X, pady=5)

        tk.Button(
            reset_btn_frame,
            text="重置当前页面ROI为自动背景",
            command=self.gui._reset_current_backgrounds,
            width=30
        ).pack()
    
    def create_smooth_panel(self):
        """创建平滑面板"""
        panel = self.ui_helper.create_collapsible_panel(self.parent, "曲线平滑")

        tk.Checkbutton(
            panel,
            text="启用 Savitzky-Golay 平滑",
            variable=self.gui.show_smoothed,
            command=self.gui._update_plots
        ).pack(anchor=tk.W, pady=5)

        window_frame = tk.Frame(panel)
        window_frame.pack(fill=tk.X, pady=5)

        tk.Label(window_frame, text="窗口大小:").pack(side=tk.LEFT)

        window_spinbox = tk.Spinbox(
            window_frame,
            from_=3,
            to=21,
            increment=2,
            textvariable=self.gui.smooth_window,
            width=5,
            command=self.gui._update_plots
        )
        window_spinbox.pack(side=tk.LEFT, padx=5)
    
    def create_channel_panel(self):
        """创建通道显示面板"""
        if self.gui.is_three_color:
            panel = self.ui_helper.create_collapsible_panel(self.parent, "通道显示（三色）")

            tk.Checkbutton(
                panel,
                text="显示 488nm",
                variable=self.gui.show_488,
                command=self.gui._update_plots
            ).pack(anchor=tk.W, pady=2)

            # 532通道
            frame_532 = tk.Frame(panel)
            frame_532.pack(anchor=tk.W, pady=2)

            tk.Checkbutton(
                frame_532,
                text="显示 532nm",
                variable=self.gui.show_532,
                command=self.gui._update_plots
            ).pack(side=tk.LEFT)

            # 只在背景校正启用时显示单选按钮
            if self.gui.bg_correction_enabled.get():
                tk.Radiobutton(
                    frame_532,
                    text="原始",
                    variable=self.gui.show_corrected_532,
                    value=False,
                    command=self.gui._update_plots
                ).pack(side=tk.LEFT, padx=5)

            tk.Radiobutton(
                frame_532,
                text="校正",
                variable=self.gui.show_corrected_532,
                value=True,
                command=self.gui._update_plots
            ).pack(side=tk.LEFT)

            # 638通道
            frame_638 = tk.Frame(panel)
            frame_638.pack(anchor=tk.W, pady=2)

            tk.Checkbutton(
                frame_638,
                text="显示 638nm",
                variable=self.gui.show_638,
                command=self.gui._update_plots
            ).pack(side=tk.LEFT)

            # 只在背景校正启用时显示单选按钮
            if self.gui.bg_correction_enabled.get():
                tk.Radiobutton(
                    frame_638,
                    text="原始",
                    variable=self.gui.show_corrected_638,
                    value=False,
                    command=self.gui._update_plots
                ).pack(side=tk.LEFT, padx=5)

            tk.Radiobutton(
                frame_638,
                text="校正",
                variable=self.gui.show_corrected_638,
                value=True,
                command=self.gui._update_plots
            ).pack(side=tk.LEFT)
        else:
            panel = self.ui_helper.create_collapsible_panel(self.parent, "通道显示（双色）")

            # 532通道
            frame_532 = tk.Frame(panel)
            frame_532.pack(anchor=tk.W, pady=2)

            tk.Checkbutton(
                frame_532,
                text="显示 532nm",
                variable=self.gui.show_532,
                command=self.gui._update_plots
            ).pack(side=tk.LEFT)

            # 只在背景校正启用时显示单选按钮
            if self.gui.bg_correction_enabled.get():
                tk.Radiobutton(
                    frame_532,
                    text="原始",
                    variable=self.gui.show_corrected_532,
                    value=False,
                    command=self.gui._update_plots
                ).pack(side=tk.LEFT, padx=5)

            tk.Radiobutton(
                frame_532,
                text="校正",
                variable=self.gui.show_corrected_532,
                value=True,
                command=self.gui._update_plots
            ).pack(side=tk.LEFT)

            # 638通道
            frame_638 = tk.Frame(panel)
            frame_638.pack(anchor=tk.W, pady=2)

            tk.Checkbutton(
                frame_638,
                text="显示 638nm",
                variable=self.gui.show_638,
                command=self.gui._update_plots
            ).pack(side=tk.LEFT)

            # 只在背景校正启用时显示单选按钮
            if self.gui.bg_correction_enabled.get():
                tk.Radiobutton(
                    frame_638,
                    text="原始",
                    variable=self.gui.show_corrected_638,
                    value=False,
                    command=self.gui._update_plots
                ).pack(side=tk.LEFT, padx=5)

            tk.Radiobutton(
                frame_638,
                text="校正",
                variable=self.gui.show_corrected_638,
                value=True,
                command=self.gui._update_plots
            ).pack(side=tk.LEFT)
    
    def create_correction_panel(self):
        """创建校正参数面板"""
        panel = self.ui_helper.create_collapsible_panel(self.parent, "信号校正参数")
        system_font = self.ui_helper.get_system_font()

        # 串色校正（仅三色）
        if self.gui.is_three_color:
            tk.Label(panel, text="488→532 串色校正:", 
                    font=(system_font[0], 9, "bold")).pack(anchor=tk.W, pady=(5, 2))

            self.gui.crosstalk_slider = tk.Scale(
                panel,
                from_=0.0,
                to=1.0,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                command=self.gui._update_crosstalk
            )
            self.gui.crosstalk_slider.set(self.gui.crosstalk_coef)
            self.gui.crosstalk_slider.pack(fill=tk.X)

            self.gui.crosstalk_value_label = tk.Label(
                panel,
                text=f"Crosstalk: {self.gui.crosstalk_coef:.2f}",
                font=(system_font[0], 9),
                fg="blue"
            )
            self.gui.crosstalk_value_label.pack(pady=2)

        # 泄漏校正
        tk.Label(panel, text="532→638 泄漏校正:", 
                font=(system_font[0], 9, "bold")).pack(anchor=tk.W, pady=(10, 2))

        self.gui.leaking_slider = tk.Scale(
            panel,
            from_=0.0,
            to=0.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self.gui._update_leaking
        )
        self.gui.leaking_slider.set(self.gui.leaking_coef)
        self.gui.leaking_slider.pack(fill=tk.X)

        self.gui.leaking_value_label = tk.Label(
            panel,
            text=f"Leaking: {self.gui.leaking_coef:.2f}",
            font=(system_font[0], 9),
            fg="blue"
        )
        self.gui.leaking_value_label.pack(pady=2)

        tk.Label(
            panel,
            text="💡 校正顺序: 背景→串色(488→532)→泄漏(532→638)",
            font=(system_font[0], 8),
            fg="#666666"
        ).pack(anchor=tk.W, pady=5)
    
    def create_overlap_panel(self):
        """创建时间重叠筛选面板"""
        panel = self.ui_helper.create_collapsible_panel(self.parent, "时间重叠筛选")

        tk.Checkbutton(
            panel,
            text="启用时间重叠筛选",
            variable=self.gui.overlap_enabled
        ).pack(anchor=tk.W, pady=5)

        tk.Label(
            panel,
            text="至少选择两个通道，并填写各自阈值与连续帧数。",
            fg="#666666",
            font=("Arial", 9)
        ).pack(anchor=tk.W, pady=2)

        channels = []
        if self.gui.is_three_color:
            channels.append('488')
        channels.append('532')
        channels.append('638')

        for ch in channels:
            frame = tk.Frame(panel)
            frame.pack(fill=tk.X, pady=4)

            var = tk.BooleanVar(value=False)
            self.gui.overlap_channel_vars[ch] = var

            tk.Checkbutton(
                frame,
                text=f"{self.gui.channel_display_names[ch]}",
                variable=var
            ).pack(side=tk.LEFT)

            tk.Label(frame, text="阈值:").pack(side=tk.LEFT, padx=(10, 2))
            thresh_entry = tk.Entry(frame, width=7)
            thresh_entry.pack(side=tk.LEFT)

            tk.Label(frame, text="连续帧:").pack(side=tk.LEFT, padx=(10, 2))
            frames_entry = tk.Entry(frame, width=5)
            frames_entry.pack(side=tk.LEFT)

            self.gui.overlap_channel_entries[ch] = {
                'threshold': thresh_entry,
                'frames': frames_entry
            }

        overlap_frame = tk.Frame(panel)
        overlap_frame.pack(fill=tk.X, pady=(10, 2))

        tk.Label(overlap_frame, text="最小重叠帧数:").pack(side=tk.LEFT)
        self.gui.overlap_frames_entry = tk.Entry(overlap_frame, width=8)
        self.gui.overlap_frames_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(
            panel,
            text="所有勾选通道需存在长度 ≥ 设定值的共同重叠片段。",
            fg="#666666",
            font=("Arial", 9)
        ).pack(anchor=tk.W, pady=2)
    
    def create_filter_panel(self):
        """创建筛选参数面板"""
        panel = self.ui_helper.create_collapsible_panel(self.parent, "筛选参数")
        system_font = self.ui_helper.get_system_font()

        tk.Label(
            panel,
            text="💡 筛选使用校正后的信号",
            font=(system_font[0], 8),
            fg="#0066cc"
        ).pack(anchor=tk.W, pady=2)

        tk.Label(panel, text="532nm 通道:", 
                font=(system_font[0], 9, "bold")).pack(anchor=tk.W, pady=(5, 2))
        frame_532_threshold = tk.Frame(panel)
        frame_532_threshold.pack(fill=tk.X, pady=2)
        tk.Label(frame_532_threshold, text="阈值:").pack(side=tk.LEFT)
        self.gui.entry_532_threshold = tk.Entry(frame_532_threshold, width=10)
        self.gui.entry_532_threshold.pack(side=tk.LEFT, padx=5)

        frame_532_frames = tk.Frame(panel)
        frame_532_frames.pack(fill=tk.X, pady=2)
        tk.Label(frame_532_frames, text="连续帧:").pack(side=tk.LEFT)
        self.gui.entry_532_frames = tk.Entry(frame_532_frames, width=10)
        self.gui.entry_532_frames.pack(side=tk.LEFT, padx=5)

        tk.Label(panel, text="638nm 通道:", 
                font=(system_font[0], 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
        frame_638_threshold = tk.Frame(panel)
        frame_638_threshold.pack(fill=tk.X, pady=2)
        tk.Label(frame_638_threshold, text="阈值:").pack(side=tk.LEFT)
        self.gui.entry_638_threshold = tk.Entry(frame_638_threshold, width=10)
        self.gui.entry_638_threshold.pack(side=tk.LEFT, padx=5)

        frame_638_frames = tk.Frame(panel)
        frame_638_frames.pack(fill=tk.X, pady=2)
        tk.Label(frame_638_frames, text="连续帧:").pack(side=tk.LEFT)
        self.gui.entry_638_frames = tk.Entry(frame_638_frames, width=10)
        self.gui.entry_638_frames.pack(side=tk.LEFT, padx=5)

        if self.gui.is_three_color:
            tk.Label(panel, text="488nm 通道:", 
                    font=(system_font[0], 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
            frame_488_threshold = tk.Frame(panel)
            frame_488_threshold.pack(fill=tk.X, pady=2)
            tk.Label(frame_488_threshold, text="阈值:").pack(side=tk.LEFT)
            self.gui.entry_488_threshold = tk.Entry(frame_488_threshold, width=10)
            self.gui.entry_488_threshold.pack(side=tk.LEFT, padx=5)

            frame_488_frames = tk.Frame(panel)
            frame_488_frames.pack(fill=tk.X, pady=2)
            tk.Label(frame_488_frames, text="连续帧:").pack(side=tk.LEFT)
            self.gui.entry_488_frames = tk.Entry(frame_488_frames, width=10)
            self.gui.entry_488_frames.pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(panel)
        btn_frame.pack(pady=15)

        tk.Button(
            btn_frame,
            text="执行筛选",
            command=self.gui._execute_filter,
            bg="#4CAF50",
            fg="white",
            width=12
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="重置筛选",
            command=self.gui._reset_filter,
            bg="#FF9800",
            fg="white",
            width=12
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            panel,
            text="保存筛选结果",
            command=self.gui._save_filter_results,
            bg="#2196F3",
            fg="white",
            width=25
        ).pack(pady=10)

        self.gui.filter_status_label = tk.Label(
            panel,
            text="筛选状态: 未筛选",
            font=(system_font[0], 9),
            fg="gray"
        )
        self.gui.filter_status_label.pack(pady=5)
