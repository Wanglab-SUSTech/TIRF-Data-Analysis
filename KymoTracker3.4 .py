"""  
KymoTracker GUI - Multi-Channel Track Display Version  
Interactive kymograph tracking with multi-channel overlay capability  
Multi-file support with automatic unit conversion  
"""  

import tkinter as tk  
from tkinter import ttk, filedialog, messagebox, scrolledtext  
import matplotlib  
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  
from matplotlib.figure import Figure  
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector  
import lumicks.pylake as lk  
import numpy as np  
import pandas as pd  
from pathlib import Path  
from datetime import datetime  
import math
import csv
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.backend_bases import TimerBase
import tkinter.filedialog as filedialog
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

DNA_BP_TO_NM = 0.34
DNA_UM_TO_BP = 1000.0 / DNA_BP_TO_NM



def polygon_to_mask(vertices, shape):
    """
    将多边形顶点转换为二值掩码 - 修复版本
    
    参数:
    - vertices: [(x_pixel, y_pixel), ...] 多边形顶点列表
    - shape: (height, width) 掩码图像尺寸
    
    返回:
    - mask: (height, width) 布尔型掩码，True=在ROI内，False=在ROI外
    """
    import numpy as np
    
    # 创建空掩码
    mask = np.zeros(shape, dtype=bool)
    
    # 检查顶点数量
    if len(vertices) < 3:
        print(f"⚠️  Warning: 需要至少3个顶点，当前只有 {len(vertices)} 个")
        return mask
    
    # 验证顶点坐标范围
    height, width = shape
    for i, (x_px, y_px) in enumerate(vertices):
        if not (0 <= x_px < width and 0 <= y_px < height):
            print(f"⚠️  Warning: 顶点 {i+1} ({x_px:.1f}, {y_px:.1f}) 超出图像范围 ({width}x{height})")
    
    # 尝试使用 scikit-image（优先选项）
    try:
        from skimage.draw import polygon as sk_polygon
        
        # 提取 x 和 y 坐标
        xs = np.array([v[0] for v in vertices], dtype=np.float32)
        ys = np.array([v[1] for v in vertices], dtype=np.float32)
        
        # 重要：scikit-image 使用 (row, col) 即 (y, x) 顺序
        rr, cc = sk_polygon(ys, xs, shape=shape)
        mask[rr, cc] = True
        
        masked_pixels = np.sum(mask)
        print(f"✓ ROI Mask 创建成功 (scikit-image): {masked_pixels} 像素被掩码")
        return mask
        
    except ImportError:
        print("⚠️  scikit-image 不可用，尝试使用 OpenCV...")
    except Exception as e:
        print(f"⚠️  scikit-image 失败: {e}，尝试使用 OpenCV...")
    
    # 备选方案：使用 OpenCV
    try:
        import cv2
        
        # 转换顶点格式为 OpenCV 需要的格式
        polygon_vertices = np.array(vertices, dtype=np.int32)
        
        # 检查多边形是否有效
        if len(polygon_vertices) < 3:
            return mask
        
        # 用 fillPoly 填充多边形
        cv2.fillPoly(mask, [polygon_vertices], True)
        
        masked_pixels = np.sum(mask)
        print(f"✓ ROI Mask 创建成功 (OpenCV): {masked_pixels} 像素被掩码")
        return mask
        
    except ImportError:
        print("⚠️  OpenCV 也不可用")
    except Exception as e:
        print(f"⚠️  OpenCV 失败: {e}")
    
    # 如果两个都失败，返回空掩码并打印警告
    print("⚠️  警告：两个库都不可用，无法创建多边形掩码！")
    return mask

class CurvedPolygonROI:
    """
    简单的弯曲多边形ROI选择工具
    • 点击添加顶点（用直线连接）
    • 不自动平滑曲线
    • 不自动缩放
    • Z: 撤销最后一个顶点
    • Enter: 完成ROI
    • ESC: 取消
    """
    
    def __init__(self, image, fig, ax, pixel_size_nm, delta_line_time):
        self.image = image
        self.fig = fig
        self.ax = ax
        self.pixel_size_nm = pixel_size_nm
        self.delta_line_time = delta_line_time
        
        self.vertices = []  # [(x_pixel, y_pixel), ...]
        self.vertex_markers = []  # 顶点标记
        self.line_polygon = None  # 连接线
        self.fill_polygon = None  # 填充
        
        self.canvas = fig.canvas
        self.connected = False
    
    def connect(self):
        """连接事件"""
        if self.connected:
            return
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.canvas.mpl_connect('key_press_event', self.on_key)
        self.connected = True
    
    def disconnect(self):
        """断开连接"""
        if self.connected:
            try:
                self.canvas.mpl_disconnect(self.cid_press)
                self.canvas.mpl_disconnect(self.cid_key)
            except:
                pass
            self.connected = False
    
    def on_click(self, event):
        """点击添加顶点"""
        if event.inaxes != self.ax or event.button != 1:
            return
        
        x_real = event.xdata  # 秒
        y_real = event.ydata  # μm
        
        if x_real is None or y_real is None:
            return
        
        # 转换为像素坐标
        x_pixel = x_real / self.delta_line_time
        y_pixel = y_real * 1000 / self.pixel_size_nm
        
        # 防止重复点（距离<0.5像素）
        if len(self.vertices) > 0:
            last_x, last_y = self.vertices[-1]
            distance = np.sqrt((x_pixel - last_x)**2 + (y_pixel - last_y)**2)
            if distance < 0.5:
                return
        
        self.vertices.append((x_pixel, y_pixel))
        
        # 绘制顶点
        marker, = self.ax.plot([x_real], [y_real], 'ro', markersize=10, 
                             zorder=11, markeredgewidth=2, markeredgecolor='darkred')
        self.vertex_markers.append(marker)
        
        # 显示序号
        text = self.ax.text(x_real, y_real + 0.8, f"V{len(self.vertices)}", 
                          fontsize=10, ha='center', va='bottom', fontweight='bold',
                          zorder=12, bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='yellow', edgecolor='orange', alpha=0.8))
        self.vertex_markers.append(text)
        
        self.update_preview()
        self.canvas.draw()
    
    def on_key(self, event):
        """处理键盘"""
        print(f"🔍 Key pressed: {event.key}")  # 调试：打印按下的键
        
        if event.key == 'z' and len(self.vertices) > 0:
            # 撤销最后一个顶点
            print("↩️  Undo: removing last vertex")
            self.vertices.pop()
            # 删除最后两个标记（marker和text）
            for _ in range(2):
                if self.vertex_markers:
                    marker = self.vertex_markers.pop()
                    try:
                        marker.remove()
                    except:
                        pass
            self.update_preview()
            self.canvas.draw()
            
        elif event.key == 'escape':
            # 取消
            print("❌ Cancel: clearing ROI")
            self.clear()
            self.disconnect()
            
        elif event.key in ['enter', 'return']:  # ✅ 添加这一部分
            # ✅ 完成ROI选择
            print(f"✅ Enter pressed: ROI completed with {len(self.vertices)} vertices")
            if len(self.vertices) >= 3:
                self.disconnect()  # 断开事件连接
                print(f"   → ROI is valid, disconnecting...")
            else:
                print(f"   ⚠️  ROI needs at least 3 vertices, but only has {len(self.vertices)}")
    
    def update_preview(self):
        """更新预览（用直线连接）"""
        # 移除旧线
        if self.line_polygon is not None:
            try:
                self.line_polygon.remove()
            except:
                pass
            self.line_polygon = None
        
        if self.fill_polygon is not None:
            try:
                self.fill_polygon.remove()
            except:
                pass
            self.fill_polygon = None
        
        if len(self.vertices) < 2:
            return
        
        # 用直线连接（不平滑）
        xs = [v[0] * self.delta_line_time for v in self.vertices + [self.vertices[0]]]
        ys = [v[1] * self.pixel_size_nm / 1000 for v in self.vertices + [self.vertices[0]]]
        
        self.line_polygon, = self.ax.plot(xs, ys, 'c-', linewidth=2.5, 
                                         alpha=0.8, zorder=8)
        
        if len(self.vertices) >= 3:
            self.fill_polygon = self.ax.fill(xs, ys, 'cyan', alpha=0.12, zorder=7)[0]
    
    def clear(self):
        """清除"""
        self.vertices = []
        
        for marker in self.vertex_markers:
            try:
                marker.remove()
            except:
                pass
        self.vertex_markers = []
        
        if self.line_polygon is not None:
            try:
                self.line_polygon.remove()
            except:
                pass
            self.line_polygon = None
        
        if self.fill_polygon is not None:
            try:
                self.fill_polygon.remove()
            except:
                pass
            self.fill_polygon = None
        
        self.canvas.draw()
class ManualSegmentInputDialog(tk.Toplevel):
    """
    手动输入时间段的对话框 - 优化版
    使用表格式输入，更加直观和方便
    """
    def __init__(self, parent, track_times):
        super().__init__(parent)
        self.title("✏️ Manual Time Segment Input")
        self.geometry("900x650")
        self.transient(parent)
        self.grab_set()
        
        self.result = None
        self.track_times = track_times
        self.segment_rows = []  # 存储每一行的Entry控件
        
        self.create_widgets()
        
        # 对话框居中
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def create_widgets(self):
        """创建界面"""
        # ===== 信息框 =====
        info_frame = ttk.LabelFrame(self, text="📊 Track Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        if self.track_times is not None and len(self.track_times) > 0:
            track_min = self.track_times[0]
            track_max = self.track_times[-1]
            track_duration = track_max - track_min
            
            info_text = f"Track Duration: {track_min:.3f} - {track_max:.3f} s  |  Total: {track_duration:.3f} s"
        else:
            info_text = "No track data available"
        
        ttk.Label(info_frame, text=info_text, foreground='blue',
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')).pack(anchor=tk.W)
        
        # ===== 快速分段按钮 =====
        quick_frame = ttk.LabelFrame(self, text="⚡ Quick Actions", padding=10)
        quick_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(quick_frame, text="Auto-divide into:", 
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=0, column=0, padx=5)
        
        n_segments_var = tk.IntVar(value=3)
        ttk.Spinbox(quick_frame, from_=2, to=10, textvariable=n_segments_var, 
                   width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(quick_frame, text="equal segments", 
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=0, column=2, padx=5)
        
        ttk.Button(quick_frame, text="🔄 Generate",
                  command=lambda: self.auto_divide(n_segments_var.get())).grid(row=0, column=3, padx=10)
        
        ttk.Button(quick_frame, text="🗑️ Clear All",
                  command=self.clear_all_rows).grid(row=0, column=4, padx=5)
        
        # ===== 表格输入区域 =====
        table_frame = ttk.LabelFrame(self, text="📝 Segment Table", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建带滚动条的容器
        container = ttk.Frame(table_frame)
        container.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas用于滚动
        self.canvas = tk.Canvas(container, yscrollcommand=scrollbar.set, 
                               highlightthickness=0, height=300)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.canvas.yview)
        
        # 内部框架
        self.table_inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.table_inner_frame, anchor='nw')
        
        # 表头
        header_labels = ["#", "Start Time (s)", "End Time (s)", "Label", "Color", ""]
        header_widths = [3, 12, 12, 15, 12, 5]
        
        for i, (label, width) in enumerate(zip(header_labels, header_widths)):
            ttk.Label(self.table_inner_frame, text=label, 
                     font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
                     width=width, anchor='center',
                     relief=tk.RIDGE, borderwidth=1).grid(row=0, column=i, sticky='ew', padx=1, pady=1)
        
        # 绑定canvas调整
        self.table_inner_frame.bind('<Configure>', 
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        
        # 鼠标滚轮
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # ===== 底部按钮 =====
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(bottom_frame, text="➕ Add Row",
                  command=self.add_row).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="✅ Confirm & Apply",
                  command=self.on_confirm).pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)
        ttk.Button(bottom_frame, text="❌ Cancel",
                  command=self.on_cancel).pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)
        
        # 默认添加3行
        for _ in range(3):
            self.add_row()
    
    def add_row(self, start_val="", end_val="", label_val="", color_val=""):
        """添加一行输入"""
        row_idx = len(self.segment_rows) + 1
        
        # 如果没有提供值，自动计算
        if not start_val and self.track_times is not None and len(self.track_times) > 0:
            track_min = self.track_times[0]
            track_max = self.track_times[-1]
            
            if row_idx == 1:
                # 🆕 使用3位小数精度，与轨迹时间对齐
                start_val = f"{track_min:.3f}"
                end_val = f"{(track_min + track_max)/2:.3f}"
            else:
                # 从上一行的end_val开始
                prev_end = self.segment_rows[-1]['end_entry'].get()
                start_val = prev_end
                try:
                    # 🆕 确保不超过轨迹最大值
                    proposed_end = float(prev_end) + 10
                    if proposed_end > track_max:
                        end_val = f"{track_max:.1f}"
                    else:
                        end_val = f"{proposed_end:.1f}"
                except:
                    end_val = f"{track_max:.1f}"
        
        if not label_val:
            label_val = f"Seg{row_idx}"
        
        # 🆕 自动循环选择颜色
        if not color_val:
            colors = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "brown"]
            color_val = colors[(row_idx - 1) % len(colors)]
        
        row_widgets = {}
        
        # 行号
        ttk.Label(self.table_inner_frame, text=str(row_idx), 
                 width=3, anchor='center').grid(row=row_idx, column=0, padx=1, pady=2)
        
        # Start Time
        start_entry = ttk.Entry(self.table_inner_frame, width=12)
        start_entry.insert(0, start_val)
        start_entry.grid(row=row_idx, column=1, padx=2, pady=2)
        row_widgets['start_entry'] = start_entry
        
        # End Time
        end_entry = ttk.Entry(self.table_inner_frame, width=12)
        end_entry.insert(0, end_val)
        end_entry.grid(row=row_idx, column=2, padx=2, pady=2)
        row_widgets['end_entry'] = end_entry
        
        # Label
        label_entry = ttk.Entry(self.table_inner_frame, width=15)
        label_entry.insert(0, label_val)
        label_entry.grid(row=row_idx, column=3, padx=2, pady=2)
        row_widgets['label_entry'] = label_entry
        
        # Color
        colors = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "brown"]
        color_combo = ttk.Combobox(self.table_inner_frame, values=colors, 
                                   width=10, state="readonly")
        color_combo.set(color_val)
        color_combo.grid(row=row_idx, column=4, padx=2, pady=2)
        row_widgets['color_combo'] = color_combo
        
        # 删除按钮
        del_btn = ttk.Button(self.table_inner_frame, text="🗑️", width=3,
                            command=lambda: self.delete_row(row_widgets))
        del_btn.grid(row=row_idx, column=5, padx=2, pady=2)
        row_widgets['del_btn'] = del_btn
        
        self.segment_rows.append(row_widgets)
        
        # 更新滚动区域
        self.table_inner_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def delete_row(self, row_widgets):
        """删除一行"""
        # 删除控件
        for widget in row_widgets.values():
            widget.destroy()
        
        # 从列表中移除
        self.segment_rows.remove(row_widgets)
        
        # 重新编号
        self.renumber_rows()
    
    def renumber_rows(self):
        """重新编号所有行"""
        for idx, row in enumerate(self.segment_rows, 1):
            # 更新行号（需要找到对应的Label）
            for widget in self.table_inner_frame.winfo_children():
                if isinstance(widget, ttk.Label) and widget.grid_info().get('row') == idx and widget.grid_info().get('column') == 0:
                    widget.config(text=str(idx))
    
    def clear_all_rows(self):
        """清除所有行"""
        if not messagebox.askyesno("Confirm", "Clear all rows?"):
            return
        
        for row_widgets in self.segment_rows[:]:
            for widget in row_widgets.values():
                widget.destroy()
        
        self.segment_rows.clear()
    
    def auto_divide(self, n_segments):
        """自动等分轨迹"""
        if self.track_times is None or len(self.track_times) == 0:
            messagebox.showwarning("Warning", "No track data available")
            return
        
        # 清空现有行
        self.clear_all_rows()
        
        track_min = self.track_times[0]
        track_max = self.track_times[-1]
        duration = track_max - track_min
        seg_duration = duration / n_segments
        
        colors = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "brown"]
        
        for i in range(n_segments):
            # 🆕 第一段直接用track_min，最后一段直接用track_max
            if i == 0:
                start_t = track_min
            else:
                start_t = track_min + i * seg_duration
            
            if i == n_segments - 1:
                end_t = track_max
            else:
                end_t = track_min + (i + 1) * seg_duration
            
            label = f"Seg{i+1}"
            color = colors[i % len(colors)]
            
            self.add_row(f"{start_t:.3f}", f"{end_t:.3f}", label, color)
    
    def on_confirm(self):
        """验证并确认输入"""
        try:
            segments = []
            
            for idx, row in enumerate(self.segment_rows, 1):
                start_str = row['start_entry'].get().strip()
                end_str = row['end_entry'].get().strip()
                label = row['label_entry'].get().strip()
                color = row['color_combo'].get()
                
                if not start_str or not end_str or not label:
                    raise ValueError(f"Row {idx}: All fields are required")
                
                try:
                    start_t = float(start_str)
                    end_t = float(end_str)
                except ValueError:
                    raise ValueError(f"Row {idx}: Start and End must be numbers")
                
                if end_t <= start_t:
                    raise ValueError(f"Row {idx}: End time must be greater than Start time")
                
                # 验证时间范围（添加浮点数容差）
                if self.track_times is not None and len(self.track_times) > 0:
                    track_min = self.track_times[0]
                    track_max = self.track_times[-1]
                    
                    # 🆕 添加0.001秒的容差来处理浮点数精度问题
                    tolerance = 0.001
                    
                    if start_t < track_min - tolerance:
                        raise ValueError(
                            f"Row {idx}: Start time {start_t:.3f}s is before track start {track_min:.3f}s"
                        )
                    
                    if end_t > track_max + tolerance:
                        raise ValueError(
                            f"Row {idx}: End time {end_t:.3f}s is after track end {track_max:.3f}s"
                        )
                    
                    if end_t <= start_t:
                        raise ValueError(
                            f"Row {idx}: End time {end_t:.3f}s must be greater than start time {start_t:.3f}s"
                        )
                
                segments.append((start_t, end_t, label, color))
            
            if not segments:
                raise ValueError("Please add at least one segment")
            
            self.result = segments
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
    
    def on_cancel(self):
        """取消"""
        self.result = None
        self.destroy()

# ===== 🆕 缓存管理 =====
class CacheManager:
    """统一缓存管理，避免重复计算"""
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = value
    
    def clear(self):
        with self.lock:
            self.cache.clear()


# ===== 🆕 优化的图像处理 =====
class ImageProcessor:
    """向量化图像处理类"""

    @staticmethod
    def normalize_image(image):
        """Normalize image data to 0..1 for display."""
        array = np.asarray(image, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        array = np.clip(array, 0.0, None)
        max_val = float(np.max(array)) if array.size else 0.0
        if max_val > 0:
            array = array / max_val
        return np.clip(array, 0.0, 1.0)

    @staticmethod
    def apply_contrast_vectorized(image, contrast):
        """快速对比度调整（无缓存）"""
        normalized = ImageProcessor.normalize_image(image)
        adjusted = normalized * float(contrast)
        return np.clip(adjusted, 0.0, 1.0)
    
    @staticmethod
    def create_rgb_composite_fast(red, green, blue, 
                                  show_r=False, show_g=False, show_b=False,
                                  c_r=1.0, c_g=1.0, c_b=1.0):
        """快速创建RGB合成"""
        h, w = red.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        
        if show_r:
            rgb[:, :, 0] = ImageProcessor.apply_contrast_vectorized(red, c_r)
        if show_g:
            rgb[:, :, 1] = ImageProcessor.apply_contrast_vectorized(green, c_g)
        if show_b:
            rgb[:, :, 2] = ImageProcessor.apply_contrast_vectorized(blue, c_b)
        
        return np.clip(rgb, 0.0, 1.0)


# ===== 🆕 MSD计算优化 =====
class MSDCalculator:
    """向量化MSD计算（快5倍以上）"""
    
    @staticmethod
    def calc_msd_vectorized(positions, times, max_lag):
        """向量化MSD计算"""
        msd_values = np.zeros(max_lag)
        lag_times = np.zeros(max_lag)
        
        for lag in range(1, max_lag + 1):
            displacements = positions[lag:] - positions[:-lag]
            msd_values[lag-1] = np.mean(displacements ** 2)
            lag_times[lag-1] = np.mean(times[lag:] - times[:-lag])
        
        return msd_values, lag_times
    
    @staticmethod
    def linear_fit_msd(lag_times, msd_values, n_fit):
        """快速线性拟合"""
        if n_fit > len(lag_times):
            n_fit = len(lag_times)
        
        slope, intercept = np.polyfit(lag_times[:n_fit], msd_values[:n_fit], 1)
        D = slope / 2
        
        y_pred = slope * lag_times[:n_fit] + intercept
        ss_res = np.sum((msd_values[:n_fit] - y_pred) ** 2)
        ss_tot = np.sum((msd_values[:n_fit] - np.mean(msd_values[:n_fit])) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return D, r_squared, slope, intercept

# ===== 🆕 新增：优化的速率计算类 =====
class RateCalculator:
    """
    优化的速率计算
    支持多种方法：简单法、线性拟合法、移动平均法
    """
    
    @staticmethod
    def linear_fit_rate(times, positions):
        """
        🔑 关键方法：线性拟合速率计算
        
        原理：
        - 使用 numpy.polyfit 对轨迹进行一阶多项式拟合
        - 斜率就是速率（μm/s）
        - 计算 R² 验证拟合质量
        
        返回字典包含：
        - rate: 速率 (μm/s)  ← 使用这个！
        - intercept: 截距 (μm)
        - r_squared: 拟合优度 (0-1)，越接近1越好
        - stderr: 标准误差 (μm)
        - y_pred: 预测的位置值
        """
        if len(times) < 2:
            return None
        
        # 📐 线性拟合：Position = slope * Time + intercept
        try:
            slope, intercept = np.polyfit(times, positions, 1)
        except:
            return None
        
        # slope 就是速率！
        rate = slope
        
        # 计算拟合质量
        y_pred = rate * times + intercept
        residuals = positions - y_pred
        
        ss_res = np.sum(residuals ** 2)  # 残差平方和
        ss_tot = np.sum((positions - np.mean(positions)) ** 2)  # 总平方和
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算标准误
        n = len(times)
        if n > 2:
            se = np.sqrt(ss_res / (n - 2))
        else:
            se = 0
        
        return {
            'rate': rate,              # ✅ 最重要的输出
            'intercept': intercept,
            'r_squared': r_squared,    # ✅ 质量指标
            'stderr': se,              # ✅ 不确定性
            'y_pred': y_pred,          # ✅ 用于绘图
            'residuals': residuals
        }
    
    @staticmethod
    def simple_rate(times, positions):
        """
        简单法：端点法
        速率 = (最后位置 - 初始位置) / (最后时间 - 初始时间)
        """
        if len(positions) < 2:
            return None
        
        displacement = positions[-1] - positions[0]
        duration = times[-1] - times[0]
        
        if duration > 0:
            return displacement / duration
        return None
    
    @staticmethod
    def moving_average_rate(times, positions, window_size=5):
        """
        移动平均法：检测速率变化
        用于检测速率随时间如何变化
        """
        if len(positions) < window_size:
            return None
        
        rates = []
        rate_times = []
        
        for i in range(len(positions) - window_size):
            dt = times[i + window_size] - times[i]
            dx = positions[i + window_size] - positions[i]
            
            if dt > 0:
                rate = dx / dt
                rates.append(rate)
                rate_times.append((times[i] + times[i + window_size]) / 2)
        
        return np.array(rate_times), np.array(rates)
# ========== Font Configuration (Balanced) ==========  
FONT_FAMILY = 'Arial'  
FONT_SIZE_TITLE = 14            # 标题 (稍微降低)
FONT_SIZE_SUBTITLE = 12         # 副标题 (保持或略低)
FONT_SIZE_NORMAL = 11           # 正文标签 (原值)
FONT_SIZE_SECTION = 12          # Section标题
FONT_SIZE_SMALL = 10            # 小文字 (从9提高到10)
FONT_SIZE_INSTRUCTION = 9       # 说明文本 (从8提高到9)
FONT_SIZE_BUTTON = 11           # 按钮文字 (保持) 

def setup_chinese_font():  
    """Setup Chinese font support"""  
    import platform  
    
    system = platform.system()  
    
    if system == 'Windows':  
        font_list = ['Arial', 'Microsoft YaHei', 'SimHei', 'SimSun']  
    elif system == 'Darwin':  
        font_list = ['Arial', 'PingFang SC', 'Heiti SC', 'STHeiti']  
    else:  
        font_list = ['Arial', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']  
    
    for font_name in font_list:  
        try:  
            plt.rcParams['font.sans-serif'] = [font_name]  
            plt.rcParams['axes.unicode_minus'] = False  
            return True  
        except:  
            continue  
    
    plt.rcParams['axes.unicode_minus'] = False  
    return False  

CHINESE_SUPPORT = setup_chinese_font()

class CustomNavigationToolbar(NavigationToolbar2Tk):
    """自定义工具栏，拦截保存功能以使用源文件名"""
    
    def __init__(self, canvas, window, app_instance=None):
        self.app_instance = app_instance  # 保存应用实例
        super().__init__(canvas, window)
    
    def save_figure(self, *args):
        """简化版：异步保存 + 右下角提示"""
        if self.app_instance is None or self.app_instance.current_file_path is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        try:
            # 获取源文件信息
            source_path = Path(self.app_instance.current_file_path)
            base_name = source_path.stem
            output_dir = source_path.parent
            
            kymo_name = self.app_instance.current_kymo_name.replace(' ', '_').replace('/', '_') \
                       if self.app_instance.current_kymo_name else 'kymo'
            
            output_filename = f"{base_name}_{kymo_name}_kymograph.png"
            output_path = output_dir / output_filename
            
            # 异步保存
            def save_in_thread():
                try:
                    self.canvas.figure.savefig(
                        str(output_path), 
                        dpi=150,
                        bbox_inches=None,
                        format='png',
                        facecolor='white'
                    )
                    
                    # 保存完成：显示右下角提示
                    self.app_instance.root.after(0, lambda: self._show_toast_notification(
                        f"✓ Saved: {output_filename}", duration=2000
                    ))
                    
                    # 输出日志
                    if self.app_instance:
                        self.app_instance.log(f"✓ Kymograph saved: {output_filename}")
                    
                except Exception as e:
                    self.app_instance.root.after(
                        0, 
                        lambda: messagebox.showerror("Save Error", f"Failed to save:\n{str(e)}")
                    )
            
            # 在线程中执行
            import threading
            thread = threading.Thread(target=save_in_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save:\n{str(e)}")

    def _show_toast_notification(self, message, duration=2000):
        """右下角浮窗提示（自动消失）"""
        try:
            root = self.canvas.get_tk_widget().winfo_toplevel()
            
            toast = tk.Toplevel(root)
            toast.wm_overrideredirect(True)  # 去掉窗口边框
            toast.wm_attributes('-alpha', 0.9)  # 半透明
            
            # 放在右下角
            x = root.winfo_x() + root.winfo_width() - 250
            y = root.winfo_y() + root.winfo_height() - 100
            toast.geometry(f"+{x}+{y}")
            
            label = ttk.Label(
                toast, 
                text=message, 
                background='#2c3e50',
                foreground='white',
                font=("Arial", 10),
                padding=15,
                relief=tk.SOLID,
                borderwidth=1
            )
            label.pack()
            
            # 自动关闭
            toast.after(duration, toast.destroy)
            
        except:
            pass  # 如果toast失败，不影响主流程


class ExportDialog(tk.Toplevel):  
    """Dialog for selecting export format and options with scrollbar"""  
    
    def __init__(self, parent):  
        super().__init__(parent)  
        self.title("Export Options")  
        self.geometry("560x650")  
        self.resizable(False, False)  
        
        self.result = None  
        
        # Make dialog modal  
        self.transient(parent)  
        self.grab_set()  
        
        self.create_widgets()  
        
        # Center on parent  
        self.update_idletasks()  
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2  
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2  
        self.geometry(f"+{x}+{y}")  
    
    def create_widgets(self):  
        """Create dialog widgets with scrollable canvas"""  
        
        # ===== MAIN CONTAINER =====  
        main_frame = ttk.Frame(self)  
        main_frame.pack(fill=tk.BOTH, expand=True)  
        
        # ===== SCROLLABLE CONTENT AREA =====  
        canvas_frame = ttk.Frame(main_frame)  
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  
        
        canvas = tk.Canvas(canvas_frame, highlightthickness=0, height=530)  
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)  
        
        # Create scrollable frame  
        scrollable_frame = ttk.Frame(canvas)  
        scrollable_frame.bind(  
            "<Configure>",  
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))  
        )  
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=520)  
        canvas.configure(yscrollcommand=scrollbar.set)  
        
        # Mouse wheel scrolling  
        def _on_mousewheel(event):  
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")  
        
        def _bind_mousewheel(event):  
            canvas.bind_all("<MouseWheel>", _on_mousewheel)  
        
        def _unbind_mousewheel(event):  
            canvas.unbind_all("<MouseWheel>")  
        
        canvas.bind('<Enter>', _bind_mousewheel)  
        canvas.bind('<Leave>', _unbind_mousewheel)  
        
        canvas.pack(side="left", fill="both", expand=True)  
        scrollbar.pack(side="right", fill="y")  
        
        # ===== CONTENT IN SCROLLABLE FRAME =====  
        
        # FILE FORMAT SELECTION  
        file_format_frame = ttk.LabelFrame(scrollable_frame, text="1. Select File Format", padding=10)  
        file_format_frame.pack(fill=tk.X, padx=10, pady=10)  
        
        self.file_format_var = tk.StringVar(value="excel")  
        
        # Excel option  
        excel_frame = ttk.Frame(file_format_frame)  
        excel_frame.pack(fill=tk.X, pady=5)  
        ttk.Radiobutton(excel_frame,   
                       text="Excel (.xlsx)",   
                       variable=self.file_format_var,   
                       value="excel").pack(side=tk.LEFT)  
        ttk.Label(excel_frame, text="All data in one file with multiple sheets",  
                 foreground='gray', font=(FONT_FAMILY, FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=10)  
        
        # CSV option  
        csv_frame = ttk.Frame(file_format_frame)  
        csv_frame.pack(fill=tk.X, pady=5)  
        ttk.Radiobutton(csv_frame,   
                       text="CSV (.csv)",   
                       variable=self.file_format_var,   
                       value="csv").pack(side=tk.LEFT)  
        ttk.Label(csv_frame, text="Separate CSV files for each trace",  
                 foreground='gray', font=(FONT_FAMILY, FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=10)  
        
        # CSV format explanation  
        csv_note_frame = ttk.Frame(file_format_frame)  
        csv_note_frame.pack(fill=tk.X, pady=5, padx=20)  
        ttk.Label(csv_note_frame,   
                 text="CSV will generate:\n"  
                      "  • Separate file for each trace\n"  
                      "  • Settings file (_KymotrackerSettings.csv)\n"  
                      "  • Metadata file (_metadata.csv)",  
                 foreground='blue', font=(FONT_FAMILY, FONT_SIZE_INSTRUCTION), justify=tk.LEFT).pack(anchor=tk.W)  
        
        # DATA ORGANIZATION  
        format_frame = ttk.LabelFrame(scrollable_frame, text="2. Data Organization", padding=10)  
        format_frame.pack(fill=tk.X, padx=10, pady=10)  
        
        self.format_var = tk.StringVar(value="standard")  
        
        ttk.Radiobutton(format_frame, text="Standard Format (separate sheet/file per trace)",   
                       variable=self.format_var, value="standard").pack(anchor=tk.W, pady=2)  
        ttk.Radiobutton(format_frame, text="Wide Format (each track in columns)",   
                       variable=self.format_var, value="wide").pack(anchor=tk.W, pady=2)  
        ttk.Radiobutton(format_frame, text="Long Format (all tracks stacked per channel)",   
                       variable=self.format_var, value="long").pack(anchor=tk.W, pady=2)  
        ttk.Radiobutton(format_frame, text="Long Format - All Channels Merged (single sheet)",   
                       variable=self.format_var, value="long_merged").pack(anchor=tk.W, pady=2)  
        ttk.Radiobutton(format_frame, text="Summary Statistics Only",   
                       variable=self.format_var, value="summary").pack(anchor=tk.W, pady=2)  
        
        # DATA SELECTION  
        data_frame = ttk.LabelFrame(scrollable_frame, text="3. Include Data", padding=10)  
        data_frame.pack(fill=tk.X, padx=10, pady=10)  
        
        self.include_time = tk.BooleanVar(value=True)  
        self.include_position = tk.BooleanVar(value=True)  
        self.include_intensity = tk.BooleanVar(value=True)  
        self.include_velocity = tk.BooleanVar(value=True)  
        self.include_indices = tk.BooleanVar(value=False)  
        
        ttk.Checkbutton(data_frame, text="Time (seconds)",   
                       variable=self.include_time).pack(anchor=tk.W, pady=2)  
        ttk.Checkbutton(data_frame, text="Position (μm)",   
                       variable=self.include_position).pack(anchor=tk.W, pady=2)  
        ttk.Checkbutton(data_frame, text="Summed Photon Counts",   
                       variable=self.include_intensity).pack(anchor=tk.W, pady=2)  
        ttk.Checkbutton(data_frame, text="Velocity (μm/s)",   
                       variable=self.include_velocity).pack(anchor=tk.W, pady=2)  
        ttk.Checkbutton(data_frame, text="Raw Indices (pixels/frames)",   
                       variable=self.include_indices).pack(anchor=tk.W, pady=2)  
        
        # ADDITIONAL OPTIONS  
        options_frame = ttk.LabelFrame(scrollable_frame, text="4. Additional Options", padding=10)  
        options_frame.pack(fill=tk.X, padx=10, pady=10)  
        
        self.include_metadata = tk.BooleanVar(value=True)  
        self.include_settings = tk.BooleanVar(value=True)  
        
        ttk.Checkbutton(options_frame, text="Include metadata",   
                       variable=self.include_metadata).pack(anchor=tk.W, pady=2)  
        ttk.Checkbutton(options_frame, text="Include tracking settings",   
                       variable=self.include_settings).pack(anchor=tk.W, pady=2)  
        
        # ===== FIXED BUTTON FRAME AT BOTTOM =====  
        button_frame = ttk.Frame(main_frame, relief=tk.RAISED, borderwidth=1)  
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)  
        
        ttk.Button(button_frame, text="Cancel",   
                  command=self.on_cancel).pack(side=tk.RIGHT, padx=5, pady=5)  
        ttk.Button(button_frame, text="Export",   
                  command=self.on_export).pack(side=tk.RIGHT, padx=5, pady=5)  
    
    def on_export(self):  
        """Handle export button"""  
        self.result = {  
            'file_format': self.file_format_var.get(),  
            'format': self.format_var.get(),  
            'include_time': self.include_time.get(),  
            'include_position': self.include_position.get(),  
            'include_intensity': self.include_intensity.get(),  
            'include_velocity': self.include_velocity.get(),  
            'include_indices': self.include_indices.get(),  
            'include_metadata': self.include_metadata.get(),  
            'include_settings': self.include_settings.get()  
        }  
        self.destroy()  
    
    def on_cancel(self):  
        """Handle cancel button"""  
        self.result = None  
        self.destroy()  


class KymoTrackerGUI:  
    """KymoTracker GUI Application with multi-channel track display and multi-file support"""  
    
    def __init__(self, root):  
        self.root = root  
        self.root.title("KymoTracker - Multi-File Multi-Channel Track Display")  
        self.root.geometry("1600x900")  
        
        # Multi-file data storage  
        self.loaded_files = {}  # {file_path: lk.File object}  
        self.current_file = None  
        self.current_file_path = None
        # 🆕 添加：用于存储删除点前的轨迹备份
        self.track_backups = {
            'red': [],
            'green': [],
            'blue': []
        }
        
        # 🆕 Deletion History for Undo
        self.deletion_history = []  # Stack of (channel, index, track_object)
        self.active_delete_mode = None  # None, 'click', 'box'
        self.rs = None  # RectangleSelector instance
        self.cid_click_delete = None  # Click event connection ID
        
        # Current kymograph data  
        self.kymo = None  
        self.current_kymo_name = None  
        
        # Y-axis flip state  
        self.y_flip_enabled = False
        
        
        # Multi-channel tracks storage  
        self.tracks_dict = {  
            'red': [],  
            'green': [],  
            'blue': []  
        }  
        
        # Image data  
        self.red_image = None  
        self.green_image = None  
        self.blue_image = None  
        self.rgb_image = None  
        
        # Original image data (before flip)  
        self.red_image_original = None  
        self.green_image_original = None  
        self.blue_image_original = None  
        
        # ROI data  
        self.roi_selection_mode = False  
        self.roi_clicks = []  
        self.roi_coords = None  
        self.roi_rect = None  
        self.temp_markers = []
        # 🆕 弯曲多边形ROI数据
        self.roi_type = 'rectangular'  # 'rectangular' 或 'polygon'
        self.roi_vertices = []  # 多边形顶点列表 [(x_pixel, y_pixel), ...]
        self.roi_mask = None  # 多边形掩码
        self.curved_roi_tool = None  # 弯曲ROI工具对象
        self.curved_roi_pending_callback = False
        
        # Metadata  
        self.pixel_size_nm = None  
        self.delta_line_time = None  
        self.metadata = {}  
        
        # Tracking settings for each channel  
        self.tracking_settings_dict = {  
            'red': {},  
            'green': {},  
            'blue': {}  
        }

        # 🆕 Independent tracking parameters per channel
        # Default values: width=5, threshold=1.0, filter_length=20, min_duration=""
        self.channel_params = {
            'red': {'line_width': 5, 'pixel_threshold': 1.0, 'filter_length': 20, 'min_duration': ''},
            'green': {'line_width': 5, 'pixel_threshold': 1.0, 'filter_length': 20, 'min_duration': ''},
            'blue': {'line_width': 5, 'pixel_threshold': 1.0, 'filter_length': 20, 'min_duration': ''}
        }
        # 🆕 优化：线程池和缓存
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.image_cache = CacheManager(max_size=6)
        self.contrast_cache = {}
        
        # Create UI  
        self.setup_ui()  
        
        self.log("KymoTracker Ready - Multi-File Multi-Channel Display Version")  
        self.log("Load H5 files to begin analysis")  
    
    def setup_ui(self):  
        """Setup user interface with scrollable left panel"""  
        
        # ===== LEFT PANEL WITH SCROLLBAR =====  
        left_container = ttk.Frame(self.root, width=480)  
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)  
        left_container.pack_propagate(False)  
        
        # Create canvas and scrollbar  
        canvas = tk.Canvas(left_container, highlightthickness=0)  
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)  
        
        # Create scrollable frame  
        self.scrollable_frame = ttk.Frame(canvas)  
        self.scrollable_frame.bind(  
            "<Configure>",  
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))  
        )  
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=460)  
        canvas.configure(yscrollcommand=scrollbar.set)  
        
        # Mouse wheel scrolling  
        def _on_mousewheel(event):  
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")  
        
        def _bind_to_mousewheel(event):  
            canvas.bind_all("<MouseWheel>", _on_mousewheel)  
        
        def _unbind_from_mousewheel(event):  
            canvas.unbind_all("<MouseWheel>")  
        
        canvas.bind('<Enter>', _bind_to_mousewheel)  
        canvas.bind('<Leave>', _unbind_from_mousewheel)  
        
        canvas.pack(side="left", fill="both", expand=True)  
        scrollbar.pack(side="right", fill="y")  
        
        # ===== RIGHT PANEL =====  
        right_panel = ttk.Frame(self.root)  
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)  
        
        # ===== CREATE ALL SECTIONS IN SCROLLABLE FRAME =====  
        # Title  
        title_frame = ttk.Frame(self.scrollable_frame)  
        title_frame.pack(pady=10, fill=tk.X, padx=5)  
        ttk.Label(title_frame, text="KymoTracker",   
                 font=(FONT_FAMILY, FONT_SIZE_TITLE, 'bold')).pack()  
        ttk.Label(title_frame, text="Multi-File Multi-Channel Track Display",   
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).pack()  
        
        # Create all sections  
        self.create_file_kymo_section(self.scrollable_frame)  
        self.create_yflip_section(self.scrollable_frame)  
        self.create_roi_section(self.scrollable_frame)  
        self.create_tracking_section(self.scrollable_frame)  
        self.create_filter_section(self.scrollable_frame)  
        self.create_display_section(self.scrollable_frame)  
        self.create_action_section(self.scrollable_frame)  
        self.create_log_section(self.scrollable_frame)  
        
        # Right side sections  
        self.create_plot_section(right_panel)
        self.create_rgb_channel_control_section(right_panel)  # ⭐ 新增

        
        # 初始更新对比度显示
        self.update_contrast_visibility()

    
    def create_file_kymo_section(self, parent):
        """File & Kymograph selection section - optimized layout"""
        frame = ttk.LabelFrame(parent, text="📂 1. LOAD FILES & SELECT", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # ===== FILES SECTION =====
        ttk.Label(frame, text="Files:", font=(FONT_FAMILY, 11, 'bold')).pack(anchor=tk.W, pady=(0,3))
        
        files_list_frame = ttk.Frame(frame)
        files_list_frame.pack(fill=tk.BOTH, expand=True, pady=3)
        files_list_frame.grid_rowconfigure(0, weight=1)
        files_list_frame.grid_columnconfigure(0, weight=1)
        
        self.files_listbox = tk.Listbox(files_list_frame, height=6, font=(FONT_FAMILY, 10))
        self.files_listbox.grid(row=0, column=0, sticky=tk.NSEW)
        
        files_scrollbar = ttk.Scrollbar(files_list_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        files_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.files_listbox.config(yscrollcommand=files_scrollbar.set)
        self.files_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # File buttons (2x2 grid)
        file_btn_frame = ttk.Frame(frame)
        file_btn_frame.pack(fill=tk.X, pady=5)
        file_btn_frame.columnconfigure([0,1], weight=1)
        
        ttk.Button(file_btn_frame, text="📁 Load Folder",
                  command=self.load_folder).grid(row=0, column=0, sticky=tk.EW, padx=(0,2), pady=2)
        ttk.Button(file_btn_frame, text="📄 Load H5 Files",
                  command=self.load_files).grid(row=0, column=1, sticky=tk.EW, padx=(2,0), pady=2)
        ttk.Button(file_btn_frame, text="❌ Remove Selected",
                  command=self.remove_selected_file).grid(row=1, column=0, sticky=tk.EW, padx=(0,2), pady=2)
        ttk.Button(file_btn_frame, text="🗑️ Clear All",
                  command=self.clear_all_files).grid(row=1, column=1, sticky=tk.EW, padx=(2,0), pady=2)
        
        # Current file
        self.current_file_var = tk.StringVar(value="No file selected")
        ttk.Label(frame, textvariable=self.current_file_var, foreground='blue',
                 font=(FONT_FAMILY, 9)).pack(anchor=tk.W, pady=3)
        
        # ===== KYMOGRAPHS SECTION =====
        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=8)
        ttk.Label(frame, text="Kymographs:", font=(FONT_FAMILY, 11, 'bold')).pack(anchor=tk.W, pady=(0,3))
        
        kymo_list_frame = ttk.Frame(frame)
        kymo_list_frame.pack(fill=tk.X, pady=3)
        kymo_list_frame.grid_rowconfigure(0, weight=0)
        kymo_list_frame.grid_columnconfigure(0, weight=1)
        
        self.kymo_listbox = tk.Listbox(kymo_list_frame, height=4, font=(FONT_FAMILY, 10))
        self.kymo_listbox.grid(row=0, column=0, sticky=tk.EW)
        
        kymo_scrollbar = ttk.Scrollbar(kymo_list_frame, orient=tk.HORIZONTAL, command=self.kymo_listbox.xview)
        kymo_scrollbar.grid(row=1, column=0, sticky=tk.EW)
        self.kymo_listbox.config(xscrollcommand=kymo_scrollbar.set)
        self.kymo_listbox.bind('<<ListboxSelect>>', self.on_kymo_select)
        
        # Current kymo
        self.current_kymo_var = tk.StringVar(value="None")
        ttk.Label(frame, textvariable=self.current_kymo_var, foreground='green',
                 font=(FONT_FAMILY, 9)).pack(anchor=tk.W, pady=2)

    def load_folder(self):
        """选择文件夹，自动加载包含'kymograph'关键词的H5文件"""
        folder_path = filedialog.askdirectory(title="Select Folder Containing H5 Files")
        
        if not folder_path:
            return
        
        self.log(f"Scanning folder: {folder_path}")
        
        # 获取文件夹中所有H5文件
        h5_files = list(Path(folder_path).glob("*.h5"))
        
        if not h5_files:
            messagebox.showwarning("Warning", "No H5 files found in selected folder")
            self.log("✗ No H5 files found")
            return
        
        self.log(f"Found {len(h5_files)} H5 files")
        
        # 筛选包含'kymograph'关键词的文件
        kymograph_files = [f for f in h5_files if 'kymograph' in f.name.lower()]
        
        if not kymograph_files:
            messagebox.showwarning("Warning", 
                f"No files containing 'kymograph' found.\n\nFound {len(h5_files)} H5 files:\n" + 
                "\n".join([f.name for f in h5_files[:10]]))
            self.log(f"✗ No files with 'kymograph' keyword found (Total H5 files: {len(h5_files)})")
            return
        
        self.log(f"Found {len(kymograph_files)} files with 'kymograph' keyword")
        
        # 加载筛选出的文件
        for file_path in kymograph_files:
            if str(file_path) in self.loaded_files:
                self.log(f"  ├─ Already loaded: {file_path.name}")
                continue
            
            try:
                self.log(f"  ├─ Loading: {file_path.name}")
                file_obj = lk.File(str(file_path))
                self.loaded_files[str(file_path)] = file_obj
                
                kymo_count = len(file_obj.kymos.keys())
                self.log(f"  ├─ ✓ Loaded: {kymo_count} kymographs")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {file_path.name}:\n{str(e)}")
                self.log(f"  ├─ ✗ Error: {str(e)}")
        
        # 刷新文件列表
        self.refresh_file_list_display()
        
        # 自动选中第一个文件
        if self.files_listbox.size() > 0 and not self.files_listbox.curselection():
            self.files_listbox.selection_set(0)
            self.on_file_select(None)
        
        total_files = len(self.loaded_files)
        total_kymos = sum(len(f.kymos.keys()) for f in self.loaded_files.values())
        self.update_stats(f"Loaded files: {total_files}\nTotal kymographs: {total_kymos}")
        self.log(f"✓ Complete: {total_files} files, {total_kymos} kymographs")

    def refresh_file_list_display(self):
        """刷新文件列表显示"""
        self.files_listbox.delete(0, tk.END)
        
        for file_path in self.loaded_files.keys():
            file_name = Path(file_path).name
            self.files_listbox.insert(tk.END, file_name)
    def on_kymo_select(self, event):
        """Handle kymograph file selection from listbox"""
        selection = self.kymo_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= len(self.kymo_files):
            return
        
        # 切换到选中的文件
        self.current_file_index = index
        filename = self.kymo_files[index]
        
        self.log(f"Switched to: {filename}")
        
        # 加载该文件的数据
        self.load_current_file_data()
        
        # 更新显示
        self.update_display()
    def load_current_file_data(self):
        """Load data for the currently selected file"""
        if not self.kymo_files or self.current_file_index >= len(self.kymo_files):
            return
        
        filename = self.kymo_files[self.current_file_index]
        
        # 加载 kymograph 数据
        if filename in self.file_data:
            data = self.file_data[filename]
            self.kymo = data['kymo']
            self.red_image = data['red']
            self.green_image = data['green']
            self.blue_image = data['blue']
            
            # 加载该文件的轨迹数据
            self.tracks_dict = {
                'red': data.get('tracks_red', []),
                'green': data.get('tracks_green', []),
                'blue': data.get('tracks_blue', [])
            }
            
            # 清除 ROI
            self.roi_coords = None
            self.roi_rect = None
            
            self.log(f"Loaded data for: {filename}")
        else:
            self.log(f"No data found for: {filename}")   
    def create_yflip_section(self, parent):
        """Y-axis flip control - cleaner layout"""
        frame = ttk.LabelFrame(parent, text="🔧 2. PREPROCESSING", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        
        # Instruction
        instruction_text = "Flip Y-axis if kymograph is inverted"
        ttk.Label(frame, text=instruction_text, foreground='blue',
                 font=(FONT_FAMILY, 9)).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0,5))
        
        # Checkbox
        self.y_flip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Flip Y-Axis (Invert Position)", variable=self.y_flip_var,
                       command=self.on_yflip_toggle).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=3)
        
        # Status
        self.yflip_status_var = tk.StringVar(value="Status: Normal orientation")
        ttk.Label(frame, textvariable=self.yflip_status_var, foreground='gray',
                 font=(FONT_FAMILY, 9)).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)  
            
    def create_roi_section(self, parent):
        """ROI selection - organized"""
        frame = ttk.LabelFrame(parent, text="🎯 3. ROI SELECTION", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        
        # Instruction
        instruction_text = (
            "Rectangular: 2 clicks | Curved Polygon: ≥3 clicks\n"
            "Z: undo | Enter: finish | ESC: cancel"
        )
        ttk.Label(frame, text=instruction_text, foreground='blue', font=(FONT_FAMILY, 9),
                 justify=tk.LEFT).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0,5))
        
        # ROI status
        self.roi_status_var = tk.StringVar(value="No ROI selected")
        ttk.Label(frame, textvariable=self.roi_status_var, foreground='darkgreen',
                 font=(FONT_FAMILY, 10, 'bold')).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=3)
        
        # Details
        self.roi_details_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.roi_details_var, foreground='gray',
                 font=(FONT_FAMILY, 9)).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # ROI buttons (2x2)
        ttk.Button(frame, text="📍 Rectangular ROI",
                  command=self.activate_two_click_roi).grid(row=3, column=0, sticky=tk.EW, padx=(0,2), pady=3)
        ttk.Button(frame, text="🌀 Curved Polygon",
                  command=self.activate_curved_roi).grid(row=3, column=1, sticky=tk.EW, padx=(2,0), pady=3)
        
        ttk.Button(frame, text="✏️ Edit Points",
                  command=self.edit_curved_roi_points).grid(row=4, column=0, sticky=tk.EW, padx=(0,2), pady=2)
        ttk.Button(frame, text="🗑️ Clear ROI",
                  command=self.clear_roi).grid(row=4, column=1, sticky=tk.EW, padx=(2,0), pady=2)  
        
    def create_tracking_section(self, parent):
        """Tracking parameters - grid organized"""
        frame = ttk.LabelFrame(parent, text="🔍 4. TRACKING SETTINGS", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Algorithm
        ttk.Label(frame, text="Algorithm:", font=(FONT_FAMILY, 10)).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.algorithm_var = tk.StringVar(value="greedy")
        algo_frame = ttk.Frame(frame)
        algo_frame.grid(row=0, column=1, sticky=tk.EW, pady=3)
        ttk.Radiobutton(algo_frame, text="Greedy", variable=self.algorithm_var, value="greedy",
                       command=self.update_params).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(algo_frame, text="Lines", variable=self.algorithm_var, value="lines",
                       command=self.update_params).pack(side=tk.LEFT, padx=2)
        
        # Channel
        ttk.Label(frame, text="Channel:", font=(FONT_FAMILY, 10)).grid(row=1, column=0, sticky=tk.W, pady=3)
        self.channel_var = tk.StringVar(value="red")
        channel_combo = ttk.Combobox(frame, textvariable=self.channel_var, values=["red", "green", "blue"],
                    state="readonly", width=15)
        channel_combo.grid(row=1, column=1, sticky=tk.W, pady=3)
        # 🆕 Bind channel switch event
        channel_combo.bind("<<ComboboxSelected>>", self.on_channel_switch)
        
        # Line Width
        ttk.Label(frame, text="Line Width:", font=(FONT_FAMILY, 10)).grid(row=2, column=0, sticky=tk.W, pady=3)
        self.line_width_var = tk.IntVar(value=5)
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.line_width_var, width=15).grid(row=2, column=1, sticky=tk.W, pady=3)
        
        # Pixel Threshold
        ttk.Label(frame, text="Threshold:", font=(FONT_FAMILY, 10)).grid(row=3, column=0, sticky=tk.W, pady=3)
        self.pixel_threshold_var = tk.DoubleVar(value=1.0)
        ttk.Entry(frame, textvariable=self.pixel_threshold_var, width=15).grid(row=3, column=1, sticky=tk.W, pady=3)
        
        # Greedy params frame
        self.greedy_frame = ttk.LabelFrame(frame, text="Greedy Options", padding=5)
        self.greedy_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        ttk.Label(self.greedy_frame, text="Window:", font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=5)
        self.window_var = tk.IntVar(value=8)
        ttk.Spinbox(self.greedy_frame, from_=1, to=20, textvariable=self.window_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.greedy_frame, text="Sigma:", font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=5)
        self.sigma_var = tk.StringVar(value="None")
        ttk.Entry(self.greedy_frame, textvariable=self.sigma_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Lines params frame
        self.lines_frame = ttk.LabelFrame(frame, text="Lines Options", padding=5)
        self.lines_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        ttk.Label(self.lines_frame, text="Max Lines:", font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=5)
        self.max_lines_var = tk.IntVar(value=10)
        ttk.Spinbox(self.lines_frame, from_=1, to=50, textvariable=self.max_lines_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Track button frame (row 6)
        track_btn_frame = ttk.Frame(frame)
        track_btn_frame.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=8)
        track_btn_frame.columnconfigure([0,1], weight=1)
        
        self.track_btn = ttk.Button(track_btn_frame, text="▶️ START TRACKING",
                                   command=self.run_tracking, state=tk.DISABLED)
        self.track_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0,2), ipady=8)
        
        ttk.Button(track_btn_frame, text="🗑️ Clear Tracks",
                  command=self.clear_all_tracks).grid(row=0, column=1, sticky=tk.EW, padx=(2,0), ipady=8)
        
        # Separator (row 7)
        ttk.Separator(frame, orient='horizontal').grid(row=7, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8)
        
        # Delete Bad Points button (row 8, full width)
        ttk.Button(frame, text="✏️ Delete Bad Points in Traces",
                  command=self.delete_trace_points).grid(row=8, column=0, columnspan=2,
                                                         sticky=tk.EW, pady=3, ipady=8)
    
    # 🆕 New methods for channel parameter handling
    def save_current_params(self):
        """Save current UI parameters to the dictionary for the current channel"""
        current_channel = self.channel_var.get()
        
        try:
            self.channel_params[current_channel]['line_width'] = self.line_width_var.get()
            self.channel_params[current_channel]['pixel_threshold'] = self.pixel_threshold_var.get()
            
            # Save filter params too
            if hasattr(self, 'filter_length_var'):
                self.channel_params[current_channel]['filter_length'] = self.filter_length_var.get()
            if hasattr(self, 'min_duration_var'):
                self.channel_params[current_channel]['min_duration'] = self.min_duration_var.get()
                
        except Exception as e:
            self.log(f"⚠️ Error saving params for {current_channel}: {e}")

    def update_ui_with_params(self, channel):
        """Update UI with parameters from the dictionary for the specific channel"""
        if channel not in self.channel_params:
            return
            
        params = self.channel_params[channel]
        
        # Update UI variables
        self.line_width_var.set(params.get('line_width', 5))
        self.pixel_threshold_var.set(params.get('pixel_threshold', 1.0))
        
        if hasattr(self, 'filter_length_var'):
            self.filter_length_var.set(params.get('filter_length', 100))
        if hasattr(self, 'min_duration_var'):
            self.min_duration_var.set(params.get('min_duration', ''))
            
    def on_channel_switch(self, event=None):
        """Handle channel switch: save old params, load new params"""
        # Note: self.channel_var is already updated when this event fires? 
        # Actually for ComboboxSelected, the variable might be updated. 
        # But we need to know the PREVIOUS channel to save it?
        # A simpler approach: we just assume the *current* UI values belong to the *previous* channel
        # WARNING: channel_var is bound to the combobox, so it updates IMMEDIATELY upon selection. 
        # So when this function runs, self.channel_var.get() is ALREADY the NEW channel.
        # Ideally, we should save params periodically or use a 'last_channel' variable.
        
        # However, since we initialized everything for 'red' (default) and the UI starts with 'red',
        # we can track the 'previous_channel' manually.
        
        new_channel = self.channel_var.get()
        
        # We need to access the PREVIOUS channel. Let's store it on the instance if not exists.
        if not hasattr(self, 'last_channel'):
            self.last_channel = 'red' # Default start
            
        if self.last_channel == new_channel:
            return

        # 1. Save params for the OLD channel (self.last_channel)
        # But wait, the UI variables currently hold the values for the OLD channel? 
        # OR did the user change them?
        # Yes, the UI Entry widgets still hold the values the user typed.
        # So we save what's in the UI to self.last_channel storage.
        
        # Helper to save specific channel
        try:
            self.channel_params[self.last_channel]['line_width'] = self.line_width_var.get()
            self.channel_params[self.last_channel]['pixel_threshold'] = self.pixel_threshold_var.get()
            if hasattr(self, 'filter_length_var'):
                self.channel_params[self.last_channel]['filter_length'] = self.filter_length_var.get()
            if hasattr(self, 'min_duration_var'):
                self.channel_params[self.last_channel]['min_duration'] = self.min_duration_var.get()
        except Exception as e:
            # print(f"Error saving params: {e}")
            pass

        # 2. Load params for the NEW channel
        self.update_ui_with_params(new_channel)
        
        # 3. Update last_channel
        self.last_channel = new_channel
        
        self.log(f"Switched to {new_channel} channel params")
    
    def create_filter_section(self, parent):  
        """Filter section"""  
        frame = ttk.LabelFrame(parent, text="5. Filter Tracks", padding=10)  
        frame.pack(fill=tk.X, padx=5, pady=5)  
        
        ttk.Label(frame, text="Min Line Length:",   
                  font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=0, column=0, sticky=tk.W, pady=2)  
        self.filter_length_var = tk.IntVar(value=20)
        ttk.Spinbox(frame, from_=0, to=1000,   
                   textvariable=self.filter_length_var, width=18,   
                   font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=0, column=1, sticky=tk.W, pady=2)  
        
        ttk.Label(frame, text="Min Duration (s):",   
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=1, column=0, sticky=tk.W, pady=2)  
        self.min_duration_var = tk.StringVar()  
        ttk.Entry(frame, textvariable=self.min_duration_var,   
                 width=18, font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=1, column=1, sticky=tk.W, pady=2)  
        
        ttk.Button(frame, text="Apply Filter",   
                  command=self.apply_filter).grid(row=2, column=0, columnspan=2,  
                                                  sticky=tk.EW, pady=5) 
        
    def create_display_section(self, parent):
        """Track display controls - organized layout"""
        frame = ttk.LabelFrame(parent, text="6️⃣ TRACK DISPLAY", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        
        # Title
        ttk.Label(frame, text="Select Channels to Display:",
                 font=(FONT_FAMILY, 11, 'bold')).grid(row=0, column=0, columnspan=2, 
                                                        sticky=tk.W, pady=(0, 8))
        
        # Row 1: Red channel
        self.show_red_var = tk.BooleanVar(value=False)
        red_frame = ttk.Frame(frame)
        red_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=3)
        
        ttk.Checkbutton(red_frame, text="🔴 Red Tracks", variable=self.show_red_var,
                       command=self.update_display).pack(side=tk.LEFT)
        self.red_count_label = ttk.Label(red_frame, text="(0 tracks)", foreground='#FF6666',
                                         font=(FONT_FAMILY, 9, 'bold'))
        self.red_count_label.pack(side=tk.LEFT, padx=10)
        
        # Row 2: Green channel
        self.show_green_var = tk.BooleanVar(value=False)
        green_frame = ttk.Frame(frame)
        green_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=3)
        
        ttk.Checkbutton(green_frame, text="🟢 Green Tracks", variable=self.show_green_var,
                       command=self.update_display).pack(side=tk.LEFT)
        self.green_count_label = ttk.Label(green_frame, text="(0 tracks)", foreground='#66FF66',
                                           font=(FONT_FAMILY, 9, 'bold'))
        self.green_count_label.pack(side=tk.LEFT, padx=10)
        
        # Row 3: Blue channel
        self.show_blue_var = tk.BooleanVar(value=False)
        blue_frame = ttk.Frame(frame)
        blue_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=3)
        
        ttk.Checkbutton(blue_frame, text="🔵 Blue Tracks", variable=self.show_blue_var,
                       command=self.update_display).pack(side=tk.LEFT)
        self.blue_count_label = ttk.Label(blue_frame, text="(0 tracks)", foreground='#6666FF',
                                          font=(FONT_FAMILY, 9, 'bold'))
        self.blue_count_label.pack(side=tk.LEFT, padx=10)
        
        # Row 4: Quick action buttons (separator)
        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8)
        
        # Row 5: Show/Hide All buttons (2 columns)
        ttk.Button(frame, text="✅ Show All Tracks",
                  command=self.show_all_tracks).grid(row=5, column=0, sticky=tk.EW, padx=(0, 2), pady=3)
        ttk.Button(frame, text="❌ Hide All Tracks",
                  command=self.hide_all_tracks).grid(row=5, column=1, sticky=tk.EW, padx=(2, 0), pady=3)  
    
    def create_action_section(self, parent):
        """Action buttons section - organized with hierarchy"""
        frame = ttk.LabelFrame(parent, text="7️⃣ EXPORT & ANALYSIS", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        
        # ===== PRIMARY ACTION =====
        ttk.Label(frame, text="Main Export:",
                 font=(FONT_FAMILY, 11, 'bold')).grid(row=0, column=0, columnspan=2, 
                                                        sticky=tk.W, pady=(0, 5))
        
        ttk.Button(frame, text="💾 EXPORT DATA (Excel/CSV)",
                  command=self.export_results).grid(row=1, column=0, columnspan=2,
                                                    sticky=tk.EW, pady=5, ipady=10)
        
        # Separator
        ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8)
        
        # ===== DATA CLEANING =====
        ttk.Label(frame, text="Data Cleaning:",
                 font=(FONT_FAMILY, 11, 'bold')).grid(row=3, column=0, columnspan=2,
                                                        sticky=tk.W, pady=(0, 5))
        
        # Separator
        ttk.Separator(frame, orient='horizontal').grid(row=5, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8)

        # ===== MANUAL DELETION TOOLS =====
        ttk.Label(frame, text="Manual Deletion:",
                 font=(FONT_FAMILY, 11, 'bold')).grid(row=6, column=0, columnspan=2,
                                                        sticky=tk.W, pady=(0, 5))
        
        # Deletion Buttons
        del_frame = ttk.Frame(frame)
        del_frame.grid(row=7, column=0, columnspan=2, sticky=tk.EW)
        del_frame.columnconfigure([0, 1, 2], weight=1)
        
        self.btn_click_del = ttk.Button(del_frame, text="🖱️ Click Del",
                  command=self.start_trace_click_delete)
        self.btn_click_del.grid(row=0, column=0, padx=2, sticky=tk.EW)
        
        self.btn_box_del = ttk.Button(del_frame, text="🔲 Box Del",
                  command=self.start_box_delete)
        self.btn_box_del.grid(row=0, column=1, padx=2, sticky=tk.EW)
        
        ttk.Button(del_frame, text="↩️ Undo",
                  command=self.undo_last_deletion).grid(row=0, column=2, padx=2, sticky=tk.EW)

        # Separator
        ttk.Separator(frame, orient='horizontal').grid(row=8, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8)

        # ===== ANALYSIS TOOLS =====
        ttk.Label(frame, text="Analysis Tools:",
                 font=(FONT_FAMILY, 11, 'bold')).grid(row=9, column=0, columnspan=2,
                                                        sticky=tk.W, pady=(0, 5))
        
        # Row 10: Velocity & MSD (2 columns)
        ttk.Button(frame, text="📉 Velocity",
                  command=self.analyze_velocity).grid(row=10, column=0, sticky=tk.EW, padx=(0, 2), pady=3)
        ttk.Button(frame, text="📈 Multi-Seg MSD",
                  command=self.calculate_multi_segment_msd).grid(row=10, column=1, sticky=tk.EW, padx=(2, 0), pady=3)
        
        # Row 11: Rate & Sliding Window (2 columns)
        ttk.Button(frame, text="📊 Rate Analysis",
                  command=self.calculate_multi_segment_rate).grid(row=11, column=0, sticky=tk.EW, padx=(0, 2), pady=3)
        ttk.Button(frame, text="🔍 Sliding Window MSD",
                  command=self.sliding_window_msd_analysis).grid(row=11, column=1, sticky=tk.EW, padx=(2, 0), pady=3)
        
        # Row 12: Fluorescence tools
        ttk.Button(frame, text="Fluorescence Trace",
                  command=self.analyze_trace_fluorescence).grid(row=12, column=0, sticky=tk.EW, padx=(0, 2), pady=3)
        ttk.Button(frame, text="ROI Fluorescence",
                  command=self.analyze_roi_fluorescence).grid(row=12, column=1, sticky=tk.EW, padx=(2, 0), pady=3)

        ttk.Separator(frame, orient='horizontal').grid(row=13, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8)
        
        # Row 14: Reset (large button)
        ttk.Button(frame, text="🔄 Reset All",
                  command=self.reset_all).grid(row=14, column=0, columnspan=2,
                                              sticky=tk.EW, pady=3, ipady=8)     
    
    def create_log_section(self, parent):  
        """Log section - cleaner display"""  
        frame = ttk.LabelFrame(parent, text="📝 LOG OUTPUT", padding=10)  
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(frame, height=8, wrap=tk.WORD,  
                                                 font=(FONT_FAMILY, FONT_SIZE_SMALL))  
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)  
        
        # Button frame
        button_frame = ttk.Frame(frame)  
        button_frame.pack(fill=tk.X, pady=5)  
        
        button_frame.columnconfigure(0, weight=1)  
        button_frame.columnconfigure(1, weight=1)  
        
        ttk.Button(button_frame, text="🗑️ Clear Log",  
                  command=lambda: self.log_text.delete(1.0, tk.END)).grid(row=0, column=0,   
                                                                           sticky=tk.EW, padx=(0, 2))  
        
        ttk.Button(button_frame, text="💾 Copy All",  
                  command=lambda: self.root.clipboard_clear() or 
                                self.root.clipboard_append(self.log_text.get(1.0, tk.END))).grid(row=0, column=1, 
                                                                                                   sticky=tk.EW, padx=(2, 0)) 
        
    def create_plot_section(self, parent):  
        """Plot section"""  
        frame = ttk.LabelFrame(parent, text="Kymograph Display", padding=5)  
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  
        
        self.fig = Figure(figsize=(10, 6), dpi=100)  
        self.ax = self.fig.add_subplot(111)  
        self.ax.set_title("Load file and select kymograph", fontsize=FONT_SIZE_SUBTITLE)  
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)  
        self.canvas.draw()  
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  
        

        # 🆕 使用自定义工具栏，传入应用实例
        toolbar = CustomNavigationToolbar(self.canvas, frame, app_instance=self)
        toolbar.update()
    def save_kymograph_png(self):
        """Save current kymograph with source filename"""
        if self.kymo is None:
            messagebox.showwarning("Warning", "Please load a kymograph first")
            return
        
        if not self.current_file_path:
            messagebox.showerror("Error", "No source file path available")
            return
        
        try:
            # 获取源文件名（不含路径和扩展名）
            source_path = Path(self.current_file_path)
            base_name = source_path.stem
            output_dir = source_path.parent
            
            # 清理 kymograph 名称
            kymo_name = self.current_kymo_name.replace(' ', '_').replace('/', '_') if self.current_kymo_name else 'kymo'
            
            # 生成输出路径
            output_path = output_dir / f"{base_name}_{kymo_name}_kymograph.png"
            
            # 保存图像
            self.fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
            
            self.log(f"✓ Kymograph saved: {output_path.name}")
            messagebox.showinfo("Save Complete", 
                              f"Kymograph saved successfully!\n\n"
                              f"File: {output_path.name}\n"
                              f"Location: {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save kymograph:\n{str(e)}")
            self.log(f"✗ Save error: {str(e)}")
            
    def create_rgb_channel_control_section(self, parent):
        """RGB Channel Display & Contrast Control"""
        frame = ttk.LabelFrame(parent, text="Background Channels & Contrast", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # ===== 左右两部分布局 =====
        main_layout = ttk.Frame(frame)
        main_layout.pack(fill=tk.BOTH, expand=True)
        
        # --- 左侧区域 ---
        left_section = ttk.Frame(main_layout)
        left_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # === Display Mode (缩小框，放大字) ===
        mode_frame = ttk.LabelFrame(left_section, text="Display Mode", padding=3)
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.display_mode_var = tk.StringVar(value="pseudocolor")
        
        ttk.Radiobutton(mode_frame, text="🎨 Pseudocolor", 
                       variable=self.display_mode_var, value="pseudocolor",
                       command=self.update_display,
                       style='Large.TRadiobutton').pack(anchor=tk.W, pady=1)
        ttk.Radiobutton(mode_frame, text="🌈 RGB Composite", 
                       variable=self.display_mode_var, value="rgb",
                       command=self.update_display,
                       style='Large.TRadiobutton').pack(anchor=tk.W, pady=1)
        
        # === Display Channels (缩小色块) ===
        channel_frame = ttk.LabelFrame(left_section, text="Display Channels", padding=3)
        channel_frame.pack(fill=tk.X)
        
        self.show_bg_red_var = tk.BooleanVar(value=False)
        self.show_bg_green_var = tk.BooleanVar(value=True)
        self.show_bg_blue_var = tk.BooleanVar(value=False)
        
        channels_config = [
            ('R', self.show_bg_red_var, '#FF4444'),
            ('G', self.show_bg_green_var, '#44DD44'),
            ('B', self.show_bg_blue_var, '#4444FF')
        ]
        
        self.bg_channel_buttons = {}
        
        btn_container = ttk.Frame(channel_frame)
        btn_container.pack(pady=2)
        
        for label, var, color in channels_config:
            btn = tk.Button(btn_container,
                           text=label,
                           font=(FONT_FAMILY, 11, 'bold'),  # 缩小字体
                           fg='white',
                           bg=color,
                           activebackground=color,
                           relief=tk.RAISED,
                           bd=2,
                           width=4,   # 缩小宽度
                           height=1,  # 缩小高度
                           command=lambda v=var, c=color, l=label: self.toggle_bg_channel(v, c, l))
            btn.pack(side=tk.LEFT, padx=2)
            self.bg_channel_buttons[label] = btn
        
        self.update_bg_button_states()
        
        # --- 右侧区域：Contrast Adjustment ---
        right_section = ttk.Frame(main_layout)
        right_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        contrast_label_frame = ttk.LabelFrame(right_section, text="Contrast Adjustment", padding=5)
        contrast_label_frame.pack(fill=tk.BOTH, expand=True)
        
        # 对比度控制容器
        self.contrast_container = ttk.Frame(contrast_label_frame)
        self.contrast_container.pack(fill=tk.BOTH, expand=True)
        
        self.contrast_frames = {}
        self.contrast_vars = {}
        self.contrast_labels = {}
        
        contrast_config = [
            ('red', 'R:', '#FF0000'),
            ('green', 'G:', '#00AA00'),
            ('blue', 'B:', '#0000FF')
        ]
        
        for channel, label_text, color in contrast_config:
            contrast_frame = ttk.Frame(self.contrast_container, padding=(2, 3))
            
            ttk.Label(contrast_frame, text=label_text, width=3, 
                     foreground=color, 
                     font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')).pack(side=tk.LEFT, padx=(0, 3))
            
            var = tk.DoubleVar(value=1.0)
            
            scale = ttk.Scale(contrast_frame, from_=0.1, to=10.0,
                             variable=var,
                             orient=tk.HORIZONTAL,
                             command=lambda v, ch=channel: self.update_channel_contrast(ch))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            label = ttk.Label(contrast_frame, text="1.0x", width=6,
                             font=(FONT_FAMILY, FONT_SIZE_SMALL))
            label.pack(side=tk.LEFT)
            
            self.contrast_frames[channel] = contrast_frame
            self.contrast_vars[channel] = var
            self.contrast_labels[channel] = label
        
        # Reset button
        reset_frame = ttk.Frame(contrast_label_frame)
        reset_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.reset_contrast_btn = ttk.Button(reset_frame, text="Reset", 
                                             command=self.reset_all_contrasts)
        self.reset_contrast_btn.pack(side=tk.RIGHT)
        
        # 初始更新对比度显示
        self.update_contrast_visibility()
          
    
    # ========== Y-Axis Flip Methods ==========  
    
    def on_yflip_toggle(self):  
        """Handle Y-flip toggle"""  
        if self.kymo is None:  
            messagebox.showwarning("Warning", "Please select a kymograph first")  
            self.y_flip_var.set(False)  
            return  
        
        self.y_flip_enabled = self.y_flip_var.get()  
        
        # Apply flip  
        if self.y_flip_enabled:  
            self.red_image = np.flipud(self.red_image_original)  
            self.green_image = np.flipud(self.green_image_original)  
            self.blue_image = np.flipud(self.blue_image_original)  
            self.yflip_status_var.set("Status: Y-axis FLIPPED (inverted)")  
            self.log("Y-axis flipped (position inverted)")  
        else:  
            self.red_image = self.red_image_original.copy()  
            self.green_image = self.green_image_original.copy()  
            self.blue_image = self.blue_image_original.copy()  
            self.yflip_status_var.set("Status: Normal orientation")  
            self.log("Y-axis restored to normal orientation")  
          
        
        # Clear existing tracks if any  
        if any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue']):  
            if messagebox.askyesno("Clear Tracks?",   
                                   "Changing orientation will clear existing tracks.\nContinue?"):  
                self.tracks_dict = {'red': [], 'green': [], 'blue': []}  
                self.update_track_counts()  
            else:  
                # Restore flip state  
                self.y_flip_var.set(not self.y_flip_enabled)  
                self.on_yflip_toggle()  
                return  
        
        # Update display  
        self.update_display()
    def update_display(self):
        """Update display based on selected channels"""
        if self.kymo is None:
            return

        self.plot_kymo_with_selected_tracks()
        
    def toggle_bg_channel(self, var, color, label):
        """切换背景通道显示"""
        var.set(not var.get())
        self.update_bg_button_states()
        self.update_contrast_visibility()  # ⭐ 新增：更新对比度显示
        self.update_display()
        
        status = "ON" if var.get() else "OFF"
        self.log(f"Background {label} channel: {status}")
        
    def update_contrast_visibility(self):
        """Show/hide contrast sliders based on active channels"""
        # 隐藏所有对比度控制
        for frame in self.contrast_frames.values():
            frame.pack_forget()
        
        # 显示激活通道的对比度控制
        if self.show_bg_red_var.get():
            self.contrast_frames['red'].pack(fill=tk.X, pady=2)
        
        if self.show_bg_green_var.get():
            self.contrast_frames['green'].pack(fill=tk.X, pady=2)
        
        if self.show_bg_blue_var.get():
            self.contrast_frames['blue'].pack(fill=tk.X, pady=2)

    def update_bg_button_states(self):
        """更新按钮视觉状态"""
        states = {
            'R': self.show_bg_red_var.get(),
            'G': self.show_bg_green_var.get(),
            'B': self.show_bg_blue_var.get()
        }
        
        colors = {
            'R': '#FF4444',
            'G': '#44DD44',
            'B': '#4444FF'
        }
        
        for label, btn in self.bg_channel_buttons.items():
            if states[label]:
                # 选中：凹陷+加深
                darker = self.darken_color(colors[label])
                btn.config(relief=tk.SUNKEN, bd=4, bg=darker)
            else:
                # 未选中：凸起+原色
                btn.config(relief=tk.RAISED, bd=3, bg=colors[label])

    def darken_color(self, hex_color):
        """将颜色变暗"""
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        darkened = tuple(int(c * 0.7) for c in rgb)
        return f'#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}'

    def update_channel_contrast(self, channel):
        """Update contrast for a specific channel（优化版）"""
        if channel not in self.contrast_vars:
            return
            
        value = self.contrast_vars[channel].get()
        self.contrast_labels[channel].config(text=f"{value:.1f}x")
        
        # 🆕 清除缓存
        keys_to_delete = [k for k in self.contrast_cache.keys() if k[0] == channel]
        for key in keys_to_delete:
            del self.contrast_cache[key]
        
        self.update_display()

    def reset_all_contrasts(self):
        """Reset all contrast values to 1.0"""
        for channel in ['red', 'green', 'blue']:
            if channel in self.contrast_vars:
                self.contrast_vars[channel].set(1.0)
                self.contrast_labels[channel].config(text="1.0x")
        self.update_display()  # ✅ 修改这里

    
    # ========== Multi-File Management Methods ==========  
    
    def load_files(self):  
        """Load multiple H5 files"""  
        file_paths = filedialog.askopenfilenames(  
            title="Select H5 Files (can select multiple)",  
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]  
        )  
        
        if not file_paths:  
            return  
        
        for file_path in file_paths:  
            if file_path in self.loaded_files:  
                self.log(f"Already loaded: {Path(file_path).name}")  
                continue  
            
            try:  
                self.log(f"Loading: {Path(file_path).name}")  
                file_obj = lk.File(file_path)  
                self.loaded_files[file_path] = file_obj  
                
                # Add to listbox  
                self.files_listbox.insert(tk.END, Path(file_path).name)  
                
                kymo_count = len(file_obj.kymos.keys())  
                self.log(f"✓ Loaded: {Path(file_path).name} ({kymo_count} kymographs)")  
                
            except Exception as e:  
                messagebox.showerror("Error", f"Failed to load {Path(file_path).name}:\n{str(e)}")  
                self.log(f"✗ Error loading {Path(file_path).name}: {str(e)}")  
        
        # Auto-select first file if nothing selected  
        if self.files_listbox.size() > 0 and not self.files_listbox.curselection():  
            self.files_listbox.selection_set(0)  
            self.on_file_select(None)  
        
        total_files = len(self.loaded_files)  
        total_kymos = sum(len(f.kymos.keys()) for f in self.loaded_files.values())  
        self.update_stats(f"Loaded files: {total_files}\nTotal kymographs: {total_kymos}")  
    
    def on_file_select(self, event):
        """Handle file selection from listbox - auto-load if single kymograph"""
        selection = self.files_listbox.curselection()
        if not selection:
            return
        
        selected_name = self.files_listbox.get(selection[0])
        
        # Find the file path matching this name
        for file_path, file_obj in self.loaded_files.items():
            if Path(file_path).name == selected_name:
                self.current_file = file_obj
                self.current_file_path = file_path
                self.current_file_var.set(f"Current: {selected_name}")
                
                # Update kymograph list
                self.kymo_listbox.delete(0, tk.END)
                kymo_names = list(file_obj.kymos.keys())
                for name in kymo_names:
                    self.kymo_listbox.insert(tk.END, name)
                
                self.log(f"Selected file: {selected_name} ({len(kymo_names)} kymographs)")
                
                # ✅ 如果只有一个kymograph，自动选择并加载
                if len(kymo_names) == 1:
                    self.kymo_listbox.selection_set(0)  # 选中第一个
                    self.log(f"  → Auto-loading: {kymo_names[0]}")
                    # 直接调用加载逻辑
                    self.load_single_kymograph(kymo_names[0])
                elif len(kymo_names) == 0:
                    self.log("⚠ No kymographs found in this file")
                    self.current_kymo_var.set("⚠ No kymographs")
                    self.kymo = None
                    self.current_kymo_name = None
                else:
                    # 多个kymograph，用户需要手动选择
                    self.log("  → Please select a kymograph from the list")
                    self.current_kymo_var.set("Select from list below")
                    self.kymo = None
                    self.current_kymo_name = None
                
                # Reset flip state
                self.y_flip_enabled = False
                self.y_flip_var.set(False)
                self.track_btn.config(state=tk.NORMAL if len(kymo_names) == 1 else tk.DISABLED)
                
                # Clear tracks for current kymograph
                self.tracks_dict = {
                    'red': [],
                    'green': [],
                    'blue': []
                }
                self.update_track_counts()
                
                # 如果没有自动加载，清空显示
                if len(kymo_names) != 1:
                    self.ax.clear()
                    self.ax.set_title("Select a kymograph", fontsize=FONT_SIZE_SUBTITLE)
                    self.canvas.draw()
                
                break 
    
    def remove_selected_file(self):  
        """Remove selected file from loaded files"""  
        selection = self.files_listbox.curselection()  
        if not selection:  
            messagebox.showwarning("Warning", "Please select a file to remove")  
            return  
        
        selected_name = self.files_listbox.get(selection[0])  
        
        if messagebox.askyesno("Confirm", f"Remove file:\n{selected_name}?"):  
            # Find and remove the file  
            file_path_to_remove = None  
            for file_path in self.loaded_files.keys():  
                if Path(file_path).name == selected_name:  
                    file_path_to_remove = file_path  
                    break  
            
            if file_path_to_remove:  
                del self.loaded_files[file_path_to_remove]  
                self.files_listbox.delete(selection[0])  
                self.log(f"Removed file: {selected_name}")  
                
                # If this was the current file, clear current data  
                if self.current_file_path == file_path_to_remove:  
                    self.current_file = None  
                    self.current_file_path = None  
                    self.current_file_var.set("No file selected")  
                    self.kymo_listbox.delete(0, tk.END)  
                    self.kymo = None  
                    self.current_kymo_name = None  
                    self.current_kymo_var.set("None")  
                    self.tracks_dict = {'red': [], 'green': [], 'blue': []}  
                    
                    # Reset flip state  
                    self.y_flip_enabled = False  
                    self.y_flip_var.set(False)    
                      
                    self.track_btn.config(state=tk.DISABLED)  
                    
                    self.update_track_counts()  
                    self.ax.clear()  
                    self.ax.set_title("Load file and select kymograph", fontsize=FONT_SIZE_SUBTITLE)  
                    self.canvas.draw()  
    
    def clear_all_files(self):  
        """Clear all loaded files"""  
        if not self.loaded_files:  
            return  
        
        if messagebox.askyesno("Confirm", "Clear all loaded files?"):  
            self.loaded_files = {}  
            self.files_listbox.delete(0, tk.END)  
            self.current_file = None  
            self.current_file_path = None  
            self.current_file_var.set("No file selected")  
            self.kymo_listbox.delete(0, tk.END)  
            self.kymo = None  
            self.current_kymo_name = None  
            self.current_kymo_var.set("None")  
            self.tracks_dict = {'red': [], 'green': [], 'blue': []}  
            
            # Reset flip state  
            self.y_flip_enabled = False  
            self.y_flip_var.set(False)      
            self.track_btn.config(state=tk.DISABLED)  
            
            self.update_track_counts()  
            self.ax.clear()  
            self.ax.set_title("Load files to begin", fontsize=FONT_SIZE_SUBTITLE)  
            self.canvas.draw()  
            self.update_stats("All files cleared")  
            self.log("All files cleared")  
    
    # ========== Display Control Methods ==========  
    
    def update_display(self):  
        """Update display based on selected channels"""  
        if self.kymo is None:  
            return  
        
        self.plot_kymo_with_selected_tracks()  
    
    def show_all_tracks(self):  
        """Show all available tracks"""  
        if len(self.tracks_dict['red']) > 0:  
            self.show_red_var.set(True)  
        if len(self.tracks_dict['green']) > 0:  
            self.show_green_var.set(True)  
        if len(self.tracks_dict['blue']) > 0:  
            self.show_blue_var.set(True)  
        self.update_display()  
    
    def hide_all_tracks(self):  
        """Hide all tracks"""  
        self.show_red_var.set(False)  
        self.show_green_var.set(False)  
        self.show_blue_var.set(False)  
        self.update_display()  
    
    def clear_all_tracks(self):  
        """Clear all tracks from all channels"""  
        if messagebox.askyesno("Confirm", "Clear all tracks from all channels?"):  
            self.tracks_dict = {  
                'red': [],  
                'green': [],  
                'blue': []  
            }  
            self.show_red_var.set(False)  
            self.show_green_var.set(False)  
            self.show_blue_var.set(False)  
            self.update_track_counts()  
            self.update_display()  
            self.log("All tracks cleared")  
    
    def update_track_counts(self):  
        """Update track count labels"""  
        red_count = len(self.tracks_dict['red'])  
        green_count = len(self.tracks_dict['green'])  
        blue_count = len(self.tracks_dict['blue'])  
        
        self.red_count_label.config(text=f"({red_count} tracks)")  
        self.green_count_label.config(text=f"({green_count} tracks)")  
        self.blue_count_label.config(text=f"({blue_count} tracks)")  
        
        # Update stats  
        total_tracks = red_count + green_count + blue_count  
        stats_text = f"Total tracks: {total_tracks}\n"  
        stats_text += f"  Red: {red_count}\n"  
        stats_text += f"  Green: {green_count}\n"  
        stats_text += f"  Blue: {blue_count}"  
        self.update_stats(stats_text)
        
    
    def plot_kymo_with_selected_tracks(self):
        """Plot kymograph with multiple background channels (pseudocolor/RGB/auto)"""
        if self.kymo is None:
            return
        
        self.ax.clear()
        
        try:
            # 获取要显示的通道
            show_red = self.show_bg_red_var.get()
            show_green = self.show_bg_green_var.get()
            show_blue = self.show_bg_blue_var.get()
            
            # 🆕 获取显示模式
            display_mode = self.display_mode_var.get()  # auto, pseudocolor, rgb
            
            active_channels = sum([show_red, show_green, show_blue])
            
            if active_channels == 0:
                self.ax.text(0.5, 0.5, 'Please select at least one background channel',
                            ha='center', va='center', transform=self.ax.transAxes,
                            fontsize=14, color='gray')
                self.canvas.draw()
                return
            
            height, width = self.red_image.shape
            time_extent = width * self.delta_line_time
            position_extent = height * self.pixel_size_nm / 1000
            
            # ===== 🆕 核心：根据模式判断 =====
            use_pseudocolor = False
            
            if display_mode == "auto":
                # 自动模式：单通道用伪彩色，多通道用RGB
                use_pseudocolor = (active_channels == 1)
            elif display_mode == "pseudocolor":
                # 强制伪彩色模式
                use_pseudocolor = True
            else:  # display_mode == "rgb"
                # 强制RGB模式
                use_pseudocolor = False
            
            # ===== 伪彩色模式（优化版） =====
            if use_pseudocolor:
                # 选择第一个激活的通道
                if show_red:
                    channel_data = self.red_image
                    contrast = self.contrast_vars['red'].get()
                    cmap = 'Reds'
                    channel_name = 'Red'
                elif show_green:
                    channel_data = self.green_image
                    contrast = self.contrast_vars['green'].get()
                    cmap = 'Greens'
                    channel_name = 'Green'
                else:  # show_blue
                    channel_data = self.blue_image
                    contrast = self.contrast_vars['blue'].get()
                    cmap = 'Blues'
                    channel_name = 'Blue'
                
                # 🆕 使用向量化对比度调整
                normalized_data = ImageProcessor.apply_contrast_vectorized(channel_data, contrast)
                
                self.ax.imshow(normalized_data, aspect='auto',
                              extent=[0, time_extent, 0, position_extent],
                              origin='lower',
                              cmap=cmap)
                
                title_suffix = f'({channel_name} - Pseudocolor)'
            
            # ===== RGB 合成模式 =====
            # ===== RGB 合成模式（优化版） =====
            else:
                # 🆕 使用向量化RGB合成
                rgb_composite = ImageProcessor.create_rgb_composite_fast(
                    self.red_image, self.green_image, self.blue_image,
                    show_red, show_green, show_blue,
                    self.contrast_vars['red'].get() if show_red else 1.0,
                    self.contrast_vars['green'].get() if show_green else 1.0,
                    self.contrast_vars['blue'].get() if show_blue else 1.0
                )
                
                self.ax.imshow(rgb_composite, aspect='auto',
                              extent=[0, time_extent, 0, position_extent],
                              origin='lower')
                
                channels_shown = []
                if show_red: channels_shown.append('R')
                if show_green: channels_shown.append('G')
                if show_blue: channels_shown.append('B')
                title_suffix = f'({"+".join(channels_shown)} - RGB)'
            
            # ===== 绘制 ROI =====
            if self.roi_type == 'polygon' and len(self.roi_vertices) > 0:
                # 🆕 绘制多边形ROI
                xs = [v[0] * self.delta_line_time for v in self.roi_vertices]
                ys = [v[1] * self.pixel_size_nm / 1000 for v in self.roi_vertices]
                xs_closed = xs + [xs[0]]
                ys_closed = ys + [ys[0]]
                
                self.ax.plot(xs_closed, ys_closed, 'c-', linewidth=2, linestyle='-', 
                           alpha=0.8, zorder=10)
                self.ax.fill(xs_closed, ys_closed, 'cyan', alpha=0.05, zorder=9)
                
            elif self.roi_coords:
                # 原有矩形ROI
                x1, y1, x2, y2 = self.roi_coords
                x1_real = x1 * self.delta_line_time
                x2_real = x2 * self.delta_line_time
                y1_real = y1 * self.pixel_size_nm / 1000
                y2_real = y2 * self.pixel_size_nm / 1000
                
                roi_width = x2_real - x1_real
                roi_height = y2_real - y1_real
                self.roi_rect = Rectangle((x1_real, y1_real), roi_width, roi_height,
                                          fill=False, edgecolor='cyan',
                                          linewidth=2, linestyle='--')
                self.ax.add_patch(self.roi_rect)
            
            # ===== 绘制轨迹 =====
            track_colors = {
                'red': '#FF6666',
                'green': '#66FF66',
                'blue': '#6666FF'
            }
            
            tracks_shown = []
            
            if self.show_red_var.get() and len(self.tracks_dict['red']) > 0:
                for track in self.tracks_dict['red']:
                    time_coords, pos_coords = self.get_track_display_coords_real_units(track)
                    self.ax.plot(time_coords, pos_coords,
                               color=track_colors['red'],
                               linewidth=2.5, alpha=0.8,
                               label='Red tracks' if 'Red tracks' not in tracks_shown else "")
                    if 'Red tracks' not in tracks_shown:
                        tracks_shown.append('Red tracks')
            
            if self.show_green_var.get() and len(self.tracks_dict['green']) > 0:
                for track in self.tracks_dict['green']:
                    time_coords, pos_coords = self.get_track_display_coords_real_units(track)
                    self.ax.plot(time_coords, pos_coords,
                               color=track_colors['green'],
                               linewidth=2.5, alpha=0.8,
                               label='Green tracks' if 'Green tracks' not in tracks_shown else "")
                    if 'Green tracks' not in tracks_shown:
                        tracks_shown.append('Green tracks')
            
            if self.show_blue_var.get() and len(self.tracks_dict['blue']) > 0:
                for track in self.tracks_dict['blue']:
                    time_coords, pos_coords = self.get_track_display_coords_real_units(track)
                    self.ax.plot(time_coords, pos_coords,
                               color=track_colors['blue'],
                               linewidth=2.5, alpha=0.8,
                               label='Blue tracks' if 'Blue tracks' not in tracks_shown else "")
                    if 'Blue tracks' not in tracks_shown:
                        tracks_shown.append('Blue tracks')
            
            # ===== 图例 =====
            if tracks_shown:
                self.ax.legend(loc='upper right', fontsize=FONT_SIZE_NORMAL)
            
            # ===== 标题 =====
            # 获取源文件名（不含路径和扩展名）
            source_name = Path(self.current_file_path).stem if self.current_file_path else 'Kymograph'
            # 🆕 改为"文件夹名-Kymograph.编号"格式
            folder_name = Path(self.current_file_path).parent.name
            try:
                kymo_number = self.current_kymo_name.split('-')[-1].strip()
            except:
                kymo_number = "1"

            title = f'{folder_name}-Kymograph.{kymo_number} {title_suffix}'
                        
            if self.y_flip_enabled:
                title += " [Y-FLIPPED]"
            
            if self.current_file_path and self.current_kymo_name:
                folder_name = Path(self.current_file_path).parent.name
                try:
                    kymo_number = self.current_kymo_name.split('-')[-1].strip()
                except:
                    kymo_number = "1"
                full_title = f'{folder_name}-Kymograph.{kymo_number} {title_suffix}'
            else:
                full_title = f'Kymograph {title_suffix}'

            self.ax.set_title(full_title, fontsize=FONT_SIZE_SUBTITLE)
            self.ax.set_xlabel('Time (s)', fontsize=FONT_SIZE_NORMAL)
            self.ax.set_ylabel('Position (μm)', fontsize=FONT_SIZE_NORMAL)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"Plot error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_track_display_coords_real_units(self, track):  
        """Get display coordinates for a track in real units (s and μm)"""  
        time_indices = np.array(track.time_idx)  
        pos_indices = np.array(track.coordinate_idx)  
        
        # Adjust for ROI offset if applicable  
        if self.roi_coords:  
            x1, y1, _, _ = self.roi_coords  
            time_indices = time_indices + x1  
            pos_indices = pos_indices + y1  
        
        # Convert to real units  
        time_coords = time_indices * self.delta_line_time  # seconds  
        pos_coords = pos_indices * self.pixel_size_nm / 1000  # nm to μm  
        
        return time_coords, pos_coords  
    
    # ========== Core Methods ==========  
    
    def update_params(self):  
        """Update parameter visibility based on algorithm"""  
        if self.algorithm_var.get() == "greedy":  
            self.greedy_frame.grid()  
            self.lines_frame.grid_remove()  
        else:  
            self.greedy_frame.grid_remove()  
            self.lines_frame.grid()  
    
    # ===== 🆕 Interactive Deletion Tools =====

    def disable_roi_modes(self):
        """Force disable any active ROI modes"""
        # Disable Rectangular ROI
        if self.roi_selection_mode:
            self.roi_selection_mode = False
            if hasattr(self, 'cid_canvas_click') and self.cid_canvas_click:
                self.canvas.mpl_disconnect(self.cid_canvas_click)
            if hasattr(self, 'clear_temp_markers'):
                self.clear_temp_markers()
            self.log("Rectangular ROI selection cancelled")
            
        # Disable Curved ROI
        if hasattr(self, 'curved_roi_tool') and self.curved_roi_tool:
            self.curved_roi_tool.disconnect()
            self.curved_roi_tool.clear()
            self.curved_roi_tool = None
            self.log("Curved ROI selection cancelled")

    def disable_delete_modes(self):
        """Turn off any active deletion modes"""
        if self.active_delete_mode == 'click':
            if self.cid_click_delete:
                self.canvas.mpl_disconnect(self.cid_click_delete)
                self.cid_click_delete = None
            self.log("Exited Click Delete Mode")
            
        elif self.active_delete_mode == 'box':
            if self.rs:
                self.rs.set_active(False)
                self.rs = None
            self.log("Exited Box Delete Mode")
            
        self.active_delete_mode = None

    def start_trace_click_delete(self):
        """Toggle Trace Click Delete Mode"""
        if self.active_delete_mode == 'click':
            self.disable_delete_modes()
            return
            
        self.disable_delete_modes()  # Ensure others are off
        self.disable_roi_modes()     # Ensure ROI modes are off
        self.active_delete_mode = 'click'
        self.cid_click_delete = self.canvas.mpl_connect('button_press_event', self.on_click_delete)
        self.log("🖱️ Click Delete Mode Active: Click on a trace to remove it")
        messagebox.showinfo("Mode Active", "Click Delete Mode Active.\nClick near any trace to delete it.")

    def start_box_delete(self):
        """Toggle Area Box Delete Mode"""
        if self.active_delete_mode == 'box':
            self.disable_delete_modes()
            return
            
        self.disable_delete_modes()
        self.disable_roi_modes()    # Ensure ROI modes are off
        self.active_delete_mode = 'box'
        self.rs = RectangleSelector(self.ax, self.on_box_select,
                                    useblit=True,
                                    button=[1],  # Left click
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
        self.log("🔲 Box Delete Mode Active: Drag to select area")
        messagebox.showinfo("Mode Active", "Box Delete Mode Active.\nDrag a box to delete intersecting tracks.")

    def on_click_delete(self, event):
        """Handle click for trace deletion"""
        if event.inaxes != self.ax or self.active_delete_mode != 'click':
            return
            
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return

        # Find closest track
        min_dist = float('inf')
        closest_track_info = None  # (channel, index, track)
        
        for channel in ['red', 'green', 'blue']:
            for idx, track in enumerate(self.tracks_dict[channel]):
                t_coords, p_coords = self.get_track_display_coords_real_units(track)
                
                # Vectorized distance calculation
                distances = np.sqrt((t_coords - click_x)**2 + (p_coords - click_y)**2)
                local_min = np.min(distances)
                
                if local_min < min_dist:
                    min_dist = local_min
                    closest_track_info = (channel, idx, track)

        # Threshold for deletion (e.g., radius in data units)
        # 0.5 unit tolerance (adjust based on typical scales)
        tolerance = 0.5 
        
        if closest_track_info and min_dist < tolerance:
            channel, idx, track = closest_track_info
            response = messagebox.askyesno("Confirm Delete", f"Delete {channel} track #{idx+1}?")
            if response:
                self.perform_track_deletion(channel, idx)

    def on_box_select(self, eclick, erelease):
        """Handle box selection for deletion"""
        if self.active_delete_mode != 'box':
            return
            
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        to_delete = []  # List of (channel, index) - reverse sorted later
        
        for channel in ['red', 'green', 'blue']:
            for idx, track in enumerate(self.tracks_dict[channel]):
                t_coords, p_coords = self.get_track_display_coords_real_units(track)
                
                # Check for any intersection
                # Point in box: xmin <= t <= xmax AND ymin <= p <= ymax
                in_box = (t_coords >= xmin) & (t_coords <= xmax) & (p_coords >= ymin) & (p_coords <= ymax)
                
                if np.any(in_box):
                    to_delete.append((channel, idx))
        
        if not to_delete:
            self.log("No tracks found in selection")
            return
            
        if messagebox.askyesno("Confirm Delete", f"Delete {len(to_delete)} tracks found in area?"):
            # Sort by index descending to avoid shifting issues when deleting multiple
            
            # Group by channel
            grouped = {'red': [], 'green': [], 'blue': []}
            for ch, idx in to_delete:
                grouped[ch].append(idx)
            
            # Process each channel, deleting from high to low index
            count = 0 
            
            for ch in grouped:
                indices = sorted(grouped[ch], reverse=True)
                for idx in indices:
                    self.perform_track_deletion(ch, idx, refresh=False)
                    count += 1
            
            self.update_display()
            self.log(f"Deleted {count} tracks via box selection")

    def perform_track_deletion(self, channel, index, refresh=True):
        """Execute deletion and save to history"""
        if index >= len(self.tracks_dict[channel]):
            return
            
        track = self.tracks_dict[channel].pop(index)
        self.deletion_history.append((channel, index, track))
        
        self.log(f"Deleted {channel} track #{index+1}")
        
        if refresh:
            self.update_display()

    def undo_last_deletion(self):
        """Undo the last track deletion"""
        if not self.deletion_history:
            self.log("Nothing to undo")
            return
            
        channel, index, track = self.deletion_history.pop()
        
        if index > len(self.tracks_dict[channel]):
             self.tracks_dict[channel].append(track)
        else:
             self.tracks_dict[channel].insert(index, track)
             
        self.log(f"Undid deletion of {channel} track #{index+1}")
        self.update_display()

    def calculate_track_duration(self, track):  
        """Calculate duration of a track"""  
        if hasattr(track, 'time_idx') and len(track.time_idx) > 0:  
            time_array = np.array(track.time_idx) * self.delta_line_time  
            return time_array[-1] - time_array[0]  
        return 0

    def get_track_index_arrays(self, track):
        """Return sorted track indices after applying ROI offsets."""
        time_indices = np.asarray(track.time_idx, dtype=float)
        pos_indices = np.asarray(track.coordinate_idx, dtype=float)

        if self.roi_coords:
            x1, y1, _, _ = self.roi_coords
            time_indices = time_indices + x1
            pos_indices = pos_indices + y1

        order = np.arange(len(time_indices))
        if len(time_indices) > 1:
            order = np.argsort(time_indices)
            time_indices = time_indices[order]
            pos_indices = pos_indices[order]

        return time_indices, pos_indices, order

    def get_track_time_position_arrays(self, track):
        """Return sorted track coordinates in seconds and micrometers."""
        time_indices, pos_indices, _ = self.get_track_index_arrays(track)
        times = time_indices * self.delta_line_time
        positions = pos_indices * self.pixel_size_nm / 1000.0

        return times, positions

    def get_track_intensity_array(self, track, channel=None):
        """Sample fluorescence intensity along one track in the same sorted order."""
        line_width = self.line_width_var.get()
        if channel in self.channel_params:
            line_width = self.channel_params[channel].get('line_width', line_width)

        num_pixels = max(1, int(math.ceil(line_width / 2)))
        intensities = np.asarray(track.sample_from_image(num_pixels=num_pixels), dtype=float).reshape(-1)

        time_indices, _, order = self.get_track_index_arrays(track)
        if intensities.size != time_indices.size:
            padded = np.full(time_indices.shape, np.nan, dtype=float)
            valid_len = min(intensities.size, padded.size)
            padded[:valid_len] = intensities[:valid_len]
            intensities = padded

        if intensities.size == order.size and intensities.size > 1:
            intensities = intensities[order]

        return intensities

    def get_track_fluorescence_profile(self, track, include_intensity=True, channel=None):
        """Build a reusable time-position-intensity bundle for one trace."""
        time_indices, pos_indices, _ = self.get_track_index_arrays(track)
        times = time_indices * self.delta_line_time
        relative_times = times - times[0] if len(times) else np.array([], dtype=float)
        positions = pos_indices * self.pixel_size_nm / 1000.0

        intensities = np.array([], dtype=float)
        intensity_error = None
        if include_intensity and len(time_indices):
            try:
                intensities = self.get_track_intensity_array(track, channel=channel)
            except Exception as exc:
                intensities = np.full(len(time_indices), np.nan, dtype=float)
                intensity_error = str(exc)

        if len(time_indices):
            lifetime_frames = int(round(time_indices[-1] - time_indices[0])) + 1
            lifetime_seconds = float(lifetime_frames * self.delta_line_time)
            duration_seconds = float(times[-1] - times[0]) if len(times) > 1 else 0.0
            start_time = float(times[0])
            end_time = float(times[-1])
        else:
            lifetime_frames = 0
            lifetime_seconds = 0.0
            duration_seconds = 0.0
            start_time = 0.0
            end_time = 0.0

        return {
            'time_indices': time_indices,
            'position_indices': pos_indices,
            'times': times,
            'relative_times': relative_times,
            'positions': positions,
            'intensities': intensities,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': duration_seconds,
            'lifetime_frames': lifetime_frames,
            'lifetime_seconds': lifetime_seconds,
            'point_count': int(len(time_indices)),
            'intensity_error': intensity_error
        }

    def build_track_summary_row(self, track, track_num, channel, include_intensity=True):
        """Build one summary row with lifetime and fluorescence statistics."""
        profile = self.get_track_fluorescence_profile(track, include_intensity=include_intensity, channel=channel)
        times = profile['times']
        positions = profile['positions']
        intensities = profile['intensities']

        total_distance = float(np.sum(np.abs(np.diff(positions)))) if len(positions) > 1 else 0.0
        start_pos = float(positions[0]) if len(positions) else np.nan
        end_pos = float(positions[-1]) if len(positions) else np.nan
        displacement = end_pos - start_pos if len(positions) else np.nan

        mean_velocity = np.nan
        mean_abs_velocity = np.nan
        std_velocity = np.nan
        if len(positions) >= 2:
            delta_t = np.diff(times)
            delta_x = np.diff(positions)
            valid = np.abs(delta_t) > np.finfo(float).eps
            if np.any(valid):
                velocities = delta_x[valid] / delta_t[valid]
                mean_velocity = float(np.mean(velocities))
                mean_abs_velocity = float(np.mean(np.abs(velocities)))
                std_velocity = float(np.std(velocities))

        summary = {
            'Track_ID': track_num,
            'Channel': channel.capitalize(),
            'Points': profile['point_count'],
            'Start_Time (s)': profile['start_time'],
            'End_Time (s)': profile['end_time'],
            'Duration (s)': profile['duration_seconds'],
            'Lifetime (frames)': profile['lifetime_frames'],
            'Lifetime (s)': profile['lifetime_seconds'],
            'Start_Position (渭m)': start_pos,
            'End_Position (渭m)': end_pos,
            'Net_Displacement (渭m)': displacement,
            'Total_Distance (渭m)': total_distance,
            'Mean_Velocity (渭m/s)': mean_velocity,
            'Mean_|Velocity| (渭m/s)': mean_abs_velocity,
            'Std_Velocity (渭m/s)': std_velocity
        }

        if include_intensity:
            finite_intensities = intensities[np.isfinite(intensities)]
            if finite_intensities.size:
                start_intensity = float(intensities[0]) if np.isfinite(intensities[0]) else np.nan
                end_intensity = float(intensities[-1]) if np.isfinite(intensities[-1]) else np.nan
                delta_intensity = end_intensity - start_intensity if np.isfinite(start_intensity) and np.isfinite(end_intensity) else np.nan
                normalized_change = (
                    delta_intensity / start_intensity
                    if np.isfinite(delta_intensity) and np.isfinite(start_intensity) and abs(start_intensity) > np.finfo(float).eps
                    else np.nan
                )

                summary.update({
                    'Mean_Intensity': float(np.mean(finite_intensities)),
                    'Max_Intensity': float(np.max(finite_intensities)),
                    'Integrated_Intensity': float(np.sum(finite_intensities)),
                    'Start_Intensity': start_intensity,
                    'End_Intensity': end_intensity,
                    'Delta_Intensity': delta_intensity,
                    'Relative_Intensity_Change': normalized_change
                })
            else:
                summary.update({
                    'Mean_Intensity': np.nan,
                    'Max_Intensity': np.nan,
                    'Integrated_Intensity': np.nan,
                    'Start_Intensity': np.nan,
                    'End_Intensity': np.nan,
                    'Delta_Intensity': np.nan,
                    'Relative_Intensity_Change': np.nan
                })

        if profile['intensity_error']:
            summary['Intensity_Error'] = profile['intensity_error']

        return summary

    def compute_local_velocity_series(self, times, positions, window_points=7):
        """Estimate instantaneous velocity from local linear fits in a moving window."""
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)

        if times.size < 2:
            return np.array([]), np.array([])

        window_points = max(int(window_points), 3)
        if window_points % 2 == 0:
            window_points += 1

        n_points = times.size
        half_window = window_points // 2
        local_times = []
        local_velocity_um_s = []

        for idx in range(n_points):
            left = max(0, idx - half_window)
            right = min(n_points, idx + half_window + 1)

            window_times = times[left:right]
            window_positions = positions[left:right]

            if window_times.size < 2 or np.ptp(window_times) <= 0:
                continue

            slope, _ = np.polyfit(window_times, window_positions, 1)
            local_times.append(times[idx])
            local_velocity_um_s.append(slope)

        return np.asarray(local_times, dtype=float), np.asarray(local_velocity_um_s, dtype=float)

    def compute_weighted_fit_rate_bp_s(self, segment_results):
        """Compute a duration-weighted fit rate across selected segments."""
        if not segment_results:
            return np.nan

        weighted_sum = 0.0
        total_duration = 0.0
        fallback_rates = []

        for result in segment_results:
            fit_rate_bp_s = result.get('fit_rate_bp_s', np.nan)
            duration = float(result.get('duration', 0.0))

            if np.isfinite(fit_rate_bp_s):
                fallback_rates.append(fit_rate_bp_s)
                if duration > 0:
                    weighted_sum += fit_rate_bp_s * duration
                    total_duration += duration

        if total_duration > 0:
            return weighted_sum / total_duration
        if fallback_rates:
            return float(np.mean(fallback_rates))
        return np.nan

    def analyze_velocity_segment(self, times, positions, start, end, label, color, local_window_points=7):
        """Use the original instantaneous-velocity method: np.diff(position) / np.diff(time)."""
        um_to_bp = DNA_UM_TO_BP

        mask = (times >= start) & (times <= end)
        seg_times = np.asarray(times[mask], dtype=float)
        seg_positions = np.asarray(positions[mask], dtype=float)

        if seg_times.size < 2:
            return None

        order = np.argsort(seg_times)
        seg_times = seg_times[order]
        seg_positions = seg_positions[order]

        delta_t = np.diff(seg_times)
        delta_position = np.diff(seg_positions)
        valid = delta_t > 0

        if not np.any(valid):
            return None

        interval_start = seg_times[:-1][valid]
        interval_end = seg_times[1:][valid]
        interval_mid = (interval_start + interval_end) / 2.0
        delta_t = delta_t[valid]
        delta_position = delta_position[valid]

        instant_velocity_um_s = delta_position / delta_t
        instant_velocity_bp_s = instant_velocity_um_s * um_to_bp
        abs_velocity_bp_s = np.abs(instant_velocity_bp_s)

        displacement = seg_positions[-1] - seg_positions[0]
        duration = seg_times[-1] - seg_times[0]
        simple_rate_um_s = displacement / duration if duration > 0 else np.nan

        fit_rate_um_s = np.nan
        fit_intercept = np.nan
        fitted_positions = np.full_like(seg_positions, np.nan, dtype=float)
        r_squared = np.nan

        if seg_times.size >= 2:
            fit_rate_um_s, fit_intercept = np.polyfit(seg_times, seg_positions, 1)
            fitted_positions = fit_rate_um_s * seg_times + fit_intercept

            ss_res = np.sum((seg_positions - fitted_positions) ** 2)
            ss_tot = np.sum((seg_positions - np.mean(seg_positions)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        local_time_s, local_velocity_um_s = self.compute_local_velocity_series(
            seg_times,
            seg_positions,
            window_points=local_window_points,
        )
        local_velocity_bp_s = local_velocity_um_s * um_to_bp

        return {
            'label': label,
            'color': color,
            'start': float(seg_times[0]),
            'end': float(seg_times[-1]),
            'duration': float(duration),
            'point_count': int(seg_times.size),
            'interval_count': int(interval_mid.size),
            'times': seg_times,
            'positions_um': seg_positions,
            'positions_bp': seg_positions * um_to_bp,
            'time_from_start_s': seg_times - seg_times[0],
            'position_from_start_um': seg_positions - seg_positions[0],
            'position_from_start_bp': (seg_positions - seg_positions[0]) * um_to_bp,
            'interval_start_s': interval_start,
            'interval_end_s': interval_end,
            'interval_mid_s': interval_mid,
            'interval_from_start_s': interval_mid - seg_times[0],
            'delta_t_s': delta_t,
            'delta_position_um': delta_position,
            'delta_position_bp': delta_position * um_to_bp,
            'instant_velocity_um_s': instant_velocity_um_s,
            'instant_velocity_bp_s': instant_velocity_bp_s,
            'abs_velocity_bp_s': abs_velocity_bp_s,
            'local_velocity_time_s': local_time_s,
            'local_velocity_time_from_start_s': local_time_s - seg_times[0] if local_time_s.size else np.array([]),
            'local_velocity_um_s': local_velocity_um_s,
            'local_velocity_bp_s': local_velocity_bp_s,
            'local_window_points': local_window_points,
            'mean_velocity_um_s': float(np.mean(instant_velocity_um_s)),
            'mean_velocity_bp_s': float(np.mean(instant_velocity_bp_s)),
            'mean_abs_velocity_bp_s': float(np.mean(abs_velocity_bp_s)),
            'max_abs_velocity_bp_s': float(np.max(abs_velocity_bp_s)),
            'displacement_um': float(displacement),
            'displacement_bp': float(displacement * um_to_bp),
            'simple_rate_um_s': float(simple_rate_um_s),
            'simple_rate_bp_s': float(simple_rate_um_s * um_to_bp) if np.isfinite(simple_rate_um_s) else np.nan,
            'fit_rate_um_s': float(fit_rate_um_s) if np.isfinite(fit_rate_um_s) else np.nan,
            'fit_rate_bp_s': float(fit_rate_um_s * um_to_bp) if np.isfinite(fit_rate_um_s) else np.nan,
            'abs_fit_rate_bp_s': float(abs(fit_rate_um_s) * um_to_bp) if np.isfinite(fit_rate_um_s) else np.nan,
            'fit_intercept_um': float(fit_intercept) if np.isfinite(fit_intercept) else np.nan,
            'fitted_positions_um': fitted_positions,
            'fitted_positions_bp': fitted_positions * um_to_bp,
            'r_squared': float(r_squared) if np.isfinite(r_squared) else np.nan,
            'um_to_bp': um_to_bp,
        }

    def build_velocity_distribution_table(self, velocity_bp_s, bin_count=30):
        """Build histogram data so OriginLab can reproduce the same histogram bins as the preview plot."""
        velocity_bp_s = np.asarray(velocity_bp_s, dtype=float)
        velocity_bp_s = velocity_bp_s[np.isfinite(velocity_bp_s)]

        if velocity_bp_s.size == 0:
            empty_df = pd.DataFrame(
                columns=[
                    'Bin_Index',
                    'Bin_Left_bp_s',
                    'Bin_Right_bp_s',
                    'Bin_Center_bp_s',
                    'Bin_Width_bp_s',
                    'Count',
                    'Probability',
                    'Density',
                ]
            )
            return empty_df, {'bin_count': 0, 'mean': np.nan, 'std': np.nan, 'median': np.nan}

        if np.allclose(velocity_bp_s, velocity_bp_s[0]):
            spread = max(abs(float(velocity_bp_s[0])) * 0.05, 1.0)
            bin_edges = np.linspace(float(velocity_bp_s[0]) - spread, float(velocity_bp_s[0]) + spread, bin_count + 1)
        else:
            data_min = float(np.min(velocity_bp_s))
            data_max = float(np.max(velocity_bp_s))
            bin_edges = np.linspace(data_min, data_max, bin_count + 1)

        counts, bin_edges = np.histogram(velocity_bp_s, bins=bin_edges)
        widths = np.diff(bin_edges)
        probability = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
        density = probability / widths

        distribution_df = pd.DataFrame({
            'Bin_Index': np.arange(1, len(counts) + 1),
            'Bin_Left_bp_s': bin_edges[:-1],
            'Bin_Right_bp_s': bin_edges[1:],
            'Bin_Center_bp_s': (bin_edges[:-1] + bin_edges[1:]) / 2.0,
            'Bin_Width_bp_s': widths,
            'Count': counts,
            'Probability': probability,
            'Density': density,
        })

        stats = {
            'bin_count': len(counts),
            'mean': float(np.mean(velocity_bp_s)),
            'std': float(np.std(velocity_bp_s)),
            'median': float(np.median(velocity_bp_s)),
            'min': float(np.min(velocity_bp_s)),
            'max': float(np.max(velocity_bp_s)),
            'n_values': int(velocity_bp_s.size),
        }
        return distribution_df, stats

    def build_velocity_export_frames(self, segment_results, channel, track_idx, total_track_points,
                                     max_abs_velocity_bp_s=None, velocity_method='local_fit', local_window_points=7):
        """Build Origin-friendly Excel sheets for segment velocity analysis."""
        if not segment_results:
            return {}

        um_to_bp = segment_results[0]['um_to_bp']

        summary_rows = [
            {'Parameter': 'Channel', 'Value': channel.capitalize()},
            {'Parameter': 'Track Index', 'Value': track_idx + 1},
            {'Parameter': 'Track Points', 'Value': total_track_points},
            {'Parameter': 'Segments Exported', 'Value': len(segment_results)},
            {'Parameter': 'DNA Conversion (bp/um)', 'Value': um_to_bp},
            {'Parameter': 'Analysis Timestamp', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
        ]

        point_rows = []
        instant_rows = []
        local_rows = []
        selected_rows = []
        fit_rows = []

        for result in segment_results:
            for idx, (time_s, time_from_start, pos_um, pos_bp, pos_rel_um, pos_rel_bp) in enumerate(
                zip(
                    result['times'],
                    result['time_from_start_s'],
                    result['positions_um'],
                    result['positions_bp'],
                    result['position_from_start_um'],
                    result['position_from_start_bp'],
                ),
                start=1,
            ):
                point_rows.append({
                    'Segment': result['label'],
                    'Point_Index': idx,
                    'Time_s': time_s,
                    'Time_From_Segment_Start_s': time_from_start,
                    'Position_um': pos_um,
                    'Position_bp': pos_bp,
                    'Position_From_Segment_Start_um': pos_rel_um,
                    'Position_From_Segment_Start_bp': pos_rel_bp,
                })

            for idx, values in enumerate(
                zip(
                    result['interval_start_s'],
                    result['interval_end_s'],
                    result['interval_mid_s'],
                    result['interval_from_start_s'],
                    result['delta_t_s'],
                    result['delta_position_um'],
                    result['delta_position_bp'],
                    result['instant_velocity_um_s'],
                    result['instant_velocity_bp_s'],
                    result['abs_velocity_bp_s'],
                ),
                start=1,
            ):
                (
                    time_start,
                    time_end,
                    time_mid,
                    time_mid_from_start,
                    delta_t,
                    delta_pos_um,
                    delta_pos_bp,
                    velocity_um_s,
                    velocity_bp_s,
                    abs_velocity_bp_s,
                ) = values

                instant_rows.append({
                    'Segment': result['label'],
                    'Interval_Index': idx,
                    'Time_Start_s': time_start,
                    'Time_End_s': time_end,
                    'Time_Mid_s': time_mid,
                    'Time_Mid_From_Segment_Start_s': time_mid_from_start,
                    'Delta_t_s': delta_t,
                    'Delta_Position_um': delta_pos_um,
                    'Delta_Position_bp': delta_pos_bp,
                    'Velocity_um_s': velocity_um_s,
                    'Velocity_bp_s': velocity_bp_s,
                    'Abs_Velocity_bp_s': abs_velocity_bp_s,
                })

            for idx, values in enumerate(
                zip(
                    result['local_velocity_time_s'],
                    result['local_velocity_time_from_start_s'],
                    result['local_velocity_um_s'],
                    result['local_velocity_bp_s'],
                ),
                start=1,
            ):
                time_s, time_from_start_s, velocity_um_s, velocity_bp_s = values
                local_rows.append({
                    'Segment': result['label'],
                    'Point_Index': idx,
                    'Time_s': time_s,
                    'Time_From_Segment_Start_s': time_from_start_s,
                    'Velocity_LocalFit_um_s': velocity_um_s,
                    'Velocity_LocalFit_bp_s': velocity_bp_s,
                })

            fit_rows.append({
                'Segment': result['label'],
                'Start_s': result['start'],
                'End_s': result['end'],
                'Duration_s': result['duration'],
                'Point_Count': result['point_count'],
                'Interval_Count': result['interval_count'],
                'Displacement_um': result['displacement_um'],
                'Displacement_bp': result['displacement_bp'],
                'Mean_Velocity_um_s': result['mean_velocity_um_s'],
                'Mean_Velocity_bp_s': result['mean_velocity_bp_s'],
                'Mean_Abs_Velocity_bp_s': result['mean_abs_velocity_bp_s'],
                'Max_Abs_Velocity_bp_s': result['max_abs_velocity_bp_s'],
                'Simple_Rate_um_s': result['simple_rate_um_s'],
                'Simple_Rate_bp_s': result['simple_rate_bp_s'],
                'Fit_Rate_um_s': result['fit_rate_um_s'],
                'Fit_Rate_bp_s': result['fit_rate_bp_s'],
                'Abs_Fit_Rate_bp_s': result['abs_fit_rate_bp_s'],
                'Fit_Intercept_um': result['fit_intercept_um'],
                'R_squared': result['r_squared'],
            })

        instant_df = pd.DataFrame(instant_rows)
        local_df = pd.DataFrame(local_rows)

        method_name = 'Local linear fit' if velocity_method == 'local_fit' else 'Raw point-to-point'

        if velocity_method == 'local_fit':
            for row in local_rows:
                selected_rows.append({
                    'Segment': row['Segment'],
                    'Index': row['Point_Index'],
                    'Time_s': row['Time_s'],
                    'Time_From_Segment_Start_s': row['Time_From_Segment_Start_s'],
                    'Velocity_um_s': row['Velocity_LocalFit_um_s'],
                    'Velocity_bp_s': row['Velocity_LocalFit_bp_s'],
                })
        else:
            for row in instant_rows:
                selected_rows.append({
                    'Segment': row['Segment'],
                    'Index': row['Interval_Index'],
                    'Time_s': row['Time_Mid_s'],
                    'Time_From_Segment_Start_s': row['Time_Mid_From_Segment_Start_s'],
                    'Velocity_um_s': row['Velocity_um_s'],
                    'Velocity_bp_s': row['Velocity_bp_s'],
                })

        selected_df = pd.DataFrame(selected_rows)

        if not selected_df.empty:
            if max_abs_velocity_bp_s is not None and np.isfinite(max_abs_velocity_bp_s) and max_abs_velocity_bp_s > 0:
                selected_df['Pass_Filter'] = np.abs(selected_df['Velocity_bp_s']) <= max_abs_velocity_bp_s
            else:
                selected_df['Pass_Filter'] = True
        else:
            selected_df['Pass_Filter'] = pd.Series(dtype=bool)

        raw_distribution_df, raw_distribution_stats = self.build_velocity_distribution_table(
            selected_df['Velocity_bp_s'].to_numpy() if not selected_df.empty else np.array([])
        )
        filtered_distribution_df, filtered_distribution_stats = self.build_velocity_distribution_table(
            selected_df.loc[selected_df['Pass_Filter'], 'Velocity_bp_s'].to_numpy() if not selected_df.empty else np.array([])
        )

        summary_rows.extend([
            {'Parameter': 'Velocity Method Used', 'Value': method_name},
            {'Parameter': 'Local Fit Window (points)', 'Value': local_window_points if velocity_method == 'local_fit' else 'N/A'},
            {'Parameter': 'Max |Velocity| Filter (bp/s)', 'Value': max_abs_velocity_bp_s if max_abs_velocity_bp_s is not None else 'None'},
            {'Parameter': 'Instant Velocity Points - Raw', 'Value': raw_distribution_stats.get('n_values', 0)},
            {'Parameter': 'Instant Velocity Points - Filtered', 'Value': filtered_distribution_stats.get('n_values', 0)},
            {'Parameter': 'Removed by Filter', 'Value': max(raw_distribution_stats.get('n_values', 0) - filtered_distribution_stats.get('n_values', 0), 0)},
            {'Parameter': 'Distribution Bin Count', 'Value': raw_distribution_stats.get('bin_count', 0)},
            {'Parameter': 'Raw Mean (bp/s)', 'Value': raw_distribution_stats.get('mean', np.nan)},
            {'Parameter': 'Raw Std (bp/s)', 'Value': raw_distribution_stats.get('std', np.nan)},
            {'Parameter': 'Filtered Mean (bp/s)', 'Value': filtered_distribution_stats.get('mean', np.nan)},
            {'Parameter': 'Filtered Std (bp/s)', 'Value': filtered_distribution_stats.get('std', np.nan)},
            {'Parameter': 'Recommended Origin Sheet', 'Value': 'VelocityDistribution_Filtered'},
            {'Parameter': 'Recommended Origin X', 'Value': 'Bin_Center_bp_s'},
            {'Parameter': 'Recommended Origin Y', 'Value': 'Count'},
            {'Parameter': 'Raw Data Sheet', 'Value': 'InstantVelocity'},
            {'Parameter': 'Raw Data Column', 'Value': 'Velocity_bp_s'},
        ])

        return {
            'Summary': pd.DataFrame(summary_rows),
            'SegmentPoints': pd.DataFrame(point_rows),
            'InstantVelocity': instant_df,
            'LocalInstantVelocity': local_df,
            'VelocitySeries_Selected': selected_df,
            'VelocityDistribution_Raw': raw_distribution_df,
            'VelocityDistribution_Filtered': filtered_distribution_df,
            'SegmentFit': pd.DataFrame(fit_rows),
        }
    def delete_trace_points(self):
        """
        ✏️ 交互式删除Trace中的不好点 - 修复版本
        底部按钮正确显示，不被matplotlib工具栏遮挡
        """
        if not hasattr(self, 'tracks_dict') or not any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue']):
            messagebox.showwarning("Warning", "No tracks available")
            return
        
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("✏️ Delete Bad Points - Interactive Editor")
        dialog.geometry("1350x900")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # ===== 主容器：使用Frame而不是PanedWindow =====
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # ===== 左侧面板 =====
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        # 轨迹列表
        track_frame = ttk.LabelFrame(left_frame, text="Select Track", padding=10)
        track_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        track_listbox = tk.Listbox(track_frame, height=10, font=(FONT_FAMILY, FONT_SIZE_SMALL))
        track_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 生成轨迹列表
        track_list = []
        for channel in ['red', 'green', 'blue']:
            for i, track in enumerate(self.tracks_dict[channel]):
                n_points = len(track.time_idx)
                track_id = f"{channel.upper()} #{i+1} ({n_points} pts)"
                track_listbox.insert(tk.END, track_id)
                track_list.append((channel, i, track))
        
        # 统计信息
        stats_frame = ttk.LabelFrame(left_frame, text="Track Info", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)
        
        stats_var = tk.StringVar(value="Select a track")
        ttk.Label(stats_frame, textvariable=stats_var, foreground='blue',
                 font=(FONT_FAMILY, FONT_SIZE_SMALL), wraplength=250,
                 justify=tk.LEFT).pack(anchor=tk.W)
        
        # 删除统计
        delete_stats_var = tk.StringVar(value="Remove: 0\nKeep: 0")
        ttk.Label(stats_frame, textvariable=delete_stats_var, foreground='red',
                 font=(FONT_FAMILY, FONT_SIZE_SMALL), wraplength=250,
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(5,0))
        
        # ===== 右侧面板：图形 =====
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        fig, ax = plt.subplots(figsize=(11, 6))
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # ✅ 重要：工具栏放在canvas下方，而不是上方
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(fill=tk.X, side=tk.TOP)
        
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # 状态
        plot_state = {
            'current_track': None,
            'marked_indices': set(),
        }
        
        def load_track():
            selection = track_listbox.curselection()
            if not selection:
                return
            
            channel, track_idx, track = track_list[selection[0]]
            plot_state['current_track'] = (channel, track_idx, track)
            plot_state['marked_indices'] = set()
            
            n_points = len(track.time_idx)
            duration = self.calculate_track_duration(track)
            
            stats_var.set(
                f"Channel: {channel.upper()}\n"
                f"Track #{track_idx+1}\n"
                f"Points: {n_points}\n"
                f"Duration: {duration:.3f} s"
            )
            
            redraw_plot()
        
        def redraw_plot():
            ax.clear()
            
            if plot_state['current_track'] is None:
                ax.text(0.5, 0.5, 'Select a track',
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                canvas.draw()
                return
            
            channel, track_idx, track = plot_state['current_track']
            
            time_indices = np.array(track.time_idx)
            pos_indices = np.array(track.coordinate_idx)
            
            if self.roi_coords:
                x1, y1, _, _ = self.roi_coords
                time_indices = time_indices + x1
                pos_indices = pos_indices + y1
            
            times = time_indices * self.delta_line_time
            positions = pos_indices * self.pixel_size_nm / 1000
            
            # 完整轨迹
            ax.plot(times, positions, 'o-', color='lightgray', linewidth=2, markersize=6,
                   alpha=0.5, label='Full track', zorder=1)
            
            # 保留的点（绿色）
            keep_mask = np.array([pt_idx not in plot_state['marked_indices'] 
                                 for pt_idx in range(len(times))])
            if np.any(keep_mask):
                ax.plot(times[keep_mask], positions[keep_mask], 'o', color='green',
                       markersize=8, alpha=0.8, label='Keep', zorder=3)
            
            # 标记删除的点（红色）
            marked_indices = sorted(list(plot_state['marked_indices']))
            if marked_indices:
                marked_mask = np.array([i in plot_state['marked_indices'] for i in range(len(times))])
                ax.plot(times[marked_mask], positions[marked_mask], 'x', color='red',
                       markersize=12, markeredgewidth=3, label='Delete', zorder=4)
            
            ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Position (um)', fontsize=11, fontweight='bold')
            ax.set_title(f'{channel.upper()} Track #{track_idx+1} - Delete Points\n'
                        f'Remove: {len(marked_indices)} | Keep: {len(times) - len(marked_indices)}',
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 更新删除统计
            delete_stats_var.set(f"Remove: {len(marked_indices)}\nKeep: {len(times) - len(marked_indices)}")
            
            canvas.draw()
        
        def on_canvas_click(event):
            if event.inaxes != ax or plot_state['current_track'] is None:
                return
            
            if event.button == 1:  # 左键
                channel, track_idx, track = plot_state['current_track']
                
                time_indices = np.array(track.time_idx)
                pos_indices = np.array(track.coordinate_idx)
                
                if self.roi_coords:
                    x1, y1, _, _ = self.roi_coords
                    time_indices = time_indices + x1
                    pos_indices = pos_indices + y1
                
                times = time_indices * self.delta_line_time
                positions = pos_indices * self.pixel_size_nm / 1000
                
                distances = np.sqrt((times - event.xdata)**2 + (positions - event.ydata)**2)
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < 0.5:
                    if closest_idx in plot_state['marked_indices']:
                        plot_state['marked_indices'].remove(closest_idx)
                        self.log(f"Unmarked point {closest_idx+1}")
                    else:
                        plot_state['marked_indices'].add(closest_idx)
                        self.log(f"Marked point {closest_idx+1}")
                    
                    redraw_plot()
            
            elif event.button == 3:  # 右键：撤销
                if plot_state['marked_indices']:
                    last_marked = max(plot_state['marked_indices'])
                    plot_state['marked_indices'].remove(last_marked)
                    self.log(f"Unmarked point {last_marked+1}")
                    redraw_plot()
        
        def apply_deletion():
            """✅ 执行删除操作"""
            if plot_state['current_track'] is None:
                return
            
            channel, track_idx, track = plot_state['current_track']
            to_delete = sorted(list(plot_state['marked_indices']), reverse=True)
            
            if not to_delete:
                messagebox.showinfo("Info", "No points marked for deletion")
                return
            
            msg = (f"Delete {len(to_delete)} points from {channel.upper()} Track #{track_idx+1}?\n\n"
                   f"Original: {len(track.time_idx)} points\n"
                   f"After: {len(track.time_idx) - len(to_delete)} points")
            # 🆕 保存删除前的备份
            import copy
            backup_track = copy.deepcopy(track)
            
            # 保存到备份列表，使用track_idx作为索引
            while len(self.track_backups[channel]) <= track_idx:
                self.track_backups[channel].append(None)
            self.track_backups[channel][track_idx] = backup_track
            
            self.log(f"✓ Backup created for {channel.upper()} Track #{track_idx+1}")
            
            if not messagebox.askyesno("Confirm", msg):
                return
            
            try:
                for idx in to_delete:
                    track.time_idx = np.delete(track.time_idx, idx)
                    track.coordinate_idx = np.delete(track.coordinate_idx, idx)
                    
                    if hasattr(track, 'intensity') and track.intensity is not None:
                        track.intensity = np.delete(track.intensity, idx)
                
                messagebox.showinfo("Success", 
                                  f"✓ Deleted {len(to_delete)} points\n"
                                  f"Remaining: {len(track.time_idx)} points")
                
                self.log(f"✓ Deleted {len(to_delete)} points from {channel.upper()} Track #{track_idx+1}")
                
                # ✅ 更新主界面
                self.update_display()

                # 🆕 强制刷新主界面
                self.canvas.draw_idle()
                self.root.update()
                
                # ✅ 关闭对话框
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed:\n{str(e)}")
        
        # 事件绑定
        track_listbox.bind('<<ListboxSelect>>', lambda e: load_track())
        canvas.mpl_connect('button_press_event', on_canvas_click)
        def restore_track():
            """🆕 恢复删除前的轨迹"""
            if plot_state['current_track'] is None:
                return
            
            channel, track_idx, track = plot_state['current_track']
            
            # 检查是否有备份
            if (track_idx >= len(self.track_backups[channel]) or 
                self.track_backups[channel][track_idx] is None):
                messagebox.showwarning("No Backup", 
                    f"No backup found for {channel.upper()} Track #{track_idx+1}")
                return
            
            if not messagebox.askyesno("Confirm Restore", 
                f"Restore {channel.upper()} Track #{track_idx+1} to previous state?\n\n"
                f"Current: {len(track.time_idx)} points\n"
                f"Backup: {len(self.track_backups[channel][track_idx].time_idx)} points"):
                return
            
            try:
                # 恢复轨迹
                backup_track = self.track_backups[channel][track_idx]
                self.tracks_dict[channel][track_idx] = copy.deepcopy(backup_track)
                
                messagebox.showinfo("Success", f"✓ Track restored successfully")
                self.log(f"✓ Restored {channel.upper()} Track #{track_idx+1} from backup")
                
                # 更新显示
                self.update_display()
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to restore:\n{str(e)}")
        
        # 🆕 添加恢复按钮
        
        # ===== 底部按钮框（单独放在对话框最底部） =====
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=10)
        ttk.Button(button_frame, text="🔄 Restore Backup",
                  command=restore_track).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        def preview_on_main():
            """在主界面预览删除效果"""
            if plot_state['current_track'] is None:
                return
            
            channel, track_idx, track = plot_state['current_track']
            marked_indices = plot_state['marked_indices']
            
            if len(marked_indices) == 0:
                messagebox.showinfo("Info", "No points marked for deletion")
                return
            
            # 获取轨迹数据
            time_indices = np.array(track.time_idx)
            pos_indices = np.array(track.coordinate_idx)
            
            if self.roi_coords:
                x1, y1, _, _ = self.roi_coords
                time_indices = time_indices + x1
                pos_indices = pos_indices + y1
            
            times = time_indices * self.delta_line_time
            positions = pos_indices * self.pixel_size_nm / 1000
            
            # 创建预览mask
            keep_mask = np.array([i not in marked_indices for i in range(len(times))])
            preview_times = times[keep_mask]
            preview_positions = positions[keep_mask]
            
            # 在主界面绘制预览
            try:
                # 清除主图上的旧预览（如果有）
                for line in self.ax.lines[:]:
                    if hasattr(line, '_is_preview'):
                        line.remove()
                
                # 绘制预览轨迹
                color_map = {'red': '#FF4444', 'green': '#44DD44', 'blue': '#4444FF'}
                preview_color = color_map.get(channel, 'gray')
                
                preview_line, = self.ax.plot(preview_times, preview_positions, 'o-',
                           color=preview_color, linewidth=3, markersize=6,
                           alpha=0.9, label=f'{channel.upper()} T{track_idx+1} Preview (After Delete)',
                           zorder=15)
                preview_line._is_preview = True  # 标记为预览线
                
                self.ax.legend(fontsize=9, loc='best')
                self.canvas.draw()
                
                self.log(f"👁️ Preview: {len(preview_times)} points will remain (deleted {len(marked_indices)} points)")
                messagebox.showinfo("Preview", 
                    f"Preview shown on main plot\n\n"
                    f"Original: {len(times)} points\n"
                    f"After deletion: {len(preview_times)} points\n"
                    f"Deleted: {len(marked_indices)} points")
                
            except Exception as e:
                messagebox.showerror("Preview Error", f"Failed to preview:\n{str(e)}")
        
        ttk.Button(button_frame, text="👁️ Preview on Main",
                  command=preview_on_main).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        ttk.Button(button_frame, text="✅ Apply & Return",
                  command=apply_deletion).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(button_frame, text="Reset",
                  command=lambda: [plot_state['marked_indices'].clear(), redraw_plot()]).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(button_frame, text="Close",
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 初始化
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
    def log(self, message):  
        """Log output with timestamp"""  
        timestamp = datetime.now().strftime("%H:%M:%S")  
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")  
        self.log_text.see(tk.END)  
        self.root.update()  
    
    # ========== ROI Methods ==========  
    
    def clear_temp_markers(self):
        """清除临时标记（用于ROI选择）"""
        for marker in self.temp_markers:
            try:
                marker.remove()
            except:
                pass
        self.temp_markers = []
        self.canvas.draw()
    def activate_two_click_roi(self):  
        """Activate two-click Y-axis ROI selection mode"""  
        # Disable conflicting modes
        self.disable_delete_modes()
        
        if self.kymo is None:  
            messagebox.showwarning("Warning", "Please load a kymograph first")  
            return  
        
        # ⭐ 清除旧的连接
        try:
            self.canvas.mpl_disconnect(self.cid_canvas_click) if hasattr(self, 'cid_canvas_click') else None
        except:
            pass
        
        self.roi_selection_mode = True  
        self.roi_clicks = []  
        self.clear_temp_markers()  
        
        # ⭐ 重新连接点击事件（关键！）
        self.cid_canvas_click = self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        self.log("ROI selection activated - Click TWO points on Y-axis")  
        self.roi_status_var.set("ROI Mode: Click 2 points to define Y-axis range")  
        messagebox.showinfo("ROI Selection",   
                           "Click TWO points on the kymograph\n"  
                           "to define the Y-axis range.\n\n"  
                           "X-axis will use full kymograph width.")
    def on_canvas_click(self, event):
        """处理canvas点击事件 - 用于ROI选择"""
        # 如果处于矩形ROI模式
        if self.roi_selection_mode and event.inaxes == self.ax:
            y_pos_real = event.ydata
            if y_pos_real is not None:
                y_pos_pixel = y_pos_real * 1000 / self.pixel_size_nm
                self.roi_clicks.append(y_pos_pixel)
                
                # Draw marker
                marker = self.ax.axhline(y=y_pos_real, color='yellow', linestyle='--', linewidth=2)
                self.temp_markers.append(marker)
                self.canvas.draw()
                
                self.log(f"Click {len(self.roi_clicks)}: Y = {y_pos_real:.3f} μm")
                
                if len(self.roi_clicks) == 2:
                    self.finalize_two_click_roi()
    def finalize_two_click_roi(self):
        """完成两点ROI选择"""
        if len(self.roi_clicks) != 2:
            return
        
        y1_pixel, y2_pixel = self.roi_clicks
        
        # 确保y1 < y2
        if y1_pixel > y2_pixel:
            y1_pixel, y2_pixel = y2_pixel, y1_pixel
        
        # 转换为整数像素坐标
        y1_pixel = int(y1_pixel)
        y2_pixel = int(y2_pixel)
        
        # X轴使用整个kymograph宽度
        x1_pixel = 0
        x2_pixel = self.red_image.shape[1]
        
        self.roi_coords = (x1_pixel, y1_pixel, x2_pixel, y2_pixel)
        self.roi_type = 'rectangular'
        
        # 更新UI
        y1_real = y1_pixel * self.pixel_size_nm / 1000
        y2_real = y2_pixel * self.pixel_size_nm / 1000
        
        self.roi_status_var.set(f"✅ Rectangular ROI Active: Y [{y1_real:.2f}, {y2_real:.2f}] μm")
        self.roi_details_var.set(f"Height: {abs(y2_real - y1_real):.2f} μm | Area: {(x2_pixel - x1_pixel) * (y2_pixel - y1_pixel)} pixels")
        
        self.log(f"✓ ROI finalized: Y range [{y1_real:.2f}, {y2_real:.2f}] μm")
        
        # 清理状态
        self.roi_selection_mode = False
        self.roi_clicks = []
        self.clear_temp_markers()
        
        # 断开点击事件（不再接受点击）
        try:
            self.canvas.mpl_disconnect(self.cid_canvas_click)
        except:
            pass
        
        # 重绘显示
        self.update_display()
    
    def activate_curved_roi(self):
        """激活弯曲多边形ROI选择模式"""
        # Disable conflicting modes
        self.disable_delete_modes()
        
        if self.kymo is None:
            messagebox.showwarning("Warning", "Please load a kymograph first")
            return
        
        # 创建弯曲ROI工具
        self.curved_roi_tool = CurvedPolygonROI(
            self.red_image,
            self.fig,
            self.ax,
            self.pixel_size_nm,
            self.delta_line_time
        )
        
        # ✅ 连接状态回调
        def status_update(msg):
            self.log(msg)
            self.roi_details_var.set(msg)
        
        self.curved_roi_tool.status_callback = status_update
        self.curved_roi_tool.connect()
        
        instructions = (
            "🌀 CURVED POLYGON ROI MODE\n\n"
            "ACTIONS:\n"
            "  • LEFT CLICK on kymograph: Add vertex\n"
            "  • Z KEY: Undo last vertex\n"
            "  • ENTER KEY: Finish ROI (need ≥3 vertices)\n"
            "  • ESC KEY: Cancel\n\n"
            "TIPS:\n"
            "  • Click 5-10 points along your diffusive trace\n"
            "  • Don't click too close together\n"
            "  • Curve will be auto-smoothed\n"
            "  • ROI will tightly fit your trace"
        )
        
        messagebox.showinfo("🌀 Curved ROI Mode", instructions)
        
        self.log("🌀 Curved ROI selection activated")
        self.log("   Start clicking on kymograph to add vertices")
        self.roi_status_var.set("🔄 Curved ROI Mode: Waiting for clicks...")
        self.roi_details_var.set(f"Vertices: 0 / Need: ≥3")
        
        # 使用消息循环等待ROI完成
        self.wait_for_curved_roi_completion()


    def wait_for_curved_roi_completion(self):
        """
        ✅ 改进版：等待用户完成弯曲ROI的输入
        处理用户按 Enter 键完成、按 ESC 取消、或按 Z 撤销顶点的情况
        """
        print("🔄 Waiting for ROI completion...")
        print("   - Left-click to add vertices")
        print("   - Z key to undo")
        print("   - Enter to complete")
        print("   - ESC to cancel")
        
        def check_roi_status():
            """定期检查ROI状态"""
            if not hasattr(self, 'curved_roi_tool') or not self.curved_roi_tool:
                print("❌ ROI tool not available")
                return
            
            tool = self.curved_roi_tool
            n_vertices = len(tool.vertices)
            
            # 显示当前顶点数（供调试使用）
            # print(f"   Current vertices: {n_vertices}")
            
            # 更新UI中的顶点计数
            self.roi_details_var.set(f"Vertices: {n_vertices} / Need: ≥3")
            
            # ✅ 关键：检查工具是否还在连接状态
            if not tool.connected:
                # 工具已断开连接（用户按了Enter或ESC）
                print(f"✅ ROI tool disconnected (connected={tool.connected})")
                
                if n_vertices >= 3:
                    print(f"✅ ROI is valid: {n_vertices} vertices")
                    self.apply_curved_roi()
                else:
                    print(f"❌ ROI is invalid: only {n_vertices} vertices (need ≥3)")
                    self.log(f"⚠️  ROI cancelled: insufficient vertices ({n_vertices})")
                    self.roi_details_var.set(f"⚠️  Cancelled: need ≥3 vertices")
                    # 清理
                    if hasattr(self, 'curved_roi_tool') and self.curved_roi_tool:
                        self.curved_roi_tool = None
                return
            
            # 继续监视
            self.root.after(100, check_roi_status)
        
        # 开始监视
        check_roi_status()

    def apply_curved_roi(self):
        """应用弯曲多边形ROI - 修复版本（带验证）"""
        if not hasattr(self, 'curved_roi_tool') or not self.curved_roi_tool:
            messagebox.showerror("Error", "ROI tool not initialized")
            return
        
        vertices = self.curved_roi_tool.vertices
        
        # ✅ 检查顶点数量
        if len(vertices) < 3:
            messagebox.showwarning("Error", "ROI requires at least 3 vertices")
            return
        
        # ✅ 检查图像是否已加载
        if self.red_image is None:
            messagebox.showerror("Error", "No image data available")
            return
        
        height, width = self.red_image.shape
        
        # ✅ 检查所有顶点是否在有效范围内
        print(f"🔍 验证ROI顶点...")
        print(f"   图像尺寸: {width}×{height}")
        for i, (x_px, y_px) in enumerate(vertices):
            print(f"   顶点 {i+1}: ({x_px:.1f}, {y_px:.1f}) px", end="")
            if not (0 <= x_px < width and 0 <= y_px < height):
                messagebox.showerror("Error", 
                    f"Vertex {i+1} ({x_px:.1f}, {y_px:.1f}) is outside image bounds "
                    f"({width}×{height})")
                self.log(f"❌ ROI verification failed: Vertex {i+1} out of bounds")
                return
            print(" ✓")
        
        # ✅ 创建掩码
        print(f"🔄 创建多边形掩码...")
        self.roi_type = 'polygon'
        self.roi_vertices = vertices.copy()
        self.roi_mask = polygon_to_mask(self.roi_vertices, self.red_image.shape)
        
        # ✅ 验证掩码是否有效
        masked_pixels = np.sum(self.roi_mask)
        print(f"   掩码像素数: {masked_pixels}")
        
        if masked_pixels == 0:
            messagebox.showwarning("Warning", 
                "ROI mask contains no pixels. Please draw a larger polygon.")
            self.roi_mask = None
            self.roi_vertices = []
            self.roi_type = 'rectangular'
            self.log(f"⚠️  ROI mask is empty - reverting to rectangular mode")
            return
        
        # ✅ 计算统计信息
        n_vertices = len(self.roi_vertices)
        area_real = masked_pixels * (self.pixel_size_nm / 1000) ** 2
        
        print(f"✅ ROI creation success:")
        print(f"   顶点数: {n_vertices}")
        print(f"   掩码像素: {masked_pixels}")
        print(f"   面积: {area_real:.2f} μm²")
        
        # ✅ 更新UI
        self.roi_status_var.set(f"✅ Curved ROI Active: {n_vertices} vertices, {masked_pixels} pixels")
        self.roi_details_var.set(f"Area: {area_real:.2f} μm² | Type: Curved Polygon")
        
        self.log(f"✓ Curved ROI finalized: {n_vertices} vertices, {masked_pixels} pixels, {area_real:.2f} μm²")
        self.log(f"  Image shape: {self.red_image.shape}")
        self.log(f"  ROI bounds: X[0-{width}], Y[0-{height}]")
        
        # 重绘以显示ROI
        self.update_display()
        
        # 清理
        self.curved_roi_tool.disconnect()
        self.curved_roi_tool = None
        
    def edit_curved_roi_points(self):
        """编辑现有的弯曲ROI顶点"""
        if self.roi_type != 'polygon' or not self.roi_vertices:
            messagebox.showwarning("Warning", "No curved ROI to edit")
            return
        
        # 创建编辑对话框
        edit_dialog = tk.Toplevel(self.root)
        edit_dialog.title("✏️ Edit Curved ROI Vertices")
        edit_dialog.geometry("600x400")
        edit_dialog.transient(self.root)
        edit_dialog.grab_set()
        
        # 信息框
        info_frame = ttk.LabelFrame(edit_dialog, text="Current Vertices", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶点列表
        text_widget = tk.Text(info_frame, height=15, width=70, font=('Courier', 9))
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # 显示顶点信息
        text_widget.insert('1.0', "Current Vertices:\n" + "="*60 + "\n")
        for i, (x_px, y_px) in enumerate(self.roi_vertices):
            x_real = x_px * self.delta_line_time
            y_real = y_px * self.pixel_size_nm / 1000
            text_widget.insert(tk.END, f"V{i+1}: ({x_real:.2f}s, {y_real:.2f}μm) [{x_px:.1f}px, {y_px:.1f}px]\n")
        
        text_widget.config(state=tk.DISABLED)
        
        # 按钮
        btn_frame = ttk.Frame(edit_dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def reset_roi():
            if messagebox.askyesno("Confirm", "Reset curved ROI and start over?"):
                self.clear_roi()
                edit_dialog.destroy()
        
        ttk.Button(btn_frame, text="🗑️ Delete This ROI",
                  command=reset_roi).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Close",
                  command=edit_dialog.destroy).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)


    def clear_roi(self):  
        """Clear ROI selection（包括多边形）"""  
        self.roi_coords = None  
        self.roi_selection_mode = False  
        self.roi_clicks = []  
        self.clear_temp_markers()  
        
        # 🆕 清除弯曲ROI
        if hasattr(self, 'curved_roi_tool') and self.curved_roi_tool:
            self.curved_roi_tool.clear()
            self.curved_roi_tool.disconnect()
            self.curved_roi_tool = None
        
        self.roi_type = 'rectangular'
        self.roi_vertices = []
        self.roi_mask = None
        
        if self.roi_rect:  
            try:
                self.roi_rect.remove()
            except:
                pass
            self.roi_rect = None  
        
        self.canvas.draw()  
        
        self.roi_status_var.set("No ROI selected")
        self.roi_details_var.set("")
        self.log("ROI cleared")

    def get_roi_image(self, image):  
        """Extract ROI from image（支持矩形和多边形）"""  
        # ⭐ 多边形ROI优先级更高
        if self.roi_type == 'polygon' and self.roi_mask is not None:
            roi_image = image.copy()
            roi_image[~self.roi_mask] = 0  # 关键：设置ROI外的像素为0
            return roi_image
        
        # 矩形ROI
        if self.roi_coords is None:  
            return image  
        
        x1, y1, x2, y2 = self.roi_coords  
        return image[y1:y2, x1:x2] 

    def get_channel_image(self, channel):
        """Return the currently displayed image for a specific channel."""
        if channel == 'red':
            return self.red_image
        if channel == 'green':
            return self.green_image
        if channel == 'blue':
            return self.blue_image
        raise ValueError(f"Unknown channel: {channel}")

    def get_roi_intensity_profile(self, channel):
        """Calculate ROI fluorescence change over time without trace tracking."""
        image = self.get_channel_image(channel)
        if image is None:
            raise ValueError("No channel image loaded")

        image = np.asarray(image, dtype=float)
        height, width = image.shape

        if self.roi_type == 'polygon' and self.roi_mask is not None:
            mask = np.asarray(self.roi_mask, dtype=bool)
            if mask.shape != image.shape:
                raise ValueError(f"ROI mask shape {mask.shape} does not match image shape {image.shape}")

            active_pixels = np.sum(mask, axis=0).astype(float)
            summed_intensity = np.sum(np.where(mask, image, 0.0), axis=0)
            times = np.arange(width, dtype=float) * self.delta_line_time
            x_indices = np.arange(width, dtype=int)
            roi_type = 'polygon'
            roi_area_pixels = int(np.sum(mask))
        elif self.roi_coords is not None:
            x1, y1, x2, y2 = self.roi_coords
            x1 = max(0, min(int(x1), width))
            x2 = max(0, min(int(x2), width))
            y1 = max(0, min(int(y1), height))
            y2 = max(0, min(int(y2), height))

            if x2 <= x1 or y2 <= y1:
                raise ValueError("ROI has zero size")

            roi_image = image[y1:y2, x1:x2]
            summed_intensity = np.sum(roi_image, axis=0)
            active_pixels = np.full(roi_image.shape[1], roi_image.shape[0], dtype=float)
            x_indices = np.arange(x1, x2, dtype=int)
            times = x_indices.astype(float) * self.delta_line_time
            roi_type = 'rectangular'
            roi_area_pixels = int(roi_image.shape[0] * roi_image.shape[1])
        else:
            raise ValueError("No ROI selected")

        mean_intensity = np.divide(
            summed_intensity,
            active_pixels,
            out=np.full_like(summed_intensity, np.nan, dtype=float),
            where=active_pixels > 0
        )

        return {
            'channel': channel,
            'roi_type': roi_type,
            'times': times,
            'relative_times': times - times[0] if len(times) else np.array([], dtype=float),
            'x_indices': x_indices,
            'summed_intensity': np.asarray(summed_intensity, dtype=float),
            'mean_intensity': np.asarray(mean_intensity, dtype=float),
            'active_pixels': np.asarray(active_pixels, dtype=float),
            'roi_area_pixels': roi_area_pixels
        }

    def summarize_roi_intensity_profile(self, profile):
        """Create a compact summary for ROI fluorescence over time."""
        summed = np.asarray(profile['summed_intensity'], dtype=float)
        mean_int = np.asarray(profile['mean_intensity'], dtype=float)
        times = np.asarray(profile['times'], dtype=float)
        active_pixels = np.asarray(profile['active_pixels'], dtype=float)

        finite_sum = summed[np.isfinite(summed)]
        finite_mean = mean_int[np.isfinite(mean_int)]

        start_sum = float(summed[0]) if summed.size and np.isfinite(summed[0]) else np.nan
        end_sum = float(summed[-1]) if summed.size and np.isfinite(summed[-1]) else np.nan
        delta_sum = end_sum - start_sum if np.isfinite(start_sum) and np.isfinite(end_sum) else np.nan
        rel_change = (
            delta_sum / start_sum
            if np.isfinite(delta_sum) and np.isfinite(start_sum) and abs(start_sum) > np.finfo(float).eps
            else np.nan
        )

        return {
            'Channel': profile['channel'].capitalize(),
            'ROI_Type': profile['roi_type'],
            'ROI_Area_Pixels': profile['roi_area_pixels'],
            'Frames': int(len(summed)),
            'Start_Time (s)': float(times[0]) if times.size else np.nan,
            'End_Time (s)': float(times[-1]) if times.size else np.nan,
            'Duration (s)': float(times[-1] - times[0]) if times.size > 1 else 0.0,
            'Mean_Active_Pixels': float(np.nanmean(active_pixels)) if active_pixels.size else np.nan,
            'Mean_Summed_Intensity': float(np.nanmean(finite_sum)) if finite_sum.size else np.nan,
            'Max_Summed_Intensity': float(np.nanmax(finite_sum)) if finite_sum.size else np.nan,
            'Integrated_Summed_Intensity': float(np.nansum(summed)) if summed.size else np.nan,
            'Mean_Pixel_Intensity': float(np.nanmean(finite_mean)) if finite_mean.size else np.nan,
            'Max_Pixel_Intensity': float(np.nanmax(finite_mean)) if finite_mean.size else np.nan,
            'Start_Summed_Intensity': start_sum,
            'End_Summed_Intensity': end_sum,
            'Delta_Summed_Intensity': delta_sum,
            'Relative_Summed_Intensity_Change': rel_change
        }
    
    # ========== Kymograph Selection Methods ==========  
    
    def load_single_kymograph(self, kymo_name):
        """Load a single kymograph (extracted common logic)"""
        try:
            self.log(f"Loading kymograph: {kymo_name}")
            self.kymo = self.current_file.kymos[kymo_name]
            self.current_kymo_name = kymo_name
            
            self.clear_roi()
            
            # ✅ 修复：Mac/pylake 版本兼容性
            try:
                # 尝试新版 API
                self.red_image_original = self.kymo.red_image.copy()
                self.green_image_original = self.kymo.green_image.copy()
                self.blue_image_original = self.kymo.blue_image.copy()
                self.log("✓ Using red_image/green_image/blue_image attributes")
            except AttributeError:
                # Mac 旧版本可能用不同的属性名
                try:
                    # 尝试备选属性名
                    self.red_image_original = np.array(self.kymo.red)
                    self.green_image_original = np.array(self.kymo.green)
                    self.blue_image_original = np.array(self.kymo.blue)
                    self.log("✓ Using red/green/blue attributes (Mac compatibility)")
                except:
                    # 最后的备选方案：从分光数据构建
                    try:
                        # 某些版本可能用 colors 或其他名称
                        if hasattr(self.kymo, 'get_image'):
                            self.red_image_original = self.kymo.get_image('red')
                            self.green_image_original = self.kymo.get_image('green')
                            self.blue_image_original = self.kymo.get_image('blue')
                            self.log("✓ Using get_image() method")
                        else:
                            raise AttributeError("Cannot find image attributes in kymo object")
                    except Exception as e:
                        messagebox.showerror(
                            "Mac Compatibility Error",
                            f"Cannot access kymograph image data.\n\n"
                            f"Error: {str(e)}\n\n"
                            f"This is likely a pylake version issue on Mac.\n\n"
                            f"Solutions:\n"
                            f"1. Update pylake: pip install --upgrade lumicks-pylake\n"
                            f"2. Or reinstall: pip uninstall lumicks-pylake -y && pip install lumicks-pylake\n"
                            f"3. Check your pylake version: python -m pip show lumicks.pylake"
                        )
                        self.log(f"❌ Error: {str(e)}")
                        return
            
            # Set current images (based on flip state)
            if self.y_flip_enabled:
                self.red_image = np.flipud(self.red_image_original)
                self.green_image = np.flipud(self.green_image_original)
                self.blue_image = np.flipud(self.blue_image_original)
            else:
                self.red_image = self.red_image_original.copy()
                self.green_image = self.green_image_original.copy()
                self.blue_image = self.blue_image_original.copy()
            
            max_val = max(np.max(self.red_image), np.max(self.green_image), np.max(self.blue_image))
            self.rgb_image = np.dstack([
                self.red_image * 255 / max_val,
                self.green_image * 255 / max_val,
                self.blue_image * 255 / max_val
            ]).astype(int)
            
            self.pixel_size_nm = self.kymo.pixelsize_um[0] * 1000
            self.delta_line_time = (self.kymo.timestamps[0,1] - self.kymo.timestamps[0,0]) / 1e9
            
            shape = self.red_image.shape
            time_extent = shape[1] * self.delta_line_time
            position_extent = shape[0] * self.pixel_size_nm / 1000
            
            self.current_kymo_var.set(f"{kymo_name} ({shape[0]}×{shape[1]})")
            
            self.track_btn.config(state=tk.NORMAL)  # 直接启用
            
            self.plot_kymo_with_selected_tracks()
            self.log("✓ Kymograph loaded successfully")
            
            self.update_stats(
                f"Kymo: {kymo_name}\n"
                f"Shape: {shape[0]}×{shape[1]} pixels\n"
                f"Time: {time_extent:.2f} s\n"
                f"Position: {position_extent:.2f} μm\n"
                f"Pixel size: {self.pixel_size_nm:.2f} nm\n"
                f"Line time: {self.delta_line_time:.4f} s"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load kymograph:\n{str(e)}")
            self.log(f"✗ Error: {str(e)}")
    def on_kymo_select(self, event):
        """Handle kymograph selection"""
        selection = self.kymo_listbox.curselection()
        if not selection or self.current_file is None:
            return
        
        kymo_name = self.kymo_listbox.get(selection[0])
        self.load_single_kymograph(kymo_name)
    
    # ========== Tracking Methods ==========
    def run_tracking(self):  
        """Run tracking for selected channel（异步版本）"""
        
        if self.kymo is None:  
            messagebox.showwarning("Warning", "Please select a kymograph first")  
            return  
        
        # 禁用按钮，防止重复点击
        self.track_btn.config(state=tk.DISABLED)
        self.log("🔄 Tracking started in background...")
        
        def tracking_task():
            """在后台线程执行的tracking任务"""
            try:  
                # 🆕 Ensure we save any pending UI edits to the params dict first (if running from UI thread, this helps)
                # Note: This runs in a thread, so accessing UI vars directly is technically unsafe if not careful,
                # but tkinter vars (StringVar, etc) are generally thread-safe for reading.
                # BETTER: The run_tracking method (on main thread) already triggered this.
                # Actually, let's just use the values from the dict directly which is safer.
                # But we need to make sure the dict is up to date.
                # We will rely on self.save_current_params() being called BEFORE submitting this task? 
                # Or we just read from the UI because the user hasn't switched channels yet.
                
                # Let's read the channel from the var (it was captured when task started? No, it's looked up now)
                # To be safe, we should have passed 'channel' into the thread. But `self.channel_var` is used.
                
                channel = self.channel_var.get()  
                
                # 🆕 Force update params from UI if we haven't switched channels? 
                # Actually, if we use the dict, we assume on_channel_switch updated it. 
                # But if the user edited values and didn't switch, the dict is STALE.
                # So we MUST read from the UI for the *current* channel's active values.
                # AND we should update the dict with these values.
                
                # In a thread, calling .get() on Tk vars is usually okay.
                line_width = self.line_width_var.get()
                pixel_threshold = self.pixel_threshold_var.get()
                filter_length = self.filter_length_var.get()
                
                # Update the dict for consistency
                self.channel_params[channel]['line_width'] = line_width
                self.channel_params[channel]['pixel_threshold'] = pixel_threshold
                self.channel_params[channel]['filter_length'] = filter_length
                # Note: min_duration is used in UI but maybe not here directly?
                
                self.log("=" * 50)  
                self.log(f"Tracking started for {channel.upper()} channel...")  
                
                # Record flip state  
                orientation = "Y-FLIPPED" if self.y_flip_enabled else "NORMAL"  
                self.log(f"Orientation: {orientation}")  
                
                # Store tracking settings for this channel  
                self.tracking_settings_dict[channel] = {  
                    'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  
                    'source_file': Path(self.current_file_path).name if self.current_file_path else 'N/A',  
                    'kymograph_name': self.current_kymo_name or 'N/A',  
                    'y_axis_flipped': self.y_flip_enabled,  
                    'algorithm': self.algorithm_var.get(),  
                    'channel': channel,  
                    'line_width': line_width,  
                    'pixel_threshold': pixel_threshold,  
                    'filter_line_length': filter_length  
                }  
                
                if channel == "red":  
                    channel_image = self.red_image  
                elif channel == "green":  
                    channel_image = self.green_image  
                else:  
                    channel_image = self.blue_image  
                

                if self.roi_type == 'polygon' and self.roi_mask is not None:
                    # ✅ 多边形ROI - 修复版本
                    print(f"🔍 应用多边形ROI:")
                    print(f"   通道: {channel}")
                    print(f"   图像尺寸: {channel_image.shape}")
                    print(f"   顶点数: {len(self.roi_vertices)}")
                    
                    if len(channel_image.shape) == 2:
                        roi_image = channel_image.copy()
                        
                        # 验证掩码尺寸
                        if self.roi_mask.shape != channel_image.shape:
                            print(f"⚠️  警告：掩码尺寸 {self.roi_mask.shape} 与图像尺寸 {channel_image.shape} 不匹配！")
                            self.log(f"⚠️  ROI mask size mismatch: {self.roi_mask.shape} vs {channel_image.shape}")
                            # 尝试自动调整
                            if self.roi_mask.shape[0] == channel_image.shape[0] and self.roi_mask.shape[1] == channel_image.shape[1]:
                                pass  # 尺寸匹配，继续
                            else:
                                print(f"❌ 无法应用ROI：尺寸不匹配")
                                channel_image = channel_image  # 不应用ROI
                        else:
                            # 应用ROI掩码
                            roi_image[~self.roi_mask] = 0
                            channel_image = roi_image
                            
                            # 统计信息
                            roi_pixel_count = np.sum(self.roi_mask)
                            roi_signal = np.sum(channel_image)
                            
                            print(f"✓ ROI已应用:")
                            print(f"   ROI像素数: {roi_pixel_count}")
                            print(f"   ROI内信号强度: {roi_signal:.0f}")
                            
                            if roi_signal == 0:
                                self.log(f"⚠️  警告: ROI区域内无信号!")
                            
                            self.tracking_settings_dict[channel]['roi_type'] = 'polygon'
                            self.tracking_settings_dict[channel]['roi'] = f"{len(self.roi_vertices)} vertices, {roi_pixel_count} pixels"
                            self.log(f"Using curved ROI with {len(self.roi_vertices)} vertices")
                    else:
                        print(f"⚠️  错误: 通道图像的形状异常: {channel_image.shape}")
                        self.log(f"⚠️  错误: Channel image shape unexpected: {channel_image.shape}")
                elif self.roi_coords:  
                    # 矩形ROI
                    channel_image = self.get_roi_image(channel_image)  
                    self.log(f"Using rectangular ROI: {self.roi_coords}")  
                    self.tracking_settings_dict[channel]['roi_type'] = 'rectangular'
                    self.tracking_settings_dict[channel]['roi'] = str(self.roi_coords)  
                else:  
                    self.tracking_settings_dict[channel]['roi'] = 'None'  

                algorithm = self.algorithm_var.get()  
                channel_image = np.asarray(channel_image, dtype=float)
                channel_min = float(np.nanmin(channel_image)) if channel_image.size else 0.0
                channel_max = float(np.nanmax(channel_image)) if channel_image.size else 0.0
                channel_mean = float(np.nanmean(channel_image)) if channel_image.size else 0.0
                self.log(
                    f"Signal stats: min={channel_min:.3f}, max={channel_max:.3f}, "
                    f"mean={channel_mean:.3f}, threshold={pixel_threshold:.3f}"
                )
                if algorithm == "greedy" and channel_max > 0 and pixel_threshold >= channel_max:
                    self.log(
                        "Warning: pixel threshold is >= image max intensity. "
                        "This usually prevents trace detection."
                    )
                
                # line_width and pixel_threshold already retrieved above
                
                if algorithm == "greedy":  
                    self.log(f"Algorithm: track_greedy")  
                    self.log(f"Parameters: line_width={line_width}, window={self.window_var.get()}")  
                    
                    sigma_str = self.sigma_var.get()  
                    sigma = None if sigma_str.lower() in ["none", ""] else float(sigma_str)  
                    
                    self.tracking_settings_dict[channel]['window'] = self.window_var.get()  
                    self.tracking_settings_dict[channel]['sigma'] = sigma_str  
                    
                    tracks = lk.track_greedy(  
                        channel_image,  
                        line_width=line_width,  
                        pixel_threshold=pixel_threshold,  
                        window=self.window_var.get(),  
                        sigma=sigma  
                    )  
                else:  
                    self.log(f"Algorithm: track_lines")  
                    self.log(f"Parameters: line_width={line_width}, max_lines={self.max_lines_var.get()}")  
                    
                    self.tracking_settings_dict[channel]['max_lines'] = self.max_lines_var.get()  
                    
                    tracks = lk.track_lines(  
                        channel_image,  
                        line_width=line_width,  
                        max_lines=self.max_lines_var.get()  
                    )  
                
                # Filter tracks  
                # Filter tracks  
                # Use local var filtered_length instead of self.filter_length_var.get()
                filtered_length = filter_length
                filtered_tracks = [track for track in tracks if len(track.time_idx) >= filtered_length]  
                
                total_count = len(tracks)
                filtered_count = len(filtered_tracks)

                if total_count > 0 and filtered_count == 0 and filtered_length > 1:
                    self.log(
                        f"Warning: filter length {filtered_length} removed all detected tracks. "
                        "Keeping raw tracks so they remain visible."
                    )
                    filtered_tracks = tracks
                    filtered_count = total_count
                
                self.log(f"Found {total_count} tracks, filtered to {filtered_count} tracks")  
                self.log(f"(Filter: min length = {filter_length} points)")  
                if total_count == 0:
                    self.log(
                        "No tracks detected. Try lowering pixel threshold, reducing line width, "
                        "or drawing a smaller ROI around the trace."
                    )
                self.log("=" * 50)
                
                # 🆕 回到主线程更新UI
                self.root.after(0, lambda: self.on_tracking_complete(
                    channel, filtered_tracks, total_count
                ))
                
            except Exception as e:  
                # 错误也要回到主线程处理
                self.root.after(0, lambda: messagebox.showerror(
                    "Tracking Error", f"Tracking failed:\n{str(e)}"
                ))
                self.root.after(0, lambda: self.log(f"Error: {str(e)}"))
                self.root.after(0, lambda: self.on_tracking_error())
                
                import traceback  
                self.log(traceback.format_exc())
        
        # 🆕 在线程池中执行（不阻塞UI）
        self.executor.submit(tracking_task)


    def on_tracking_complete(self, channel, filtered_tracks, total_count):
        """Tracking完成的回调（在主线程执行）"""
        # 更新数据
        self.tracks_dict[channel] = filtered_tracks
        
        # 自动启用该通道显示
        if channel == "red":  
            self.show_red_var.set(True)  
        elif channel == "green":  
            self.show_green_var.set(True)  
        else:  
            self.show_blue_var.set(True)  
        
        # 更新UI
        self.update_track_counts()  
        self.update_display()  
        
        # 重新启用按钮
        self.track_btn.config(state=tk.NORMAL)
        
        self.log(f"✅ Tracking complete!")


    def on_tracking_error(self):
        """Tracking出错的回调"""
        # 重新启用按钮
        self.track_btn.config(state=tk.NORMAL)
        self.log("❌ Tracking failed - please check error message")
    
    def apply_filter(self):  
        """Apply additional filters to all tracks using channel-specific settings"""  
        # Save current params first to ensure latest value for current channel is stored
        self.save_current_params()
        
        try:  
            applied_any = False
            for channel in ['red', 'green', 'blue']:  
                # Get param for this channel
                min_dur_str = self.channel_params[channel].get('min_duration', '')
                
                if not min_dur_str:
                    continue
                    
                try:
                    min_dur = float(min_dur_str)
                except ValueError:
                    continue
                
                if len(self.tracks_dict[channel]) > 0:  
                    original = len(self.tracks_dict[channel])  
                    filtered = [t for t in self.tracks_dict[channel]   
                               if self.calculate_track_duration(t) >= min_dur]  
                    self.tracks_dict[channel] = filtered  
                    if original != len(filtered):
                        self.log(f"{channel.upper()}: Filtered {original} -> {len(filtered)} tracks (Duration >= {min_dur}s)")
                        applied_any = True
            
            if applied_any:
                self.update_track_counts()  
                self.update_display()  
                self.log(f"Filters applied successfully")
            else:
                self.log("No filters applied (check min duration settings)")
            
        except Exception as e:  
            messagebox.showerror("Error", str(e))  
            self.log(f"Error: {str(e)}")  
    
    
    def calculate_diffusion_coefficient(self):  
        """  
        Calculate diffusion coefficient with methods from Michalet & Berglund (2012)  
        Supports: MSD, CVE, OLS methods with interactive time segment selection  
        """  
        if not hasattr(self, 'tracks_dict') or not any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue']):  
            messagebox.showwarning("Warning", "No tracks available for analysis")  
            return  
        
        # Create dialog with scrollbar  
        dialog = tk.Toplevel(self.root)  
        dialog.title("📊 Diffusion Coefficient Analysis (Michalet & Berglund 2012 Methods)")  
        dialog.geometry("700x1000")  
        dialog.transient(self.root)  
        dialog.grab_set()  
        
        # ===== MAIN CONTAINER WITH SCROLLBAR =====  
        main_container = ttk.Frame(dialog)  
        main_container.pack(fill=tk.BOTH, expand=True)  
        
        canvas = tk.Canvas(main_container, highlightthickness=0)  
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)  
        
        main_frame = ttk.Frame(canvas, padding=10)  
        main_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))  
        
        canvas.create_window((0, 0), window=main_frame, anchor="nw", width=680)  
        canvas.configure(yscrollcommand=scrollbar.set)  
        
        # Mouse wheel scrolling  
        def _on_mousewheel(event):  
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")  
        canvas.bind('<Enter>', lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))  
        canvas.bind('<Leave>', lambda e: canvas.unbind_all("<MouseWheel>"))  
        
        canvas.pack(side="left", fill="both", expand=True)  
        scrollbar.pack(side="right", fill="y")  
        
        # ===== 1. TRACK SELECTION =====  
        track_frame = ttk.LabelFrame(main_frame, text="1. Select Track", padding=10)  
        track_frame.pack(fill=tk.BOTH, expand=True, pady=5)  
        
        ttk.Label(track_frame, text="Available tracks:",   
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')).pack(anchor=tk.W)  
        
        list_frame = ttk.Frame(track_frame)  
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)  
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)  
        track_listbox = tk.Listbox(list_frame, height=6,   
                                   yscrollcommand=list_scrollbar.set,  
                                   font=(FONT_FAMILY, FONT_SIZE_SMALL))  
        list_scrollbar.config(command=track_listbox.yview)  
        
        track_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  
        
        # Populate track list  
        track_list = []  
        for channel in ['red', 'green', 'blue']:  
            for i, track in enumerate(self.tracks_dict[channel]):  
                duration = self.calculate_track_duration(track)  
                track_id = f"{channel.capitalize()} - Track {i+1} ({len(track.time_idx)} pts, {duration:.2f}s)"  
                track_listbox.insert(tk.END, track_id)  
                track_list.append((channel, i, track))  
        
        track_info_var = tk.StringVar(value="Select a track to see details")  
        ttk.Label(track_frame, textvariable=track_info_var,  
                 foreground='blue', font=(FONT_FAMILY, FONT_SIZE_SMALL)).pack(anchor=tk.W, pady=5)  
        
        # ===== 2. TIME RANGE SELECTION =====  
        time_frame = ttk.LabelFrame(main_frame, text="2. Time Range Selection", padding=10)  
        time_frame.pack(fill=tk.X, pady=5)  
        
        use_custom_time_var = tk.BooleanVar(value=False)  
        ttk.Checkbutton(time_frame, text="Use custom time range (otherwise use full track)",   
                       variable=use_custom_time_var).pack(anchor=tk.W, pady=3)  
        
        # Time inputs  
        time_input_frame = ttk.Frame(time_frame)  
        time_input_frame.pack(fill=tk.X, pady=5, padx=20)  
        
        ttk.Label(time_input_frame, text="Start time (s):",   
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)  
        start_time_var = tk.StringVar(value="0")  
        start_time_entry = ttk.Entry(time_input_frame, textvariable=start_time_var, width=15)  
        start_time_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)  
        
        ttk.Label(time_input_frame, text="End time (s):",   
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)  
        end_time_var = tk.StringVar(value="")  
        end_time_entry = ttk.Entry(time_input_frame, textvariable=end_time_var, width=15)  
        end_time_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)  
        
        # Visual selection button  
        visual_btn = ttk.Button(time_input_frame, text="🎯 Visual Selection",   
                               command=lambda: visual_time_selection())  
        visual_btn.grid(row=0, column=2, rowspan=2, padx=10, pady=3)  
        # 🆕 Click切割按钮
        click_cut_btn = ttk.Button(time_input_frame, text="✂️ Click to Cut",
                                   command=lambda: click_cut_segment())
        click_cut_btn.grid(row=0, column=3, rowspan=2, padx=10, pady=3)
        
        # Time info  
        time_info_var = tk.StringVar(value="")  
        ttk.Label(time_frame, textvariable=time_info_var,  
                 foreground='green', font=(FONT_FAMILY, FONT_SIZE_SMALL)).pack(anchor=tk.W, pady=3)  
        
        # Enable/disable controls  
        def toggle_time_entries(*args):  
            state = 'normal' if use_custom_time_var.get() else 'disabled'  
            start_time_entry.config(state=state)  
            end_time_entry.config(state=state)  
            visual_btn.config(state=state)  
        
        use_custom_time_var.trace('w', toggle_time_entries)  
        toggle_time_entries()  
        
        # ===== 3. ANALYSIS METHOD =====  
        method_frame = ttk.LabelFrame(main_frame, text="3. Analysis Method", padding=10)  
        method_frame.pack(fill=tk.X, pady=5)  
        
        analysis_mode_var = tk.StringVar(value="msd")  
        
        ttk.Radiobutton(method_frame, text="MSD Linear Fit (Classical, with R²)",   
                       variable=analysis_mode_var, value="msd").pack(anchor=tk.W, pady=2)  
        ttk.Radiobutton(method_frame, text="CVE Method (Covariance-based, corrects error)",   
                       variable=analysis_mode_var, value="cve").pack(anchor=tk.W, pady=2)  
        ttk.Radiobutton(method_frame, text="OLS Method (Simple & fast)",   
                       variable=analysis_mode_var, value="ols").pack(anchor=tk.W, pady=2)  
        ttk.Radiobutton(method_frame, text="Time Window Analysis (50s segments)",   
                       variable=analysis_mode_var, value="window").pack(anchor=tk.W, pady=2)  
        ttk.Radiobutton(method_frame, text="Multiple Tracks Histogram",   
                       variable=analysis_mode_var, value="histogram").pack(anchor=tk.W, pady=2)  
        
        # ===== 4. PARAMETERS =====  
        param_frame = ttk.LabelFrame(main_frame, text="4. Parameters", padding=10)  
        param_frame.pack(fill=tk.X, pady=5)  
        
        ttk.Label(param_frame, text="MSD fit points:",   
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)  
        max_lag_var = tk.IntVar(value=10)  
        ttk.Spinbox(param_frame, from_=5, to=50,   
                   textvariable=max_lag_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)  
        
        ttk.Label(param_frame, text="Localization error σ (nm):",   
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)  
        loc_error_var = tk.StringVar(value="30")  
        ttk.Entry(param_frame, textvariable=loc_error_var, width=20).grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)  
        
        ttk.Label(param_frame, text="Window duration (s):",   
                 font=(FONT_FAMILY, FONT_SIZE_NORMAL)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)  
        window_duration_var = tk.DoubleVar(value=50.0)  
        ttk.Entry(param_frame, textvariable=window_duration_var, width=20).grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)  
        
        # ===== 5. RESULTS =====  
        result_frame = ttk.LabelFrame(main_frame, text="5. Results", padding=10)  
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)  
        
        result_text = scrolledtext.ScrolledText(result_frame, height=20, width=80,  
                                               font=('Courier', 9))  
        result_text.pack(fill=tk.BOTH, expand=True)  
        
        # ===== HELPER FUNCTIONS =====  
        
        def calculate_msd_curve(positions, times, max_lag=10):  
            """Calculate MSD curve and fit diffusion coefficient"""  
            msd_values = []  
            lags_time = []  
            
            for lag in range(1, min(max_lag + 1, len(positions))):  
                displacements = positions[lag:] - positions[:-lag]  
                msd = np.mean(displacements**2)  
                msd_values.append(msd)  
                time_lags = times[lag:] - times[:-lag]  
                lags_time.append(np.mean(time_lags))  
            
            if len(lags_time) < 2:  
                return None, None, None, None, None  
            
            # Linear fit to first 10 points  
            fit_n = min(10, len(lags_time))  
            slope, intercept = np.polyfit(lags_time[:fit_n], msd_values[:fit_n], 1)  
            
            # D = slope / 2 for 1D diffusion  
            D = slope / 2  
            
            # Calculate R²  
            fit_values = slope * np.array(lags_time[:fit_n]) + intercept  
            residuals = msd_values[:fit_n] - fit_values  
            ss_res = np.sum(residuals**2)  
            ss_tot = np.sum((msd_values[:fit_n] - np.mean(msd_values[:fit_n]))**2)  
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0  
            
            return D, msd_values, lags_time, slope, r_squared  
        
        def calculate_D_cve(positions, times, sigma_nm=30):  
            """Covariance-based estimator (CVE)"""  
            displacements = np.diff(positions)  
            delta_t = np.mean(np.diff(times))  
            
            sigma_um = sigma_nm / 1000  
            var_dx = np.var(displacements, ddof=1)  
            D_cve = (var_dx - 2 * sigma_um**2) / (2 * delta_t)  
            
            se_D = np.sqrt(2 * var_dx**2 / len(displacements)) / (2 * delta_t)  
            
            return D_cve, se_D, var_dx, delta_t  
        
        def calculate_D_ols(positions, times):  
            """Ordinary Least Squares"""  
            displacements = np.diff(positions)  
            delta_t = np.mean(np.diff(times))  
            
            var_dx = np.var(displacements, ddof=1)  
            D_ols = var_dx / (2 * delta_t)  
            
            se_D = np.sqrt(2 * var_dx**2 / len(displacements)) / (2 * delta_t)  
            
            return D_ols, se_D, var_dx, delta_t  
        
        def calculate_D_per_window(track, window_duration=50.0, method='cve', sigma_nm=30):  
            """Calculate D for each non-overlapping time window"""  
            time_indices = np.array(track.time_idx)  
            pos_indices = np.array(track.coordinate_idx)  
            
            if self.roi_coords:  
                x1, y1, _, _ = self.roi_coords  
                time_indices = time_indices + x1  
                pos_indices = pos_indices + y1  
            
            times = time_indices * self.delta_line_time  
            positions = pos_indices * self.pixel_size_nm / 1000  
            
            start_time = times[0]  
            end_time = times[-1]  
            
            D_values = []  
            window_centers = []  
            
            current_time = start_time  
            while current_time + window_duration <= end_time:  
                mask = (times >= current_time) & (times < current_time + window_duration)  
                window_times = times[mask]  
                window_positions = positions[mask]  
                
                if len(window_positions) >= 10:  
                    if method == 'cve':  
                        D, _, _, _ = calculate_D_cve(window_positions, window_times, sigma_nm)  
                    elif method == 'ols':  
                        D, _, _, _ = calculate_D_ols(window_positions, window_times)  
                    else:  
                        D, _, _, _, _ = calculate_msd_curve(window_positions, window_times, max_lag=10)  
                    
                    if D is not None and D > 0:  
                        D_values.append(D)  
                        window_centers.append(current_time + window_duration/2)  
                
                current_time += window_duration  
            
            return D_values, window_centers  
        
        def update_track_info(event=None):  
            """Update track info"""  
            selection = track_listbox.curselection()  
            if not selection:  
                return  
            
            channel, track_idx, track = track_list[selection[0]]  
            
            time_indices = np.array(track.time_idx)  
            if self.roi_coords:  
                x1, y1, _, _ = self.roi_coords  
                time_indices = time_indices + x1  
            
            times = time_indices * self.delta_line_time  
            duration = times[-1] - times[0]  
            
            track_info_var.set(f"Duration: {duration:.3f} s, Points: {len(times)}, Δt: {np.mean(np.diff(times))*1000:.2f} ms")  
            
            # Auto-fill time range  
            if end_time_var.get() == "":  
                end_time_var.set(f"{times[-1]:.3f}")  
            if start_time_var.get() == "0":  
                start_time_var.set(f"{times[0]:.3f}")  
        
        track_listbox.bind('<<ListboxSelect>>', update_track_info)  
        
        def visual_time_selection():  
            """Visual interactive time selection with draggable boundaries"""  
            selection = track_listbox.curselection()  
            if not selection:  
                messagebox.showwarning("Warning", "Please select a track first")  
                return  
            
            channel, track_idx, track = track_list[selection[0]]  
            
            try:  
                start_t = float(start_time_var.get())  
                end_t_str = end_time_var.get().strip()  
                
                # Get track data  
                time_indices = np.array(track.time_idx)  
                pos_indices = np.array(track.coordinate_idx)  
                
                if self.roi_coords:  
                    x1, y1, _, _ = self.roi_coords  
                    time_indices = time_indices + x1  
                    pos_indices = pos_indices + y1  
                
                times = time_indices * self.delta_line_time  
                positions = pos_indices * self.pixel_size_nm / 1000  
                
                if end_t_str == "":  
                    end_t = times[-1]  
                else:  
                    end_t = float(end_t_str)  
                
                if start_t >= end_t:  
                    messagebox.showerror("Error", "Start time must be less than end time")  
                    return  
                
                # Create interactive plot  
                fig, ax = plt.subplots(figsize=(14, 6))  
                
                # Plot full track  
                ax.plot(times, positions, 'o-', color='lightgray',   
                       linewidth=2, markersize=4, alpha=0.5, label='Full track')  
                
                # Initial segment  
                mask = (times >= start_t) & (times <= end_t)  
                seg_line, = ax.plot(times[mask], positions[mask], 'o-', color='red',   
                                   linewidth=3, markersize=6, label='Selected segment')  
                
                # Draggable boundaries  
                start_line = ax.axvline(start_t, color='green', linestyle='--', linewidth=3,   
                                       label=f'Start: {start_t:.3f}s', picker=5)  
                end_line = ax.axvline(end_t, color='blue', linestyle='--', linewidth=3,   
                                     label=f'End: {end_t:.3f}s', picker=5)  
                
                # Info text  
                info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,  
                                   verticalalignment='top', fontsize=10,  
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))  
                
                dragging = {'line': None, 'start_val': start_t, 'end_val': end_t}  
                
                def update_segment():  
                    s = dragging['start_val']  
                    e = dragging['end_val']  
                    
                    mask = (times >= s) & (times <= e)  
                    seg_times = times[mask]  
                    seg_positions = positions[mask]  
                    
                    if len(seg_times) > 0:  
                        seg_line.set_data(seg_times, seg_positions)  
                        
                        duration = seg_times[-1] - seg_times[0] if len(seg_times) > 1 else 0  
                        displacement = seg_positions[-1] - seg_positions[0] if len(seg_positions) > 1 else 0  
                        
                        info_text.set_text(  
                            f'Segment: {len(seg_times)} points\n'  
                            f'Duration: {duration:.3f} s\n'  
                            f'Displacement: {displacement:.3f} μm\n'  
                            f'Range: [{s:.3f}, {e:.3f}] s'  
                        )  
                    else:  
                        seg_line.set_data([], [])  
                        info_text.set_text('No points in segment')  
                    
                    fig.canvas.draw_idle()  
                
                def on_pick(event):  
                    if event.artist == start_line:  
                        dragging['line'] = 'start'  
                    elif event.artist == end_line:  
                        dragging['line'] = 'end'  
                
                def on_motion(event):  
                    if dragging['line'] is None or event.xdata is None:  
                        return  
                    
                    if dragging['line'] == 'start':  
                        new_val = event.xdata  
                        if new_val < dragging['end_val'] and new_val >= times[0]:  
                            dragging['start_val'] = new_val  
                            start_line.set_xdata([new_val, new_val])  
                            start_line.set_label(f'Start: {new_val:.3f}s')  
                            update_segment()  
                            
                    elif dragging['line'] == 'end':  
                        new_val = event.xdata  
                        if new_val > dragging['start_val'] and new_val <= times[-1]:  
                            dragging['end_val'] = new_val  
                            end_line.set_xdata([new_val, new_val])  
                            end_line.set_label(f'End: {new_val:.3f}s')  
                            update_segment()  
                    
                    ax.legend(fontsize=10, loc='upper right')  
                
                def on_release(event):  
                    if dragging['line'] is not None:  
                        start_time_var.set(f"{dragging['start_val']:.3f}")  
                        end_time_var.set(f"{dragging['end_val']:.3f}")  
                        
                        s = dragging['start_val']  
                        e = dragging['end_val']  
                        mask = (times >= s) & (times <= e)  
                        seg_times = times[mask]  
                        seg_positions = positions[mask]  
                        
                        if len(seg_times) > 0:  
                            duration = seg_times[-1] - seg_times[0] if len(seg_times) > 1 else 0  
                            displacement = seg_positions[-1] - seg_positions[0] if len(seg_positions) > 1 else 0  
                            time_info_var.set(  
                                f"✓ Segment: {len(seg_times)} points, "  
                                f"duration: {duration:.3f} s, "  
                                f"displacement: {displacement:.3f} μm"  
                            )  
                        
                        dragging['line'] = None  
                
                fig.canvas.mpl_connect('pick_event', on_pick)  
                fig.canvas.mpl_connect('motion_notify_event', on_motion)  
                fig.canvas.mpl_connect('button_release_event', on_release)  
                
                update_segment()  
                
                ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')  
                ax.set_ylabel('Position (μm)', fontsize=12, fontweight='bold')  
                ax.set_title(  
                    f'Interactive Segment Selection - {channel.capitalize()} Track {track_idx+1}\n'  
                    f'💡 Drag green/blue lines to adjust time range',  
                    fontsize=14, fontweight='bold'  
                )  
                ax.legend(fontsize=10, loc='upper right')  
                ax.grid(True, alpha=0.3)  
                plt.tight_layout()  
                plt.show()  
                
            except ValueError:  
                messagebox.showerror("Error", "Invalid time values")  
            except Exception as e:  
                messagebox.showerror("Error", f"Preview failed:\n{str(e)}")
                
        
        # ========== ⭐⭐ NEW: Multi-Segment MSD Analysis  ==========
    
    def calculate_multi_segment_msd(self):
        """
        Calculate MSD for multiple time segments of ONE track
        🎯 OPTIMIZED: Drag-and-drop to add segments instantly
        """
        if not hasattr(self, 'tracks_dict') or not any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue']):
            messagebox.showwarning("Warning", "No tracks available for analysis")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("📈 Multi-Segment MSD Analysis")
        dialog.geometry("1400x950")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main layout
        main_paned = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # LEFT PANEL
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=0)
        
    
        # ===== TRACK SELECTION =====
        track_frame = ttk.LabelFrame(left_frame, text="📍 1. Select Track", padding=10)
        track_frame.pack(fill=tk.X, pady=5)
        
        track_listbox = tk.Listbox(track_frame, height=6, font=(FONT_FAMILY, FONT_SIZE_SMALL))
        track_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Populate tracks
        track_list = []
        for channel in ['red', 'green', 'blue']:
            for i, track in enumerate(self.tracks_dict[channel]):
                duration = self.calculate_track_duration(track)
                track_id = f"{channel.capitalize()} #{i+1} ({len(track.time_idx)}pts, {duration:.1f}s)"
                track_listbox.insert(tk.END, track_id)
                track_list.append((channel, i, track))
        
        # ===== QUICK ACTIONS =====
        quick_frame = ttk.LabelFrame(left_frame, text="⚡ 2. Quick Actions", padding=10)
        quick_frame.pack(fill=tk.X, pady=5)

         # RIGHT PANEL
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        # ===== 🆕 新增：手动输入按钮 =====
        manual_input_frame = ttk.LabelFrame(quick_frame, text="✏️ Manual Input", padding=5)
        manual_input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(manual_input_frame, text="Directly input time segments:",
                 foreground='blue', font=(FONT_FAMILY, FONT_SIZE_SMALL)).pack(anchor=tk.W, pady=2)

        def open_manual_input():
            """打开手动输入对话框"""
            if current_track_data['times'] is None:
                messagebox.showwarning("Warning", "Please load a track first")
                return
            
            # 创建对话框
            manual_dialog = ManualSegmentInputDialog(dialog, current_track_data['times'])
            dialog.wait_window(manual_dialog)
            
            # 检查是否有结果
            if manual_dialog.result:
                segments.clear()
                segments_listbox.delete(0, tk.END)
                
                # 添加所有分段
                for start, end, label, color in manual_dialog.result:
                    segments.append((start, end, label, color))
                    segments_listbox.insert(tk.END, f"✓ {label}: {start:.2f}-{end:.2f}s ({color})")
                
                update_plot()
                messagebox.showinfo("Success", 
                    f"✅ Loaded {len(segments)} segments from manual input")

        ttk.Button(manual_input_frame, text="✏️ Enter Segments Manually",
                  command=open_manual_input).pack(fill=tk.X, pady=2)
        
        # Auto-divide options
        divide_frame = ttk.Frame(quick_frame)
        divide_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(divide_frame, text="Divide into:").pack(side=tk.LEFT, padx=5)
        n_segments_var = tk.IntVar(value=2)
        ttk.Spinbox(divide_frame, from_=2, to=10, textvariable=n_segments_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(divide_frame, text="segments").pack(side=tk.LEFT, padx=2)
        
        # Segments list and state
        segments = []
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot state
        current_track_data = {'times': None, 'positions': None, 'channel': None, 'track_idx': None}
        drag_state = {'dragging': False, 'start_x': None, 'rect': None}
        
        def auto_divide():
            if current_track_data['times'] is None:
                messagebox.showwarning("Warning", "Please load a track first")
                return
            
            times = current_track_data['times']
            n_seg = n_segments_var.get()
            
            segments.clear()
            segments_listbox.delete(0, tk.END)
            
            duration = times[-1] - times[0]
            seg_duration = duration / n_seg
            
            for i in range(n_seg):
                start = times[0] + i * seg_duration
                end = times[0] + (i + 1) * seg_duration
                label = f"Seg{i+1}"
                color = color_palette[i % len(color_palette)]
                
                segments.append((start, end, label, color))
                segments_listbox.insert(tk.END, f"✓ {label}: {start:.2f}-{end:.2f}s")
            
            update_plot()
        
        ttk.Button(quick_frame, text="⚡ Auto Divide Track", 
                  command=auto_divide).pack(fill=tk.X, pady=2)
        
        # 🆕 Click切割功能
        def click_cut_trace():
            """通过点击轨迹来切割分段"""
            if current_track_data['times'] is None:
                messagebox.showwarning("Warning", "Please load a track first")
                return
            
            times = current_track_data['times']
            positions = current_track_data['positions']
            channel = current_track_data['channel']
            track_idx = current_track_data['track_idx']
            
            # 创建切割窗口
            cut_window = tk.Toplevel(dialog)
            cut_window.title("✂️ Click to Cut Segments")
            cut_window.geometry("1200x900")
            cut_window.transient(dialog)
            cut_window.grab_set()
            
            # 切割状态
            cut_state = {
                'cut_points': [],
                'cut_lines': []
            }
            
            # 🆕 先创建底部控制面板（这样它就不会被canvas覆盖）
            control_frame = ttk.Frame(cut_window)
            control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
            
            ttk.Label(control_frame, 
                     text="Instructions: LEFT CLICK = Add cut | RIGHT CLICK = Remove nearest cut",
                     font=(FONT_FAMILY, FONT_SIZE_SMALL), 
                     foreground='blue').pack(pady=5)
            
            btn_frame = ttk.Frame(control_frame)
            btn_frame.pack()
            
            ttk.Button(btn_frame, text="✅ Apply Cuts & Add to Segments",
                      command=lambda: apply_cuts()).pack(side=tk.LEFT, padx=5, ipady=5)
            ttk.Button(btn_frame, text="🗑️ Clear All Cuts",
                      command=lambda: [cut_state['cut_points'].clear(), 
                                      redraw_cut_plot()]).pack(side=tk.LEFT, padx=5, ipady=5)
            ttk.Button(btn_frame, text="❌ Close",
                      command=cut_window.destroy).pack(side=tk.LEFT, padx=5, ipady=5)
            
            # 然后创建绘图（填充剩余空间）
            fig_cut, ax_cut = plt.subplots(figsize=(14, 6))
            canvas_cut = FigureCanvasTkAgg(fig_cut, master=cut_window)
            canvas_cut.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            toolbar_frame = ttk.Frame(cut_window)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar_cut = NavigationToolbar2Tk(canvas_cut, toolbar_frame)
            
            def redraw_cut_plot():
                ax_cut.clear()
                
                # 绘制完整轨迹
                ax_cut.plot(times, positions, 'o-', color='gray',
                           linewidth=2, markersize=4, alpha=0.5,
                           label='Full trajectory')
                
                # 绘制切割线
                for i, cut_time in enumerate(sorted(cut_state['cut_points'])):
                    ax_cut.axvline(cut_time, color='red', linestyle='--',
                                  linewidth=2, alpha=0.8)
                    ax_cut.text(cut_time, ax_cut.get_ylim()[1] * 0.95,
                               f'Cut{i+1}', ha='center', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
                
                # 显示分段
                if len(cut_state['cut_points']) > 0:
                    sorted_cuts = sorted([times[0]] + cut_state['cut_points'] + [times[-1]])
                    colors_cycle = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
                    
                    for i in range(len(sorted_cuts) - 1):
                        seg_start = sorted_cuts[i]
                        seg_end = sorted_cuts[i + 1]
                        
                        mask = (times >= seg_start) & (times <= seg_end)
                        color = colors_cycle[i % len(colors_cycle)]
                        ax_cut.plot(times[mask], positions[mask], 'o',
                                   color=color, markersize=6, 
                                   label=f'Seg{i+1}')
                        
                        seg_mid = (seg_start + seg_end) / 2
                        ax_cut.text(seg_mid, ax_cut.get_ylim()[0] * 1.05,
                                   f'Seg{i+1}', ha='center', fontsize=9,
                                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
                
                ax_cut.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_cut.set_ylabel('Position (μm)', fontsize=12, fontweight='bold')
                ax_cut.set_title(
                    f'{channel.upper()} Track {track_idx+1} - Click to Cut\n'
                    f'Cuts: {len(cut_state["cut_points"])} | Segments: {len(cut_state["cut_points"]) + 1}',
                    fontsize=13, fontweight='bold')
                ax_cut.legend(loc='best', fontsize=9, ncol=2)
                ax_cut.grid(True, alpha=0.3)
                
                canvas_cut.draw()
            
            def on_cut_click(event):
                if event.inaxes != ax_cut:
                    return
                
                if toolbar_cut.mode != '':
                    return
                
                if event.button == 1:  # 左键添加切割点
                    cut_time = event.xdata
                    
                    if cut_time < times[0] or cut_time > times[-1]:
                        return
                    
                    cut_state['cut_points'].append(cut_time)
                    self.log(f"Added cut at t={cut_time:.3f}s")
                    redraw_cut_plot()
                
                elif event.button == 3:  # 右键删除最近的切割点
                    if cut_state['cut_points']:
                        distances = [abs(cp - event.xdata) for cp in cut_state['cut_points']]
                        nearest_idx = distances.index(min(distances))
                        removed = cut_state['cut_points'].pop(nearest_idx)
                        self.log(f"Removed cut at t={removed:.3f}s")
                        redraw_cut_plot()
            
            def apply_cuts():
                if len(cut_state['cut_points']) == 0:
                    messagebox.showwarning("Warning", "Please add at least one cut point")
                    return
                
                # 🆕 添加确认对话框
                sorted_cuts = sorted([times[0]] + cut_state['cut_points'] + [times[-1]])
                n_segments = len(sorted_cuts) - 1
                
                confirm_msg = (
                    f"Apply {len(cut_state['cut_points'])} cuts to create {n_segments} segments?\n\n"
                    f"Current segments in list: {len(segments)}\n"
                    f"After adding: {len(segments) + n_segments} segments\n\n"
                    f"Continue?"
                )
                
                if not messagebox.askyesno("Confirm Apply Cuts", confirm_msg):
                    return
                
                # 🆕 定义颜色列表
                colors_cycle = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
                
                for i in range(n_segments):
                    seg_start = sorted_cuts[i]
                    seg_end = sorted_cuts[i + 1]
                    
                    label = f"Seg{len(segments)+1}_Click"
                    color = colors_cycle[i % len(colors_cycle)]
                    
                    segments.append((seg_start, seg_end, label, color))
                    segments_listbox.insert(tk.END, 
                        f"{label}: [{seg_start:.3f} - {seg_end:.3f}]s ({color})")
                
                self.log(f"✓ Added {n_segments} segments from click cuts")
                update_plot()
                
                messagebox.showinfo("Success", f"✓ Added {n_segments} segments successfully!")
                cut_window.destroy()
            
            canvas_cut.mpl_connect('button_press_event', on_cut_click)
            
            
            redraw_cut_plot()
        
        ttk.Button(quick_frame, text="✂️ Click to Cut Trace", 
                  command=click_cut_trace).pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="🗑️ Clear All Segments", 
                  command=lambda: [segments.clear(), segments_listbox.delete(0, tk.END), update_plot()]
                  ).pack(fill=tk.X, pady=2)
        
        # ===== SEGMENTS LIST =====
        seg_frame = ttk.LabelFrame(left_frame, text="📊 3. Segments (Drag on Plot to Add)", padding=10)
        seg_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Instructions
        instruction_text = "🖱️ Left-drag: Add segment\n🗑️ Right-click: Delete selected"
        ttk.Label(seg_frame, text=instruction_text, foreground='blue', 
                 font=(FONT_FAMILY, 9), justify=tk.LEFT).pack(anchor=tk.W, pady=5)
        
        segments_listbox = tk.Listbox(seg_frame, height=12, font=('Courier', 9))
        segments_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Right-click to delete
        def delete_segment(event):
            selection = segments_listbox.curselection()
            if selection:
                idx = selection[0]
                segments.pop(idx)
                segments_listbox.delete(idx)
                update_plot()
        
        segments_listbox.bind('<Button-3>', delete_segment)
        segments_listbox.bind('<Delete>', delete_segment)
        
        # ===== MSD PARAMETERS =====
        param_frame = ttk.LabelFrame(left_frame, text="⚙️ 4. MSD Settings", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Max lag (s):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        max_lag_time_var = tk.DoubleVar(value=20.0)
        ttk.Spinbox(param_frame, from_=1.0, to=100.0, increment=5.0, 
                   textvariable=max_lag_time_var, width=10).grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=3)
        
        ttk.Label(param_frame, text="Fit range:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        fit_frame = ttk.Frame(param_frame)
        fit_frame.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=3)
        
        fit_percent_var = tk.IntVar(value=50)
        ttk.Spinbox(fit_frame, from_=20, to=100, increment=10,
                   textvariable=fit_percent_var, width=6).pack(side=tk.LEFT)
        ttk.Label(fit_frame, text="% of points", foreground='gray', 
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=5)
        
        # ===== PLOT CANVAS =====
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        
        fig, ax = plt.subplots(figsize=(11, 7))
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, right_frame)
        toolbar.update()
        
        def update_plot():
            """Redraw plot"""
            ax.clear()
            
            if current_track_data['times'] is None:
                ax.text(0.5, 0.5, 'Select a track to start →',
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=16, color='gray')
                canvas.draw()
                return
            
            times = current_track_data['times']
            positions = current_track_data['positions']
            channel = current_track_data['channel']
            track_idx = current_track_data['track_idx']
            
            # Full track
            ax.plot(times, positions, 'o-', color='lightgray', 
                   linewidth=1.5, markersize=3, alpha=0.5, label='Full track', zorder=1)
            
            # Segments
            for i, (start, end, label, color) in enumerate(segments):
                mask = (times >= start) & (times <= end)
                n_pts = np.sum(mask)
                if n_pts > 0:
                    ax.plot(times[mask], positions[mask], 'o-', 
                           color=color, linewidth=2.5, markersize=5,
                           label=f'{label} ({n_pts}pts)', zorder=2)
                    ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)
            
            ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Position (μm)', fontsize=13, fontweight='bold')
            ax.set_title(f'{channel.capitalize()} Track #{track_idx+1} - Drag to Add Segment',
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            canvas.draw()
        
        def load_track():
            """Load selected track"""
            selection = track_listbox.curselection()
            if not selection:
                return
            
            channel, track_idx, track = track_list[selection[0]]
            
            time_indices = np.array(track.time_idx)
            pos_indices = np.array(track.coordinate_idx)
            
            if self.roi_coords:
                x1, y1, _, _ = self.roi_coords
                time_indices = time_indices + x1
                pos_indices = pos_indices + y1
            
            times = time_indices * self.delta_line_time
            positions = pos_indices * self.pixel_size_nm / 1000
            
            current_track_data['times'] = times
            current_track_data['positions'] = positions
            current_track_data['channel'] = channel
            current_track_data['track_idx'] = track_idx
            
            segments.clear()
            segments_listbox.delete(0, tk.END)
            
            update_plot()
        
        track_listbox.bind('<<ListboxSelect>>', lambda e: load_track())
        
        # ===== MOUSE DRAG TO ADD SEGMENT =====
        def on_press(event):
            if event.inaxes != ax or current_track_data['times'] is None:
                return
            if toolbar.mode != '':
                return
            
            drag_state['dragging'] = True
            drag_state['start_x'] = event.xdata
            drag_state['rect'] = ax.axvspan(event.xdata, event.xdata, 
                                           alpha=0.4, color='gold', zorder=15)
            canvas.draw()
        
        def on_motion(event):
            if not drag_state['dragging'] or event.inaxes != ax:
                return
            
            if drag_state['rect']:
                drag_state['rect'].remove()
            
            x_start = drag_state['start_x']
            x_end = event.xdata
            drag_state['rect'] = ax.axvspan(min(x_start, x_end), max(x_start, x_end),
                                           alpha=0.4, color='gold', zorder=15)
            canvas.draw()
        
        def on_release(event):
            if not drag_state['dragging']:
                return
            
            drag_state['dragging'] = False
            
            if drag_state['rect']:
                drag_state['rect'].remove()
            
            if event.inaxes != ax or drag_state['start_x'] is None:
                canvas.draw()
                return
            
            start = min(drag_state['start_x'], event.xdata)
            end = max(drag_state['start_x'], event.xdata)
            
            if abs(end - start) < 0.5:
                canvas.draw()
                return
            
            label = f"Seg{len(segments)+1}"
            color = color_palette[len(segments) % len(color_palette)]
            
            segments.append((start, end, label, color))
            segments_listbox.insert(tk.END, f"✓ {label}: {start:.2f}-{end:.2f}s")
            
            update_plot()
        
        canvas.mpl_connect('button_press_event', on_press)
        canvas.mpl_connect('motion_notify_event', on_motion)
        canvas.mpl_connect('button_release_event', on_release)
        
        # ===== CALCULATE MSD =====
        # 容器用于保存结果数据
        results_container = {'data': None, 'segments_copy': None, 'times': None, 'positions': None}
        
        def calculate_msd_analysis():
            """Calculate MSD for all segments"""
            try:
                if not segments:
                    messagebox.showwarning("Warning", "Please add at least one segment")
                    return
                
                if current_track_data['times'] is None:
                    messagebox.showwarning("Warning", "No track loaded")
                    return
                
                max_lag_time = max_lag_time_var.get()
                fit_percent = fit_percent_var.get()
                times = current_track_data['times']
                positions = current_track_data['positions']
                
                # Create results window
                result_window = tk.Toplevel(dialog)
                result_window.title("Multi-Segment MSD Results")
                result_window.geometry("1100x900")
                
                # Create figure
                fig_result = Figure(figsize=(12, 10))
                gs = fig_result.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
                ax_msd = fig_result.add_subplot(gs[0:2, :])
                ax_log = fig_result.add_subplot(gs[2, 0])
                ax_residual = fig_result.add_subplot(gs[2, 1])
                
                results_data = []
                all_lag_arrays = []
                all_msd_arrays = []
                all_msd_std_arrays = []
                all_fit_slopes = []
                
                for start_t, end_t, label, color in segments:
                    mask = (times >= start_t) & (times <= end_t)

                    seg_times = times[mask]
                    seg_positions = positions[mask]
                    
                    if len(seg_positions) < 3:
                        messagebox.showwarning("Warning", f"{label}: Too few points ({len(seg_positions)})")
                        continue
                    
                    n_points = len(seg_positions)
                    dt = seg_times[1] - seg_times[0]
                    max_lag_points = int(max_lag_time / dt)
                    actual_max_lag = min(max_lag_points, n_points // 3, n_points - 1)
                    actual_max_lag = max(5, actual_max_lag)
                    
                    lag_times = []
                    msd_values = []
                    msd_std = []
                    
                    # 🆕 使用向量化MSD计算
                    msd_values_arr, lag_times_arr = MSDCalculator.calc_msd_vectorized(
                        seg_positions, seg_times, actual_max_lag
                    )
                    # ===== DNA单位转换 =====
                    BP_TO_NM = DNA_BP_TO_NM  # 1 bp = 0.34 nm
                    UM_TO_BP = DNA_UM_TO_BP  # 1 μm ≈ 2941.176 bp
                    
                    lag_times_arr = list(lag_times_arr)  # Convert to list if needed
                    msd_values = list(msd_values_arr)
                    lag_times = list(lag_times_arr)

                    # 计算标准差（如需要）
                    msd_std = []
                    for lag in range(1, actual_max_lag + 1):
                        displacements = seg_positions[lag:] - seg_positions[:-lag]
                        msd_std.append(np.std(displacements ** 2))
                    
                    if len(lag_times) < 2:
                        continue
                    
                    lag_times_arr = np.array(lag_times)
                    msd_values_arr = np.array(msd_values)
                    msd_std_arr = np.array(msd_std)
                    
                    n_fit = max(5, int(len(lag_times) * fit_percent / 100))
                    n_fit = min(n_fit, len(lag_times))
                    
                    slope, intercept = np.polyfit(lag_times_arr[:n_fit], msd_values_arr[:n_fit], 1)
                    
                    y_pred = slope * lag_times_arr[:n_fit] + intercept
                    ss_res = np.sum((msd_values_arr[:n_fit] - y_pred) ** 2)
                    ss_tot = np.sum((msd_values_arr[:n_fit] - np.mean(msd_values_arr[:n_fit])) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    D = slope / 2
                    D_bp2 = D * (UM_TO_BP ** 2)
        
                    
                    if np.all(lag_times_arr > 0) and np.all(msd_values_arr > 0):
                        log_t = np.log(lag_times_arr[:n_fit])
                        log_msd = np.log(msd_values_arr[:n_fit])
                        alpha, log_K = np.polyfit(log_t, log_msd, 1)
                    else:
                        alpha = np.nan
                    
                    # Plot MSD
                    ax_msd.errorbar(lag_times_arr, msd_values_arr, 
                                   yerr=msd_std_arr/np.sqrt(n_points),
                                   fmt='o', color=color, markersize=6, capsize=3,
                                   alpha=0.7, zorder=2)
                    
                    fit_line = slope * lag_times_arr + intercept
                    
                    ax_msd.plot(lag_times_arr, fit_line, '--', 
                               color=color, linewidth=2.5,
                               label=f'{label}: D={D:.4f} μm²/s ({D_bp2:.2e} bp²/s), α={alpha:.2f}, R²={r_squared:.3f}',
                               zorder=3)
                    
                    ax_msd.axvline(lag_times_arr[n_fit-1], color=color, 
                                  linestyle=':', alpha=0.4, linewidth=1.5)
                    
                    ax_log.loglog(lag_times_arr, msd_values_arr, 'o-', 
                                 color=color, markersize=4, alpha=0.7,
                                 label=f'{label} (α={alpha:.2f})')
                    
                    residuals = msd_values_arr[:n_fit] - y_pred
                    ax_residual.scatter(lag_times_arr[:n_fit], residuals, 
                                      color=color, alpha=0.7, s=40)
                    
                    D_bp2 = D * (UM_TO_BP ** 2)  # 将 D 从 μm²/s 转换为 bp²/s
                    results_data.append({
                        'Segment': label,
                        'Start (s)': start_t,
                        'End (s)': end_t,
                        'Duration (s)': end_t - start_t,
                        'N points': len(seg_positions),
                        'D_um2_s': D,
                        'D_bp2_s': D_bp2,
                        'R2': r_squared,
                        'R\u864f': r_squared,
                        'D (μm²/s)': D,
                        'D (bp²/s)': D_bp2,  # ✅ 新增bp²单位
                        'Alpha': alpha,
                        'R²': r_squared,
                        'Fit pts': n_fit
                    })
                    
                    # 保存数据供导出使用
                    all_lag_arrays.append(lag_times_arr)
                    all_msd_arrays.append(msd_values_arr)
                    all_msd_std_arrays.append(msd_std_arr)
                    all_fit_slopes.append((slope, intercept))
                
                ax_msd.set_xlabel('Time lag (s)', fontsize=13, fontweight='bold')
                ax_msd.set_ylabel('MSD (μm²)', fontsize=13, fontweight='bold')
                ax_msd.set_title('Multi-Segment MSD Analysis\n(Dotted line marks fit range)', 
                               fontsize=14, fontweight='bold')
                ax_msd.legend(fontsize=9, loc='best', framealpha=0.9)
                ax_msd.grid(True, alpha=0.3)
                
                ax_log.set_xlabel('Time lag (s)', fontsize=11)
                ax_log.set_ylabel('MSD (μm²)', fontsize=11)
                ax_log.set_title('Log-Log Plot', fontsize=12, fontweight='bold')
                ax_log.legend(fontsize=8)
                ax_log.grid(True, alpha=0.3, which='both')
                
                ax_residual.axhline(0, color='black', linestyle='--', linewidth=1.5)
                ax_residual.set_xlabel('Time lag (s)', fontsize=11)
                ax_residual.set_ylabel('Residuals (μm²)', fontsize=11)
                ax_residual.set_title('Fit Residuals', fontsize=12, fontweight='bold')
                ax_residual.grid(True, alpha=0.3)
                
                canvas_result = FigureCanvasTkAgg(fig_result, master=result_window)
                canvas_result.draw()
                canvas_result.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                toolbar_result = NavigationToolbar2Tk(canvas_result, result_window)
                toolbar_result.update()
                
                # ===== 显示结果表格 =====
                if results_data:
                    table_frame = ttk.Frame(result_window)
                    table_frame.pack(fill=tk.X, padx=10, pady=5)
                    
                    text_widget = tk.Text(table_frame, height=5, font=('Courier', 9))
                    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=text_widget.yview)
                    text_widget.config(yscrollcommand=scrollbar.set)
                    
                    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    
                    # 表头
                    header = f"{'Seg':<6} {'Start':<7} {'End':<7} {'Dur':<7} {'Npts':<6} {'D(μm²/s)':<11} {'Alpha':<7} {'R²':<6} {'FitPts':<7}\n"
                    header = f"{'Seg':<6} {'Start':<7} {'End':<7} {'Dur':<7} {'Npts':<6} {'D(um^2/s)':<11} {'D(bp^2/s)':<13} {'Alpha':<7} {'R^2':<6} {'FitPts':<7}\n"
                    text_widget.insert(tk.END, header)
                    text_widget.insert(tk.END, "=" * 118 + "\n")
                    
                    # 数据行
                    for row in results_data:
                        line = f"{row['Segment']:<6} {row['Start (s)']:<7.1f} {row['End (s)']:<7.1f} {row['Duration (s)']:<7.1f} {row['N points']:<6} {row['D (μm²/s)']:<11.5f} {row['Alpha']:<7.2f} {row['R²']:<6.3f} {row['Fit pts']:<7}\n"
                        line = f"{row['Segment']:<6} {row['Start (s)']:<7.1f} {row['End (s)']:<7.1f} {row['Duration (s)']:<7.1f} {row['N points']:<6} {row['D_um2_s']:<11.5f} {row['D_bp2_s']:<13.3e} {row['Alpha']:<7.2f} {row['R虏']:<6.3f} {row['Fit pts']:<7}\n"
                        line = f"{row['Segment']:<6} {row['Start (s)']:<7.1f} {row['End (s)']:<7.1f} {row['Duration (s)']:<7.1f} {row['N points']:<6} {row['D_um2_s']:<11.5f} {row['D_bp2_s']:<13.3e} {row['Alpha']:<7.2f} {row['R虏']:<6.3f} {row['Fit pts']:<7}\n"
                        line = f"{row['Segment']:<6} {row['Start (s)']:<7.1f} {row['End (s)']:<7.1f} {row['Duration (s)']:<7.1f} {row['N points']:<6} {row['D_um2_s']:<11.5f} {row['D_bp2_s']:<13.3e} {row['Alpha']:<7.2f} {row['R2']:<6.3f} {row['Fit pts']:<7}\n"
                        text_widget.insert(tk.END, line)
                    
                    text_widget.insert(tk.END, "\n" + "="*90 + "\n")
                    text_widget.insert(tk.END, "💡 α≈1: Normal | α<1: Subdiffusion | α>1: Superdiffusion\n")
                    
                    text_widget.config(state=tk.DISABLED)
                    
                    # 保存到容器
                    results_container['data'] = results_data
                    results_container['segments_copy'] = list(segments)
                    results_container['times'] = times
                    results_container['positions'] = positions
                    results_container['all_lag_arrays'] = all_lag_arrays
                    results_container['all_msd_arrays'] = all_msd_arrays
                    results_container['all_msd_std_arrays'] = all_msd_std_arrays
                    results_container['all_fit_slopes'] = all_fit_slopes
                    results_container['max_lag_time'] = max_lag_time
                    results_container['fit_percent'] = fit_percent
            
            except Exception as e:
                messagebox.showerror("Calculation Error", f"Failed:\n{str(e)}")
                import traceback
                traceback.print_exc()
        
        def export_results():
            """Export MSD results"""
            if results_container['data'] is None:
                messagebox.showwarning("Warning", "Please calculate MSD first")
                return
            
            try:
                if not self.current_file_path:
                    base_dir = Path.cwd()
                else:
                    base_dir = Path(self.current_file_path).parent
                
                # ✅ 直接使用base_dir，不要求用户选择
                folder = base_dir
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"MultiSegMSD_{current_track_data['channel']}_Track{current_track_data['track_idx']+1}_{timestamp}"
                      
        
                # ===== 🆕 统一Excel导出 =====
                excel_path = base_dir / f"{base_name}.xlsx"  # 
                
                import pandas as pd
                from openpyxl.styles import Font, PatternFill, Alignment
                
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    
                    # ===== SHEET 1: MSD_Summary =====
                    results_data = results_container['data']
                    summary_df = pd.DataFrame(results_data)
                    summary_df.to_excel(writer, sheet_name='MSD_Summary', index=False)
                    
                    # 格式化Summary Sheet
                    ws_summary = writer.sheets['MSD_Summary']
                    for column in ws_summary.columns:
                        ws_summary.column_dimensions[column[0].column_letter].width = 14
                    
                    # 表头加粗和着色
                    for cell in ws_summary[1]:
                        cell.font = Font(bold=True, color='FFFFFF', size=11)
                        cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
                        cell.alignment = Alignment(horizontal='center', vertical='center')

                    def pad_array(values, target_length):
                        values = np.asarray(values, dtype=float)
                        if len(values) >= target_length:
                            return values
                        return np.concatenate([values, np.full(target_length - len(values), np.nan)])
                    
                    # ===== SHEET 2: MSD_RawData =====
                    segments_copy = results_container['segments_copy']
                    all_lag_arrays = results_container['all_lag_arrays']
                    all_msd_arrays = results_container['all_msd_arrays']
                    all_msd_std_arrays = results_container['all_msd_std_arrays']
                    times = results_container['times']
                    
                    # 创建多个小表格，每个分段一个
                    raw_data_sheets = []
                    
                    for seg_idx, (start_t, end_t, label, _) in enumerate(segments_copy):
                        if seg_idx < len(all_msd_arrays):
                            seg_data = {
                                'Lag_Time_s': all_lag_arrays[seg_idx],
                                f'{label}_MSD_um2': all_msd_arrays[seg_idx],
                            }
                            # 计算标准误
                            if seg_idx < len(all_msd_std_arrays):
                                n_points_seg = len(times[(times >= start_t) & (times <= end_t)])
                                stderr = all_msd_std_arrays[seg_idx] / np.sqrt(max(1, n_points_seg))
                                seg_data[f'{label}_StdErr'] = stderr
                            
                            raw_data_sheets.append((label, seg_data))
                    
                    # 将所有分段的数据保存到一个Sheet中（分列显示）
                    if raw_data_sheets:
                        # 合并所有分段数据到同一个Sheet
                        raw_data_dict = {}
                        max_lag_length = max(len(seg_data['Lag_Time_s']) for _, seg_data in raw_data_sheets)
                        
                        
                        
                        # 使用最长的lag时间作为基准
                        
                        
                        # 添加所有分段的MSD数据
                        for label, seg_data in raw_data_sheets:
                            lag_times = seg_data['Lag_Time_s']
                            msd_values = seg_data[f'{label}_MSD_um2']
                            # ✅ 添加bp²列
                            stderr_values = seg_data.get(f'{label}_StdErr', np.full(len(msd_values), np.nan))
                            
                            # 将较短的数组补齐为NaN
                            raw_data_dict[f'{label}_Lag_Time_s'] = pad_array(lag_times, max_lag_length)
                            raw_data_dict[f'{label}_MSD_um2'] = pad_array(msd_values, max_lag_length)
                            raw_data_dict[f'{label}_StdErr_um2'] = pad_array(stderr_values, max_lag_length)
                        
                        raw_df = pd.DataFrame(raw_data_dict)
                        raw_df.to_excel(writer, sheet_name='MSD_RawData', index=False)
                    
                        ws_raw = writer.sheets['MSD_RawData']
                        for column in ws_raw.columns:
                            ws_raw.column_dimensions[column[0].column_letter].width = 14
                    else:
                        raw_df = pd.DataFrame()
                    
                    # ===== SHEET 3: MSD_FitLines =====
                    if all_lag_arrays and len(all_lag_arrays) > 0:
                        try:
                            all_fit_slopes = results_container['all_fit_slopes']
                            fit_data_dict = {}
                            fit_lengths = [
                                len(all_lag_arrays[idx])
                                for idx in range(min(len(all_lag_arrays), len(all_fit_slopes), len(segments_copy)))
                                if len(all_lag_arrays[idx]) > 0
                            ]
                            max_fit_length = max(fit_lengths) if fit_lengths else 0
                            
                            for seg_idx, (_, _, label, _) in enumerate(segments_copy):
                                if seg_idx < len(all_fit_slopes) and seg_idx < len(all_lag_arrays):
                                    slope, intercept = all_fit_slopes[seg_idx]
                                    lag_times = np.asarray(all_lag_arrays[seg_idx], dtype=float)
                                    fit_values = slope * lag_times + intercept
                                    fit_data_dict[f'{label}_Fit_Lag_Time_s'] = pad_array(lag_times, max_fit_length)
                                    fit_data_dict[f'{label}_Fit_um2'] = pad_array(fit_values, max_fit_length)
                            
                            fit_df = pd.DataFrame(fit_data_dict)
                            fit_df.to_excel(writer, sheet_name='MSD_FitLines', index=False)
                            
                            ws_fit = writer.sheets['MSD_FitLines']
                            for column in ws_fit.columns:
                                ws_fit.column_dimensions[column[0].column_letter].width = 14
                        except:
                            fit_df = pd.DataFrame()
                    else:
                        fit_df = pd.DataFrame()
                    
                    # ===== SHEET 4: MSD_OriginReady (μm²) =====
                    origin_data_dict = {}
                    origin_bp_data_dict = {}
                    origin_lengths = []
                    for arr in all_lag_arrays:
                        if len(arr) > 0:
                            origin_lengths.append(len(arr))
                    max_origin_length = max(origin_lengths) if origin_lengths else 0

                    if max_origin_length > 0:
                        all_fit_slopes = results_container['all_fit_slopes']
                        um2_to_bp2 = DNA_UM_TO_BP ** 2
                        for seg_idx, (start_t, end_t, label, _) in enumerate(segments_copy):
                            if seg_idx < len(all_lag_arrays):
                                lag_times = np.asarray(all_lag_arrays[seg_idx], dtype=float)
                                msd_values_um2 = np.asarray(all_msd_arrays[seg_idx], dtype=float)
                                if seg_idx < len(all_msd_std_arrays):
                                    n_points_seg = len(times[(times >= start_t) & (times <= end_t)])
                                    stderr_values_um2 = (
                                        np.asarray(all_msd_std_arrays[seg_idx], dtype=float)
                                        / np.sqrt(max(1, n_points_seg))
                                    )
                                else:
                                    stderr_values_um2 = np.full(len(msd_values_um2), np.nan)

                                msd_values_bp2 = msd_values_um2 * um2_to_bp2
                                stderr_values_bp2 = stderr_values_um2 * um2_to_bp2

                                origin_data_dict[f'{label}_X_TimeLag_s'] = pad_array(lag_times, max_origin_length)
                                origin_data_dict[f'{label}_Y_MSD_um2'] = pad_array(msd_values_um2, max_origin_length)
                                origin_data_dict[f'{label}_YErr_StdErr_um2'] = pad_array(stderr_values_um2, max_origin_length)

                                origin_bp_data_dict[f'{label}_X_TimeLag_s'] = pad_array(lag_times, max_origin_length)
                                origin_bp_data_dict[f'{label}_Y_MSD_bp2'] = pad_array(msd_values_bp2, max_origin_length)
                                origin_bp_data_dict[f'{label}_YErr_StdErr_bp2'] = pad_array(stderr_values_bp2, max_origin_length)

                                if seg_idx < len(all_fit_slopes):
                                    slope, intercept = all_fit_slopes[seg_idx]
                                    fit_values_um2 = slope * lag_times + intercept
                                    fit_values_bp2 = fit_values_um2 * um2_to_bp2
                                    origin_data_dict[f'{label}_Fit_X_TimeLag_s'] = pad_array(lag_times, max_origin_length)
                                    origin_data_dict[f'{label}_Fit_Y_MSD_um2'] = pad_array(fit_values_um2, max_origin_length)
                                    origin_bp_data_dict[f'{label}_Fit_X_TimeLag_s'] = pad_array(lag_times, max_origin_length)
                                    origin_bp_data_dict[f'{label}_Fit_Y_MSD_bp2'] = pad_array(fit_values_bp2, max_origin_length)

                        origin_df = pd.DataFrame(origin_data_dict)
                        origin_df.to_excel(writer, sheet_name='MSD_OriginReady', index=False)

                        ws_origin = writer.sheets['MSD_OriginReady']
                        for column in ws_origin.columns:
                            ws_origin.column_dimensions[column[0].column_letter].width = 18

                        origin_bp_df = pd.DataFrame(origin_bp_data_dict)
                        origin_bp_df.to_excel(writer, sheet_name='MSD_OriginReady_bp2', index=False)

                        ws_origin_bp = writer.sheets['MSD_OriginReady_bp2']
                        for column in ws_origin_bp.columns:
                            ws_origin_bp.column_dimensions[column[0].column_letter].width = 18
                    else:
                        origin_df = pd.DataFrame()
                        origin_bp_df = pd.DataFrame()

                    # ===== SHEET 6: TrackData =====
                    positions = results_container['positions']
                    
                    # 为每个分段添加标签列
                    segment_labels = [''] * len(times)
                    for start_t, end_t, label, _ in segments_copy:
                        mask = (times >= start_t) & (times <= end_t)
                        for idx in np.where(mask)[0]:
                            segment_labels[idx] = label
                    
                    track_data_dict = {
                        'Time_s': times,
                        'Position_um': positions,
                        'Segment': segment_labels
                    }
                    
                    track_df = pd.DataFrame(track_data_dict)
                    track_df.to_excel(writer, sheet_name='TrackData', index=False)
                    
                    ws_track = writer.sheets['TrackData']
                    for column in ws_track.columns:
                        ws_track.column_dimensions[column[0].column_letter].width = 14
                    
                    # ===== SHEET 7: Analysis_Info =====
                    info_data = {
                        'Parameter': [
                            'Channel',
                            'Track Index',
                            'Total Points',
                            'Total Duration (s)',
                            'Max Lag Time (s)',
                            'Fit Percent (%)',
                            'Number of Segments',
                            'Analysis Timestamp',
                            'Recommended Plot Sheet',
                            'Traceability Plot Sheet',
                            'Origin RawData Usage',
                            'Origin FitLines Usage',
                            'Origin One-Sheet Overlay'
                        ],
                        'Value': [
                            current_track_data.get('channel', 'N/A').capitalize(),
                            current_track_data.get('track_idx', 0) + 1,
                            len(times),
                            times[-1] - times[0] if len(times) > 1 else 0,
                            results_container['max_lag_time'],
                            results_container['fit_percent'],
                            len(segments_copy),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'MSD_OriginReady_bp2',
                            'MSD_OriginReady',
                            'Use each segment as its own X/Y/YErr column group: *_Lag_Time_s, *_MSD_um2, *_StdErr_um2',
                            'Overlay each *_Fit_Lag_Time_s with the matching *_Fit_um2 series',
                            'Use MSD_OriginReady for μm² traceability, or MSD_OriginReady_bp2 for direct bp²/s fitting in Origin'
                        ]
                    }
                    
                    info_df = pd.DataFrame(info_data)
                    info_df.to_excel(writer, sheet_name='Analysis_Info', index=False)
                    
                    ws_info = writer.sheets['Analysis_Info']
                    ws_info.column_dimensions['A'].width = 25
                    ws_info.column_dimensions['B'].width = 30
                    
                    # ===== SHEET 8: Save Figure =====
                    try:
                        figure_path = folder / f"{base_name}_MSD_Figure.png"
                        self.fig.savefig(str(figure_path), dpi=300, bbox_inches='tight')
                        self.log(f"✓ Figure saved: {figure_path.name}")
                    except Exception as e:
                        self.log(f"⚠ Warning: Could not save figure: {str(e)}")
                    
                    # ===== SHEET 9: Diffusion_Parameters =====
                    param_details = []
                    for seg_idx, (start_t, end_t, label, _) in enumerate(segments_copy):
                        if seg_idx < len(results_data):
                            result = results_data[seg_idx]
                            alpha = result.get('Alpha', 1.0)
                            
                            # 分类扩散类型
                            if np.isnan(alpha):
                                diff_type = 'Unknown'
                            elif abs(alpha - 1.0) < 0.1:
                                diff_type = 'Normal (α≈1)'
                            elif alpha < 0.9:
                                diff_type = 'Subdiffusion (α<1)'
                            else:
                                diff_type = 'Superdiffusion (α>1)'
                            
                            param_details.append({
                                'Segment': label,
                                'D (um^2/s)': result.get('D_um2_s', ''),
                                'D (bp^2/s)': result.get('D_bp2_s', ''),
                                'D (μm²/s)': result.get('D (μm²/s)', ''),
                                'Alpha': result.get('Alpha', ''),
                                'R² (Fit Quality)': result.get('R²', ''),
                                'Fit Points': result.get('Fit pts', ''),
                                'N Points': result.get('N points', ''),
                                'Duration (s)': result.get('Duration (s)', ''),
                                'Diffusion Type': diff_type
                            })
                    
                    if param_details:
                        detail_df = pd.DataFrame(param_details)
                        detail_df.to_excel(writer, sheet_name='Diffusion_Parameters', index=False)
                        
                        ws_detail = writer.sheets['Diffusion_Parameters']
                        for column in ws_detail.columns:
                            ws_detail.column_dimensions[column[0].column_letter].width = 14
                        
                        # 高亮Diffusion Type列
                        diffusion_type_col = None
                        for cell in ws_detail[1]:
                            if cell.value == 'Diffusion Type':
                                diffusion_type_col = cell.column_letter
                                break

                        if diffusion_type_col is None:
                            diffusion_type_col = 'H'

                        for row_idx, cell in enumerate(ws_detail[diffusion_type_col], start=2):
                            if 'Normal' in str(cell.value):
                                cell.fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
                            elif 'Subdiffusion' in str(cell.value):
                                cell.fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
                            elif 'Superdiffusion' in str(cell.value):
                                cell.fill = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
                
                # 显示导出成功信息
                messagebox.showinfo(
                    "Export Complete",
                    f"✅ MSD analysis exported successfully!\n\n"
                    f"File: {excel_path.name}\n"
                    f"Location: {base_dir}\n\n"  # ✅ 改成 base_dir
                    f"Sheets included:\n"
                    f"  • MSD_Summary - Segment statistics\n"
                    f"  • MSD_RawData - Raw MSD curves\n"
                    f"  • MSD_FitLines - Linear fit lines\n"
                    f"  • MSD_OriginReady - Origin plotting data in μm²\n"
                    f"  • MSD_OriginReady_bp2 - Origin plotting data in bp²\n"
                    f"  • TrackData - Full trajectory\n"
                    f"  • Diffusion_Parameters - Detailed analysis\n"
                    f"  • Analysis_Info - Metadata"
                )
                
                self.log(f"✓ MSD analysis exported: {excel_path.name}")
                

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed:\n{str(e)}")
                import traceback
                traceback.print_exc()
                
        # ===== BOTTOM BUTTONS =====
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10, padx=5)
        
        ttk.Button(button_frame, text="Calculate MSD", 
                  command=calculate_msd_analysis).pack(fill=tk.X, pady=3, ipady=8)
        
        export_btn = ttk.Button(button_frame, text="Export Results", 
                        state=tk.DISABLED, command=export_results)
        export_btn.pack(fill=tk.X, pady=3, ipady=8)
        
        # 在计算后启用导出按钮
        original_calculate = calculate_msd_analysis
        def calculate_with_export_enable():
            original_calculate()
            export_btn.config(state=tk.NORMAL)
        
        button_frame.winfo_children()[0].config(command=calculate_with_export_enable)
        
        # Initialize
        update_plot()
        
        # Center dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
    
    def calculate_multi_segment_rate(self):
        """
        Calculate rate (velocity) for multiple segments of ONE track
        Interactive: drag on plot to select time segments
        """
        if not hasattr(self, 'tracks_dict') or not any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue']):
            messagebox.showwarning("Warning", "No tracks available for analysis")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("📊 Multi-Segment Rate Analysis")
        dialog.geometry("1200x800")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main layout
        main_paned = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # LEFT PANEL
        left_frame = ttk.Frame(main_paned, width=350)
        main_paned.add(left_frame, weight=0)
        
        # RIGHT PANEL
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # ===== TRACK SELECTION =====
        track_frame = ttk.LabelFrame(left_frame, text="1. Select Track", padding=10)
        track_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        track_listbox = tk.Listbox(track_frame, height=8, font=(FONT_FAMILY, FONT_SIZE_SMALL))
        track_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        track_list = []
        for channel in ['red', 'green', 'blue']:
            for i, track in enumerate(self.tracks_dict[channel]):
                duration = self.calculate_track_duration(track)
                track_id = f"{channel.capitalize()} - Track {i+1} ({len(track.time_idx)} pts, {duration:.2f}s)"
                track_listbox.insert(tk.END, track_id)
                track_list.append((channel, i, track))
        
        # ===== SEGMENTS LIST =====
        segments_frame = ttk.LabelFrame(left_frame, text="2. Segments (Drag on Plot)", padding=10)
        segments_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(segments_frame, text="🖱️ Drag on the plot to select time range",
                 foreground='blue', font=(FONT_FAMILY, FONT_SIZE_SMALL)).pack(anchor=tk.W)
        
        segments = []
        
        segments_listbox = tk.Listbox(segments_frame, height=10, font=(FONT_FAMILY, FONT_SIZE_SMALL))
        segments_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        control_frame = ttk.Frame(segments_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Next Label:").grid(row=0, column=0, sticky=tk.W, padx=2)
        seg_label_var = tk.StringVar(value="Segment 1")
        ttk.Entry(control_frame, textvariable=seg_label_var, width=15).grid(row=0, column=1, padx=2)
        
        ttk.Label(control_frame, text="Color:").grid(row=1, column=0, sticky=tk.W, padx=2)
        color_var = tk.StringVar(value="blue")
        color_combo = ttk.Combobox(control_frame, textvariable=color_var, width=13,
                                   values=["blue", "red", "green", "orange", "purple", "cyan", "magenta", "brown"],
                                   state="readonly")
        color_combo.grid(row=1, column=1, padx=2)
        
        def remove_segment():
            selection = segments_listbox.curselection()
            if selection:
                idx = selection[0]
                segments.pop(idx)
                segments_listbox.delete(idx)
                update_plot()
        
        ttk.Button(segments_frame, text="➖ Remove Selected",
                  command=remove_segment).pack(fill=tk.X, pady=2)
        # 🆕 Click切割功能
        def click_cut_trace():
            """通过点击轨迹来切割分段"""
            if current_track_data['times'] is None:
                messagebox.showwarning("Warning", "Please load a track first")
                return
            
            times = current_track_data['times']
            positions = current_track_data['positions']
            channel = current_track_data['channel']
            track_idx = current_track_data['track_idx']
            
            # 创建切割窗口
            cut_window = tk.Toplevel(dialog)
            cut_window.title("✂️ Click to Cut Segments")
            cut_window.geometry("1200x900")
            cut_window.transient(dialog)
            cut_window.grab_set()
            
            # 切割状态
            cut_state = {
                'cut_points': [],
                'cut_lines': []
            }
            
            # 创建绘图
            fig_cut, ax_cut = plt.subplots(figsize=(14, 7))
            canvas_cut = FigureCanvasTkAgg(fig_cut, master=cut_window)
            canvas_cut.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar_frame = ttk.Frame(cut_window)
            toolbar_frame.pack(fill=tk.X)
            toolbar_cut = NavigationToolbar2Tk(canvas_cut, toolbar_frame)
            
            def redraw_cut_plot():
                ax_cut.clear()
                
                # 绘制完整轨迹
                ax_cut.plot(times, positions, 'o-', color='gray',
                           linewidth=2, markersize=4, alpha=0.5,
                           label='Full trajectory')
                
                # 绘制切割线
                for i, cut_time in enumerate(sorted(cut_state['cut_points'])):
                    ax_cut.axvline(cut_time, color='red', linestyle='--',
                                  linewidth=2, alpha=0.8)
                    ax_cut.text(cut_time, ax_cut.get_ylim()[1] * 0.95,
                               f'Cut{i+1}', ha='center', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
                
                # 显示分段
                if len(cut_state['cut_points']) > 0:
                    sorted_cuts = sorted([times[0]] + cut_state['cut_points'] + [times[-1]])
                    colors_cycle = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
                    
                    for i in range(len(sorted_cuts) - 1):
                        seg_start = sorted_cuts[i]
                        seg_end = sorted_cuts[i + 1]
                        
                        mask = (times >= seg_start) & (times <= seg_end)
                        color = colors_cycle[i % len(colors_cycle)]
                        ax_cut.plot(times[mask], positions[mask], 'o',
                                   color=color, markersize=6, 
                                   label=f'Seg{i+1}')
                        
                        seg_mid = (seg_start + seg_end) / 2
                        ax_cut.text(seg_mid, ax_cut.get_ylim()[0] * 1.05,
                                   f'Seg{i+1}', ha='center', fontsize=9,
                                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
                
                ax_cut.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_cut.set_ylabel('Position (μm)', fontsize=12, fontweight='bold')
                ax_cut.set_title(
                    f'{channel.upper()} Track {track_idx+1} - Click to Cut\n'
                    f'Cuts: {len(cut_state["cut_points"])} | Segments: {len(cut_state["cut_points"]) + 1}',
                    fontsize=13, fontweight='bold')
                ax_cut.legend(loc='best', fontsize=9, ncol=2)
                ax_cut.grid(True, alpha=0.3)
                
                canvas_cut.draw()
            
            def on_cut_click(event):
                if event.inaxes != ax_cut:
                    return
                
                if toolbar_cut.mode != '':
                    return
                
                if event.button == 1:  # 左键添加切割点
                    cut_time = event.xdata
                    
                    if cut_time < times[0] or cut_time > times[-1]:
                        return
                    
                    cut_state['cut_points'].append(cut_time)
                    self.log(f"Added cut at t={cut_time:.3f}s")
                    redraw_cut_plot()
                
                elif event.button == 3:  # 右键删除最近的切割点
                    if cut_state['cut_points']:
                        distances = [abs(cp - event.xdata) for cp in cut_state['cut_points']]
                        nearest_idx = distances.index(min(distances))
                        removed = cut_state['cut_points'].pop(nearest_idx)
                        self.log(f"Removed cut at t={removed:.3f}s")
                        redraw_cut_plot()
            
            def apply_cuts():
                if len(cut_state['cut_points']) == 0:
                    messagebox.showwarning("Warning", "Please add at least one cut point")
                    return
                
                sorted_cuts = sorted([times[0]] + cut_state['cut_points'] + [times[-1]])
                colors_cycle = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
                for i in range(len(sorted_cuts) - 1):
                    seg_start = sorted_cuts[i]
                    seg_end = sorted_cuts[i + 1]
                    
                    label = f"Seg{i+1}_Click"
                    color = colors_cycle[i % len(colors_cycle)]
                    
                    segments.append((seg_start, seg_end, label, color))
                    segments_listbox.insert(tk.END, f"✓ {label}: {seg_start:.2f}-{seg_end:.2f}s")
                
                self.log(f"✓ Added {len(sorted_cuts) - 1} segments from click cuts")
                update_plot()
                cut_window.destroy()
            
            canvas_cut.mpl_connect('button_press_event', on_cut_click)
            
            # 控制面板
            control_frame = ttk.Frame(cut_window)
            control_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(control_frame, 
                     text="LEFT CLICK = Add cut | RIGHT CLICK = Remove nearest cut",
                     font=(FONT_FAMILY, FONT_SIZE_SMALL), 
                     foreground='blue').pack(pady=5)
            
            btn_frame = ttk.Frame(control_frame)
            btn_frame.pack()
            
            ttk.Button(btn_frame, text="✂️ Apply Cuts",
                      command=apply_cuts).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Clear All",
                      command=lambda: [cut_state['cut_points'].clear(), 
                                      redraw_cut_plot()]).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Close",
                      command=cut_window.destroy).pack(side=tk.LEFT, padx=5)
            
            redraw_cut_plot()
        
        ttk.Button(segments_frame, text="✂️ Click to Cut Trace",
                  command=click_cut_trace).pack(fill=tk.X, pady=2)
        # ===== PLOT CANVAS =====
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        fig, ax = plt.subplots(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, right_frame)
        toolbar.update()
        
        current_track_data = {'times': None, 'positions': None, 'channel': None, 'track_idx': None}
        drag_state = {'dragging': False, 'start_x': None, 'rect': None}
        
        def update_plot():
            ax.clear()
            
            if current_track_data['times'] is None:
                ax.text(0.5, 0.5, 'Select a track to begin',
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                canvas.draw()
                return
            
            times = current_track_data['times']
            positions = current_track_data['positions']
            channel = current_track_data['channel']
            track_idx = current_track_data['track_idx']
            
            ax.plot(times, positions, 'o-', color='lightgray',
                   linewidth=2, markersize=4, alpha=0.6, label='Full track', zorder=1)
            
            for i, (start, end, label, color) in enumerate(segments):
                mask = (times >= start) & (times <= end)
                if np.sum(mask) > 1:
                    seg_times = times[mask]
                    seg_positions = positions[mask]
                    
                    displacement = seg_positions[-1] - seg_positions[0]
                    duration = seg_times[-1] - seg_times[0]
                    rate = displacement / duration if duration > 0 else 0
                    
                    ax.plot(seg_times, seg_positions, 'o-',
                           color=color, linewidth=3, markersize=6,
                           label=f'{label}: {rate:.3f} μm/s', zorder=2)
                    
                    ax.axvspan(start, end, alpha=0.2, color=color, zorder=0)
                    
                    fit_line = seg_positions[0] + rate * (seg_times - seg_times[0])
                    ax.plot(seg_times, fit_line, '--', color=color, linewidth=2, alpha=0.7, zorder=1)
            
            ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Position (μm)', fontsize=12, fontweight='bold')
            ax.set_title(f'{channel.capitalize()} Track {track_idx+1} - Drag to Select Segments',
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            canvas.draw()
        
        def load_track():
            selection = track_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a track")
                return
            
            channel, track_idx, track = track_list[selection[0]]
            
            time_indices = np.array(track.time_idx)
            pos_indices = np.array(track.coordinate_idx)
            
            if self.roi_coords:
                x1, y1, _, _ = self.roi_coords
                time_indices = time_indices + x1
                pos_indices = pos_indices + y1
            
            times = time_indices * self.delta_line_time
            positions = pos_indices * self.pixel_size_nm / 1000
            
            current_track_data['times'] = times
            current_track_data['positions'] = positions
            current_track_data['channel'] = channel
            current_track_data['track_idx'] = track_idx
            
            segments.clear()
            segments_listbox.delete(0, tk.END)
            
            update_plot()
        
        track_listbox.bind('<<ListboxSelect>>', lambda e: load_track())
        
        # ===== MOUSE INTERACTION =====
        def on_press(event):
            if event.inaxes != ax or current_track_data['times'] is None:
                return
            if toolbar.mode != '':
                return
            
            drag_state['dragging'] = True
            drag_state['start_x'] = event.xdata
            
            y_min, y_max = ax.get_ylim()
            drag_state['rect'] = ax.axvspan(event.xdata, event.xdata,
                                           alpha=0.3, color='yellow', zorder=10)
            canvas.draw()
        
        def on_motion(event):
            if not drag_state['dragging'] or event.inaxes != ax:
                return
            
            if drag_state['rect']:
                drag_state['rect'].remove()
            
            x_start = drag_state['start_x']
            x_end = event.xdata
            
            drag_state['rect'] = ax.axvspan(min(x_start, x_end), max(x_start, x_end),
                                           alpha=0.3, color='yellow', zorder=10)
            canvas.draw()
        
        def on_release(event):
            if not drag_state['dragging']:
                return
            
            drag_state['dragging'] = False
            
            if drag_state['rect']:
                drag_state['rect'].remove()
                drag_state['rect'] = None
            
            if event.inaxes != ax or drag_state['start_x'] is None:
                canvas.draw()
                return
            
            start = min(drag_state['start_x'], event.xdata)
            end = max(drag_state['start_x'], event.xdata)
            
            if abs(end - start) < 0.01:
                canvas.draw()
                return
            
            label = seg_label_var.get().strip() or f"Segment {len(segments)+1}"
            color = color_var.get()
            
            segments.append((start, end, label, color))
            segments_listbox.insert(tk.END, f"{label}: [{start:.3f} - {end:.3f}]s ({color})")
            
            seg_label_var.set(f"Segment {len(segments)+1}")
            
            update_plot()
        
        canvas.mpl_connect('button_press_event', on_press)
        canvas.mpl_connect('motion_notify_event', on_motion)
        canvas.mpl_connect('button_release_event', on_release)
        
        # ===== CALCULATE RATE =====
        def calculate_rate_analysis():
            """
            🆕 改进版：计算轨迹分段的速率
            包含三种计算方法和详细的结果输出
            """
            if current_track_data['times'] is None:
                messagebox.showwarning("Warning", "Please load a track first")
                return
            
            if len(segments) == 0:
                messagebox.showwarning("Warning", "Please add at least one segment (drag on plot)")
                return
            
            channel = current_track_data['channel']
            track_idx = current_track_data['track_idx']
            times = current_track_data['times']
            positions = current_track_data['positions']
            
            try:
                # ===== 🆕 DNA单位转换常数 =====
                BP_TO_NM = DNA_BP_TO_NM
                UM_TO_BP = DNA_UM_TO_BP
                
                # ===== 结果汇总容器 =====
                results_text = f"{'='*70}\n"
                results_text += f"🔬 Multi-Segment Rate Analysis (OPTIMIZED VERSION)\n"
                results_text += f"{'='*70}\n\n"
                results_text += f"📊 Track Information:\n"
                results_text += f"   Channel: {channel.capitalize()}\n"
                results_text += f"   Track Index: {track_idx + 1}\n"
                results_text += f"   Total Points: {len(positions)}\n"
                results_text += f"   Total Duration: {times[-1] - times[0]:.4f} s\n"
                results_text += f"   Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                all_segment_data = []
                total_displacement = 0
                total_duration = 0
                
                # ===== 分段计算 =====
                results_text += f"{'='*70}\n"
                results_text += f"📈 DETAILED SEGMENT ANALYSIS\n"
                results_text += f"{'='*70}\n\n"
                
                for seg_idx, (start, end, label, color) in enumerate(segments):
                    mask = (times >= start) & (times <= end)
                    seg_times = times[mask]
                    seg_positions = positions[mask]
                    
                    if len(seg_positions) < 2:
                        results_text += f"⚠️  {label}: INSUFFICIENT DATA ({len(seg_positions)} point)\n\n"
                        continue
                    
                    # ===== 方法1：简单法（端点法）=====
                    displacement = seg_positions[-1] - seg_positions[0]
                    duration = seg_times[-1] - seg_times[0]
                    simple_rate = displacement / duration if duration > 0 else 0
                    
                    # ===== 方法2：线性拟合法（推荐用于OriginLab）=====
                    # 使用numpy多项式拟合
                    try:
                        fit_coeffs = np.polyfit(seg_times, seg_positions, 1)
                        fit_rate = fit_coeffs[0]  # 斜率 = 速率
                        fit_intercept = fit_coeffs[1]
                        
                        # 计算拟合质量（R²）
                        y_pred = fit_rate * seg_times + fit_intercept
                        residuals = seg_positions - y_pred
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((seg_positions - np.mean(seg_positions)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        # 计算标准误
                        n_points = len(seg_positions)
                        if n_points > 2:
                            se = np.sqrt(ss_res / (n_points - 2))
                        else:
                            se = 0
                        
                        # 拟合后的位置值
                        y_fitted = y_pred
                        
                    except Exception as e:
                        self.log(f"Fit error in {label}: {str(e)}")
                        fit_rate = simple_rate
                        r_squared = 0
                        se = 0
                        y_fitted = seg_positions
                    
                    # ===== 数据转换（加上bp单位，取绝对值）=====
                    displacement_bp = abs(displacement) * UM_TO_BP  # 🆕 取绝对值
                    simple_rate_bp = abs(simple_rate) * UM_TO_BP    # 🆕 取绝对值
                    fit_rate_bp = abs(fit_rate) * UM_TO_BP          # 🆕 取绝对值
                    
                    # ===== 累积统计 =====
                    total_displacement += displacement
                    total_duration += duration
                    
                    # ===== 保存所有数据 =====
                    all_segment_data.append({
                        'label': label,
                        'start': start,
                        'end': end,
                        'duration': duration,
                        'n_points': len(seg_positions),
                        'displacement': displacement,
                        'displacement_bp': displacement_bp,
                        
                        # 简单法
                        'simple_rate': simple_rate,
                        'simple_rate_bp': simple_rate_bp,
                        
                        # 拟合法
                        'fit_rate': fit_rate,
                        'fit_rate_bp': fit_rate_bp,
                        'r_squared': r_squared,
                        'stderr': se,
                        
                        'color': color,
                        'times': seg_times,
                        'positions': seg_positions,
                        'y_fitted': y_fitted  # 用于绘图
                    })
                    
                    # ===== 详细输出 =====
                    results_text += f"[{seg_idx + 1}] {label}\n"
                    results_text += f"─" * 70 + "\n"
                    results_text += f"  ⏱️  Time Range: {start:.4f} - {end:.4f} s\n"
                    results_text += f"  📏 Duration: {duration:.4f} s\n"
                    results_text += f"  📊 Data Points: {len(seg_positions)}\n"
                    results_text += f"  🎯 Position Range: {seg_positions[0]:.4f} - {seg_positions[-1]:.4f} μm\n"
                    results_text += f"  ↔️  Displacement: {abs(displacement):.4f} μm ({abs(displacement_bp):.1f} bp)\n\n"
            
                    results_text += f"  📈 RATE CALCULATION METHODS:\n"
                    results_text += f"     └─ Method 1 - Simple (Endpoint):\n"
                    results_text += f"        • Rate: {abs(simple_rate):.6f} μm/s ({abs(simple_rate_bp):.2f} bp/s)\n"
                    results_text += f"        • Formula: |ΔPosition| / ΔTime\n\n"
                    
                    results_text += f"     └─ Method 2 - Linear Fit (RECOMMENDED for OriginLab):\n"
                    results_text += f"        • Rate: {abs(fit_rate):.6f} μm/s ({abs(fit_rate_bp):.2f} bp/s)\n"
                    results_text += f"        • R² (fit quality): {r_squared:.6f}  ← 1.0=perfect\n"
                    results_text += f"        • Std.Error: {se:.6f} μm  ← lower=more reliable\n"
                    results_text += f"        • Formula: Position = {fit_rate:.6f} × Time + {fit_intercept:.4f}\n\n"
                    
                    # ⭐ 对比说明
                    if abs(fit_rate - simple_rate) > abs(simple_rate) * 0.1:
                        results_text += f"     ⚠️  WARNING: Methods differ by >{10}%\n"
                        results_text += f"        (Linear fit may be more accurate)\n\n"
                    
                # ===== 汇总统计 =====
                results_text += f"\n{'='*70}\n"
                results_text += f"📊 SUMMARY STATISTICS\n"
                results_text += f"{'='*70}\n\n"
                
                if all_segment_data:
                    # 统计信息
                    results_text += f"  Segments Analyzed: {len(all_segment_data)}\n"
                    results_text += f"  Total Duration: {total_duration:.4f} s\n"
                    results_text += f"  Total Displacement: {total_displacement:.4f} μm ({total_displacement * UM_TO_BP:.1f} bp)\n\n"
                    
                    # 平均速率
                    avg_simple_rate = total_displacement / total_duration if total_duration > 0 else 0
                    avg_fit_rate = np.mean([seg['fit_rate'] for seg in all_segment_data])
                    avg_r2 = np.mean([seg['r_squared'] for seg in all_segment_data])
                    # ===== 添加归零基准点计算 =====
                    # 从所有segment数据中找到第一个时间点和位置点
                    all_times = []
                    all_positions = []
                    for seg in all_segment_data:
                        all_times.extend(seg['times'])
                        all_positions.extend(seg['positions'])

                    time_zero = min(all_times) if all_times else 0
                    position_zero = all_positions[all_times.index(time_zero)] if all_times else 0
                    
                    results_text += f"  Average Rate (Simple Method): {abs(avg_simple_rate):.6f} μm/s ({abs(avg_simple_rate) * UM_TO_BP:.2f} bp/s)\n"
                    results_text += f"  Average Rate (Fit Method): {abs(avg_fit_rate):.6f} μm/s ({abs(avg_fit_rate) * UM_TO_BP:.2f} bp/s)\n"
                    results_text += f"  Average R² Score: {avg_r2:.6f}\n\n"
                    
                    # 速率范围
                    fit_rates = [seg['fit_rate'] for seg in all_segment_data]
                    results_text += f"  Rate Range (Fit):\n"
                    results_text += f"    • Min: {abs(min(fit_rates, key=abs)):.6f} μm/s ({abs(min(fit_rates, key=abs)) * UM_TO_BP:.2f} bp/s)\n"
                    results_text += f"    • Max: {max([abs(r) for r in fit_rates]):.6f} μm/s ({max([abs(r) for r in fit_rates]) * UM_TO_BP:.2f} bp/s)\n"
                    results_text += f"    • Std: {np.std([abs(r) for r in fit_rates]):.6f} μm/s ({np.std([abs(r) for r in fit_rates]) * UM_TO_BP:.2f} bp/s)\n\n"
                
                results_text += f"{'='*70}\n"
                results_text += f"💾 DNA CONVERSION FACTOR\n"
                results_text += f"{'='*70}\n"
                results_text += f"  1 μm = {UM_TO_BP:.1f} bp (B-form DNA)\n"
                results_text += f"  1 μm/s = {UM_TO_BP:.1f} bp/s\n\n"
                
                results_text += f"{'='*70}\n"
                results_text += f"📁 DATA EXPORT\n"
                results_text += f"{'='*70}\n"
                results_text += f"  Three CSV files will be generated:\n"
                results_text += f"  1️⃣  RawData.csv - Original trajectory data\n"
                results_text += f"  2️⃣  FittedData.csv - Linear fit results\n"
                results_text += f"  3️⃣  Statistics.csv - Summary statistics\n"
                results_text += f"  ✅ Ready to import into OriginLab!\n\n"
                
                # ===== 创建图表 =====
                fig_rate = plt.figure(figsize=(16, 10))
                gs = fig_rate.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                
                ax1 = fig_rate.add_subplot(gs[0, 0])
                ax2 = fig_rate.add_subplot(gs[0, 1])
                ax3 = fig_rate.add_subplot(gs[1, :])
                
                # ===== 🆕 单位转换常数 =====
                UM_TO_BP = DNA_UM_TO_BP  # 1 μm ≈ 2941.176 bp
                
                # ===== 图1：简单速率柱状图 =====
                labels = [seg['label'] for seg in all_segment_data]
                simple_rates_um = [seg['simple_rate'] for seg in all_segment_data]
                simple_rates_bp = [abs(rate) * UM_TO_BP for rate in simple_rates_um]  # 🆕 取绝对值，转换为 bp/s
                colors_list = [seg['color'] for seg in all_segment_data]
                
                bars1 = ax1.bar(labels, simple_rates_bp, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
                avg_simple_bp = np.mean(simple_rates_bp) if simple_rates_bp else 0
                ax1.axhline(avg_simple_bp, color='red', linestyle='--', linewidth=2.5,
                           label=f'Avg: {avg_simple_bp:.2f} bp/s')
                ax1.set_ylabel('Rate (bp/s)', fontsize=12, fontweight='bold')  # 🆕 改为 bp/s
                ax1.set_xlabel('Segments', fontsize=12, fontweight='bold')
                ax1.set_title(f'Method 1: Simple Rate (Endpoint Method)\n{channel.capitalize()} Track {track_idx+1}',
                            fontsize=13, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3, axis='y')
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # ===== 图2：拟合速率柱状图 =====
                fit_rates_um = [seg['fit_rate'] for seg in all_segment_data]
                fit_rates_bp = [abs(rate) * UM_TO_BP for rate in fit_rates_um]  # 🆕 取绝对值，转换为 bp/s
                r2_values = [seg['r_squared'] for seg in all_segment_data]
                
                bars2 = ax2.bar(labels, fit_rates_bp, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
                avg_fit_bp = np.mean(fit_rates_bp) if fit_rates_bp else 0
                ax2.axhline(avg_fit_bp, color='red', linestyle='--', linewidth=2.5,
                           label=f'Avg: {avg_fit_bp:.2f} bp/s')
                
                # 在柱子上显示R²值
                for i, (bar, r2) in enumerate(zip(bars2, r2_values)):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'R²={r2:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax2.set_ylabel('Rate (bp/s)', fontsize=12, fontweight='bold')  # 🆕 改为 bp/s
                ax2.set_xlabel('Segments', fontsize=12, fontweight='bold')
                ax2.set_title(f'Method 2: Linear Fit Rate (RECOMMENDED)\n{channel.capitalize()} Track {track_idx+1}',
                            fontsize=13, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3, axis='y')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # ===== 图3：累积位移曲线（包含原始数据和拟合线）=====
                # 绘制原始轨迹
                ax3.plot(times, positions, 'o-', color='lightgray', linewidth=1.5, 
                        markersize=4, alpha=0.5, label='Full trajectory', zorder=1)
                
                # 为每个分段绘制原始数据和拟合线
                for seg in all_segment_data:
                    # 原始数据点
                    ax3.scatter(seg['times'], seg['positions'], color=seg['color'], 
                               s=60, alpha=0.7, edgecolor='black', linewidth=1, 
                               label=f"{seg['label']} (Raw)", zorder=3)
                    
                    # 拟合直线（速率显示为bp/s，取绝对值）
                    fit_rate_bp = abs(seg['fit_rate']) * UM_TO_BP  # 🆕 取绝对值，转换为 bp/s
                    ax3.plot(seg['times'], seg['y_fitted'], '--', color=seg['color'],
                            linewidth=2.5, alpha=0.9, 
                            label=f"{seg['label']} (Fit: {fit_rate_bp:.2f} bp/s)", zorder=2)
                
                ax3.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
                ax3.set_ylabel('Position (μm)', fontsize=13, fontweight='bold')
                ax3.set_title(f'Trajectory with Linear Fits - {channel.capitalize()} Track {track_idx+1}\n' +
                             f'Scatter=Raw Data | Dashed Lines=Linear Fits',
                            fontsize=14, fontweight='bold')
                ax3.legend(fontsize=9, loc='best', ncol=2, framealpha=0.95)
                ax3.grid(True, alpha=0.3, linestyle='--')
                
                # 使用subplots_adjust代替tight_layout以避免兼容性警告
                plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.96, hspace=0.35, wspace=0.3)
                        
                # ===== 创建结果显示对话框 =====
                export_dialog = tk.Toplevel(dialog)
                export_dialog.title("📊 Rate Analysis Results - OPTIMIZED")
                export_dialog.geometry("800x700")
                export_dialog.transient(dialog)
                export_dialog.grab_set()
                
                result_frame = ttk.Frame(export_dialog, padding=10)
                result_frame.pack(fill=tk.BOTH, expand=True)
                
                # 标题
                ttk.Label(result_frame, text="✅ Analysis Complete!",
                         font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'), 
                         foreground='green').pack(anchor=tk.W, pady=5)
                
                # 结果文本框
                text_frame = ttk.Frame(result_frame)
                text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
                
                text_scrollbar = ttk.Scrollbar(text_frame)
                text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                result_text_widget = tk.Text(text_frame, height=25, wrap=tk.WORD,
                                            font=('Courier', 8),
                                            yscrollcommand=text_scrollbar.set)
                result_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                text_scrollbar.config(command=result_text_widget.yview)
                
                result_text_widget.insert('1.0', results_text)
                result_text_widget.config(state='disabled')
                
                # 导出选项
                export_opt_frame = ttk.LabelFrame(result_frame, text="📁 Export Options", padding=10)
                export_opt_frame.pack(fill=tk.X, pady=10)
                
                export_plot_var = tk.BooleanVar(value=True)
                export_csv_var = tk.BooleanVar(value=True)
                
                ttk.Checkbutton(export_opt_frame, text="📈 Save Analysis Plot (PNG, 300dpi)",
                               variable=export_plot_var).pack(anchor=tk.W, pady=2)
                ttk.Checkbutton(export_opt_frame, text="📊 Export Data for OriginLab (Excel with 4 sheets)",
                               variable=export_csv_var).pack(anchor=tk.W, pady=2)
                
                # ===== 🆕 导出函数 =====
                def export_results():
                    """导出结果到文件"""
                    if not self.current_file_path:
                        messagebox.showerror("Error", "No file path available")
                        return
                    
                    base_name = Path(self.current_file_path).stem
                    output_dir = Path(self.current_file_path).parent
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    export_log = ""
                    
                    try:
                        # 导出PNG图表
                        if export_plot_var.get():
                            plot_path = output_dir / f"{base_name}_{channel}_Track{track_idx+1}_RateAnalysis_{timestamp}.png"
                            fig_rate.savefig(str(plot_path), dpi=300, bbox_inches='tight')
                            export_log += f"✅ Plot: {plot_path.name}\n"
                        
                        if export_csv_var.get():
                            # ===== Excel导出（带归零列） =====
                            from openpyxl import Workbook
                            from openpyxl.styles import Font, PatternFill, Alignment
                            from openpyxl.utils.dataframe import dataframe_to_rows
                            
                            excel_path = output_dir / f"{base_name}_{channel}_Track{track_idx+1}_Analysis_{timestamp}.xlsx"
                            
                            wb = Workbook()
                            wb.remove(wb.active)
                            
                            # ===== SHEET 1: RawData =====
                            raw_data = {
                                'Time_s': [],
                                'Time_Normalized_s': [],
                                'Position_um': [],
                                'Position_Normalized_um': [],
                                'Segment': []
                            }
                            
                            for seg in all_segment_data:
                                raw_data['Time_s'].extend(seg['times'])
                                raw_data['Time_Normalized_s'].extend(seg['times'] - time_zero)
                                raw_data['Position_um'].extend(seg['positions'])
                                raw_data['Position_Normalized_um'].extend(seg['positions'] - position_zero)
                                raw_data['Segment'].extend([seg['label']] * len(seg['times']))
                            
                            raw_df = pd.DataFrame(raw_data)
                            ws_raw = wb.create_sheet("RawData")
                            
                            for r_idx, row in enumerate(dataframe_to_rows(raw_df, index=False, header=True), 1):
                                for c_idx, value in enumerate(row, 1):
                                    cell = ws_raw.cell(row=r_idx, column=c_idx, value=value)
                                    if r_idx == 1:
                                        cell.font = Font(bold=True, color="FFFFFF")
                                        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            # ===== SHEET 2: FittedData =====
                            fitted_data = {
                                'Time_s': [],
                                'Time_Normalized_s': [],
                                'Position_um': [],
                                'Position_Normalized_um': [],
                                'Fitted_Position_um': [],
                                'Fitted_Position_Normalized_um': [],
                                'Segment': [],
                                'Fit_Rate_um_s': [],
                                'Fit_Rate_bp_s': [],
                                'R_squared': []
                            }
                            
                            for seg in all_segment_data:
                                fitted_data['Time_s'].extend(seg['times'])
                                fitted_data['Time_Normalized_s'].extend(seg['times'] - time_zero)
                                fitted_data['Position_um'].extend(seg['positions'])
                                fitted_data['Position_Normalized_um'].extend(seg['positions'] - position_zero)
                                fitted_data['Fitted_Position_um'].extend(seg['y_fitted'])
                                fitted_data['Fitted_Position_Normalized_um'].extend(seg['y_fitted'] - position_zero)
                                fitted_data['Segment'].extend([seg['label']] * len(seg['times']))
                                fitted_data['Fit_Rate_um_s'].extend([abs(seg['fit_rate'])] * len(seg['times']))
                                fitted_data['Fit_Rate_bp_s'].extend([abs(seg['fit_rate_bp'])] * len(seg['times']))
                                fitted_data['R_squared'].extend([seg['r_squared']] * len(seg['times']))
                            
                            fitted_df = pd.DataFrame(fitted_data)
                            ws_fitted = wb.create_sheet("FittedData")
                            
                            for r_idx, row in enumerate(dataframe_to_rows(fitted_df, index=False, header=True), 1):
                                for c_idx, value in enumerate(row, 1):
                                    cell = ws_fitted.cell(row=r_idx, column=c_idx, value=value)
                                    if r_idx == 1:
                                        cell.font = Font(bold=True, color="FFFFFF")
                                        cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            # ===== SHEET 3: Statistics =====
                            stats_rows = []
                            for seg in all_segment_data:
                                stats_rows.append({
                                    'Segment': seg['label'],
                                    'Start_s': seg['start'],
                                    'End_s': seg['end'],
                                    'Duration_s': seg['duration'],
                                    'N_Points': seg['n_points'],
                                    'Displacement_um': abs(seg['displacement']),
                                    'Displacement_bp': abs(seg['displacement_bp']),
                                    'Simple_Rate_um_s': abs(seg['simple_rate']),
                                    'Simple_Rate_bp_s': abs(seg['simple_rate_bp']),
                                    'Fit_Rate_um_s': abs(seg['fit_rate']),
                                    'Fit_Rate_bp_s': abs(seg['fit_rate_bp']),
                                    'R_squared': seg['r_squared'],
                                    'Std_Error_um': seg['stderr']
                                })
                            
                            stats_df = pd.DataFrame(stats_rows)
                            ws_stats = wb.create_sheet("Statistics")
                            
                            for r_idx, row in enumerate(dataframe_to_rows(stats_df, index=False, header=True), 1):
                                for c_idx, value in enumerate(row, 1):
                                    cell = ws_stats.cell(row=r_idx, column=c_idx, value=value)
                                    if r_idx == 1:
                                        cell.font = Font(bold=True, color="FFFFFF")
                                        cell.fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            # ===== SHEET 4: Summary =====
                            summary_data = {
                                'Parameter': [
                                    'Channel', 'Track Index', 'Total Points',
                                    'Total Duration (s)', 'Segments Analyzed',
                                    'Average Rate (μm/s)', 'Average Rate (bp/s)',
                                    'Average R²', 'Time Zero (s)', 'Position Zero (μm)',
                                    'Analysis Date'
                                ],
                                'Value': [
                                    channel.capitalize(), track_idx + 1, len(positions),
                                    times[-1] - times[0], len(all_segment_data),
                                    abs(avg_fit_rate), abs(avg_fit_rate) * UM_TO_BP,
                                    avg_r2, time_zero, position_zero,
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                ]
                            }
                            
                            summary_df = pd.DataFrame(summary_data)
                            ws_summary = wb.create_sheet("Summary", 0)
                            
                            for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
                                for c_idx, value in enumerate(row, 1):
                                    cell = ws_summary.cell(row=r_idx, column=c_idx, value=value)
                                    if r_idx == 1:
                                        cell.font = Font(bold=True, color="FFFFFF")
                                        cell.fill = PatternFill(start_color="203864", end_color="203864", fill_type="solid")
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            wb.save(str(excel_path))
                            export_log += f"✅ Excel: {excel_path.name}\n"
                        
                        # 显示成功消息
                        success_msg = (
                            f"✅ Export Successful!\n\n"
                            f"📁 Location: {output_dir}\n\n"
                            f"📊 Excel File (4 Sheets):\n{export_log}\n"
                            f"🔄 归零数据说明:\n"
                            f"  • Time_Normalized_s = Time_s - {time_zero:.4f}\n"
                            f"  • Position_Normalized_um = Position_um - {position_zero:.4f}\n\n"
                            f"💡 OriginLab Usage:\n"
                            f"  1. Open RawData sheet (X=Time_Normalized_s, Y=Position_Normalized_um)\n"
                            f"  2. Add FittedData sheet as overlay\n"
                            f"  3. Use Statistics sheet for bar charts"
                        )
                        
                        messagebox.showinfo("Export Complete", success_msg)
                        self.log(export_log)
                        
                        export_dialog.destroy()
                        
                    except Exception as e:
                        messagebox.showerror("Export Error", f"Failed:\n{str(e)}")
                        self.log(f"❌ Export failed: {str(e)}")
                
                # 按钮
                btn_frame = ttk.Frame(result_frame)
                btn_frame.pack(fill=tk.X, pady=10)
                
                ttk.Button(btn_frame, text="💾 Confirm & Export",
                          command=export_results).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X, ipady=8)
                ttk.Button(btn_frame, text="📊 Show Plot",
                          command=lambda: plt.show()).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X, ipady=8)
                ttk.Button(btn_frame, text="❌ Cancel",
                          command=export_dialog.destroy).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X, ipady=8)
                
                # 居中对话框
                export_dialog.update_idletasks()
                x = dialog.winfo_x() + (dialog.winfo_width() - export_dialog.winfo_width()) // 2
                y = dialog.winfo_y() + (dialog.winfo_height() - export_dialog.winfo_height()) // 2
                export_dialog.geometry(f"+{x}+{y}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Calculation failed:\n{str(e)}")
                import traceback
                traceback.print_exc()
        
        update_plot()
        
        # BOTTOM BUTTONS
        bottom_frame = ttk.Frame(left_frame)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(bottom_frame, text="📊 Calculate Rates",
                  command=calculate_rate_analysis).pack(fill=tk.X, pady=2, ipady=8)
        ttk.Button(bottom_frame, text="Close",
                  command=dialog.destroy).pack(fill=tk.X, pady=2)
        
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    # ========== Export Methods ==========   
    def export_rate_analysis_to_originlab(self, all_segment_data, channel, track_idx, 
                                          all_times, all_positions):
        """
        🆕 导出优化的数据格式供 OriginLab 使用
        
        输出三个文件：
        1. RawData.csv - 原始轨迹点 （用于散点图）
        2. FittedData.csv - 拟合直线 （用于对比）
        3. Statistics.csv - 统计表格 （用于柱状图）
        
        这个函数是独立的，可以在其他地方调用
        """
        
        if not self.current_file_path:
            messagebox.showerror("Error", "No file path available")
            return
        
        base_name = Path(self.current_file_path).stem
        output_dir = Path(self.current_file_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # ===== FILE 1: RAW DATA (原始轨迹) =====
            # 用途：在 OriginLab 中创建散点图
            
            raw_data = {
                'Time_s': [],  # X 轴
                'Position_um': [],  # Y 轴
                'Segment': [],  # 分段标签（用不同颜色）
            }
            
            for seg in all_segment_data:
                raw_data['Time_s'].extend(seg['times'])
                raw_data['Position_um'].extend(seg['positions'])
                raw_data['Segment'].extend([seg['label']] * len(seg['times']))
            
            raw_df = pd.DataFrame(raw_data)
            raw_path = output_dir / f"{base_name}_{channel}_Track{track_idx+1}_RawData_{timestamp}.csv"
            raw_df.to_csv(raw_path, index=False, encoding='utf-8')
            
            self.log(f"✓ RawData exported: {raw_path.name}")
            
            
            # ===== FILE 2: FITTED DATA (拟合直线) =====
            # 用途：在 OriginLab 中叠加到散点图上，对比效果
            
            fitted_data = {
                'Time_s': [],
                'Fitted_Position_um': [],  # Y 轴（拟合值）
                'Segment': [],
                'Fit_Rate_um_s': [],  # 斜率=速率
                'R_squared': [],  # 拟合质量
            }
            
            for seg in all_segment_data:
                fitted_data['Time_s'].extend(seg['times'])
                fitted_data['Fitted_Position_um'].extend(seg['y_fitted'])  # ✅ 使用拟合值
                fitted_data['Segment'].extend([seg['label']] * len(seg['times']))
                fitted_data['Fit_Rate_um_s'].extend([seg['fit_rate']] * len(seg['times']))
                fitted_data['R_squared'].extend([seg['r_squared']] * len(seg['times']))
            
            fitted_df = pd.DataFrame(fitted_data)
            fitted_path = output_dir / f"{base_name}_{channel}_Track{track_idx+1}_FittedData_{timestamp}.csv"
            fitted_df.to_csv(fitted_path, index=False, encoding='utf-8')
            
            self.log(f"✓ FittedData exported: {fitted_path.name}")
            
            
            # ===== FILE 3: STATISTICS (统计表格) =====
            # 用途：在 OriginLab 中制作柱状图
            
            stats_rows = []
            
            for seg in all_segment_data:
                stats_rows.append({
                    'Segment': seg['label'],
                    'Start_s': seg['start'],
                    'End_s': seg['end'],
                    'Duration_s': seg['duration'],
                    'N_Points': seg['n_points'],
                    'Displacement_um': seg['displacement'],
                    'Displacement_bp': seg['displacement_bp'],
                    'Simple_Rate_um_s': seg['simple_rate'],
                    'Simple_Rate_bp_s': seg['simple_rate_bp'],
                    'Fit_Rate_um_s': seg['fit_rate'],  # ⭐ 主要指标
                    'Fit_Rate_bp_s': seg['fit_rate_bp'],
                    'R_squared': seg['r_squared'],      # ⭐ 质量指标
                    'Std_Error_um': seg['stderr']      # ⭐ 误差条
                })
            
            stats_df = pd.DataFrame(stats_rows)
            stats_path = output_dir / f"{base_name}_{channel}_Track{track_idx+1}_Statistics_{timestamp}.csv"
            stats_df.to_csv(stats_path, index=False, encoding='utf-8')
            
            self.log(f"✓ Statistics exported: {stats_path.name}")
            
            
            # 显示成功消息
            msg = (
                f"✅ 数据已导出！可用于 OriginLab 作图\n\n"
                f"📊 文件清单：\n"
                f"1️⃣ {raw_path.name}\n"
                f"   → 用于创建散点图 (X=Time_s, Y=Position_um)\n\n"
                f"2️⃣ {fitted_path.name}\n"
                f"   → 用于叠加拟合直线 (线图模式)\n\n"
                f"3️⃣ {stats_path.name}\n"
                f"   → 用于制作柱状图 (X=Segment, Y=Fit_Rate_um_s)\n"
                f"     误差条使用 Std_Error_um\n\n"
                f"📁 位置: {output_dir}\n\n"
                f"💡 OriginLab 操作步骤：\n"
                f"1. 在 OriginLab 中导入这三个 CSV 文件\n"
                f"2. 使用 RawData 制作散点图\n"
                f"3. 将 FittedData 添加为线图层\n"
                f"4. 用 Statistics 制作对比柱状图"
            )
            
            messagebox.showinfo("Export Complete", msg)
            
            self.log("✅ All files exported successfully!")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")
            self.log(f"❌ Export failed: {str(e)}")
    def export_results(self):  
        """Export tracks from all channels"""  
        # Check if any tracks exist  
        has_tracks = any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue'])  
        
        if not has_tracks:  
            messagebox.showwarning("Warning", "No tracks to export")  
            return  
        
        # Show export dialog  
        dialog = ExportDialog(self.root)  
        self.root.wait_window(dialog)  
        
        if dialog.result is None:  
            return  
        
        options = dialog.result  
        export_format = options['format']  
        file_format = options['file_format']  
        
        # Ask for file path  
        if file_format == 'excel':  
            file_path = filedialog.asksaveasfilename(  
                title="Save as Excel file",  
                defaultextension=".xlsx",  
                filetypes=[("Excel files", "*.xlsx")]  
            )  
        else:  
            file_path = filedialog.asksaveasfilename(  
                title="Save CSV files (base name)",  
                defaultextension=".csv",  
                filetypes=[("CSV files", "*.csv")]  
            )  
        
        if not file_path:  
            return  
        
        try:  
            self.log("=" * 50)  
            self.log("Exporting data...")  
            self.log(f"File format: {file_format.upper()}")  
            self.log(f"Data organization: {export_format}")  
            
            # For Excel, create writer once  
            if file_format == 'excel':  
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:  
                    if export_format == 'long_merged':  
                        self.export_all_channels_merged_excel(writer, options)  
                    else:  
                        # Export each channel that has tracks  
                        for channel in ['red', 'green', 'blue']:  
                            if len(self.tracks_dict[channel]) > 0:  
                                self.log(f"Exporting {channel.upper()} channel tracks...")  
                                
                                if export_format == 'standard':  
                                    self.export_channel_standard_excel(writer, channel, options)  
                                elif export_format == 'wide':  
                                    self.export_channel_wide_excel(writer, channel, options)  
                                elif export_format == 'long':  
                                    self.export_channel_long_excel(writer, channel, options)  
                                elif export_format == 'summary':  
                                    self.export_channel_summary_excel(writer, channel, options)  
                    
                    # Add metadata and settings sheets  
                    if options['include_metadata']:  
                        self.export_metadata_sheet(writer)  
                    if options['include_settings']:  
                        self.export_settings_sheet(writer)  
            else:  
                # CSV format  
                base_path = Path(file_path)  
                
                if export_format == 'long_merged':  
                    self.export_all_channels_merged_csv(base_path, options)  
                else:  
                    for channel in ['red', 'green', 'blue']:  
                        if len(self.tracks_dict[channel]) > 0:  
                            self.log(f"Exporting {channel.upper()} channel tracks...")  
                            
                            if export_format == 'standard':  
                                self.export_channel_standard_csv(base_path, channel, options)  
                            elif export_format == 'wide':  
                                self.export_channel_wide_csv(base_path, channel, options)  
                            elif export_format == 'long':  
                                self.export_channel_long_csv(base_path, channel, options)  
                            elif export_format == 'summary':  
                                self.export_channel_summary_csv(base_path, channel, options)  
                
                # Export metadata and settings as CSV  
                if options['include_settings']:  
                    settings_path = base_path.parent / f"{base_path.stem}_KymotrackerSettings.csv"  
                    self.export_settings_csv(settings_path)  
                if options['include_metadata']:  
                    metadata_path = base_path.parent / f"{base_path.stem}_metadata.csv"  
                    self.export_metadata_csv(metadata_path)  
            
            self.log("=" * 50)  
            self.log(f"✓ Export complete: {Path(file_path).name}")  
            
            messagebox.showinfo("Export Successful",   
                              f"Data exported successfully!\n\n"  
                              f"File: {Path(file_path).name}\n"  
                              f"Format: {file_format.upper()}\n"  
                              f"Location: {Path(file_path).parent}")  
            
        except Exception as e:  
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")  
            self.log(f"✗ Export error: {str(e)}")  
            import traceback  
            self.log(traceback.format_exc())  
    
    def prepare_track_data(self, track, track_num, channel, options):  
        """Prepare data for a single track (units: s and μm)"""  
        data = {}
        profile = self.get_track_fluorescence_profile(
            track,
            include_intensity=options.get('include_intensity', True),
            channel=channel
        )
        time_indices = profile['time_indices']
        position_indices = profile['position_indices']
        times = profile['times']
        positions = profile['positions']
        n_points = len(time_indices)

        if n_points == 0:
            return data

        data['Track_ID'] = [track_num] * n_points
        data['Channel'] = [channel.capitalize()] * n_points
        data['Point_ID'] = list(range(n_points))
        data['Relative Time (s)'] = profile['relative_times']
        data['Lifetime (frames)'] = [profile['lifetime_frames']] * n_points
        data['Lifetime (s)'] = [profile['lifetime_seconds']] * n_points
        
        # Time data  
        if options['include_time']:  
            data['Time (s)'] = times  
        
        # Position data (convert nm to μm)  
        if options['include_position']:  
            data['Position (μm)'] = positions  
        
        # Intensity data  
        if options['include_intensity']:  
            data['Summed Photon Counts'] = profile['intensities']
            if profile['intensity_error']:
                self.log(f"Warning: Could not fully extract intensity for {channel} track {track_num}: {profile['intensity_error']}")
        
        # Velocity data (convert nm/s to μm/s)  
        if options['include_velocity'] and len(time_indices) >= 2:  
            delta_t = np.diff(times)
            delta_x = np.diff(positions)
            velocities = np.full(len(delta_t), np.nan, dtype=float)
            valid = np.abs(delta_t) > np.finfo(float).eps
            velocities[valid] = delta_x[valid] / delta_t[valid]
            data['Velocity (μm/s)'] = np.append(velocities, np.nan)  
        
        # Raw indices  
        if options['include_indices']:  
            data['Time Index'] = time_indices  
            data['Position Index'] = position_indices  
            data['Relative Frame Index'] = time_indices - time_indices[0]
        
        return data  

    def pad_export_values(self, values, target_length):
        """Pad export columns while supporting both numeric and text data."""
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.number):
            padded = np.full(target_length, np.nan, dtype=float)
        else:
            padded = np.full(target_length, None, dtype=object)
        padded[:len(array)] = array
        return padded  
    
    # ========== Merged Export Methods ==========  
    
    def export_all_channels_merged_excel(self, writer, options):  
        """Export ALL channels merged into ONE sheet in long format (Excel)"""  
        all_data = []  
        
        self.log("Merging all channels into single sheet...")  
        
        # Collect data from all channels  
        for channel in ['red', 'green', 'blue']:  
            tracks = self.tracks_dict[channel]  
            if len(tracks) == 0:  
                continue  
            
            self.log(f"  - Processing {channel.upper()} channel ({len(tracks)} tracks)")  
            
            for i, track in enumerate(tracks):  
                track_num = i + 1  
                track_data = self.prepare_track_data(track, track_num, channel, options)  
                
                if track_data:  
                    n_points = len(next(iter(track_data.values())))  
                    
                    # Add identifying columns  
                    track_data['Track_Label'] = [f"{channel.capitalize()}_{track_num}"] * n_points  
                    
                    all_data.append(pd.DataFrame(track_data))  
        
        if all_data:  
            # Concatenate all tracks  
            df = pd.concat(all_data, ignore_index=True)  
            
            # Reorder columns: identifying columns first  
            cols = ['Track_Label', 'Track_ID', 'Channel', 'Point_ID'] + \
                   [c for c in df.columns if c not in ['Track_Label', 'Track_ID', 'Channel', 'Point_ID']]  
            df = df[cols]  
            
            sheet_name = 'All_Tracks_Merged'  
            df.to_excel(writer, sheet_name=sheet_name, index=False)  
            self.log(f"✓ Exported {sheet_name} sheet with {len(df)} total points")  
            
            # Log summary  
            for channel in ['red', 'green', 'blue']:  
                count = df[df['Channel'] == channel.capitalize()]['Track_Label'].nunique()  
                if count > 0:  
                    self.log(f"    {channel.capitalize()}: {count} tracks")  
        else:  
            self.log("⚠ No data to export")  
    
    def export_all_channels_merged_csv(self, base_path, options):  
        """Export ALL channels merged into ONE CSV file"""  
        all_data = []  
        
        self.log("Merging all channels into single CSV...")  
        
        for channel in ['red', 'green', 'blue']:  
            tracks = self.tracks_dict[channel]  
            if len(tracks) == 0:  
                continue  
            
            self.log(f"  - Processing {channel.upper()} channel ({len(tracks)} tracks)")  
            
            for i, track in enumerate(tracks):  
                track_num = i + 1  
                track_data = self.prepare_track_data(track, track_num, channel, options)  
                
                if track_data:  
                    n_points = len(next(iter(track_data.values())))  
                    track_data['Track_Label'] = [f"{channel.capitalize()}_{track_num}"] * n_points  
                    
                    all_data.append(pd.DataFrame(track_data))  
        
        if all_data:  
            df = pd.concat(all_data, ignore_index=True)  
            
            # Reorder columns  
            cols = ['Track_Label', 'Track_ID', 'Channel', 'Point_ID'] + \
                   [c for c in df.columns if c not in ['Track_Label', 'Track_ID', 'Channel', 'Point_ID']]  
            df = df[cols]  
            
            csv_path = base_path.parent / f"{base_path.stem}_AllTracks_Merged.csv"  
            df.to_csv(csv_path, index=False)  
            self.log(f"✓ Exported: {csv_path.name} ({len(df)} total points)")  
        else:  
            self.log("⚠ No data to export")  
    
    # ========== Standard Format Export ==========  
    
    def export_channel_standard_excel(self, writer, channel, options):  
        """Export channel tracks in standard format (separate sheet per track) - Excel"""  
        tracks = self.tracks_dict[channel]  
        
        for i, track in enumerate(tracks):  
            track_num = i + 1  
            track_data = self.prepare_track_data(track, track_num, channel, options)  
            
            if track_data:  
                df = pd.DataFrame(track_data)  
                sheet_name = f'{channel.capitalize()}_Track{track_num}'  
                df.to_excel(writer, sheet_name=sheet_name, index=False)  
                self.log(f"  ✓ {sheet_name}: {len(df)} points")  
    
    def export_channel_standard_csv(self, base_path, channel, options):  
        """Export channel tracks in standard format (separate CSV per track)"""  
        tracks = self.tracks_dict[channel]  
        
        for i, track in enumerate(tracks):  
            track_num = i + 1  
            track_data = self.prepare_track_data(track, track_num, channel, options)  
            
            if track_data:  
                df = pd.DataFrame(track_data)  
                csv_path = base_path.parent / f"{base_path.stem}_{channel.capitalize()}_Track{track_num}.csv"  
                df.to_csv(csv_path, index=False)  
                self.log(f"  ✓ {csv_path.name}: {len(df)} points")  
    
    # ========== Wide Format Export ==========  
    
    def export_channel_wide_excel(self, writer, channel, options):  
        """Export channel tracks in wide format (all tracks in columns) - Excel"""  
        tracks = self.tracks_dict[channel]  
        
        # Find maximum track length  
        max_len = max(len(track.time_idx) for track in tracks)  
        
        # Prepare wide format data  
        wide_data = {}  
        
        for i, track in enumerate(tracks):  
            track_num = i + 1  
            track_data = self.prepare_track_data(track, track_num, channel, options)  
            
            # Pad shorter tracks with NaN  
            for key, values in track_data.items():  
                col_name = f'Track{track_num}_{key}'  
                padded = self.pad_export_values(values, max_len)
                wide_data[col_name] = padded  
        
        if wide_data:  
            df = pd.DataFrame(wide_data)  
            sheet_name = f'{channel.capitalize()}_Wide'  
            df.to_excel(writer, sheet_name=sheet_name, index=False)  
            self.log(f"  ✓ {sheet_name}: {len(tracks)} tracks × {max_len} max points")  
    
    def export_channel_wide_csv(self, base_path, channel, options):  
        """Export channel tracks in wide format - CSV"""  
        tracks = self.tracks_dict[channel]  
        max_len = max(len(track.time_idx) for track in tracks)  
        
        wide_data = {}  
        for i, track in enumerate(tracks):  
            track_num = i + 1  
            track_data = self.prepare_track_data(track, track_num, channel, options)  
            
            for key, values in track_data.items():  
                col_name = f'Track{track_num}_{key}'  
                padded = self.pad_export_values(values, max_len)
                wide_data[col_name] = padded  
        
        if wide_data:  
            df = pd.DataFrame(wide_data)  
            csv_path = base_path.parent / f"{base_path.stem}_{channel.capitalize()}_Wide.csv"  
            df.to_csv(csv_path, index=False)  
            self.log(f"  ✓ {csv_path.name}: {len(tracks)} tracks × {max_len} max points")  
    
    # ========== Long Format Export ==========  
    
    def export_channel_long_excel(self, writer, channel, options):  
        """Export channel tracks in long format (stacked) - Excel"""  
        tracks = self.tracks_dict[channel]  
        
        all_data = []  
        for i, track in enumerate(tracks):  
            track_num = i + 1  
            track_data = self.prepare_track_data(track, track_num, channel, options)  
            
            if track_data:  
                n_points = len(next(iter(track_data.values())))  
                track_data['Track_ID'] = [track_num] * n_points  
                track_data['Point_ID'] = list(range(n_points))  
                
                all_data.append(pd.DataFrame(track_data))  
        
        if all_data:  
            df = pd.concat(all_data, ignore_index=True)  
            
            # Reorder columns: Track_ID and Point_ID first  
            cols = ['Track_ID', 'Point_ID'] + \
                   [c for c in df.columns if c not in ['Track_ID', 'Point_ID']]  
            df = df[cols]  
            
            sheet_name = f'{channel.capitalize()}_Long'  
            df.to_excel(writer, sheet_name=sheet_name, index=False)  
            self.log(f"  ✓ {sheet_name}: {len(tracks)} tracks, {len(df)} total points")  
    
    def export_channel_long_csv(self, base_path, channel, options):  
        """Export channel tracks in long format - CSV"""  
        tracks = self.tracks_dict[channel]  
        
        all_data = []  
        for i, track in enumerate(tracks):  
            track_num = i + 1  
            track_data = self.prepare_track_data(track, track_num, channel, options)  
            
            if track_data:  
                n_points = len(next(iter(track_data.values())))  
                track_data['Track_ID'] = [track_num] * n_points  
                track_data['Point_ID'] = list(range(n_points))  
                
                all_data.append(pd.DataFrame(track_data))  
        
        if all_data:  
            df = pd.concat(all_data, ignore_index=True)  
            
            cols = ['Track_ID', 'Point_ID'] + \
                   [c for c in df.columns if c not in ['Track_ID', 'Point_ID']]  
            df = df[cols]  
            
            csv_path = base_path.parent / f"{base_path.stem}_{channel.capitalize()}_Long.csv"  
            df.to_csv(csv_path, index=False)  
            self.log(f"  ✓ {csv_path.name}: {len(tracks)} tracks, {len(df)} total points")  
    
    # ========== Summary Format Export ==========  
    
    def export_channel_summary_excel(self, writer, channel, options):  
        """Export summary statistics for channel tracks - Excel"""  
        tracks = self.tracks_dict[channel]  

        summary_data = [
            self.build_track_summary_row(
                track,
                i + 1,
                channel,
                include_intensity=options.get('include_intensity', True)
            )
            for i, track in enumerate(tracks)
        ]

        df = pd.DataFrame(summary_data)
        sheet_name = f'{channel.capitalize()}_Summary'
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        self.log(f"  鉁?{sheet_name}: {len(tracks)} tracks")
        return  
        
        summary_data = []  
        for i, track in enumerate(tracks):  
            track_num = i + 1  
            
            time_indices = np.array(track.time_idx)  
            position_indices = np.array(track.coordinate_idx)  
            
            if self.roi_coords:  
                x1, y1, _, _ = self.roi_coords  
                time_indices = time_indices + x1  
                position_indices = position_indices + y1  
            
            times = time_indices * self.delta_line_time  
            positions = position_indices * self.pixel_size_nm / 1000  
            
            duration = times[-1] - times[0] if len(times) > 1 else 0  
            start_pos = positions[0]  
            end_pos = positions[-1]  
            displacement = end_pos - start_pos  
            
            # Calculate velocity statistics  
            if len(positions) >= 2:  
                velocities = np.diff(positions) / np.diff(times)  
                mean_velocity = np.mean(velocities)  
                mean_abs_velocity = np.mean(np.abs(velocities))  
                std_velocity = np.std(velocities)  
            else:  
                mean_velocity = 0  
                mean_abs_velocity = 0  
                std_velocity = 0  
            
            summary = {  
                'Track_ID': track_num,  
                'Points': len(track.time_idx),  
                'Duration (s)': duration,  
                'Start_Position (μm)': start_pos,  
                'End_Position (μm)': end_pos,  
                'Net_Displacement (μm)': displacement,  
                'Total_Distance (μm)': np.sum(np.abs(np.diff(positions))),  
                'Mean_Velocity (μm/s)': mean_velocity,  
                'Mean_|Velocity| (μm/s)': mean_abs_velocity,  
                'Std_Velocity (μm/s)': std_velocity  
            }  
            
            summary_data.append(summary)  
        
        df = pd.DataFrame(summary_data)  
        sheet_name = f'{channel.capitalize()}_Summary'  
        df.to_excel(writer, sheet_name=sheet_name, index=False)  
        self.log(f"  ✓ {sheet_name}: {len(tracks)} tracks")  
    
    def export_channel_summary_csv(self, base_path, channel, options):  
        """Export summary statistics - CSV"""  
        tracks = self.tracks_dict[channel]  

        summary_data = [
            self.build_track_summary_row(
                track,
                i + 1,
                channel,
                include_intensity=options.get('include_intensity', True)
            )
            for i, track in enumerate(tracks)
        ]

        df = pd.DataFrame(summary_data)
        csv_path = base_path.parent / f"{base_path.stem}_{channel.capitalize()}_Summary.csv"
        df.to_csv(csv_path, index=False)
        self.log(f"  鉁?{csv_path.name}: {len(tracks)} tracks")
        return  
        
        summary_data = []  
        for i, track in enumerate(tracks):  
            track_num = i + 1  
            
            time_indices = np.array(track.time_idx)  
            position_indices = np.array(track.coordinate_idx)  
            
            if self.roi_coords:  
                x1, y1, _, _ = self.roi_coords  
                time_indices = time_indices + x1  
                position_indices = position_indices + y1  
            
            times = time_indices * self.delta_line_time  
            positions = position_indices * self.pixel_size_nm / 1000  
            
            duration = times[-1] - times[0] if len(times) > 1 else 0  
            start_pos = positions[0]  
            end_pos = positions[-1]  
            displacement = end_pos - start_pos  
            
            if len(positions) >= 2:  
                velocities = np.diff(positions) / np.diff(times)  
                mean_velocity = np.mean(velocities)  
                mean_abs_velocity = np.mean(np.abs(velocities))  
                std_velocity = np.std(velocities)  
            else:  
                mean_velocity = 0  
                mean_abs_velocity = 0  
                std_velocity = 0  
            
            summary = {  
                'Track_ID': track_num,  
                'Points': len(track.time_idx),  
                'Duration (s)': duration,  
                'Start_Position (μm)': start_pos,  
                'End_Position (μm)': end_pos,  
                'Net_Displacement (μm)': displacement,  
                'Total_Distance (μm)': np.sum(np.abs(np.diff(positions))),  
                'Mean_Velocity (μm/s)': mean_velocity,  
                'Mean_|Velocity| (μm/s)': mean_abs_velocity,  
                'Std_Velocity (μm/s)': std_velocity  
            }  
            
            summary_data.append(summary)  
        
        df = pd.DataFrame(summary_data)  
        csv_path = base_path.parent / f"{base_path.stem}_{channel.capitalize()}_Summary.csv"  
        df.to_csv(csv_path, index=False)  
        self.log(f"  ✓ {csv_path.name}: {len(tracks)} tracks")  
    
    # ========== Metadata & Settings Export ==========  
    
    def export_metadata_sheet(self, writer):  
        """Export metadata sheet - Excel"""  
        if self.kymo is None:  
            return  
        
        metadata = {  
            'Parameter': [  
                'Source File',  
                'Kymograph Name',  
                'Analysis Date',  
                'Kymo Shape (H×W)',  
                'Pixel Size (nm)',  
                'Line Time (s)',  
                'Total Time (s)',  
                'Total Position (μm)',  
                'Y-Axis Flipped',  
                'ROI Applied'  
            ],  
            'Value': [  
                Path(self.current_file_path).name if self.current_file_path else 'N/A',  
                self.current_kymo_name or 'N/A',  
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  
                f"{self.red_image.shape[0]}×{self.red_image.shape[1]}",  
                f"{self.pixel_size_nm:.2f}",  
                f"{self.delta_line_time:.6f}",  
                f"{self.red_image.shape[1] * self.delta_line_time:.3f}",  
                f"{self.red_image.shape[0] * self.pixel_size_nm / 1000:.3f}",  
                'Yes' if self.y_flip_enabled else 'No',  
                str(self.roi_coords) if self.roi_coords else 'None'  
            ]  
        }  
        
        df = pd.DataFrame(metadata)  
        df.to_excel(writer, sheet_name='Metadata', index=False)  
        self.log("  ✓ Metadata sheet exported")  
    
    def export_metadata_csv(self, csv_path):  
        """Export metadata - CSV"""  
        if self.kymo is None:  
            return  
        
        metadata = {  
            'Parameter': [  
                'Source File',  
                'Kymograph Name',  
                'Analysis Date',  
                'Kymo Shape (H×W)',  
                'Pixel Size (nm)',  
                'Line Time (s)',  
                'Total Time (s)',  
                'Total Position (μm)',  
                'Y-Axis Flipped',  
                'ROI Applied'  
            ],  
            'Value': [  
                Path(self.current_file_path).name if self.current_file_path else 'N/A',  
                self.current_kymo_name or 'N/A',  
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  
                f"{self.red_image.shape[0]}×{self.red_image.shape[1]}",  
                f"{self.pixel_size_nm:.2f}",  
                f"{self.delta_line_time:.6f}",  
                f"{self.red_image.shape[1] * self.delta_line_time:.3f}",  
                f"{self.red_image.shape[0] * self.pixel_size_nm / 1000:.3f}",  
                'Yes' if self.y_flip_enabled else 'No',  
                str(self.roi_coords) if self.roi_coords else 'None'  
            ]  
        }  
        
        df = pd.DataFrame(metadata)  
        df.to_csv(csv_path, index=False)  
        self.log(f"  ✓ Metadata exported: {csv_path.name}")  
    
    def export_settings_sheet(self, writer):  
        """Export tracking settings for all channels - Excel"""  
        all_settings = []  
        
        for channel in ['red', 'green', 'blue']:  
            if channel in self.tracking_settings_dict and self.tracking_settings_dict[channel]:  
                settings = self.tracking_settings_dict[channel].copy()  
                settings['channel'] = channel.capitalize()  
                all_settings.append(settings)  
        
        if all_settings:  
            df = pd.DataFrame(all_settings)  
            
            # Reorder columns to put channel first  
            if 'channel' in df.columns:  
                cols = ['channel'] + [c for c in df.columns if c != 'channel']  
                df = df[cols]  
            
            df.to_excel(writer, sheet_name='Tracking_Settings', index=False)  
            self.log("  ✓ Tracking settings sheet exported")  
    
    def export_settings_csv(self, csv_path):  
        """Export tracking settings - CSV"""  
        all_settings = []  
        
        for channel in ['red', 'green', 'blue']:  
            if channel in self.tracking_settings_dict and self.tracking_settings_dict[channel]:  
                settings = self.tracking_settings_dict[channel].copy()  
                settings['channel'] = channel.capitalize()  
                all_settings.append(settings)  
        
        if all_settings:  
            df = pd.DataFrame(all_settings)  
            
            if 'channel' in df.columns:  
                cols = ['channel'] + [c for c in df.columns if c != 'channel']  
                df = df[cols]  
            
            df.to_csv(csv_path, index=False)  
            self.log(f"  ✓ Tracking settings exported: {csv_path.name}")  
    
    # ========== Velocity Analysis ==========  

    def analyze_roi_fluorescence(self):
        """Inspect fluorescence change over time inside the selected ROI."""
        if self.kymo is None:
            messagebox.showwarning("Warning", "Please load a kymograph first")
            return

        if self.roi_coords is None and not (self.roi_type == 'polygon' and self.roi_mask is not None):
            messagebox.showwarning("Warning", "Please select an ROI first")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("ROI Fluorescence Analysis")
        dialog.geometry("1320x860")
        dialog.transient(self.root)
        dialog.grab_set()

        main_paned = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(main_paned, width=360)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=0)
        main_paned.add(right_frame, weight=1)

        option_frame = ttk.LabelFrame(left_frame, text="1. ROI Settings", padding=10)
        option_frame.pack(fill=tk.X, pady=(0, 6))

        channel_var = tk.StringVar(value=self.channel_var.get() if self.channel_var.get() in ['red', 'green', 'blue'] else 'red')
        use_relative_time_var = tk.BooleanVar(value=True)
        normalize_sum_var = tk.BooleanVar(value=False)
        normalize_mean_var = tk.BooleanVar(value=False)
        smooth_var = tk.BooleanVar(value=False)
        smooth_window_var = tk.IntVar(value=5)

        ttk.Label(option_frame, text="Channel:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Combobox(
            option_frame,
            textvariable=channel_var,
            values=['red', 'green', 'blue'],
            state='readonly',
            width=12
        ).grid(row=0, column=1, sticky=tk.W, pady=3)

        ttk.Checkbutton(option_frame, text="Use relative time", variable=use_relative_time_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(option_frame, text="Normalize summed intensity", variable=normalize_sum_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(option_frame, text="Normalize mean intensity", variable=normalize_mean_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(option_frame, text="Apply smoothing", variable=smooth_var).grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(option_frame, from_=3, to=51, increment=2, textvariable=smooth_window_var, width=8).grid(row=4, column=1, sticky=tk.W, pady=2)

        info_frame = ttk.LabelFrame(left_frame, text="2. ROI Summary", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        info_var = tk.StringVar(value="ROI selected. Choose a channel to visualize fluorescence over time.")
        ttk.Label(info_frame, textvariable=info_var, justify=tk.LEFT, wraplength=320).pack(fill=tk.BOTH, expand=True)

        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill=tk.X, pady=(6, 0))

        fig = Figure(figsize=(9.8, 7.4), dpi=100)
        ax_sum = fig.add_subplot(211)
        ax_mean = fig.add_subplot(212, sharex=ax_sum)
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, right_frame)
        toolbar.update()

        channel_colors = {'red': '#d62728', 'green': '#2ca02c', 'blue': '#1f77b4'}
        plot_state = {'profile': None, 'summary': None}

        def maybe_smooth(values):
            arr = np.asarray(values, dtype=float)
            if not smooth_var.get() or arr.size < 3:
                return arr

            window = max(3, int(smooth_window_var.get()))
            if window % 2 == 0:
                window += 1
            if window > arr.size:
                window = arr.size if arr.size % 2 == 1 else max(1, arr.size - 1)
            if window < 3:
                return arr

            kernel = np.ones(window, dtype=float) / window
            return np.convolve(arr, kernel, mode='same')

        def normalize(values):
            arr = np.asarray(values, dtype=float)
            finite = np.isfinite(arr)
            if not np.any(finite):
                return arr
            min_val = np.min(arr[finite])
            max_val = np.max(arr[finite])
            if max_val <= min_val:
                return arr
            out = arr.copy()
            out[finite] = (arr[finite] - min_val) / (max_val - min_val)
            return out

        def update_plot(*_args):
            ax_sum.clear()
            ax_mean.clear()

            try:
                profile = self.get_roi_intensity_profile(channel_var.get())
            except Exception as exc:
                info_var.set(f"ROI fluorescence analysis failed:\n{exc}")
                ax_sum.text(0.5, 0.5, str(exc), ha='center', va='center', transform=ax_sum.transAxes, fontsize=12)
                canvas.draw()
                return

            summary = self.summarize_roi_intensity_profile(profile)
            plot_state['profile'] = profile
            plot_state['summary'] = summary

            x_values = profile['relative_times'] if use_relative_time_var.get() else profile['times']
            x_label = 'Relative time (s)' if use_relative_time_var.get() else 'Time (s)'
            color = channel_colors[channel_var.get()]

            summed = maybe_smooth(profile['summed_intensity'])
            mean_values = maybe_smooth(profile['mean_intensity'])

            if normalize_sum_var.get():
                summed = normalize(summed)
                sum_label = 'Normalized summed intensity'
            else:
                sum_label = 'Summed photon counts in ROI'

            if normalize_mean_var.get():
                mean_values = normalize(mean_values)
                mean_label = 'Normalized mean pixel intensity'
            else:
                mean_label = 'Mean pixel intensity in ROI'

            ax_sum.plot(x_values, summed, color=color, linewidth=2.0)
            ax_sum.fill_between(x_values, summed, color=color, alpha=0.16)
            ax_sum.set_ylabel(sum_label)
            ax_sum.set_title(
                f"{channel_var.get().capitalize()} ROI fluorescence ({summary['ROI_Type']}, "
                f"{summary['ROI_Area_Pixels']} px)"
            )
            ax_sum.grid(True, alpha=0.25)

            ax_mean.plot(x_values, mean_values, color=color, linewidth=1.8)
            ax_mean.fill_between(x_values, mean_values, color=color, alpha=0.14)
            ax_mean.set_xlabel(x_label)
            ax_mean.set_ylabel(mean_label)
            ax_mean.grid(True, alpha=0.25)

            info_lines = [
                f"Channel: {summary['Channel']}",
                f"ROI type: {summary['ROI_Type']}",
                f"ROI area: {summary['ROI_Area_Pixels']} pixels",
                f"Frames: {summary['Frames']}",
                f"Start / End: {summary['Start_Time (s)']:.3f} s -> {summary['End_Time (s)']:.3f} s",
                f"Duration: {summary['Duration (s)']:.3f} s",
                f"Mean summed intensity: {summary['Mean_Summed_Intensity']:.3f}",
                f"Max summed intensity: {summary['Max_Summed_Intensity']:.3f}",
                f"Integrated summed intensity: {summary['Integrated_Summed_Intensity']:.3f}",
                f"Mean pixel intensity: {summary['Mean_Pixel_Intensity']:.3f}",
                f"Summed intensity delta: {summary['Delta_Summed_Intensity']:.3f}",
                f"Relative change: {summary['Relative_Summed_Intensity_Change']:.3f}"
            ]
            info_var.set("\n".join(info_lines))

            fig.tight_layout()
            canvas.draw()

        def save_current_plot():
            if plot_state['profile'] is None:
                messagebox.showwarning("Warning", "No ROI profile to save", parent=dialog)
                return

            default_name = (
                f"{Path(self.current_file_path).stem if self.current_file_path else 'kymo'}_"
                f"{channel_var.get()}_ROIFluorescence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            file_path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Save ROI fluorescence plot",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                initialfile=default_name
            )
            if not file_path:
                return
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.log(f"ROI fluorescence plot saved: {Path(file_path).name}")

        def export_current_profile():
            if plot_state['profile'] is None or plot_state['summary'] is None:
                messagebox.showwarning("Warning", "No ROI profile to export", parent=dialog)
                return

            profile = plot_state['profile']
            summary = plot_state['summary']
            default_name = (
                f"{Path(self.current_file_path).stem if self.current_file_path else 'kymo'}_"
                f"{channel_var.get()}_ROIFluorescence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            file_path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Export ROI fluorescence data",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
                initialfile=default_name
            )
            if not file_path:
                return

            points_df = pd.DataFrame({
                'Channel': [profile['channel'].capitalize()] * len(profile['times']),
                'ROI_Type': [profile['roi_type']] * len(profile['times']),
                'Time_Index': profile['x_indices'],
                'Time (s)': profile['times'],
                'Relative Time (s)': profile['relative_times'],
                'Summed Photon Counts': profile['summed_intensity'],
                'Mean Pixel Intensity': profile['mean_intensity'],
                'Active Pixels': profile['active_pixels']
            })
            summary_df = pd.DataFrame([summary])

            export_path = Path(file_path)
            if export_path.suffix.lower() == '.csv':
                points_path = export_path.with_name(f"{export_path.stem}_points.csv")
                summary_path = export_path.with_name(f"{export_path.stem}_summary.csv")
                points_df.to_csv(points_path, index=False)
                summary_df.to_csv(summary_path, index=False)
                self.log(f"ROI fluorescence exported: {points_path.name}, {summary_path.name}")
            else:
                with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                    points_df.to_excel(writer, sheet_name='ROIPoints', index=False)
                    summary_df.to_excel(writer, sheet_name='ROISummary', index=False)
                self.log(f"ROI fluorescence exported: {export_path.name}")

        ttk.Button(action_frame, text="Save Plot", command=save_current_plot).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(action_frame, text="Export ROI", command=export_current_profile).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        for var in [channel_var, use_relative_time_var, normalize_sum_var, normalize_mean_var, smooth_var, smooth_window_var]:
            try:
                var.trace_add('write', update_plot)
            except AttributeError:
                var.trace('w', lambda *_args: update_plot())

        update_plot()

    def analyze_trace_fluorescence(self):
        """Interactive inspection of trace lifetime and fluorescence intensity changes."""
        track_entries = []
        for channel in ['red', 'green', 'blue']:
            for idx, track in enumerate(self.tracks_dict[channel]):
                profile = self.get_track_fluorescence_profile(track, include_intensity=False, channel=channel)
                label = (
                    f"{channel.capitalize()} - Track {idx + 1} "
                    f"({profile['point_count']} pts, lifetime {profile['lifetime_seconds']:.2f}s)"
                )
                track_entries.append((channel, idx, track, label))

        if not track_entries:
            messagebox.showwarning("Warning", "No tracks available for fluorescence analysis")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Trace Fluorescence Analysis")
        dialog.geometry("1320x860")
        dialog.transient(self.root)
        dialog.grab_set()

        main_paned = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(main_paned, width=360)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=0)
        main_paned.add(right_frame, weight=1)

        select_frame = ttk.LabelFrame(left_frame, text="1. Select Trace", padding=10)
        select_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        listbox = tk.Listbox(select_frame, height=18, exportselection=False, font=(FONT_FAMILY, 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(select_frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)

        for _, _, _, label in track_entries:
            listbox.insert(tk.END, label)

        option_frame = ttk.LabelFrame(left_frame, text="2. Plot Options", padding=10)
        option_frame.pack(fill=tk.X, pady=6)

        use_relative_time_var = tk.BooleanVar(value=True)
        normalize_intensity_var = tk.BooleanVar(value=False)
        show_position_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(option_frame, text="Use relative time", variable=use_relative_time_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(option_frame, text="Normalize intensity", variable=normalize_intensity_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(option_frame, text="Show position subplot", variable=show_position_var).pack(anchor=tk.W, pady=2)

        info_frame = ttk.LabelFrame(left_frame, text="3. Trace Summary", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        info_var = tk.StringVar(value="Select one trace to inspect lifetime and fluorescence intensity.")
        ttk.Label(info_frame, textvariable=info_var, justify=tk.LEFT, wraplength=320).pack(fill=tk.BOTH, expand=True)

        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill=tk.X, pady=(6, 0))

        fig = Figure(figsize=(9.5, 7.2), dpi=100)
        ax_position = fig.add_subplot(211)
        ax_intensity = fig.add_subplot(212, sharex=ax_position)
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, right_frame)
        toolbar.update()

        plot_state = {'selection': None, 'profile': None}
        channel_colors = {'red': '#d62728', 'green': '#2ca02c', 'blue': '#1f77b4'}

        def update_plot(*_args):
            selection = listbox.curselection()
            ax_position.clear()
            ax_intensity.clear()

            if not selection:
                ax_position.set_visible(show_position_var.get())
                ax_intensity.text(
                    0.5, 0.5, 'Select a trace from the list',
                    ha='center', va='center', transform=ax_intensity.transAxes, fontsize=13
                )
                canvas.draw()
                return

            channel, track_idx, track, _ = track_entries[selection[0]]
            profile = self.get_track_fluorescence_profile(track, include_intensity=True, channel=channel)
            plot_state['selection'] = (channel, track_idx, track)
            plot_state['profile'] = profile

            x_values = profile['relative_times'] if use_relative_time_var.get() else profile['times']
            x_label = 'Relative time (s)' if use_relative_time_var.get() else 'Time (s)'
            color = channel_colors[channel]

            ax_position.set_visible(show_position_var.get())
            if show_position_var.get():
                ax_position.plot(x_values, profile['positions'], 'o-', color=color, linewidth=1.8, markersize=4)
                ax_position.set_ylabel('Position (um)')
                ax_position.set_title(f"{channel.capitalize()} Track {track_idx + 1} Position")
                ax_position.grid(True, alpha=0.25)

            intensities = np.asarray(profile['intensities'], dtype=float)
            finite_mask = np.isfinite(intensities)
            plotted_intensity = intensities.copy()
            intensity_label = 'Summed photon counts'
            if normalize_intensity_var.get() and np.any(finite_mask):
                finite_values = intensities[finite_mask]
                min_val = np.min(finite_values)
                max_val = np.max(finite_values)
                if max_val > min_val:
                    plotted_intensity[finite_mask] = (finite_values - min_val) / (max_val - min_val)
                    intensity_label = 'Normalized intensity'

            ax_intensity.plot(x_values, plotted_intensity, 'o-', color=color, linewidth=2.0, markersize=4)
            if np.any(np.isfinite(plotted_intensity)):
                ax_intensity.fill_between(x_values, plotted_intensity, alpha=0.16, color=color)

            ax_intensity.set_xlabel(x_label)
            ax_intensity.set_ylabel(intensity_label)
            ax_intensity.set_title(
                f"{channel.capitalize()} Track {track_idx + 1} Fluorescence "
                f"(lifetime {profile['lifetime_seconds']:.2f}s / {profile['lifetime_frames']} frames)"
            )
            ax_intensity.grid(True, alpha=0.25)

            summary = self.build_track_summary_row(track, track_idx + 1, channel, include_intensity=True)
            info_lines = [
                f"Channel: {summary['Channel']}",
                f"Track ID: {summary['Track_ID']}",
                f"Points: {summary['Points']}",
                f"Start / End: {summary['Start_Time (s)']:.3f} s -> {summary['End_Time (s)']:.3f} s",
                f"Lifetime: {summary['Lifetime (s)']:.3f} s ({summary['Lifetime (frames)']} frames)",
                f"Duration: {summary['Duration (s)']:.3f} s",
                f"Mean intensity: {summary.get('Mean_Intensity', np.nan):.3f}",
                f"Max intensity: {summary.get('Max_Intensity', np.nan):.3f}",
                f"Integrated intensity: {summary.get('Integrated_Intensity', np.nan):.3f}",
                f"Intensity delta: {summary.get('Delta_Intensity', np.nan):.3f}",
                f"Relative change: {summary.get('Relative_Intensity_Change', np.nan):.3f}"
            ]
            if summary.get('Intensity_Error'):
                info_lines.append(f"Intensity note: {summary['Intensity_Error']}")
            info_var.set("\n".join(info_lines))

            fig.tight_layout()
            canvas.draw()

        def save_current_plot():
            if plot_state['selection'] is None:
                messagebox.showwarning("Warning", "Please select a trace first", parent=dialog)
                return

            channel, track_idx, _ = plot_state['selection']
            default_name = (
                f"{Path(self.current_file_path).stem if self.current_file_path else 'kymo'}_"
                f"{channel}_Track{track_idx + 1}_FluorescenceTrace_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            file_path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Save fluorescence plot",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                initialfile=default_name
            )
            if not file_path:
                return
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.log(f"Fluorescence plot saved: {Path(file_path).name}")

        def export_current_trace():
            if plot_state['selection'] is None or plot_state['profile'] is None:
                messagebox.showwarning("Warning", "Please select a trace first", parent=dialog)
                return

            channel, track_idx, track = plot_state['selection']
            profile = plot_state['profile']
            summary = self.build_track_summary_row(track, track_idx + 1, channel, include_intensity=True)

            default_name = (
                f"{Path(self.current_file_path).stem if self.current_file_path else 'kymo'}_"
                f"{channel}_Track{track_idx + 1}_FluorescenceTrace_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            file_path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Export fluorescence trace data",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
                initialfile=default_name
            )
            if not file_path:
                return

            points_df = pd.DataFrame({
                'Track_ID': [track_idx + 1] * profile['point_count'],
                'Channel': [channel.capitalize()] * profile['point_count'],
                'Point_ID': list(range(profile['point_count'])),
                'Time (s)': profile['times'],
                'Relative Time (s)': profile['relative_times'],
                'Position (um)': profile['positions'],
                'Summed Photon Counts': profile['intensities'],
                'Lifetime (frames)': [profile['lifetime_frames']] * profile['point_count'],
                'Lifetime (s)': [profile['lifetime_seconds']] * profile['point_count']
            })
            summary_df = pd.DataFrame([summary])

            export_path = Path(file_path)
            if export_path.suffix.lower() == '.csv':
                points_path = export_path.with_name(f"{export_path.stem}_points.csv")
                summary_path = export_path.with_name(f"{export_path.stem}_summary.csv")
                points_df.to_csv(points_path, index=False)
                summary_df.to_csv(summary_path, index=False)
                self.log(f"Fluorescence trace exported: {points_path.name}, {summary_path.name}")
            else:
                with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                    points_df.to_excel(writer, sheet_name='TracePoints', index=False)
                    summary_df.to_excel(writer, sheet_name='TraceSummary', index=False)
                self.log(f"Fluorescence trace exported: {export_path.name}")

        ttk.Button(action_frame, text="Save Plot", command=save_current_plot).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(action_frame, text="Export Trace", command=export_current_trace).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        listbox.bind('<<ListboxSelect>>', update_plot)
        use_relative_time_var.trace_add('write', update_plot)
        normalize_intensity_var.trace_add('write', update_plot)
        show_position_var.trace_add('write', update_plot)

        listbox.selection_set(0)
        update_plot()

    def analyze_velocity(self):
        """Interactive velocity analysis for selected trace segments."""
        has_tracks = any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue'])

        if not has_tracks:
            messagebox.showwarning("Warning", "No tracks to analyze")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Velocity Analysis - Selected Trace Segment")
        dialog.geometry("1280x860")
        dialog.transient(self.root)
        dialog.grab_set()

        main_paned = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(main_paned, width=360)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=0)
        main_paned.add(right_frame, weight=1)

        track_frame = ttk.LabelFrame(left_frame, text="1. Select Trace", padding=10)
        track_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        ttk.Label(
            track_frame,
            text="Choose one trace, then drag on the upper plot to define the segment for instantaneous velocity.",
            wraplength=300,
            foreground='blue',
        ).pack(anchor=tk.W, pady=(0, 6))

        track_listbox = tk.Listbox(track_frame, height=10, font=(FONT_FAMILY, FONT_SIZE_SMALL))
        track_listbox.pack(fill=tk.BOTH, expand=True)

        track_items = []
        for channel in ['red', 'green', 'blue']:
            for idx, track in enumerate(self.tracks_dict[channel]):
                duration = self.calculate_track_duration(track)
                label = f"{channel.capitalize()} - Track {idx+1} ({len(track.time_idx)} pts, {duration:.2f}s)"
                track_listbox.insert(tk.END, label)
                track_items.append((channel, idx, track))

        segment_frame = ttk.LabelFrame(left_frame, text="2. Segment Selection", padding=10)
        segment_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        ttk.Label(
            segment_frame,
            text="Drag on the top plot to add one or more segments. Instantaneous velocity is calculated from adjacent-point differences.",
            wraplength=300,
            foreground='blue',
        ).pack(anchor=tk.W, pady=(0, 6))

        segments = []
        segment_listbox = tk.Listbox(segment_frame, height=9, font=(FONT_FAMILY, FONT_SIZE_SMALL))
        segment_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        segment_controls = ttk.Frame(segment_frame)
        segment_controls.pack(fill=tk.X, pady=(0, 6))
        segment_controls.columnconfigure(1, weight=1)

        ttk.Label(segment_controls, text="Label").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        segment_label_var = tk.StringVar(value="Segment 1")
        ttk.Entry(segment_controls, textvariable=segment_label_var, width=18).grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(segment_controls, text="Color").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        segment_color_var = tk.StringVar(value="blue")
        ttk.Combobox(
            segment_controls,
            textvariable=segment_color_var,
            state="readonly",
            values=["blue", "red", "green", "orange", "purple", "cyan", "magenta", "brown"],
            width=16,
        ).grid(row=1, column=1, sticky=tk.EW, pady=2)

        method_frame = ttk.LabelFrame(left_frame, text="3. Velocity Method", padding=10)
        method_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(
            method_frame,
            text="Local linear fit is usually much more stable than frame-to-frame differences when the trace slope is steady.",
            wraplength=300,
            foreground='blue',
        ).pack(anchor=tk.W, pady=(0, 6))

        method_controls = ttk.Frame(method_frame)
        method_controls.pack(fill=tk.X)
        method_controls.columnconfigure(1, weight=1)

        ttk.Label(method_controls, text="Method").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        velocity_method_options = {
            "Local linear fit (Recommended)": "local_fit",
            "Raw frame-to-frame diff": "raw_diff",
        }
        velocity_method_var = tk.StringVar(value="Local linear fit (Recommended)")
        ttk.Combobox(
            method_controls,
            textvariable=velocity_method_var,
            state="readonly",
            values=list(velocity_method_options.keys()),
            width=24,
        ).grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(method_controls, text="Window points").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        local_window_var = tk.StringVar(value="11")
        ttk.Entry(method_controls, textvariable=local_window_var, width=16).grid(row=1, column=1, sticky=tk.EW, pady=2)

        filter_frame = ttk.LabelFrame(left_frame, text="4. Velocity Filter", padding=10)
        filter_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(
            filter_frame,
            text="Keep only velocities with |v| <= threshold for the distribution plot and filtered export.",
            wraplength=300,
            foreground='blue',
        ).pack(anchor=tk.W, pady=(0, 6))

        filter_controls = ttk.Frame(filter_frame)
        filter_controls.pack(fill=tk.X)
        filter_controls.columnconfigure(1, weight=1)

        ttk.Label(filter_controls, text="Max |v| (bp/s)").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        max_velocity_var = tk.StringVar(value="1000")
        ttk.Entry(filter_controls, textvariable=max_velocity_var, width=16).grid(row=0, column=1, sticky=tk.EW, pady=2)

        current_track = {'times': None, 'positions': None, 'channel': None, 'track_idx': None}
        drag_state = {'dragging': False, 'start_x': None, 'span': None}

        fig, (ax_trace, ax_velocity) = plt.subplots(
            2, 1, figsize=(10.5, 7.8), gridspec_kw={'height_ratios': [2.2, 1.3]}
        )
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, right_frame)
        toolbar.update()

        def refresh_segment_listbox():
            segment_listbox.delete(0, tk.END)
            for segment in segments:
                segment_listbox.insert(
                    tk.END,
                    f"{segment['label']}: {segment['start']:.3f}-{segment['end']:.3f}s ({segment['color']})"
                )

        def get_velocity_filter_threshold(show_warning=True):
            value = max_velocity_var.get().strip()
            if not value:
                return None

            try:
                threshold = float(value)
            except ValueError:
                if show_warning:
                    messagebox.showwarning("Warning", "Max |v| must be a number in bp/s, or leave it empty to disable filtering")
                return False

            if threshold <= 0:
                if show_warning:
                    messagebox.showwarning("Warning", "Max |v| must be greater than 0")
                return False

            return threshold

        def get_velocity_method():
            return velocity_method_options.get(velocity_method_var.get().strip(), 'local_fit')

        def get_local_window_points(show_warning=True):
            value = local_window_var.get().strip()
            if not value:
                return 7

            try:
                window_points = int(value)
            except ValueError:
                if show_warning:
                    messagebox.showwarning("Warning", "Window points must be an integer")
                return False

            if window_points < 3:
                if show_warning:
                    messagebox.showwarning("Warning", "Window points must be at least 3")
                return False

            if window_points % 2 == 0:
                window_points += 1

            return window_points

        def split_velocity_values(segment_results, threshold_bp_s):
            raw_velocity_bp = np.concatenate([result['instant_velocity_bp_s'] for result in segment_results]) if segment_results else np.array([])
            raw_velocity_bp = np.asarray(raw_velocity_bp, dtype=float)
            if threshold_bp_s is None:
                filtered_velocity_bp = raw_velocity_bp.copy()
            else:
                filtered_velocity_bp = raw_velocity_bp[np.abs(raw_velocity_bp) <= threshold_bp_s]
            return raw_velocity_bp, filtered_velocity_bp

        def get_selected_velocity_values(segment_results, method_key, threshold_bp_s):
            if not segment_results:
                return np.array([]), np.array([])

            if method_key == 'local_fit':
                raw_velocity_bp = np.concatenate([result['local_velocity_bp_s'] for result in segment_results])
            else:
                raw_velocity_bp = np.concatenate([result['instant_velocity_bp_s'] for result in segment_results])

            raw_velocity_bp = np.asarray(raw_velocity_bp, dtype=float)
            if threshold_bp_s is None:
                filtered_velocity_bp = raw_velocity_bp.copy()
            else:
                filtered_velocity_bp = raw_velocity_bp[np.abs(raw_velocity_bp) <= threshold_bp_s]
            return raw_velocity_bp, filtered_velocity_bp

        def collect_segment_results(show_warning=True):
            if current_track['times'] is None:
                if show_warning:
                    messagebox.showwarning("Warning", "Please select a trace first")
                return []

            if not segments:
                if show_warning:
                    messagebox.showwarning("Warning", "Please drag on the trace plot to select at least one segment")
                return []

            local_window_points = get_local_window_points(show_warning=show_warning)
            if local_window_points is False:
                return []

            results = []
            skipped = []
            for segment in segments:
                result = self.analyze_velocity_segment(
                    current_track['times'],
                    current_track['positions'],
                    segment['start'],
                    segment['end'],
                    segment['label'],
                    segment['color'],
                    local_window_points=local_window_points,
                )
                if result is None:
                    skipped.append(segment['label'])
                    continue
                results.append(result)

            if not results and show_warning:
                messagebox.showwarning("Warning", "None of the selected segments contains enough points for velocity calculation")
            elif skipped and show_warning:
                self.log(f"Skipped segments with insufficient points: {', '.join(skipped)}")

            return results

        def update_plot():
            ax_trace.clear()
            ax_velocity.clear()

            if current_track['times'] is None:
                ax_trace.text(0.5, 0.5, 'Select a trace to start', ha='center', va='center', transform=ax_trace.transAxes, fontsize=14)
                ax_velocity.text(0.5, 0.5, 'Instantaneous velocity will appear here', ha='center', va='center', transform=ax_velocity.transAxes, fontsize=12)
                canvas.draw()
                return

            times = current_track['times']
            positions = current_track['positions']
            channel = current_track['channel']
            track_idx = current_track['track_idx']
            hist_color = {'red': '#ff6b6b', 'green': '#66cc66', 'blue': '#6b8cff'}.get(channel, '#ff6b6b')

            ax_trace.plot(times, positions, 'o-', color='lightgray', linewidth=1.8, markersize=4, alpha=0.75, label='Full trace', zorder=1)

            segment_results = collect_segment_results(show_warning=False)

            for segment in segments:
                ax_trace.axvspan(segment['start'], segment['end'], alpha=0.12, color=segment['color'], zorder=0)

            if segment_results:
                threshold_bp_s = get_velocity_filter_threshold(show_warning=False)
                if threshold_bp_s is False:
                    threshold_bp_s = None
                method_key = get_velocity_method()
                for result in segment_results:
                    ax_trace.plot(
                        result['times'],
                        result['positions_um'],
                        'o-',
                        color=result['color'],
                        linewidth=2.5,
                        markersize=5,
                        label=f"{result['label']} selected",
                        zorder=3,
                    )
                raw_velocity_bp, filtered_velocity_bp = get_selected_velocity_values(segment_results, method_key, threshold_bp_s)
                distribution_df, distribution_stats = self.build_velocity_distribution_table(filtered_velocity_bp)
                fit_rate_bp_s = self.compute_weighted_fit_rate_bp_s(segment_results)
                if not distribution_df.empty:
                    ax_velocity.hist(
                        filtered_velocity_bp,
                        bins=30,
                        color=hist_color,
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=1.2,
                    )
                    ax_velocity.axvline(distribution_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {distribution_stats['mean']:.2f} bp/s")
                    ax_velocity.axvline(distribution_stats['mean'] + distribution_stats['std'], color='orange', linestyle=':', linewidth=2, label=f"+/- Std: {distribution_stats['std']:.2f} bp/s")
                    ax_velocity.axvline(distribution_stats['mean'] - distribution_stats['std'], color='orange', linestyle=':', linewidth=2)
                    if np.isfinite(fit_rate_bp_s):
                        ax_velocity.axvline(
                            fit_rate_bp_s,
                            color='navy',
                            linestyle='-.',
                            linewidth=2,
                            label=f"Fit rate: {fit_rate_bp_s:.2f} bp/s",
                        )
                elif raw_velocity_bp.size > 0:
                    ax_velocity.text(
                        0.5,
                        0.5,
                        'All velocities were removed by the current filter',
                        ha='center',
                        va='center',
                        transform=ax_velocity.transAxes,
                        fontsize=11,
                        color='red',
                    )

            ax_trace.set_ylabel('Position (um)', fontsize=11, fontweight='bold')
            ax_trace.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
            ax_trace.set_title(
                f"{channel.capitalize()} Track {track_idx+1} - drag on this plot to choose segment(s)",
                fontsize=12,
                fontweight='bold',
            )
            ax_trace.grid(True, alpha=0.3)
            ax_trace.legend(fontsize=8, loc='best')

            ax_velocity.set_xlabel('Instantaneous Velocity (bp/s)', fontsize=11, fontweight='bold')
            ax_velocity.set_ylabel('Count', fontsize=11, fontweight='bold')
            if segment_results:
                removed_count = max(len(raw_velocity_bp) - len(filtered_velocity_bp), 0)
                filter_label = f'|v| <= {threshold_bp_s:.0f} bp/s' if threshold_bp_s is not None else 'no filter'
                method_label = 'local linear fit' if method_key == 'local_fit' else 'raw diff'
                ax_velocity.set_title(
                    f'Instantaneous velocity distribution ({method_label}, {filter_label}, removed {removed_count})',
                    fontsize=11,
                    fontweight='bold'
                )
            else:
                ax_velocity.set_title('Instantaneous velocity distribution using the original calculation method', fontsize=11, fontweight='bold')
            ax_velocity.grid(True, alpha=0.3, axis='y')

            if segment_results:
                ax_velocity.legend(fontsize=8, loc='best')
            else:
                ax_velocity.text(
                    0.5,
                    0.5,
                    'Add a segment to preview the velocity distribution',
                    ha='center',
                    va='center',
                    transform=ax_velocity.transAxes,
                    fontsize=11,
                )

            plt.subplots_adjust(top=0.94, bottom=0.08, left=0.09, right=0.97, hspace=0.25)
            canvas.draw()

        def load_track():
            selection = track_listbox.curselection()
            if not selection:
                return

            channel, track_idx, track = track_items[selection[0]]
            times, positions = self.get_track_time_position_arrays(track)

            current_track['times'] = times
            current_track['positions'] = positions
            current_track['channel'] = channel
            current_track['track_idx'] = track_idx

            segments.clear()
            refresh_segment_listbox()
            segment_label_var.set("Segment 1")
            update_plot()

        def remove_selected_segment():
            selection = segment_listbox.curselection()
            if not selection:
                return

            segments.pop(selection[0])
            refresh_segment_listbox()
            segment_label_var.set(f"Segment {len(segments) + 1}")
            update_plot()

        def clear_segments():
            segments.clear()
            refresh_segment_listbox()
            segment_label_var.set("Segment 1")
            update_plot()

        def on_press(event):
            if event.inaxes != ax_trace or current_track['times'] is None or event.xdata is None:
                return
            if toolbar.mode != '':
                return

            drag_state['dragging'] = True
            drag_state['start_x'] = event.xdata
            drag_state['span'] = ax_trace.axvspan(event.xdata, event.xdata, alpha=0.2, color='gold', zorder=10)
            canvas.draw()

        def on_motion(event):
            if not drag_state['dragging'] or event.xdata is None:
                return

            if drag_state['span'] is not None:
                drag_state['span'].remove()

            x0 = drag_state['start_x']
            x1 = event.xdata
            drag_state['span'] = ax_trace.axvspan(min(x0, x1), max(x0, x1), alpha=0.2, color='gold', zorder=10)
            canvas.draw()

        def on_release(event):
            if not drag_state['dragging']:
                return

            drag_state['dragging'] = False
            if drag_state['span'] is not None:
                drag_state['span'].remove()
                drag_state['span'] = None

            if event.inaxes != ax_trace or event.xdata is None or drag_state['start_x'] is None:
                drag_state['start_x'] = None
                canvas.draw()
                return

            start = min(drag_state['start_x'], event.xdata)
            end = max(drag_state['start_x'], event.xdata)
            drag_state['start_x'] = None

            if current_track['times'] is None or abs(end - start) < 1e-9:
                canvas.draw()
                return

            label = segment_label_var.get().strip() or f"Segment {len(segments) + 1}"
            color = segment_color_var.get()
            segments.append({'start': start, 'end': end, 'label': label, 'color': color})
            refresh_segment_listbox()
            segment_label_var.set(f"Segment {len(segments) + 1}")
            update_plot()

        def build_results_text(segment_results):
            um_to_bp = segment_results[0]['um_to_bp']
            threshold_bp_s = get_velocity_filter_threshold(show_warning=False)
            if threshold_bp_s is False:
                threshold_bp_s = None
            method_key = get_velocity_method()
            local_window_points = get_local_window_points(show_warning=False)
            if local_window_points is False:
                local_window_points = 7
            raw_velocity_bp, filtered_velocity_bp = get_selected_velocity_values(segment_results, method_key, threshold_bp_s)
            _, raw_stats = self.build_velocity_distribution_table(raw_velocity_bp)
            _, filtered_stats = self.build_velocity_distribution_table(filtered_velocity_bp)
            fit_rate_bp_s = self.compute_weighted_fit_rate_bp_s(segment_results)
            lines = [
                "=" * 72,
                "Instantaneous Velocity Distribution",
                "=" * 72,
                f"Channel: {current_track['channel'].capitalize()}",
                f"Track Index: {current_track['track_idx'] + 1}",
                f"Track Points: {len(current_track['positions'])}",
                f"DNA Conversion: 1 um = {um_to_bp:.1f} bp",
                f"Velocity method: {'Local linear fit' if method_key == 'local_fit' else 'Raw diff(position) / diff(time)'}",
                f"Local fit window: {local_window_points} points" if method_key == 'local_fit' else "Local fit window: not used",
                f"Selected-segment fit rate: {fit_rate_bp_s:.2f} bp/s" if np.isfinite(fit_rate_bp_s) else "Selected-segment fit rate: N/A",
                f"Max |velocity| filter: {threshold_bp_s:.2f} bp/s" if threshold_bp_s is not None else "Max |velocity| filter: disabled",
                f"Instant velocity count (raw): {raw_stats['n_values']}",
                f"Instant velocity count (filtered): {filtered_stats['n_values']}",
                f"Removed by filter: {max(raw_stats['n_values'] - filtered_stats['n_values'], 0)}",
                f"Filtered mean: {filtered_stats['mean']:.2f} bp/s",
                f"Filtered std: {filtered_stats['std']:.2f} bp/s",
                f"Filtered median: {filtered_stats['median']:.2f} bp/s",
                f"Mean - Fit difference: {filtered_stats['mean'] - fit_rate_bp_s:.2f} bp/s" if np.isfinite(fit_rate_bp_s) and np.isfinite(filtered_stats['mean']) else "Mean - Fit difference: N/A",
                "Distribution bins: 30 (same as original histogram style)",
                "",
            ]

            for idx, result in enumerate(segment_results, start=1):
                lines.extend([
                    f"[{idx}] {result['label']}",
                    "-" * 72,
                    f"Time range: {result['start']:.4f} - {result['end']:.4f} s",
                    f"Duration: {result['duration']:.4f} s",
                    f"Points / intervals: {result['point_count']} / {result['interval_count']}",
                    f"Displacement: {result['displacement_um']:.4f} um ({result['displacement_bp']:.1f} bp)",
                    f"Mean velocity: {result['mean_velocity_um_s']:.6f} um/s ({result['mean_velocity_bp_s']:.2f} bp/s)",
                    f"Mean |velocity|: {result['mean_abs_velocity_bp_s']:.2f} bp/s",
                    f"Max |velocity|: {result['max_abs_velocity_bp_s']:.2f} bp/s",
                    "",
                ])

            lines.extend([
                "OriginLab recommendation:",
                "  Use VelocitySeries_Selected and keep Pass_Filter = True",
                "  Or directly plot VelocityDistribution_Filtered",
                "  X = Bin_Center_bp_s, Y = Count",
            ])
            return "\n".join(lines)

        def export_excel(segment_results=None, parent_window=None):
            if segment_results is None:
                segment_results = collect_segment_results(show_warning=True)
            if not segment_results:
                return
            threshold_bp_s = get_velocity_filter_threshold(show_warning=True)
            if threshold_bp_s is False:
                return
            local_window_points = get_local_window_points(show_warning=True)
            if local_window_points is False:
                return

            channel = current_track['channel']
            track_idx = current_track['track_idx']

            if self.current_file_path:
                base_name = Path(self.current_file_path).stem
                initial_dir = str(Path(self.current_file_path).parent)
            else:
                base_name = "velocity_analysis"
                initial_dir = str(Path.cwd())

            default_name = f"{base_name}_{channel}_Track{track_idx+1}_InstantVelocity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            file_path = filedialog.asksaveasfilename(
                parent=parent_window or dialog,
                title="Save Instantaneous Velocity Excel",
                defaultextension=".xlsx",
                initialdir=initial_dir,
                initialfile=default_name,
                filetypes=[("Excel files", "*.xlsx")],
            )

            if not file_path:
                return

            frames = self.build_velocity_export_frames(
                segment_results,
                channel,
                track_idx,
                len(current_track['positions']),
                max_abs_velocity_bp_s=threshold_bp_s,
                velocity_method=get_velocity_method(),
                local_window_points=local_window_points,
            )

            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    for sheet_name, df in frames.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        worksheet = writer.sheets[sheet_name]
                        for col_idx, column in enumerate(df.columns, start=1):
                            column_letter = worksheet.cell(row=1, column=col_idx).column_letter
                            column_values = [str(column)] + [str(value) for value in df[column].tolist()]
                            width = min(max(len(value) for value in column_values) + 2, 28)
                            worksheet.column_dimensions[column_letter].width = width

                self.log(f"Velocity Excel exported: {Path(file_path).name}")
                messagebox.showinfo(
                    "Export Complete",
                    "Excel export finished.\n\n"
                    "OriginLab plot recommendation:\n"
                    "  1. Use VelocitySeries_Selected and keep rows with Pass_Filter = True\n"
                    "  2. Or directly plot VelocityDistribution_Filtered with X=Bin_Center_bp_s, Y=Count\n"
                    "  3. InstantVelocity and LocalInstantVelocity are also included for comparison.",
                    parent=parent_window or dialog,
                )
            except Exception as exc:
                messagebox.showerror("Export Error", f"Failed to export Excel:\n{exc}", parent=parent_window or dialog)
                self.log(f"Velocity export failed: {exc}")

        def preview_results():
            segment_results = collect_segment_results(show_warning=True)
            if not segment_results:
                return

            threshold_bp_s = get_velocity_filter_threshold(show_warning=True)
            if threshold_bp_s is False:
                return
            local_window_points = get_local_window_points(show_warning=True)
            if local_window_points is False:
                return
            method_key = get_velocity_method()
            raw_velocity_bp, filtered_velocity_bp = get_selected_velocity_values(segment_results, method_key, threshold_bp_s)
            distribution_df, distribution_stats = self.build_velocity_distribution_table(filtered_velocity_bp)
            fit_rate_bp_s = self.compute_weighted_fit_rate_bp_s(segment_results)
            hist_color = {'red': '#ff6b6b', 'green': '#66cc66', 'blue': '#6b8cff'}.get(current_track['channel'], '#ff6b6b')

            preview = tk.Toplevel(dialog)
            preview.title("Instantaneous Velocity Distribution")
            preview.geometry("980x760")
            preview.transient(dialog)
            preview.grab_set()

            preview_main = ttk.Frame(preview, padding=10)
            preview_main.pack(fill=tk.BOTH, expand=True)

            fig_preview, ax_preview = plt.subplots(figsize=(8.2, 4.6))
            preview_canvas = FigureCanvasTkAgg(fig_preview, master=preview_main)
            preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            preview_toolbar = NavigationToolbar2Tk(preview_canvas, preview_main)
            preview_toolbar.update()

            if not distribution_df.empty:
                ax_preview.hist(
                    filtered_velocity_bp,
                    bins=30,
                    color=hist_color,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1.2,
                )
                ax_preview.axvline(distribution_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {distribution_stats['mean']:.2f} bp/s")
                ax_preview.axvline(distribution_stats['mean'] + distribution_stats['std'], color='orange', linestyle=':', linewidth=2, label=f"+/- Std: {distribution_stats['std']:.2f} bp/s")
                ax_preview.axvline(distribution_stats['mean'] - distribution_stats['std'], color='orange', linestyle=':', linewidth=2)
                if np.isfinite(fit_rate_bp_s):
                    ax_preview.axvline(
                        fit_rate_bp_s,
                        color='navy',
                        linestyle='-.',
                        linewidth=2,
                        label=f"Fit rate: {fit_rate_bp_s:.2f} bp/s",
                    )
            else:
                ax_preview.text(
                    0.5,
                    0.5,
                    'All velocities were removed by the current filter',
                    ha='center',
                    va='center',
                    transform=ax_preview.transAxes,
                    fontsize=12,
                    color='red',
                )

            removed_count = max(len(raw_velocity_bp) - len(filtered_velocity_bp), 0)
            filter_label = f'|v| <= {threshold_bp_s:.0f} bp/s' if threshold_bp_s is not None else 'no filter'
            method_label = 'local linear fit' if method_key == 'local_fit' else 'raw diff'
            ax_preview.set_title(
                f'Instantaneous velocity distribution ({method_label}, {filter_label}, removed {removed_count})',
                fontsize=13,
                fontweight='bold'
            )
            ax_preview.set_xlabel('Instantaneous Velocity (bp/s)', fontsize=11, fontweight='bold')
            ax_preview.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax_preview.grid(True, alpha=0.3, axis='y')
            ax_preview.legend(fontsize=9, loc='best')
            fig_preview.subplots_adjust(top=0.90, bottom=0.14, left=0.11, right=0.97)
            preview_canvas.draw()

            text_widget = scrolledtext.ScrolledText(preview_main, wrap=tk.WORD, height=14, font=('Consolas', 9))
            text_widget.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
            text_widget.insert('1.0', build_results_text(segment_results))
            text_widget.config(state='disabled')

            action_frame = ttk.Frame(preview_main)
            action_frame.pack(fill=tk.X, pady=(10, 0))
            ttk.Button(
                action_frame,
                text="Confirm and Export Excel",
                command=lambda: export_excel(segment_results=segment_results, parent_window=preview),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=6)
            ttk.Button(
                action_frame,
                text="Close",
                command=preview.destroy,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0), ipady=6)

        button_frame = ttk.Frame(segment_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="Remove Selected", command=remove_selected_segment).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Clear All Segments", command=clear_segments).pack(fill=tk.X, pady=2)

        bottom_frame = ttk.Frame(left_frame)
        bottom_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(bottom_frame, text="Generate Distribution Plot", command=preview_results).pack(fill=tk.X, pady=2, ipady=6)
        ttk.Button(bottom_frame, text="Close", command=dialog.destroy).pack(fill=tk.X, pady=2)

        track_listbox.bind('<<ListboxSelect>>', lambda _event: load_track())
        canvas.mpl_connect('button_press_event', on_press)
        canvas.mpl_connect('motion_notify_event', on_motion)
        canvas.mpl_connect('button_release_event', on_release)

        update_plot()
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def analyze_velocity_overview(self):  
        """Analyze velocity for all tracks"""  
        has_tracks = any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue'])  
        
        if not has_tracks:  
            messagebox.showwarning("Warning", "No tracks to analyze")  
            return  
        
        try:  
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
            fig.suptitle("Velocity Analysis - Multi-Channel", fontsize=16, fontweight='bold')  
            
            channel_names = ['Red', 'Green', 'Blue']  
            colors = ['#FF4444', '#44FF44', '#4444FF']  
            
            for idx, (channel, color) in enumerate(zip(['red', 'green', 'blue'], colors)):  
                tracks = self.tracks_dict[channel]  
                
                if len(tracks) == 0:  
                    # Empty plots  
                    axes[0, idx].text(0.5, 0.5, f'No {channel_names[idx]} tracks',  
                                     ha='center', va='center', fontsize=12)  
                    axes[0, idx].set_title(f'{channel_names[idx]} Channel')  
                    axes[0, idx].axis('off')  
                    
                    axes[1, idx].text(0.5, 0.5, f'No {channel_names[idx]} tracks',  
                                     ha='center', va='center', fontsize=12)  
                    axes[1, idx].axis('off')  
                    continue  
                
                all_velocities = []  
                
                for track in tracks:  
                    time_indices = np.array(track.time_idx)  
                    pos_indices = np.array(track.coordinate_idx)  
                    
                    if self.roi_coords:  
                        x1, y1, _, _ = self.roi_coords  
                        time_indices = time_indices + x1  
                        pos_indices = pos_indices + y1  
                    
                    times = time_indices * self.delta_line_time  
                    positions = pos_indices * self.pixel_size_nm / 1000  # μm  
                    
                    if len(positions) >= 2:  
                        velocities = np.diff(positions) / np.diff(times)  # μm/s  
                        all_velocities.extend(velocities)  
                        
                        # Plot time vs velocity  
                        axes[0, idx].plot(times[:-1], velocities, alpha=0.5,  
                                        color=color, linewidth=1)  
                
                # Histogram  
                if all_velocities:  
                    axes[1, idx].hist(all_velocities, bins=30, color=color,  
                                    alpha=0.7, edgecolor='black')  
                    
                    mean_vel = np.mean(all_velocities)  
                    std_vel = np.std(all_velocities)  
                    
                    axes[1, idx].axvline(mean_vel, color='red', linestyle='--',  
                                       linewidth=2, label=f'Mean: {mean_vel:.3f} μm/s')  
                    axes[1, idx].axvline(mean_vel + std_vel, color='orange',  
                                       linestyle=':', linewidth=2,  
                                       label=f'±1σ: {std_vel:.3f} μm/s')  
                    axes[1, idx].axvline(mean_vel - std_vel, color='orange',  
                                       linestyle=':', linewidth=2)  
                    
                    axes[1, idx].legend(fontsize=8)  
                    axes[1, idx].set_xlabel('Velocity (μm/s)', fontsize=10)  
                    axes[1, idx].set_ylabel('Count', fontsize=10)  
                    axes[1, idx].set_title(f'Distribution (n={len(all_velocities)})',  
                                          fontsize=10)  
                    
                    self.log(f"{channel_names[idx]}: Mean velocity = {mean_vel:.3f} ± {std_vel:.3f} μm/s")  
                
                # Format time series plot  
                axes[0, idx].set_xlabel('Time (s)', fontsize=10)  
                axes[0, idx].set_ylabel('Velocity (μm/s)', fontsize=10)  
                axes[0, idx].set_title(f'{channel_names[idx]} Channel ({len(tracks)} tracks)',  
                                      fontsize=11, fontweight='bold')  
                axes[0, idx].grid(True, alpha=0.3)  
                axes[0, idx].axhline(0, color='black', linestyle='-', linewidth=0.5)  
            
            plt.tight_layout()  
            plt.show()  
            
            self.log("Velocity analysis complete")  
            
        except Exception as e:  
            messagebox.showerror("Error", f"Velocity analysis failed:\n{str(e)}")  
            self.log(f"Error: {str(e)}")  
    
    # ========== Utility Methods ==========  
    
    def update_stats(self, text):  
        """Update statistics display - now logs to console"""  
        # Stats panel removed, just log the information
        if text and text != "Waiting for data...":
            self.log(f"Stats: {text}")  
    
    def reset_all(self):  
        """Reset application to initial state"""  
        if messagebox.askyesno("Confirm Reset",   
                              "This will clear all loaded files, tracks, and settings.\n"  
                              "Continue?"):  
            # Clear all files  
            self.loaded_files = {}  
            self.files_listbox.delete(0, tk.END)  
            
            # Reset current file/kymo  
            self.current_file = None  
            self.current_file_path = None  
            self.kymo = None  
            self.current_kymo_name = None  
            
            # Reset images  
            self.red_image = None  
            self.green_image = None  
            self.blue_image = None  
            self.red_image_original = None  
            self.green_image_original = None  
            self.blue_image_original = None  
            
            # Reset tracks  
            self.tracks_dict = {'red': [], 'green': [], 'blue': []}  
            self.tracking_settings_dict = {'red': {}, 'green': {}, 'blue': {}}  
            
            # Reset ROI  
            self.roi_coords = None  
            self.roi_rect = None  
            self.roi_selection_mode = False  
            self.roi_clicks = []  
            
            # Reset flip state  
            self.y_flip_enabled = False  
            self.y_flip_var.set(False)    
            
            # Reset UI states  
            self.current_file_var.set("No file selected")  
            self.current_kymo_var.set("None")  
            self.roi_status_var.set("No ROI selected")  
            self.yflip_status_var.set("Status: Normal orientation")  
            self.tracking_confirm_var.set("Tracking: Not confirmed")  
            
            self.show_red_var.set(False)  
            self.show_green_var.set(False)  
            self.show_blue_var.set(False)  
            
            # Disable buttons    
            self.track_btn.config(state=tk.DISABLED)  
            
            # Clear kymograph list  
            self.kymo_listbox.delete(0, tk.END)  
            
            # Update display  
            self.update_track_counts()  
            self.ax.clear()  
            self.ax.set_title("Load files to begin", fontsize=FONT_SIZE_SUBTITLE)  
            self.canvas.draw()  
            
            self.update_stats("Application reset")  
            self.log("Application reset to initial state")
            
    def sliding_window_msd_analysis(self):
        """
        Sliding window MSD analysis (similar to paper method)
        - Divide track into multiple 30s windows
        - Calculate MSD for each window
        - Fit first N points to get diffusion constant
        - Generate distribution of diffusion constants
        """
        if not hasattr(self, 'tracks_dict') or not any(len(self.tracks_dict[ch]) > 0 for ch in ['red', 'green', 'blue']):
            messagebox.showwarning("Warning", "No tracks available for analysis")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("🔬 Sliding Window MSD Analysis")
        dialog.geometry("1000x750")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== TRACK SELECTION =====
        track_frame = ttk.LabelFrame(main_frame, text="1. Select Track for Analysis", padding=10)
        track_frame.pack(fill=tk.X, pady=5)
        
        track_listbox = tk.Listbox(track_frame, height=6, font=(FONT_FAMILY, FONT_SIZE_SMALL))
        track_listbox.pack(fill=tk.X, pady=5)
        
        track_list = []
        for channel in ['red', 'green', 'blue']:
            for i, track in enumerate(self.tracks_dict[channel]):
                duration = self.calculate_track_duration(track)
                track_id = f"{channel.capitalize()} - Track {i+1} ({len(track.time_idx)} pts, {duration:.2f}s)"
                track_listbox.insert(tk.END, track_id)
                track_list.append((channel, i, track))
        
        # ===== PARAMETER SETTINGS =====
        param_frame = ttk.LabelFrame(main_frame, text="2. Analysis Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        # Window duration
        row = 0
        ttk.Label(param_frame, text="Time Window (s):", 
                 font=(FONT_FAMILY, FONT_SIZE_SMALL)).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        window_var = tk.DoubleVar(value=30.0)
        ttk.Entry(param_frame, textvariable=window_var, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(param_frame, text="Duration of each analysis window", 
                 foreground='gray', font=(FONT_FAMILY, FONT_SIZE_SMALL-1)).grid(row=row, column=2, sticky=tk.W, padx=5)
        
        # Lag sampling interval
        row += 1
        ttk.Label(param_frame, text="Lag Sampling Interval:", 
                 font=(FONT_FAMILY, FONT_SIZE_SMALL)).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        lag_sampling_var = tk.IntVar(value=5)
        ttk.Entry(param_frame, textvariable=lag_sampling_var, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(param_frame, text="Sample every N line positions (e.g., 5 = 1s if line time = 200ms)", 
                 foreground='gray', font=(FONT_FAMILY, FONT_SIZE_SMALL-1)).grid(row=row, column=2, sticky=tk.W, padx=5)
        
        # Fit points
        row += 1
        ttk.Label(param_frame, text="Fit Points:", 
                 font=(FONT_FAMILY, FONT_SIZE_SMALL)).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        fit_points_var = tk.IntVar(value=10)
        ttk.Entry(param_frame, textvariable=fit_points_var, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(param_frame, text="Number of initial points for linear fitting", 
                 foreground='gray', font=(FONT_FAMILY, FONT_SIZE_SMALL-1)).grid(row=row, column=2, sticky=tk.W, padx=5)
        
        # Max lag
        row += 1
        ttk.Label(param_frame, text="Max Lag Points:", 
                 font=(FONT_FAMILY, FONT_SIZE_SMALL)).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        max_lag_var = tk.IntVar(value=15)
        ttk.Entry(param_frame, textvariable=max_lag_var, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(param_frame, text="Maximum lag for MSD calculation (typically half of window)", 
                 foreground='gray', font=(FONT_FAMILY, FONT_SIZE_SMALL-1)).grid(row=row, column=2, sticky=tk.W, padx=5)
        
        # Window overlap
        row += 1
        ttk.Label(param_frame, text="Window Overlap (%):", 
                 font=(FONT_FAMILY, FONT_SIZE_SMALL)).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        overlap_var = tk.IntVar(value=0)
        overlap_combo = ttk.Combobox(param_frame, textvariable=overlap_var, width=8,
                                     values=[0, 25, 50], state="readonly")
        overlap_combo.grid(row=row, column=1, padx=5)
        overlap_combo.current(0)
        ttk.Label(param_frame, text="0% = non-overlapping windows, 50% = 50% overlap", 
                 foreground='gray', font=(FONT_FAMILY, FONT_SIZE_SMALL-1)).grid(row=row, column=2, sticky=tk.W, padx=5)
        
        # ===== RESULTS PREVIEW =====
        preview_frame = ttk.LabelFrame(main_frame, text="3. Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        preview_text = tk.Text(preview_frame, height=8, wrap=tk.WORD, 
                              font=(FONT_FAMILY, FONT_SIZE_SMALL))
        preview_scrollbar = ttk.Scrollbar(preview_frame, command=preview_text.yview)
        preview_text.config(yscrollcommand=preview_scrollbar.set)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        def update_preview(self):
            """更新ROI预览 - 修复版本"""
            if len(self.vertices) < 2:
                return
            
            # 移除旧线
            if self.line_polygon is not None:
                try:
                    self.line_polygon.remove()
                except:
                    pass
                self.line_polygon = None
            
            # 移除旧的填充
            # 获取当前的patch并删除
            for patch in list(self.ax.patches):
                if patch.get_facecolor()[3] < 0.2:  # 半透明的cyan填充
                    try:
                        patch.remove()
                    except:
                        pass
            
            # 如果有3个以上顶点，尝试用平滑曲线连接
            if len(self.vertices) >= 3:
                vertices_closed = self.vertices + [self.vertices[0]]
                xs = [v[0] * self.delta_line_time for v in vertices_closed]
                ys = [v[1] * self.pixel_size_nm / 1000 for v in vertices_closed]
                
                # 尝试使用样条曲线平滑
                try:
                    from scipy import interpolate
                    
                    # ✅ 修复：增加错误处理和数据验证
                    # 检查是否有重复点
                    unique_vertices = []
                    for v in self.vertices:
                        if not unique_vertices or v != unique_vertices[-1]:
                            unique_vertices.append(v)
                    
                    if len(unique_vertices) < 3:
                        # 点数不足，用直线连接
                        raise ValueError("Not enough unique vertices")
                    
                    vertices_closed = unique_vertices + [unique_vertices[0]]
                    xs = [v[0] * self.delta_line_time for v in vertices_closed]
                    ys = [v[1] * self.pixel_size_nm / 1000 for v in vertices_closed]
                    
                    # 检查点是否共线（会导致splprep失败）
                    xs_arr = np.array(xs)
                    ys_arr = np.array(ys)
                    
                    # 用3点通过splprep（较小的k值）
                    if len(xs) == 4:  # 3个顶点 + 闭合点
                        # 对于3个顶点的情况，使用线性插值
                        u = np.linspace(0, 1, 50)
                        xs_smooth = np.interp(u, np.linspace(0, 1, len(xs)), xs)
                        ys_smooth = np.interp(u, np.linspace(0, 1, len(ys)), ys)
                    else:
                        # 4个以上顶点，使用样条
                        # ✅ 关键修复：使用较小的 k 值和合适的 s 参数
                        k = min(3, len(xs) - 2)  # k不能超过点数-2，也不能超过3
                        tck, u = interpolate.splprep([xs, ys], s=1.0, k=k, per=True)
                        u_new = np.linspace(0, 1, 100)
                        xs_smooth, ys_smooth = interpolate.splev(u_new, tck)
                    
                    self.line_polygon, = self.ax.plot(xs_smooth, ys_smooth, 'c-', 
                                                     linewidth=2, alpha=0.7, zorder=8)
                    
                    # 填充区域（半透明）
                    self.ax.fill(xs_smooth, ys_smooth, 'cyan', alpha=0.1, zorder=7)
                    
                except Exception as e:
                    # 如果平滑失败，用直线连接顶点
                    print(f"⚠ Smooth curve failed ({str(e)}), using linear interpolation")
                    
                    # 线性插值
                    xs = [v[0] * self.delta_line_time for v in self.vertices + [self.vertices[0]]]
                    ys = [v[1] * self.pixel_size_nm / 1000 for v in self.vertices + [self.vertices[0]]]
                    
                    self.line_polygon, = self.ax.plot(xs, ys, 'c-', linewidth=2, alpha=0.5, zorder=8)
                    self.ax.fill(xs, ys, 'cyan', alpha=0.1, zorder=7)
            
            else:
                # 少于3个顶点，直接连接线段
                xs = [v[0] * self.delta_line_time for v in self.vertices]
                ys = [v[1] * self.pixel_size_nm / 1000 for v in self.vertices]
                
                self.line_polygon, = self.ax.plot(xs, ys, 'c-', linewidth=2, alpha=0.5, zorder=8)
            
            self.canvas.draw_idle()
        
        # Bind parameter changes to update preview
        for var in [window_var, lag_sampling_var, overlap_var]:
            var.trace('w', lambda *args: update_preview())
        
        # ===== ACTION BUTTONS =====
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        def run_analysis():
            """Execute sliding window MSD analysis"""
            selection = track_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a track")
                return
            
            channel, track_idx, track = track_list[selection[0]]
            
            # Get parameters
            window_duration = window_var.get()
            lag_sampling = lag_sampling_var.get()
            fit_points = fit_points_var.get()
            max_lag = max_lag_var.get()
            overlap_percent = overlap_var.get()
            
            try:
                # Extract positions and times
                time_indices = np.array(track.time_idx)
                pos_indices = np.array(track.coordinate_idx)
                
                if self.roi_coords:
                    x1, y1, _, _ = self.roi_coords
                    time_indices = time_indices + x1
                    pos_indices = pos_indices + y1
                
                times = time_indices * self.delta_line_time
                positions = pos_indices * self.pixel_size_nm / 1000  # μm
                
                # Sampling
                sampled_positions = positions[::lag_sampling]
                sampled_times = times[::lag_sampling]
                sampling_time = lag_sampling * self.delta_line_time
                
                # DNA unit conversion
                BP_TO_NM = DNA_BP_TO_NM
                UM_TO_BP = DNA_UM_TO_BP
                
                # Window parameters
                points_per_window = int(window_duration / sampling_time)
                
                if overlap_percent == 0:
                    step = points_per_window
                else:
                    step = int(points_per_window * (1 - overlap_percent / 100))
                
                n_windows = (len(sampled_positions) - points_per_window) // step + 1
                
                if n_windows < 1:
                    messagebox.showerror("Error", "Track too short for analysis window")
                    return
                
                # Analyze each window
                all_diffusion_constants_um = []
                all_diffusion_constants_bp = []
                all_msd_curves = []
                all_fit_lines = []
                window_info = []
                
                for i in range(n_windows):
                    start_idx = i * step
                    end_idx = start_idx + points_per_window
                    
                    if end_idx > len(sampled_positions):
                        break
                    
                    window_pos = sampled_positions[start_idx:end_idx]
                    window_time = sampled_times[start_idx:end_idx]
                    
                    # Calculate MSD
                    msd_lags = []
                    msd_values = []
                    
                    # 🆕 向量化MSD计算
                    if len(window_pos) >= 2:
                        msd_values_temp, lags_temp = MSDCalculator.calc_msd_vectorized(
                            window_pos, window_time, min(max_lag, len(window_pos) - 1)
                        )
                        msd_values = list(msd_values_temp)
                        msd_lags = list(lags_temp)
                    
                    # Linear fit (first N points)
                    n_fit = min(fit_points, len(msd_values))
                    if n_fit < 3:
                        continue
                    
                    fit_lags = np.array(msd_lags[:n_fit])
                    fit_msd = np.array(msd_values[:n_fit])
                    
                    # MSD = 2D*t for 1D diffusion
                    coeffs = np.polyfit(fit_lags, fit_msd, 1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    
                    D_um2_per_s = slope / 2  # μm²/s
                    D_bp2_per_s = D_um2_per_s * (UM_TO_BP ** 2)  # bp²/s
                    
                    all_diffusion_constants_um.append(D_um2_per_s)
                    all_diffusion_constants_bp.append(D_bp2_per_s)
                    all_msd_curves.append((msd_lags, msd_values))
                    
                    # Fit line for plotting
                    fit_line = slope * np.array(msd_lags) + intercept
                    all_fit_lines.append((msd_lags, fit_line))
                    
                    window_info.append({
                        'index': i + 1,
                        'start_time': window_time[0],
                        'end_time': window_time[-1],
                        'D_um2_s': D_um2_per_s,
                        'D_bp2_s': D_bp2_per_s,
                        'fit_R2': 1 - (np.sum((fit_msd - (slope * fit_lags + intercept))**2) / 
                                      np.sum((fit_msd - np.mean(fit_msd))**2))
                    })
                
                if len(all_diffusion_constants_um) == 0:
                    messagebox.showerror("Error", "No valid windows for analysis")
                    return
                
                # Statistics
                mean_D_um = np.mean(all_diffusion_constants_um)
                std_D_um = np.std(all_diffusion_constants_um)
                mean_D_bp = np.mean(all_diffusion_constants_bp)
                std_D_bp = np.std(all_diffusion_constants_bp)
                
                # ===== CREATE PLOTS =====
                fig = plt.figure(figsize=(16, 10))
                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                
                # Plot 1: All MSD curves (top left, span 2 rows)
                ax1 = fig.add_subplot(gs[0:2, 0])
                colors = plt.cm.rainbow(np.linspace(0, 1, len(all_msd_curves)))
                
                for idx, ((lags, msd), color) in enumerate(zip(all_msd_curves, colors)):
                    ax1.plot(lags, msd, 'o-', color=color, alpha=0.6, 
                            markersize=4, linewidth=1.5, label=f'W{idx+1}')
                
                # Add fit lines
                for (lags, fit_line), color in zip(all_fit_lines, colors):
                    ax1.plot(lags, fit_line, '--', color=color, alpha=0.8, linewidth=2)
                
                ax1.set_xlabel('Δt (s)', fontsize=11, fontweight='bold')
                ax1.set_ylabel('MSD (μm²)', fontsize=11, fontweight='bold')
                ax1.set_title(f'MSD Curves - All {len(all_msd_curves)} Windows', 
                             fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                if len(all_msd_curves) <= 15:
                    ax1.legend(fontsize=7, ncol=2)
                
                # Plot 2: D distribution (μm²/s)
                ax2 = fig.add_subplot(gs[0, 1])
                # 🔧 修复：确保 bins 至少为 5
                n_bins = max(5, min(20, len(all_diffusion_constants_um) // 2))
                if n_bins <= 0:  # 双重保险
                    n_bins = 5
                
                counts, bins, patches = ax2.hist(all_diffusion_constants_um, bins=n_bins, 
                                                 color='steelblue', alpha=0.7, edgecolor='black')
                ax2.axvline(mean_D_um, color='red', linestyle='--', linewidth=2.5, 
                           label=f'Mean: {mean_D_um:.2e}')
                ax2.axvline(mean_D_um + std_D_um, color='orange', linestyle=':', linewidth=2)
                ax2.axvline(mean_D_um - std_D_um, color='orange', linestyle=':', linewidth=2)
                ax2.set_xlabel('D (μm²/s)', fontsize=10, fontweight='bold')
                ax2.set_ylabel('Count', fontsize=10, fontweight='bold')
                ax2.set_title('Diffusion Constant Distribution (μm²/s)', fontsize=11, fontweight='bold')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Plot 3: D distribution (bp²/s)
                ax3 = fig.add_subplot(gs[0, 2])
                
                # 使用相同的 bins（已经在 Plot 2 中计算并修复）
                counts, bins, patches = ax3.hist(all_diffusion_constants_bp, bins=n_bins,
                                                 color='darkgreen', alpha=0.7, edgecolor='black')
                ax3.axvline(mean_D_bp, color='red', linestyle='--', linewidth=2.5, 
                           label=f'Mean: {mean_D_bp:.2e}')
                ax3.axvline(mean_D_bp + std_D_bp, color='orange', linestyle=':', linewidth=2)
                ax3.axvline(mean_D_bp - std_D_bp, color='orange', linestyle=':', linewidth=2)
                ax3.set_xlabel('D (bp²/s)', fontsize=10, fontweight='bold')
                ax3.set_ylabel('Count', fontsize=10, fontweight='bold')
                ax3.set_title('Diffusion Constant Distribution (bp²/s)', fontsize=11, fontweight='bold')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3, axis='y')
                
                               # ===== Plot 4: Probability density (μm²/s) =====
                ax4 = fig.add_subplot(gs[1, 1])
                
                n_data = len(all_diffusion_constants_um)
                
                if n_data >= 2:
                    # 有足够数据点，使用 KDE
                    from scipy import stats
                    try:
                        kde = stats.gaussian_kde(all_diffusion_constants_um)
                        x_range = np.linspace(min(all_diffusion_constants_um) * 0.9, 
                                             max(all_diffusion_constants_um) * 1.1, 200)
                        ax4.plot(x_range, kde(x_range), 'b-', linewidth=3, label='KDE')
                        ax4.fill_between(x_range, kde(x_range), alpha=0.3)
                        ax4.axvline(mean_D_um, color='red', linestyle='--', linewidth=2, label='Mean')
                    except:
                        # KDE 失败，回退到直方图
                        ax4.hist(all_diffusion_constants_um, bins=10, density=True,
                                color='blue', alpha=0.5, edgecolor='black')
                        ax4.axvline(mean_D_um, color='red', linestyle='--', linewidth=2, label='Mean')
                else:
                    # 只有 1 个数据点
                    ax4.axvline(all_diffusion_constants_um[0], color='blue', linestyle='-', 
                               linewidth=4, label=f'D = {all_diffusion_constants_um[0]:.2e}')
                    ax4.set_ylim(0, 1)
                    ax4.text(0.5, 0.5, 'Insufficient data for KDE\n(need ≥ 2 windows)',
                            transform=ax4.transAxes, ha='center', va='center',
                            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax4.set_xlabel('D (μm²/s)', fontsize=10, fontweight='bold')
                ax4.set_ylabel('Probability Density', fontsize=10, fontweight='bold')
                ax4.set_title('Probability Density (μm²/s)', fontsize=11, fontweight='bold')
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
                
                # ===== Plot 5: Probability density (bp²/s) =====
                ax5 = fig.add_subplot(gs[1, 2])
                
                if n_data >= 2:
                    from scipy import stats
                    try:
                        kde_bp = stats.gaussian_kde(all_diffusion_constants_bp)
                        x_range_bp = np.linspace(min(all_diffusion_constants_bp) * 0.9,
                                                max(all_diffusion_constants_bp) * 1.1, 200)
                        ax5.plot(x_range_bp, kde_bp(x_range_bp), 'g-', linewidth=3, label='KDE')
                        ax5.fill_between(x_range_bp, kde_bp(x_range_bp), alpha=0.3, color='green')
                        ax5.axvline(mean_D_bp, color='red', linestyle='--', linewidth=2, label='Mean')
                    except:
                        ax5.hist(all_diffusion_constants_bp, bins=10, density=True,
                                color='green', alpha=0.5, edgecolor='black')
                        ax5.axvline(mean_D_bp, color='red', linestyle='--', linewidth=2, label='Mean')
                else:
                    ax5.axvline(all_diffusion_constants_bp[0], color='green', linestyle='-',
                               linewidth=4, label=f'D = {all_diffusion_constants_bp[0]:.2e}')
                    ax5.set_ylim(0, 1)
                    ax5.text(0.5, 0.5, 'Insufficient data for KDE\n(need ≥ 2 windows)',
                            transform=ax5.transAxes, ha='center', va='center',
                            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax5.set_xlabel('D (bp²/s)', fontsize=10, fontweight='bold')
                ax5.set_ylabel('Probability Density', fontsize=10, fontweight='bold')
                ax5.set_title('Probability Density (bp²/s)', fontsize=11, fontweight='bold')
                ax5.legend(fontsize=8)
                ax5.grid(True, alpha=0.3)
                
                # Plot 6: D values vs window index
                ax6 = fig.add_subplot(gs[2, 0])
                window_indices = [w['index'] for w in window_info]
                D_values_um = [w['D_um2_s'] for w in window_info]
                ax6.plot(window_indices, D_values_um, 'o-', color='navy', markersize=6, linewidth=2)
                ax6.axhline(mean_D_um, color='red', linestyle='--', linewidth=2, label='Mean')
                ax6.fill_between(window_indices, mean_D_um - std_D_um, mean_D_um + std_D_um, 
                                alpha=0.2, color='red', label='±1σ')
                ax6.set_xlabel('Window Index', fontsize=10, fontweight='bold')
                ax6.set_ylabel('D (μm²/s)', fontsize=10, fontweight='bold')
                ax6.set_title('D vs Window Index', fontsize=11, fontweight='bold')
                ax6.legend(fontsize=8)
                ax6.grid(True, alpha=0.3)
                
                # Plot 7: Box plot
                ax7 = fig.add_subplot(gs[2, 1])
                bp = ax7.boxplot([all_diffusion_constants_um], widths=0.6, patch_artist=True,
                                labels=['All Windows'])
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][0].set_edgecolor('navy')
                bp['medians'][0].set_color('red')
                bp['medians'][0].set_linewidth(2)
                ax7.set_ylabel('D (μm²/s)', fontsize=10, fontweight='bold')
                ax7.set_title('Distribution Summary', fontsize=11, fontweight='bold')
                ax7.grid(True, alpha=0.3, axis='y')
                
                # Plot 8: Text summary
                ax8 = fig.add_subplot(gs[2, 2])
                ax8.axis('off')
                
                summary_text = f"=== Analysis Summary ===\n\n"
                summary_text += f"Track: {channel.capitalize()} Track {track_idx+1}\n"
                summary_text += f"Total Windows: {len(all_diffusion_constants_um)}\n"
                summary_text += f"Window Duration: {window_duration}s\n"
                summary_text += f"Overlap: {overlap_percent}%\n\n"
                
                summary_text += f"--- Diffusion Constants ---\n"
                summary_text += f"Mean D: {mean_D_um:.3e} μm²/s\n"
                summary_text += f"        {mean_D_bp:.3e} bp²/s\n\n"
                summary_text += f"Std D:  {std_D_um:.3e} μm²/s\n"
                summary_text += f"        {std_D_bp:.3e} bp²/s\n\n"
                summary_text += f"CV:     {(std_D_um/mean_D_um)*100:.1f}%\n\n"
                
                summary_text += f"--- Range ---\n"
                summary_text += f"Min: {min(all_diffusion_constants_um):.3e} μm²/s\n"
                summary_text += f"Max: {max(all_diffusion_constants_um):.3e} μm²/s\n\n"
                
                summary_text += f"DNA Conversion:\n"
                summary_text += f"1 μm = {UM_TO_BP:.1f} bp"
                
                ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                
                plt.suptitle(f'Sliding Window MSD Analysis - {channel.capitalize()} Track {track_idx+1}',
                            fontsize=14, fontweight='bold')
                
                plt.show()
                
                # ... (继续导出对话框代码，在下一条消息中提供)
                # ===== EXPORT DIALOG =====
                export_dialog = tk.Toplevel(dialog)
                export_dialog.title("💾 Export Results")
                export_dialog.geometry("700x650")
                export_dialog.transient(dialog)
                export_dialog.grab_set()
                
                export_frame = ttk.Frame(export_dialog, padding=10)
                export_frame.pack(fill=tk.BOTH, expand=True)
                
                ttk.Label(export_frame, text="Analysis Complete!", 
                         font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')).pack(anchor=tk.W, pady=5)
                
                # Results text
                text_frame = ttk.Frame(export_frame)
                text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
                
                text_scrollbar = ttk.Scrollbar(text_frame)
                text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                result_text = tk.Text(text_frame, height=20, wrap=tk.WORD,
                                     font=(FONT_FAMILY, FONT_SIZE_SMALL),
                                     yscrollcommand=text_scrollbar.set)
                result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                text_scrollbar.config(command=result_text.yview)
                
                # Format detailed results
                detailed_text = f"=== Sliding Window MSD Analysis Results ===\n\n"
                detailed_text += f"Track: {channel.capitalize()} Track {track_idx+1}\n"
                detailed_text += f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                detailed_text += f"--- Parameters ---\n"
                detailed_text += f"Window Duration: {window_duration} s\n"
                detailed_text += f"Lag Sampling: every {lag_sampling} line positions ({sampling_time:.3f} s)\n"
                detailed_text += f"Fit Points: {fit_points}\n"
                detailed_text += f"Max Lag: {max_lag}\n"
                detailed_text += f"Window Overlap: {overlap_percent}%\n"
                detailed_text += f"Total Windows: {len(window_info)}\n\n"
                
                detailed_text += f"--- Statistics (μm²/s) ---\n"
                detailed_text += f"Mean:   {mean_D_um:.6e}\n"
                detailed_text += f"Std:    {std_D_um:.6e}\n"
                detailed_text += f"CV:     {(std_D_um/mean_D_um)*100:.2f}%\n"
                detailed_text += f"Min:    {min(all_diffusion_constants_um):.6e}\n"
                detailed_text += f"Max:    {max(all_diffusion_constants_um):.6e}\n"
                detailed_text += f"Median: {np.median(all_diffusion_constants_um):.6e}\n\n"
                
                detailed_text += f"--- Statistics (bp²/s) ---\n"
                detailed_text += f"Mean:   {mean_D_bp:.6e}\n"
                detailed_text += f"Std:    {std_D_bp:.6e}\n"
                detailed_text += f"CV:     {(std_D_bp/mean_D_bp)*100:.2f}%\n"
                detailed_text += f"Min:    {min(all_diffusion_constants_bp):.6e}\n"
                detailed_text += f"Max:    {max(all_diffusion_constants_bp):.6e}\n"
                detailed_text += f"Median: {np.median(all_diffusion_constants_bp):.6e}\n\n"
                
                detailed_text += f"--- Individual Windows ---\n"
                for w in window_info:
                    detailed_text += f"\nWindow {w['index']}: [{w['start_time']:.1f}-{w['end_time']:.1f}]s\n"
                    detailed_text += f"  D = {w['D_um2_s']:.6e} μm²/s ({w['D_bp2_s']:.6e} bp²/s)\n"
                    detailed_text += f"  Fit R² = {w['fit_R2']:.4f}\n"
                
                detailed_text += f"\n\nDNA Conversion Factor: 1 μm = {UM_TO_BP:.1f} bp (B-form DNA)\n"
                
                result_text.insert('1.0', detailed_text)
                result_text.config(state='disabled')
                
                # Export options
                export_opt_frame = ttk.LabelFrame(export_frame, text="Export Options", padding=10)
                export_opt_frame.pack(fill=tk.X, pady=10)
                
                export_plot_var = tk.BooleanVar(value=True)
                export_csv_var = tk.BooleanVar(value=True)
                
                ttk.Checkbutton(export_opt_frame, text="📈 Analysis Plot (PNG)",
                               variable=export_plot_var).pack(anchor=tk.W, pady=2)
                ttk.Checkbutton(export_opt_frame, text="📊 Diffusion Constants (CSV)",
                               variable=export_csv_var).pack(anchor=tk.W, pady=2)
                
                # ===== 🆕 导出函数 =====
                def export_results():
                    """导出结果到 Excel 文件（多个 sheet）"""
                    if not self.current_file_path:
                        messagebox.showerror("Error", "No file path available")
                        return
                    
                    base_name = Path(self.current_file_path).stem
                    output_dir = Path(self.current_file_path).parent
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    export_log = ""
                    
                    try:
                        # 导出PNG图表
                        if export_plot_var.get():
                            plot_path = output_dir / f"{base_name}_{channel}_Track{track_idx+1}_RateAnalysis_{timestamp}.png"
                            fig_rate.savefig(str(plot_path), dpi=300, bbox_inches='tight')
                            export_log += f"✅ Plot: {plot_path.name}\n"
                        
                        # 导出Excel数据
                        if export_csv_var.get():
                            # 🆕 创建 Excel 工作簿
                            from openpyxl import Workbook
                            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
                            from openpyxl.utils.dataframe import dataframe_to_rows
                            
                            excel_path = output_dir / f"{base_name}_{channel}_Track{track_idx+1}_Analysis_{timestamp}.xlsx"
                            
                            wb = Workbook()
                            wb.remove(wb.active)  # 删除默认 sheet
                            
                            raw_data = {
                                'Time_s': [],
                                'Time_Normalized_s': [],
                                'Position_um': [],
                                'Position_Normalized_um': [],
                                'Segment': []
                            }
                            
                            for seg in all_segment_data:
                                raw_data['Time_s'].extend(seg['times'])
                                raw_data['Time_Normalized_s'].extend(seg['times'] - time_zero)
                                raw_data['Position_um'].extend(seg['positions'])
                                raw_data['Position_Normalized_um'].extend(seg['positions'] - position_zero)
                                raw_data['Segment'].extend([seg['label']] * len(seg['times']))
                            
                            raw_df = pd.DataFrame(raw_data)
                            ws_raw = wb.create_sheet("RawData")
                            
                            # 写入数据
                            for r_idx, row in enumerate(dataframe_to_rows(raw_df, index=False, header=True), 1):
                                for c_idx, value in enumerate(row, 1):
                                    cell = ws_raw.cell(row=r_idx, column=c_idx, value=value)
                                    
                                    # 🎨 格式化标题行
                                    if r_idx == 1:
                                        cell.font = Font(bold=True, color="FFFFFF")
                                        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            # 调整列宽
                            ws_raw.column_dimensions['A'].width = 15
                            ws_raw.column_dimensions['B'].width = 15
                            ws_raw.column_dimensions['C'].width = 15
                            
                            export_log += f"✅ Sheet 'RawData' created\n"
                            
                            
                            # ===== SHEET 2: Fitted Data (拟合直线数据) =====
                            fitted_data = {
                                'Time_s': [],
                                'Time_Normalized_s': [],
                                'Position_um': [],
                                'Position_Normalized_um': [],
                                'Fitted_Position_um': [],
                                'Fitted_Position_Normalized_um': [],
                                'Segment': [],
                                'Fit_Rate_um_s': [],
                                'Fit_Rate_bp_s': [],
                                'R_squared': []
                            }
                            
                            for seg in all_segment_data:
                                fitted_data['Time_s'].extend(seg['times'])
                                fitted_data['Time_Normalized_s'].extend(seg['times'] - time_zero)
                                fitted_data['Position_um'].extend(seg['positions'])
                                fitted_data['Position_Normalized_um'].extend(seg['positions'] - position_zero)
                                fitted_data['Fitted_Position_um'].extend(seg['y_fitted'])
                                fitted_data['Fitted_Position_Normalized_um'].extend(seg['y_fitted'] - position_zero)
                                fitted_data['Segment'].extend([seg['label']] * len(seg['times']))
                                fitted_data['Fit_Rate_um_s'].extend([abs(seg['fit_rate'])] * len(seg['times']))
                                fitted_data['Fit_Rate_bp_s'].extend([abs(seg['fit_rate_bp'])] * len(seg['times']))
                                fitted_data['R_squared'].extend([seg['r_squared']] * len(seg['times']))
                            
                            fitted_df = pd.DataFrame(fitted_data)
                            ws_fitted = wb.create_sheet("FittedData")
                            
                            # 写入数据
                            for r_idx, row in enumerate(dataframe_to_rows(fitted_df, index=False, header=True), 1):
                                for c_idx, value in enumerate(row, 1):
                                    cell = ws_fitted.cell(row=r_idx, column=c_idx, value=value)
                                    
                                    # 🎨 格式化标题行
                                    if r_idx == 1:
                                        cell.font = Font(bold=True, color="FFFFFF")
                                        cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                                    else:
                                        # 数据行：数字格式
                                        if c_idx <= 2:  # Time 和 Position
                                            cell.number_format = '0.0000'
                                        elif c_idx in [4, 5]:  # Rate 列
                                            cell.number_format = '0.00'
                                        elif c_idx == 6:  # R²
                                            cell.number_format = '0.0000'
                            
                            # 调整列宽
                            ws_fitted.column_dimensions['A'].width = 15
                            ws_fitted.column_dimensions['B'].width = 20
                            ws_fitted.column_dimensions['C'].width = 15
                            ws_fitted.column_dimensions['D'].width = 18
                            ws_fitted.column_dimensions['E'].width = 18
                            ws_fitted.column_dimensions['F'].width = 15
                            
                            export_log += f"✅ Sheet 'FittedData' created\n"
                            
                            
                            # ===== SHEET 3: Statistics (统计表格 - 用于制作柱状图) =====
                            stats_rows = []
                            for seg in all_segment_data:
                                stats_rows.append({
                                    'Segment': seg['label'],
                                    'Start_s': seg['start'],
                                    'End_s': seg['end'],
                                    'Duration_s': seg['duration'],
                                    'N_Points': seg['n_points'],
                                    'Displacement_um': abs(seg['displacement']),
                                    'Displacement_bp': abs(seg['displacement_bp']),
                                    'Simple_Rate_um_s': abs(seg['simple_rate']),
                                    'Simple_Rate_bp_s': abs(seg['simple_rate_bp']),
                                    'Fit_Rate_um_s': abs(seg['fit_rate']),  # ⭐ 主要指标
                                    'Fit_Rate_bp_s': abs(seg['fit_rate_bp']),
                                    'R_squared': seg['r_squared'],     # ⭐ 质量指标
                                    'Std_Error_um': seg['stderr']      # ⭐ 误差条
                                })
                            
                            stats_df = pd.DataFrame(stats_rows)
                            ws_stats = wb.create_sheet("Statistics")
                            
                            # 写入数据
                            for r_idx, row in enumerate(dataframe_to_rows(stats_df, index=False, header=True), 1):
                                for c_idx, value in enumerate(row, 1):
                                    cell = ws_stats.cell(row=r_idx, column=c_idx, value=value)
                                    
                                    # 🎨 格式化标题行
                                    if r_idx == 1:
                                        cell.font = Font(bold=True, color="FFFFFF")
                                        cell.fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
                                        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                                    else:
                                        # 数据行：数字格式
                                        if c_idx in [2, 3, 4]:  # Time 列
                                            cell.number_format = '0.0000'
                                        elif c_idx == 5:  # N_Points
                                            cell.number_format = '0'
                                        elif c_idx in [6, 7]:  # Displacement
                                            cell.number_format = '0.0000'
                                        elif c_idx in [8, 9, 10, 11]:  # Rate
                                            cell.number_format = '0.00'
                                        elif c_idx == 12:  # R²
                                            cell.number_format = '0.0000'
                                        elif c_idx == 13:  # Std_Error
                                            cell.number_format = '0.0000'
                                        
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            # 调整列宽
                            for col_num, col_title in enumerate(stats_df.columns, 1):
                                ws_stats.column_dimensions[chr(64 + col_num)].width = 18
                            
                            export_log += f"✅ Sheet 'Statistics' created\n"
                            
                            
                            # ===== SHEET 4: Summary (汇总信息) =====
                            summary_data = {
                                'Parameter': [
                                    'Channel',
                                    'Track Index',
                                    'Total Points',
                                    'Total Duration (s)',
                                    'Segments Analyzed',
                                    'Total Displacement (μm)',
                                    'Total Displacement (bp)',
                                    'Average Rate - Simple (μm/s)',
                                    'Average Rate - Simple (bp/s)',
                                    'Average Rate - Fit (μm/s)',
                                    'Average Rate - Fit (bp/s)',
                                    'Average R² Score',
                                    'Min Rate (bp/s)',
                                    'Max Rate (bp/s)',
                                    'Std Dev Rate (bp/s)',
                                    'DNA Conversion Factor (bp/μm)',
                                    'Analysis Date',
                                    'Time Zero (s)',
                                    'Position Zero (μm)'
                                    
                                ],
                                'Value': [
                                    channel.capitalize(),
                                    track_idx + 1,
                                    len(positions),
                                    times[-1] - times[0],
                                    len(all_segment_data),
                                    total_displacement,
                                    total_displacement * UM_TO_BP,
                                    abs(avg_simple_rate),
                                    abs(avg_simple_rate) * UM_TO_BP,
                                    abs(avg_fit_rate),
                                    abs(avg_fit_rate) * UM_TO_BP,
                                    avg_r2,
                                    min([abs(r) for r in [seg['fit_rate'] for seg in all_segment_data]]) * UM_TO_BP if all_segment_data else 0,
                                    max([abs(r) for r in [seg['fit_rate'] for seg in all_segment_data]]) * UM_TO_BP if all_segment_data else 0,
                                    np.std([abs(r) for r in [seg['fit_rate'] for seg in all_segment_data]]) * UM_TO_BP if all_segment_data else 0,
                                    UM_TO_BP,
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    time_zero,
                                    position_zero
                                ]
                            }
                            
                            summary_df = pd.DataFrame(summary_data)
                            ws_summary = wb.create_sheet("Summary", 0)  # 放在第一个位置
                            
                            # 写入数据
                            for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
                                for c_idx, value in enumerate(row, 1):
                                    cell = ws_summary.cell(row=r_idx, column=c_idx, value=value)
                                    
                                    # 🎨 格式化标题行
                                    if r_idx == 1:
                                        cell.font = Font(bold=True, color="FFFFFF")
                                        cell.fill = PatternFill(start_color="203864", end_color="203864", fill_type="solid")
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                                    else:
                                        # 左边界：参数名称
                                        if c_idx == 1:
                                            cell.font = Font(bold=True)
                                            cell.fill = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")
                                        # 右边界：数值
                                        else:
                                            # 数字格式
                                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                                if isinstance(value, float):
                                                    cell.number_format = '0.00'
                                        
                                        cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            # 调整列宽
                            ws_summary.column_dimensions['A'].width = 30
                            ws_summary.column_dimensions['B'].width = 30
                            
                            export_log += f"✅ Sheet 'Summary' created\n"
                            
                            
                            # ===== 保存 Excel 文件 =====
                            wb.save(str(excel_path))
                            export_log += f"\n✅ Excel file saved: {excel_path.name}\n"
                        
                        # 显示成功消息
                        success_msg = (
                            f"✅ Export Successful!\n\n"
                            f"📁 Location: {output_dir}\n\n"
                            f"📊 Excel File Structure:\n"
                            f"  📄 Summary - 汇总统计信息\n"
                            f"  📈 RawData - 原始轨迹数据 (含归零列)\n"
                            f"  📉 FittedData - 线性拟合数据 (含归零列)\n"
                            f"  📊 Statistics - 分段统计表格\n\n"
                            f"🔄 归零数据说明:\n"
                            f"  • Time_Normalized_s = Time_s - {time_zero:.4f}\n"
                            f"  • Position_Normalized_um = Position_um - {position_zero:.4f}\n\n"
                            f"💡 OriginLab 操作步骤:\n"
                            f"  1. 打开 RawData sheet → 复制到 OriginLab\n"
                            f"     制作散点图 (X=Time_Normalized_s, Y=Position_Normalized_um)\n"
                            f"  2. 打开 FittedData sheet → 添加为叠加线图\n"
                            f"  3. 打开 Statistics sheet → 制作柱状图\n"
                            f"     X=Segment, Y=Fit_Rate_bp_s\n"
                            f"     Error Bar=Std_Error_um"
                        )
                        
                        messagebox.showinfo("Export Complete", success_msg)
                        self.log(export_log)
                        
                        export_dialog.destroy()
                        
                    except Exception as e:
                        messagebox.showerror("Export Error", f"Failed:\n{str(e)}")
                        self.log(f"❌ Export failed: {str(e)}")
                        import traceback
                        traceback.print_exc()
                btn_frame = ttk.Frame(export_frame)
                btn_frame.pack(fill=tk.X, pady=10)
                
                ttk.Button(btn_frame, text="💾 Export Results",
                          command=export_results).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
                ttk.Button(btn_frame, text="Close",
                          command=export_dialog.destroy).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
                
                export_dialog.update_idletasks()
                x = dialog.winfo_x() + (dialog.winfo_width() - export_dialog.winfo_width()) // 2
                y = dialog.winfo_y() + (dialog.winfo_height() - export_dialog.winfo_height()) // 2
                export_dialog.geometry(f"+{x}+{y}")
                
                
            except Exception as e:
                messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
                import traceback
                traceback.print_exc()
        
        ttk.Button(btn_frame, text="🔬 Run Analysis", 
                  command=run_analysis).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X, ipady=8)
        ttk.Button(btn_frame, text="Close", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Center dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")


# ========== MAIN ==========  

if __name__ == "__main__":  
    root = tk.Tk()  
    app = KymoTrackerGUI(root)  
    root.mainloop()
                    
