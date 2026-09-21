"""UI组件工具模块"""
import platform
import tkinter as tk
from tkinter import ttk


class UIHelper:
    """UI辅助工具类"""
    
    @staticmethod
    def get_system_font():
        """获取系统字体"""
        system = platform.system()
        if system == 'Darwin':
            return ('Helvetica', 10)
        elif system == 'Windows':
            return ('Arial', 10)
        else:
            return ('DejaVu Sans', 10)
    
    @staticmethod
    def create_collapsible_panel(parent, title):
        """创建可折叠面板"""
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        return frame
    
    @staticmethod
    def bind_mousewheel(canvas, system):
        """绑定鼠标滚轮事件"""
        def _on_mousewheel(event):
            if system == 'Darwin':
                canvas.yview_scroll(int(-1 * event.delta), "units")
            elif system == 'Windows':
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")

        if system in ['Windows', 'Darwin']:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        if system == 'Linux':
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)
