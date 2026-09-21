"""程序入口"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from main_gui import SignalFilterGUI


def main():
    """主函数"""
    root = tk.Tk()
    root.withdraw()

    csv_file = filedialog.askopenfilename(
        title="选择CSV文件",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Processed CSV files", "*_processed.csv"),
            ("All files", "*.*")
        ]
    )

    if not csv_file:
        print("未选择文件")
        return

    # 导入数据加载器（这里假设你有data_loader模块）
    try:
        from data_loader import ColocalizationDataLoader
    except ImportError:
        messagebox.showerror("错误", "无法导入 data_loader 模块")
        return

    loader = ColocalizationDataLoader(csv_file)
    csv_directory = os.path.dirname(os.path.abspath(csv_file))

    try:
        print("正在加载数据...")
        try:
            loader.load_processed_data()
            print("成功加载处理后的数据")
        except (KeyError, ValueError, AttributeError):
            print("检测到原始数据，正在处理...")
            loader.load_and_process()

        print(f"数据加载成功！ROI 数量: {len(loader.processed_data['ROI_ID'].unique())}")

    except Exception as e:
        print(f"加载数据失败: {e}")
        messagebox.showerror("错误", f"加载数据失败:\n{str(e)}")
        root.destroy()
        return

    # 创建主窗口
    filter_window = tk.Toplevel(root)
    filter_gui = SignalFilterGUI(filter_window, loader, default_directory=csv_directory)

    root.mainloop()


if __name__ == "__main__":
    main()
