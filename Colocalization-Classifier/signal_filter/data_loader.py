# data_loader.py - Windows 和 macOS 完全兼容版本 v2.0
import pandas as pd
import os
import re
import sys


class ColocalizationDataLoader:
    """单分子共定位数据读取与预处理模块 - 跨平台版本"""
    
    def __init__(self, csv_path):
        # 规范化路径
        self.csv_path = os.path.normpath(csv_path)
        self.raw_data = None
        self.processed_data = None
        self.is_three_color = False
        self.channels = []
        self.output_path = None
        self.time_column = None  # 新增：时间列名
        
    def load_and_process(self, output_path=None):
        """读取、处理并保存数据 - 跨平台兼容"""
        # 尝试多种编码读取
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                self.raw_data = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"✓ 成功使用 {encoding} 编码读取文件")
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise ValueError("无法识别文件编码，请检查CSV文件")
        
        self._identify_channels()
        self._process_data()
        
        if output_path is None:
            output_path = self._generate_output_path()
        
        self.output_path = output_path
        
        # 跨平台保存
        if sys.platform == 'win32':
            # Windows 使用 utf-8-sig（Excel 兼容）
            self.processed_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
            # macOS/Linux 使用 utf-8
            self.processed_data.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✓ 数据已保存至: {output_path}")
        
        return self.processed_data
    
    def _identify_time_column(self):
        """自动识别时间列"""
        possible_columns = ['Time [s]', 'Time [m:s]', 'Time']
        
        for col in possible_columns:
            if col in self.raw_data.columns:
                self.time_column = col
                print(f"✓ 检测到时间列: {col}")
                return
        
        raise ValueError(f"未找到时间列，期待以下之一: {possible_columns}")
    
    def _identify_channels(self):
        """自动识别数据类型（双色/三色）"""
        columns = self.raw_data.columns.tolist()
        
        # 先识别时间列
        self._identify_time_column()
        
        required_cols = ['ROI ID', 
                        'Intensity(532)', 'Background(532)',
                        'Intensity(638)', 'Background(638)']
        
        missing_cols = [col for col in required_cols if col not in columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        has_488 = 'Intensity(488)' in columns and 'Background(488)' in columns
        
        if has_488:
            self.is_three_color = True
            self.channels = ['488', '532', '638']
        else:
            self.is_three_color = False
            self.channels = ['532', '638']
    
    def _time_to_seconds(self, time_str):
        """
        智能时间解析 - 多格式兼容
        
        支持格式:
        1. 纯秒数: "0.399", "1.234" (适配 Time [s] 列)
        2. 分:秒.十分之一秒: "1:30.5", "00:01.3" (适配 Time [m:s] 列)
        3. 分:秒: "1:30", "00:01"
        
        Parameters:
        -----------
        time_str : str or float
            时间字符串或数值
            
        Returns:
        --------
        float
            转换后的秒数
            
        Examples:
        ---------
        >>> _time_to_seconds("0.399")
        0.399
        >>> _time_to_seconds("1:30.5")
        90.5
        >>> _time_to_seconds("00:01.3")
        1.3
        """
        time_str = str(time_str).strip()
        
        # 策略1: 直接转浮点数（最快，适配纯秒格式）
        try:
            seconds = float(time_str)
            return seconds
        except ValueError:
            pass  # 不是纯数字，继续尝试其他格式
        
        # 策略2: 匹配 mm:ss.d 格式
        match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            deciseconds = int(match.group(3))
            total_seconds = minutes * 60 + seconds + deciseconds / 10.0
            return total_seconds
        
        # 策略3: 匹配 mm:ss 格式（无小数）
        match = re.match(r'(\d+):(\d+)$', time_str)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
        
        # 所有格式都失败
        raise ValueError(
            f"无法解析时间格式: '{time_str}'\n"
            f"支持的格式:\n"
            f"  - 纯秒数: 0.399, 1.234\n"
            f"  - 分:秒.十分之一秒: 1:30.5, 00:01.3\n"
            f"  - 分:秒: 1:30, 00:01"
        )
    
    def _process_data(self):
        """处理数据：转换时间、计算净强度、重组数据"""
        # 使用识别的时间列
        time_sec = self.raw_data[self.time_column].apply(self._time_to_seconds)
        
        net_532 = self.raw_data['Intensity(532)'] - self.raw_data['Background(532)']
        net_638 = self.raw_data['Intensity(638)'] - self.raw_data['Background(638)']
        
        new_data = {
            'Time_sec': time_sec,
            'ROI_ID': self.raw_data['ROI ID'],
            'Net_532': net_532,
            'Net_638': net_638
        }
        
        if self.is_three_color:
            net_488 = self.raw_data['Intensity(488)'] - self.raw_data['Background(488)']
            new_data['Net_488'] = net_488
        
        self.processed_data = pd.DataFrame(new_data)
    
    def _generate_output_path(self):
        """自动生成输出文件路径（在原文件同一目录） - 跨平台兼容"""
        directory = os.path.dirname(self.csv_path)
        filename = os.path.basename(self.csv_path)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_processed.csv"
        
        output_path = os.path.join(directory, output_filename)
        
        # 确保目录存在
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        return output_path
    
    def get_roi_list(self):
        """获取所有ROI ID列表"""
        if self.processed_data is None:
            raise ValueError("请先运行 load_and_process() 方法")
        return sorted(self.processed_data['ROI_ID'].unique().tolist())
    
    def get_roi_data(self, roi_id):
        """
        获取特定ROI的数据
        
        Parameters:
        -----------
        roi_id : int
            ROI编号
            
        Returns:
        --------
        pd.DataFrame
            该ROI的所有时间点数据
        """
        if self.processed_data is None:
            raise ValueError("请先运行 load_and_process() 方法")
        
        return self.processed_data[self.processed_data['ROI_ID'] == roi_id].copy()
    
    def get_summary(self):
        """获取数据摘要信息"""
        if self.processed_data is None:
            raise ValueError("请先运行 load_and_process() 方法")
        
        summary = {
            '数据类型': '三色' if self.is_three_color else '双色',
            '通道': self.channels,
            '时间列格式': self.time_column,
            'ROI总数': self.processed_data['ROI_ID'].nunique(),
            '总帧数': len(self.processed_data),
            '时间范围': f"{self.processed_data['Time_sec'].min():.1f} - {self.processed_data['Time_sec'].max():.1f} 秒"
        }
        
        # 每个通道的强度统计
        for channel in self.channels:
            col_name = f'Net_{channel}'
            summary[f'{channel}nm 平均强度'] = f"{self.processed_data[col_name].mean():.1f}"
            summary[f'{channel}nm 强度范围'] = f"{self.processed_data[col_name].min():.1f} - {self.processed_data[col_name].max():.1f}"
        
        return summary
    
    def is_processed_data(self):
        """检查CSV是否是已处理的数据（含净信号列）"""
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'utf-8-sig', 'gbk']
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.csv_path, nrows=1, encoding=encoding)
                    return 'Net_532' in df.columns and 'Net_638' in df.columns
                except:
                    continue
            return False
        except:
            return False
    
    def load_processed_data(self):
        """直接加载已处理的数据（用于筛选功能） - 跨平台兼容"""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']
        
        for encoding in encodings:
            try:
                self.processed_data = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"✓ 成功使用 {encoding} 编码读取已处理数据")
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise ValueError(f"无法读取文件: {self.csv_path}")
        
        self.output_path = self.csv_path
        
        # 验证必需列
        required_cols = ['Time_sec', 'ROI_ID', 'Net_532', 'Net_638']
        missing_cols = [col for col in required_cols if col not in self.processed_data.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        # 自动识别是否为三色数据
        self.is_three_color = 'Net_488' in self.processed_data.columns
        self.channels = ['488', '532', '638'] if self.is_three_color else ['532', '638']


# ============================================================================
# 独立测试代码 - 跨平台兼容版本
# ============================================================================
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    print("=" * 60)
    print("单分子共定位数据加载器 - 测试模式")
    print("=" * 60)
    
    root = tk.Tk()
    root.withdraw()
    
    # macOS 特殊处理
    if sys.platform == 'darwin':
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
    
    csv_file = filedialog.askopenfilename(
        title="选择CSV文件",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if csv_file:
        try:
            loader = ColocalizationDataLoader(csv_file)
            
            # 测试 is_processed_data
            if loader.is_processed_data():
                print("\n✓ 检测到处理后的数据，直接加载...")
                loader.load_processed_data()
            else:
                print("\n✓ 检测到原始数据，开始处理...")
                loader.load_and_process()
            
            # 显示摘要
            summary = loader.get_summary()
            print("\n" + "=" * 60)
            print("数据摘要:")
            print("=" * 60)
            for key, value in summary.items():
                print(f"  {key}: {value}")
            
            # 显示前5个ROI
            roi_list = loader.get_roi_list()
            print(f"\n前5个ROI编号: {roi_list[:5]}")
            
            # 显示第一个ROI的前3行数据
            if roi_list:
                first_roi = roi_list[0]
                roi_data = loader.get_roi_data(first_roi)
                print(f"\nROI #{first_roi} 的前3行数据:")
                print(roi_data.head(3))
            
            print(f"\n✓ 输出文件: {loader.output_path}")
            print("=" * 60)
            print("✓ 测试完成！")
            
        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n未选择文件，退出测试")
    
    root.destroy()


# ============================================================================
# 单元测试（可选）
# ============================================================================
def run_unit_tests():
    """运行单元测试"""
    print("\n" + "=" * 60)
    print("运行单元测试...")
    print("=" * 60)
    
    # 创建测试数据
    test_data_seconds = pd.DataFrame({
        'Time [s]': [0.399, 0.424, 0.449, 0.474, 0.499],
        'ROI ID': [1, 1, 1, 1, 1],
        'Intensity(532)': [1500, 1450, 1400, 1380, 1350],
        'Background(532)': [200, 210, 205, 200, 195],
        'Intensity(638)': [800, 790, 780, 770, 760],
        'Background(638)': [150, 145, 140, 138, 135]
    })
    
    test_data_minutes = pd.DataFrame({
        'Time [m:s]': ['00:00.3', '00:00.6', '00:00.9', '00:01.2', '00:01.5'],
        'ROI ID': [1, 1, 1, 1, 1],
        'Intensity(532)': [1500, 1450, 1400, 1380, 1350],
        'Background(532)': [200, 210, 205, 200, 195],
        'Intensity(638)': [800, 790, 780, 770, 760],
        'Background(638)': [150, 145, 140, 138, 135]
    })
    
    # 保存测试文件
    test_file_seconds = 'test_data_seconds.csv'
    test_file_minutes = 'test_data_minutes.csv'
    
    test_data_seconds.to_csv(test_file_seconds, index=False, encoding='utf-8')
    test_data_minutes.to_csv(test_file_minutes, index=False, encoding='utf-8')
    
    try:
        # 测试1: 纯秒格式
        print("\n测试1: Time [s] 格式")
        loader1 = ColocalizationDataLoader(test_file_seconds)
        loader1.load_and_process()
        assert loader1.time_column == 'Time [s]'
        assert abs(loader1.processed_data['Time_sec'].iloc[0] - 0.399) < 0.001
        print("✓ Time [s] 格式测试通过")
        
        # 测试2: 分:秒格式
        print("\n测试2: Time [m:s] 格式")
        loader2 = ColocalizationDataLoader(test_file_minutes)
        loader2.load_and_process()
        assert loader2.time_column == 'Time [m:s]'
        assert abs(loader2.processed_data['Time_sec'].iloc[0] - 0.3) < 0.001
        assert abs(loader2.processed_data['Time_sec'].iloc[4] - 1.5) < 0.001
        print("✓ Time [m:s] 格式测试通过")
        
        # 测试3: 时间转换函数
        print("\n测试3: 时间转换函数")
        loader = ColocalizationDataLoader(test_file_seconds)
        loader.raw_data = test_data_seconds  # 临时设置
        
        assert loader._time_to_seconds("0.399") == 0.399
        assert loader._time_to_seconds("1.234") == 1.234
        assert loader._time_to_seconds("1:30.5") == 90.5
        assert loader._time_to_seconds("00:01.3") == 1.3
        assert loader._time_to_seconds("1:30") == 90.0
        print("✓ 时间转换函数测试通过")
        
        print("\n" + "=" * 60)
        print("✓ 所有单元测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理测试文件
        for f in [test_file_seconds, test_file_minutes, 
                  'test_data_seconds_processed.csv', 
                  'test_data_minutes_processed.csv']:
            if os.path.exists(f):
                os.remove(f)
        print("\n✓ 测试文件已清理")


# 如果要运行单元测试，取消下面的注释
# run_unit_tests()
