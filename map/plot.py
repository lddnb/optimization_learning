#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from io import StringIO

class TimingVisualizer:
    def __init__(self):
        # 设置图表样式
        sns.set_theme(style="whitegrid")
        
    def read_timing_data(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """读取计时数据和统计信息"""
        try:
            # 读取所有数据
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 分离迭代数据和统计数据
            iteration_lines = []
            summary_lines = []
            in_summary = False
            
            for line in lines:
                if line.strip() == 'Summary:':
                    in_summary = True
                    continue
                if in_summary:
                    summary_lines.append(line)
                else:
                    iteration_lines.append(line)

            # 处理迭代数据
            iterations_df = pd.read_csv(StringIO(''.join(iteration_lines)))

            # 处理统计数据
            summary_dict = {}
            for line in summary_lines:
                if ',' in line:
                    key, value = line.strip().split(',')
                    try:
                        summary_dict[key] = float(value)
                    except ValueError:
                        summary_dict[key] = value

            return iterations_df, summary_dict

        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            raise

    def plot_time_series(self, data_dict: Dict[str, pd.DataFrame], 
                        save_path: str = None):
        """绘制时间序列对比图"""
        plt.figure(figsize=(12, 6))
        
        # 计算所有数据的最大值和最小值，用于确定图例位置
        all_max = max(df['time_ms'].max() for df in data_dict.values())
        all_min = min(df['time_ms'].min() for df in data_dict.values())
        y_range = all_max - all_min
        
        for name, df in data_dict.items():
            # 绘制时间序列
            line, = plt.plot(df['iteration'], df['time_ms'], 
                            label=f'{name}', 
                            alpha=0.7)
            
            # 计算并绘制均值线
            mean_time = df['time_ms'].mean()
            plt.axhline(y=mean_time, 
                       color=line.get_color(),
                       linestyle='--',
                       alpha=0.9,
                       label=f'mean: {mean_time:.2f}ms')
        
        plt.title('Execution Time Comparison')
        plt.xlabel('Iteration')
        plt.ylabel('Time (ms)')
        
        # 将图例放在图中右上角，自动调整位置避免遮挡数据
        plt.legend(bbox_to_anchor=(1, 1), 
                  loc='upper right',
                  bbox_transform=plt.gca().transAxes,
                  borderaxespad=0.,
                  framealpha=0.8,  # 半透明背景
                  edgecolor='gray',  # 边框颜色
                  fancybox=True,    # 圆角
                  shadow=True)      # 阴影效果
        
        plt.grid(True, alpha=0.3)
        
        # 调整y轴范围，给图例留出空间
        plt.margins(x=0.02, y=0.1)
        
        if save_path:
            plt.savefig(save_path + '_series.png', 
                       dpi=300, 
                       bbox_inches='tight')
        plt.show()

    def plot_box_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                          save_path: str = None):
        """绘制箱线图对比"""
        plt.figure(figsize=(10, 6))
        
        data = []
        labels = []
        for name, df in data_dict.items():
            data.append(df['time_ms'].values)
            
        plt.boxplot(data, labels=data_dict.keys())
        plt.title('Time Distribution Comparison')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path + '_box.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_violin_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                             save_path: str = None):
        """绘制小提琴图对比"""
        plt.figure(figsize=(10, 6))
        
        all_data = []
        labels = []
        for name, df in data_dict.items():
            all_data.extend(df['time_ms'].values)
            labels.extend([name] * len(df))
            
        plot_df = pd.DataFrame({
            'Module': labels,
            'Time (ms)': all_data
        })
            
        sns.violinplot(data=plot_df, x='Module', y='Time (ms)')
        plt.title('Time Distribution Comparison (Violin Plot)')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path + '_violin.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_time_percentage(self, summary_dict: Dict[str, Dict], 
                           save_path: str = None):
        """绘制时间占比饼图"""
        plt.figure(figsize=(10, 8))
        
        # 提取平均时间
        avg_times = {name: stats['average'] for name, stats in summary_dict.items()}
        total_time = sum(avg_times.values())
        
        # 计算百分比
        percentages = {name: (time/total_time)*100 for name, time in avg_times.items()}
        
        # 绘制饼图
        plt.pie(percentages.values(), labels=[f'{name}\n({perc:.1f}%)' 
                for name, perc in percentages.items()],
                autopct='%1.1f%%', startangle=90)
        plt.title('Time Percentage Distribution')
        
        if save_path:
            plt.savefig(save_path + '_pie.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot.py <timing_file1> <timing_file2> ...")
        return

    visualizer = TimingVisualizer()
    data_dict = {}
    summary_dict = {}

    # 读取所有输入文件
    for file_path in sys.argv[1:]:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df, summary = visualizer.read_timing_data(file_path)
            data_dict[module_name] = df
            summary_dict[module_name] = summary
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not data_dict:
        print("No valid data files found!")
        return

    # 生成可视化
    save_path = 'timing_comparison'
    visualizer.plot_time_series(data_dict, save_path)
    # visualizer.plot_box_comparison(data_dict, save_path)
    visualizer.plot_violin_comparison(data_dict, save_path)
    visualizer.plot_time_percentage(summary_dict, save_path)

if __name__ == "__main__":
    main()
