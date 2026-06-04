"""
数据探索性分析（EDA）模块
分析数据集特征，生成可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re
import matplotlib

# ========== 中文显示解决方案 ==========
# 方案1：尝试使用系统中文字体
try:
    # 尝试多个中文字体
    fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    for font in fonts:
        if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用中文字体: {font}")
            break
    else:
        # 如果都没有，使用默认并显示警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("警告: 未找到中文字体，图表中的中文将无法显示")
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("字体设置失败，将使用默认字体")

sns.set_style("whitegrid")
sns.set_palette("Set2")
# =====================================


class RumorDataAnalyzer:
    """谣言检测数据集分析器"""
    
    def __init__(self, train_path: str, val_path: str = None, output_dir: str = "outputs"):
        """
        初始化分析器
        
        Args:
            train_path: 训练集CSV路径
            val_path: 验证集CSV路径（可选）
            output_dir: 输出目录
        """
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path) if val_path else None
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.table_dir = self.output_dir / "tables"
        
        # 创建输出目录
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("谣言检测数据集分析报告")
        print("=" * 60)
        print(f"训练集大小: {len(self.train_df)} 条")
        if self.val_df is not None:
            print(f"验证集大小: {len(self.val_df)} 条")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
    
    def basic_info(self):
        """1. 数据集基本信息"""
        print("\n【1. 数据集基本信息】")
        
        # 检查列名
        print(f"列名: {list(self.train_df.columns)}")
        
        # 缺失值检查
        missing = self.train_df.isnull().sum()
        if missing.sum() > 0:
            print(f"缺失值: {missing[missing > 0]}")
        else:
            print("无缺失值")
        
        # 数据类型
        print(f"\n数据类型:\n{self.train_df.dtypes}")
        
        return self.train_df.columns.tolist()
    
    def label_distribution(self):
        """2. 标签分布分析"""
        print("\n【2. 标签分布】")
        
        if 'label' not in self.train_df.columns:
            print("警告: 未找到 'label' 列")
            return
        
        # 统计标签分布
        label_counts = self.train_df['label'].value_counts()
        label_pcts = self.train_df['label'].value_counts(normalize=True) * 100
        
        # 用英文标签避免中文显示问题
        labels = ['Non-Rumor (0)', 'Rumor (1)']
        
        dist_df = pd.DataFrame({
            '数量': label_counts.values,
            '占比(%)': label_pcts.values.round(2)
        }, index=labels)
        
        print(dist_df)
        
        # 保存表格
        dist_df.to_csv(self.table_dir / 'label_distribution.csv')
        
        # 绘制饼图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 饼图
        colors = ['#4ECDC4', '#FF6B6B']
        axes[0].pie(label_counts.values, labels=labels, 
                    autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Label Distribution', fontsize=14)
        
        # 柱状图
        sns.barplot(x=labels, y=label_counts.values, 
                    ax=axes[1], palette=colors)
        axes[1].set_title('Label Distribution (Count)', fontsize=14)
        axes[1].set_ylabel('Sample Count')
        
        # 在柱子上添加数字
        for i, v in enumerate(label_counts.values):
            axes[1].text(i, v + 5, str(v), ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'label_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 标签分布图已保存: {self.fig_dir / 'label_distribution.png'}")
        
        return dist_df
    
    def text_length_analysis(self, text_col='text'):
        """3. 文本长度分析"""
        print("\n【3. 文本长度分析】")
        
        if text_col not in self.train_df.columns:
            print(f"警告: 未找到 '{text_col}' 列")
            return
        
        # 计算文本长度（字符数）
        self.train_df['text_length'] = self.train_df[text_col].astype(str).str.len()
        
        # 按标签分组统计
        length_stats = self.train_df.groupby('label')['text_length'].describe()
        length_stats.index = ['Non-Rumor (0)', 'Rumor (1)']
        print("按标签的文本长度统计:")
        print(length_stats)
        
        # 保存统计结果
        length_stats.to_csv(self.table_dir / 'text_length_stats.csv')
        
        # 绘制分布图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 整体分布直方图
        axes[0].hist(self.train_df['text_length'], bins=50, edgecolor='black', alpha=0.7, color='#95A5A6')
        axes[0].set_xlabel('Text Length (characters)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Overall Text Length Distribution')
        axes[0].axvline(self.train_df['text_length'].median(), color='red', 
                       linestyle='--', label=f'Median: {self.train_df["text_length"].median():.0f}')
        axes[0].legend()
        
        # 按标签分组密度图
        for label, color, name in [(0, '#4ECDC4', 'Non-Rumor'), (1, '#FF6B6B', 'Rumor')]:
            subset = self.train_df[self.train_df['label'] == label]['text_length']
            axes[1].hist(subset, bins=50, alpha=0.5, label=name, color=color, density=True)
        axes[1].set_xlabel('Text Length (characters)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Text Length Distribution by Label')
        axes[1].legend()
        
        # 箱线图
        data_to_plot = [self.train_df[self.train_df['label'] == 0]['text_length'],
                       self.train_df[self.train_df['label'] == 1]['text_length']]
        bp = axes[2].boxplot(data_to_plot, labels=['Non-Rumor', 'Rumor'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#4ECDC4', '#FF6B6B']):
            patch.set_facecolor(color)
        axes[2].set_ylabel('Text Length (characters)')
        axes[2].set_title('Text Length Boxplot by Label')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'text_length_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 文本长度分析图已保存: {self.fig_dir / 'text_length_analysis.png'}")
        
        # 基本统计
        print(f"\n整体统计:")
        print(f"  平均长度: {self.train_df['text_length'].mean():.1f} 字符")
        print(f"  中位数长度: {self.train_df['text_length'].median():.0f} 字符")
        print(f"  最短文本: {self.train_df['text_length'].min()} 字符")
        print(f"  最长文本: {self.train_df['text_length'].max()} 字符")
        
        return length_stats
    
    def event_distribution(self, event_col='event'):
        """4. 事件分布分析"""
        print("\n【4. 事件分布】")
        
        if event_col not in self.train_df.columns:
            print(f"警告: 未找到 '{event_col}' 列（数据集可能没有事件标记）")
            return None
        
        # 统计事件分布
        event_counts = self.train_df[event_col].value_counts()
        event_pcts = self.train_df[event_col].value_counts(normalize=True) * 100
        
        event_df = pd.DataFrame({
            'Event Name': event_counts.index,
            'Sample Count': event_counts.values,
            'Percentage (%)': event_pcts.values.round(2)
        })
        
        print(f"共有 {len(event_counts)} 个不同的事件")
        print("\n样本数最多的前5个事件:")
        print(event_df.head())
        
        # 保存完整表格
        event_df.to_csv(self.table_dir / 'event_distribution.csv', index=False)
        
        # 绘制事件分布图（前15个）
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 柱状图（前15）
        top_events = event_counts.head(15)
        axes[0].barh(range(len(top_events)), top_events.values, color='#95A5A6')
        axes[0].set_yticks(range(len(top_events)))
        axes[0].set_yticklabels(top_events.index)
        axes[0].set_xlabel('Sample Count')
        axes[0].set_title('Top 15 Events by Sample Count')
        
        # 按事件的标签分布
        event_label_dist = self.train_df.groupby(event_col)['label'].value_counts().unstack(fill_value=0)
        event_label_dist = event_label_dist.loc[event_counts.head(10).index]  # 取前10个事件
        
        event_label_dist.plot(kind='barh', ax=axes[1], color=['#4ECDC4', '#FF6B6B'])
        axes[1].set_xlabel('Sample Count')
        axes[1].set_title('Label Distribution in Top 10 Events')
        axes[1].legend(['Non-Rumor (0)', 'Rumor (1)'])
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'event_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 事件分布图已保存: {self.fig_dir / 'event_distribution.png'}")
        
        return event_df
    
    def generate_all_reports(self):
        """生成所有分析报告和图表"""
        print("\n" + "=" * 60)
        print("开始生成完整分析报告")
        print("=" * 60)
        
        # 运行所有分析
        self.basic_info()
        self.label_distribution()
        self.text_length_analysis()
        self.event_distribution()
        
        print("\n" + "=" * 60)
        print(f"✅ 分析完成！所有输出已保存到: {self.output_dir}")
        print(f"   - 图表: {self.fig_dir}")
        print(f"   - 数据表: {self.table_dir}")
        print("=" + "=" * 60)


# 主函数
if __name__ == "__main__":
    # 配置路径（根据实际路径修改）
    TRAIN_PATH = "data/train.csv"
    VAL_PATH = "data/val.csv"  # 可选
    
    # 创建分析器并运行
    analyzer = RumorDataAnalyzer(TRAIN_PATH, VAL_PATH, output_dir="outputs")
    analyzer.generate_all_reports()
