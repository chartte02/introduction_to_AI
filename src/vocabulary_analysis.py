"""
词汇与语言特征分析
分析谣言 vs 非谣言的关键词差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
from pathlib import Path
import matplotlib

# ========== 中文显示解决方案 ==========
try:
    fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    for font in fonts:
        if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用中文字体: {font}")
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("警告: 未找到中文字体")
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# =====================================


class VocabularyAnalyzer:
    """词汇分析器：对比谣言和非谣言的用词差异"""
    
    def __init__(self, data_path: str, text_col='text', label_col='label', output_dir='outputs'):
        """
        初始化词汇分析器
        
        Args:
            data_path: CSV数据路径
            text_col: 文本列名
            label_col: 标签列名
            output_dir: 输出目录
        """
        self.df = pd.read_csv(data_path)
        self.text_col = text_col
        self.label_col = label_col
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.table_dir = self.output_dir / "tables"
        
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        
        # 分离谣言和非谣言
        self.rumor_df = self.df[self.df[label_col] == 1]
        self.non_rumor_df = self.df[self.df[label_col] == 0]
        
        print("=" * 60)
        print("词汇与语言特征分析")
        print("=" * 60)
        print(f"总样本: {len(self.df)}")
        print(f"谣言样本: {len(self.rumor_df)}")
        print(f"非谣言样本: {len(self.non_rumor_df)}")
        print("=" * 60)
    
    def clean_text_for_vocab(self, text: str) -> str:
        """为词汇分析清洗文本（保留单词，去除特殊符号）"""
        if not isinstance(text, str):
            return ""
        # 转小写
        text = text.lower()
        # 去除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # 去除@用户
        text = re.sub(r'@\w+', '', text)
        # 只保留字母、数字、空格（英文为主）
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # 合并空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_top_keywords(self, df, n=20):
        """获取Top N关键词"""
        all_text = ' '.join(df[self.text_col].astype(str).apply(self.clean_text_for_vocab))
        words = all_text.split()
        word_counts = Counter(words)
        # 去除常见停用词
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'for', 'in', 
                    'on', 'at', 'by', 'with', 'without', 'is', 'are', 'was', 'were',
                    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'this', 'that',
                    'it', 'i', 'you', 'he', 'she', 'we', 'they'}
        word_counts = {w: c for w, c in word_counts.items() if w not in stopwords and len(w) > 2}
        return Counter(word_counts).most_common(n)
    
    def compare_keywords(self):
        """对比谣言和非谣言的关键词"""
        print("\n【1. Rumor Keywords Top 15】")
        rumor_keywords = self.get_top_keywords(self.rumor_df, 15)
        for i, (word, count) in enumerate(rumor_keywords, 1):
            print(f"  {i:2d}. {word}: {count}")
        
        print("\n【2. Non-Rumor Keywords Top 15】")
        non_rumor_keywords = self.get_top_keywords(self.non_rumor_df, 15)
        for i, (word, count) in enumerate(non_rumor_keywords, 1):
            print(f"  {i:2d}. {word}: {count}")
        
        # 保存到CSV
        rumor_df = pd.DataFrame(rumor_keywords, columns=['Keyword', 'Frequency'])
        rumor_df.to_csv(self.table_dir / 'rumor_keywords.csv', index=False)
        
        non_rumor_df = pd.DataFrame(non_rumor_keywords, columns=['Keyword', 'Frequency'])
        non_rumor_df.to_csv(self.table_dir / 'non_rumor_keywords.csv', index=False)
        
        # 绘制对比条形图
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # 谣言关键词
        rumor_words, rumor_counts = zip(*rumor_keywords[:10])
        axes[0].barh(range(len(rumor_words)), rumor_counts, color='#FF6B6B')
        axes[0].set_yticks(range(len(rumor_words)))
        axes[0].set_yticklabels(rumor_words)
        axes[0].set_xlabel('Frequency')
        axes[0].set_title('Top 10 Keywords in Rumors')
        
        # 非谣言关键词
        non_words, non_counts = zip(*non_rumor_keywords[:10])
        axes[1].barh(range(len(non_words)), non_counts, color='#4ECDC4')
        axes[1].set_yticks(range(len(non_words)))
        axes[1].set_yticklabels(non_words)
        axes[1].set_xlabel('Frequency')
        axes[1].set_title('Top 10 Keywords in Non-Rumors')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'keyword_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 关键词对比图已保存: {self.fig_dir / 'keyword_comparison.png'}")
        
        return rumor_keywords, non_rumor_keywords
    
    def create_wordclouds(self):
        """生成词云图"""
        print("\n【2. Generating WordClouds】")
        
        # 谣言词云
        rumor_text = ' '.join(self.rumor_df[self.text_col].astype(str).apply(self.clean_text_for_vocab))
        non_rumor_text = ' '.join(self.non_rumor_df[self.text_col].astype(str).apply(self.clean_text_for_vocab))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 谣言词云
        wc_rumor = WordCloud(width=400, height=400, background_color='white', 
                             colormap='Reds', max_words=100)
        wc_rumor.generate(rumor_text)
        axes[0].imshow(wc_rumor, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title('Rumor Word Cloud', fontsize=14)
        
        # 非谣言词云
        wc_non = WordCloud(width=400, height=400, background_color='white',
                          colormap='Blues', max_words=100)
        wc_non.generate(non_rumor_text)
        axes[1].imshow(wc_non, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('Non-Rumor Word Cloud', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'wordcloud_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 词云图已保存: {self.fig_dir / 'wordcloud_comparison.png'}")
    
    def analyze_special_characters(self):
        """分析特殊字符使用（URL、@、#、感叹号、问号）"""
        print("\n【3. Special Characters Analysis】")
        
        patterns = {
            'URL': r'https?://|www\.',
            '@mention': r'@\w+',
            'hashtag': r'#\w+',
            'exclamation': r'!+',
            'question': r'\?+'
        }
        
        results = []
        for name, pattern in patterns.items():
            rumor_count = self.rumor_df[self.text_col].astype(str).str.contains(pattern, regex=True).sum()
            non_rumor_count = self.non_rumor_df[self.text_col].astype(str).str.contains(pattern, regex=True).sum()
            
            rumor_pct = rumor_count / len(self.rumor_df) * 100
            non_pct = non_rumor_count / len(self.non_rumor_df) * 100
            
            results.append({
                'Feature': name,
                'Rumor (%)': f"{rumor_pct:.1f}",
                'Non-Rumor (%)': f"{non_pct:.1f}",
                'Difference': f"{rumor_pct - non_pct:.1f}"
            })
            
            print(f"  {name}:")
            print(f"    Rumor: {rumor_count}/{len(self.rumor_df)} ({rumor_pct:.1f}%)")
            print(f"    Non-Rumor: {non_rumor_count}/{len(self.non_rumor_df)} ({non_pct:.1f}%)")
        
        # 保存结果
        char_df = pd.DataFrame(results)
        char_df.to_csv(self.table_dir / 'special_characters_analysis.csv', index=False)
        
        # 绘制对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(patterns))
        rumor_pcts = [float(r['Rumor (%)']) for r in results]
        non_pcts = [float(r['Non-Rumor (%)']) for r in results]
        
        width = 0.35
        ax.bar([i - width/2 for i in x], rumor_pcts, width, label='Rumor', color='#FF6B6B')
        ax.bar([i + width/2 for i in x], non_pcts, width, label='Non-Rumor', color='#4ECDC4')
        
        ax.set_xlabel('Feature Type')
        ax.set_ylabel('Occurrence Rate (%)')
        ax.set_title('Special Characters Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([r['Feature'] for r in results])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'special_characters_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 特殊字符对比图已保存: {self.fig_dir / 'special_characters_comparison.png'}")
        
        return results
    
    def generate_all_analysis(self):
        """生成所有词汇分析"""
        print("\n" + "=" * 60)
        print("开始生成词汇分析报告")
        print("=" * 60)
        
        self.compare_keywords()
        self.create_wordclouds()
        self.analyze_special_characters()
        
        print("\n" + "=" * 60)
        print(f"✅ 词汇分析完成！输出目录: {self.output_dir}")
        print("=" + "=" * 60)


if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "data/train.csv"
    
    analyzer = VocabularyAnalyzer(DATA_PATH, output_dir="outputs")
    analyzer.generate_all_analysis()
