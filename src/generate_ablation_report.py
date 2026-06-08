import re
import sys
import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 使用英文（避免中文字体问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 重定向输出到文件和控制台 ==========
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

# 开始同时输出到终端和文件
tee = Tee('ablation_report_output.txt')
sys.stdout = tee

# ========== 从 log 文件提取指标 ==========
def parse_eval_log(log_path):
    """从 eval.log 文件中提取关键指标"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 {log_path}")
        return None
    
    acc_match = re.search(r'准确率:\s+([\d.]+)', content)
    accuracy = float(acc_match.group(1)) if acc_match else None
    
    f1_match = re.search(r'macro_f1:\s+([\d.]+)', content)
    macro_f1 = float(f1_match.group(1)) if f1_match else None
    
    rumor_match = re.search(r'\[谣言\]\s+P/R/F1:\s+([\d.]+)\s+/\s+([\d.]+)\s+/\s+([\d.]+)', content)
    if rumor_match:
        rumor_precision = float(rumor_match.group(1))
        rumor_recall = float(rumor_match.group(2))
        rumor_f1 = float(rumor_match.group(3))
    else:
        rumor_precision = rumor_recall = rumor_f1 = None
    
    cm_match = re.search(r'混淆矩阵:\s+TP=(\d+),\s+TN=(\d+),\s+FP=(\d+),\s+FN=(\d+)', content)
    if cm_match:
        tp = int(cm_match.group(1))
        tn = int(cm_match.group(2))
        fp = int(cm_match.group(3))
        fn = int(cm_match.group(4))
    else:
        tp = tn = fp = fn = None
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'rumor_precision': rumor_precision,
        'rumor_recall': rumor_recall,
        'rumor_f1': rumor_f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

# 解析两个版本的结果
print("Reading log files...")
without_pre = parse_eval_log('01_eval.log')   # without preprocess
with_pre = parse_eval_log('03_eval.log')      # with preprocess

if without_pre is None or with_pre is None:
    print("Error: Cannot read log files. Make sure 01_eval.log and 03_eval.log exist.")
    tee.close()
    sys.exit(1)

print("=" * 70)
print("Ablation Study Results")
print("=" * 70)

# ========== 1. Print comparison tables ==========
print("\n[Table 1: Overall Performance]")
print("-" * 70)
print(f"{'Metric':<20} {'Without Preproc':<20} {'With Preproc':<20} {'Change':<15}")
print("-" * 70)
acc_change = with_pre['accuracy'] - without_pre['accuracy']
f1_change = with_pre['macro_f1'] - without_pre['macro_f1']
print(f"{'Accuracy':<20} {without_pre['accuracy']*100:.2f}%{'':>12} {with_pre['accuracy']*100:.2f}%{'':>12} {acc_change*100:+.2f}%")
print(f"{'Macro F1':<20} {without_pre['macro_f1']:.4f}{'':>16} {with_pre['macro_f1']:.4f}{'':>16} {f1_change:+.4f}")
print("-" * 70)

print("\n[Table 2: Rumor Class Metrics]")
print("-" * 70)
print(f"{'Metric':<20} {'Without Preproc':<20} {'With Preproc':<20} {'Change':<15}")
print("-" * 70)
p_change = with_pre['rumor_precision'] - without_pre['rumor_precision']
r_change = with_pre['rumor_recall'] - without_pre['rumor_recall']
f_change = with_pre['rumor_f1'] - without_pre['rumor_f1']
print(f"{'Precision':<20} {without_pre['rumor_precision']:.4f}{'':>16} {with_pre['rumor_precision']:.4f}{'':>16} {p_change:+.4f}")
print(f"{'Recall':<20} {without_pre['rumor_recall']:.4f}{'':>16} {with_pre['rumor_recall']:.4f}{'':>16} {r_change:+.4f}")
print(f"{'F1':<20} {without_pre['rumor_f1']:.4f}{'':>16} {with_pre['rumor_f1']:.4f}{'':>16} {f_change:+.4f}")
print("-" * 70)

print("\n[Table 3: Confusion Matrix]")
print("-" * 70)
print(f"{'Version':<15} {'TP':<8} {'TN':<8} {'FP':<8} {'FN':<8}")
print("-" * 70)
print(f"{'Without Preproc':<15} {without_pre['tp']:<8} {without_pre['tn']:<8} {without_pre['fp']:<8} {without_pre['fn']:<8}")
print(f"{'With Preproc':<15} {with_pre['tp']:<8} {with_pre['tn']:<8} {with_pre['fp']:<8} {with_pre['fn']:<8}")
print("-" * 70)

# ========== 2. Generate conclusion text ==========
print("\n" + "=" * 70)
print("[Conclusion for Report]")
print("=" * 70)

conclusion = f"""
Ablation Study: Impact of Preprocessing on Model Performance

Experimental Setup:
- Without Preprocessing: Raw tweet text directly fed into the model
- With Preprocessing: Text cleaned via HTML decoding, URL replacement, @mention replacement, # removal, whitespace merging

Results Comparison:

| Metric | Without Preproc | With Preproc | Change |
|:---|:---:|:---:|:---:|
| Accuracy | {without_pre['accuracy']*100:.2f}% | {with_pre['accuracy']*100:.2f}% | {acc_change*100:+.2f}% |
| Macro F1 | {without_pre['macro_f1']:.4f} | {with_pre['macro_f1']:.4f} | {f1_change:+.4f} |
| Rumor Precision | {without_pre['rumor_precision']:.4f} | {with_pre['rumor_precision']:.4f} | {p_change:+.4f} |
| Rumor Recall | {without_pre['rumor_recall']:.4f} | {with_pre['rumor_recall']:.4f} | {r_change:+.4f} |
| Rumor F1 | {without_pre['rumor_f1']:.4f} | {with_pre['rumor_f1']:.4f} | {f_change:+.4f} |

Analysis:
1. With preprocessing achieves Accuracy of {with_pre['accuracy']*100:.2f}%, which is {acc_change*100:+.2f}% higher than without preprocessing ({without_pre['accuracy']*100:.2f}%).
2. With preprocessing achieves Macro F1 of {with_pre['macro_f1']:.4f}, higher than without preprocessing ({without_pre['macro_f1']:.4f}).
3. For rumor class, With preprocessing has significantly higher Precision ({with_pre['rumor_precision']:.4f} vs {without_pre['rumor_precision']:.4f}), meaning predictions are more reliable.
4. Without preprocessing has higher Recall ({without_pre['rumor_recall']:.4f} vs {with_pre['rumor_recall']:.4f}), meaning it catches more real rumors.
5. Both versions have very similar F1 scores ({without_pre['rumor_f1']:.4f} vs {with_pre['rumor_f1']:.4f}).

Conclusion:
Preprocessing provides a positive but limited improvement to model performance. The version with preprocessing performs better in overall accuracy and precision, while the version without preprocessing performs better in recall. Considering that false positives (misclassifying non-rumor as rumor) have a higher cost in rumor detection scenarios, we select the version with preprocessing as our final model.
"""

print(conclusion)

# ========== 3. Plot (with English labels) ==========
print("\nGenerating comparison chart...")

versions = ['Without\nPreprocess', 'With\nPreprocess']
accuracy = [without_pre['accuracy'], with_pre['accuracy']]
macro_f1 = [without_pre['macro_f1'], with_pre['macro_f1']]

x = np.arange(len(versions))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, macro_f1, width, label='Macro F1', color='darkred')

ax.set_xlabel('Version')
ax.set_ylabel('Score')
ax.set_title('Ablation Study: Impact of Preprocessing')
ax.set_xticks(x)
ax.set_xticklabels(versions)
ax.legend()
ax.set_ylim(0.85, 0.92)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('ablation_study.png', dpi=150)
plt.show()

print("\nGenerated files:")
print("  - ablation_report_output.txt (console output)")
print("  - ablation_study.png (comparison chart)")

print("\n" + "=" * 70)

# ========== Restore stdout and close ==========
sys.stdout = tee.stdout
tee.close()
print("All output saved to ablation_report_output.txt")