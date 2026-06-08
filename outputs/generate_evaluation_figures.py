import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ========== 1. 读取 JSON 文件 ==========
with open('vote_results.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("模型评估结果")
print("=" * 60)
print(f"投票策略: {data['variant']}")
print(f"投票阈值: {data['threshold']}")
print()

# ========== 2. 混淆矩阵热力图 ==========
tn = data['vote_metrics']['tn']
tp = data['vote_metrics']['tp']
fp = data['vote_metrics']['fp']
fn = data['vote_metrics']['fn']

cm = np.array([[tn, fp], [fn, tp]])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-rumor', 'Rumor'],
            yticklabels=['Non-rumor', 'Rumor'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Vote Ensemble)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("混淆矩阵:")
print(f"  TN (真负): {tn}")
print(f"  FP (假正): {fp}")
print(f"  FN (假负): {fn}")
print(f"  TP (真正): {tp}")
print()

# ========== 3. 打印评估指标 ==========
print("=" * 60)
print("投票集成模型评估指标")
print("=" * 60)
print(f"Accuracy:        {data['vote_metrics']['accuracy']:.4f} ({data['vote_metrics']['accuracy']*100:.2f}%)")
print(f"Macro F1:        {data['vote_metrics']['macro_f1']:.4f}")
print(f"Precision:       {data['vote_metrics']['precision']:.4f}")
print(f"Recall:          {data['vote_metrics']['recall']:.4f}")
print(f"F1 (谣言类):     {data['vote_metrics']['f1']:.4f}")
print(f"Balanced Acc:    {data['vote_metrics']['balanced_accuracy']:.4f}")
print()

# ========== 4. 单模型 vs 投票集成对比图 ==========
# 提取每个单模型的指标
model_names = []
accs = []
f1s = []

for m in data['per_model_at_threshold']:
    # 从路径中提取模型名称
    path = m['model_path']
    if 'final_model' in path:
        name = 'final_model'
    elif 'fold_' in path:
        fold_num = path.split('fold_')[-1].split('_')[0]
        name = f'fold_{fold_num}'
    else:
        name = path.split('/')[-1].replace('.pt', '')
    model_names.append(name)
    accs.append(m['accuracy'])
    f1s.append(m['macro_f1'])

# 添加投票集成的结果
model_names.append('Vote Ensemble')
accs.append(data['vote_metrics']['accuracy'])
f1s.append(data['vote_metrics']['macro_f1'])

# 画对比图
x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, f1s, width, label='Macro F1', color='darkred')

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Comparison: Single Models vs Vote Ensemble')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0.7, 0.92)

# 在柱子上标注数值
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', 
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), 
                textcoords="offset points", 
                ha='center', 
                va='bottom', 
                fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', 
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), 
                textcoords="offset points", 
                ha='center', 
                va='bottom', 
                fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# ========== 5. 打印单模型 vs 基准对比 ==========
print("=" * 60)
print("基准对比 (最优单模型 vs 投票集成)")
print("=" * 60)
baseline_acc = data['baseline_single_final_model']['accuracy']
baseline_f1 = data['baseline_single_final_model']['macro_f1']
print(f"最优单模型 (final_model.pt):")
print(f"  Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"  Macro F1:  {baseline_f1:.4f}")
print()
print(f"投票集成模型:")
print(f"  Accuracy: {data['vote_metrics']['accuracy']:.4f} ({data['vote_metrics']['accuracy']*100:.2f}%)")
print(f"  Macro F1:  {data['vote_metrics']['macro_f1']:.4f}")
print()
print(f"差异 (投票 - 单模型):")
print(f"  Accuracy: {data['delta_vs_baseline']['accuracy']:+.4f}")
print(f"  Macro F1:  {data['delta_vs_baseline']['macro_f1']:+.4f}")
print()

# ========== 6. 汇总表格（可直接复制到报告） ==========
print("=" * 60)
print("报告用汇总表格")
print("=" * 60)
print("| 指标 | 投票集成模型 | 最优单模型 |")
print("|:---|:---:|:---:|")
print(f"| Accuracy | {data['vote_metrics']['accuracy']:.4f} | {baseline_acc:.4f} |")
print(f"| Macro F1 | {data['vote_metrics']['macro_f1']:.4f} | {baseline_f1:.4f} |")
print(f"| Precision | {data['vote_metrics']['precision']:.4f} | - |")
print(f"| Recall | {data['vote_metrics']['recall']:.4f} | - |")
print(f"| F1 (谣言类) | {data['vote_metrics']['f1']:.4f} | {data['baseline_single_final_model']['f1']:.4f} |")
print()

print("混淆矩阵:")
print("|            | 预测非谣言 | 预测谣言 |")
print("|:---|:---:|:---:|")
print(f"| 真实非谣言 | {tn} | {fp} |")
print(f"| 真实谣言   | {fn} | {tp} |")
print()
print("=" * 60)
print("已生成图片:")
print("  - confusion_matrix.png")
print("  - model_comparison.png")
print("=" * 60)