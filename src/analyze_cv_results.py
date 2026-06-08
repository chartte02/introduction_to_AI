import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
with open('cv_results.json', 'r') as f:
    data = json.load(f)

# 提取数据
folds = [f['fold'] for f in data['folds']]
accuracies = [f['accuracy'] for f in data['folds']]
macro_f1s = [f['macro_f1'] for f in data['folds']]
f1_scores = [f['f1'] for f in data['folds']]  # 这个是正类的 f1
events = [f['val_event'] for f in data['folds']]

# 整体统计
mean_acc = data['mean_accuracy']
std_acc = data['std_accuracy']
mean_macro_f1 = data['mean_macro_f1']
std_macro_f1 = data['std_macro_f1']

print("=" * 50)
print("7折交叉验证结果统计")
print("=" * 50)
print(f"模型: {data['model_name']}")
print(f"超参数: learning_rate={data['hyperparams']['learning_rate']}, "
      f"batch_size={data['hyperparams']['batch_size']}, "
      f"epochs={data['hyperparams']['num_epochs']}")
print()
print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Macro F1:  {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
print()
print("各折详细结果:")
print("-" * 50)
for i, fold in enumerate(data['folds']):
    print(f"Fold {fold['fold']} (Event {fold['val_event']}): "
          f"Acc={fold['accuracy']:.4f}, F1={fold['f1']:.4f}, Macro_F1={fold['macro_f1']:.4f}")

# ========== 图1：Accuracy 柱状图 ==========
plt.figure(figsize=(10, 6))
bars = plt.bar([f'Fold {f}' for f in folds], accuracies, color='steelblue', alpha=0.8)
plt.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
            label=f'Mean Acc = {mean_acc:.3f} ± {std_acc:.3f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('7-fold Cross Validation Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('cv_accuracy.png', dpi=150)
plt.show()

# ========== 图2：Macro F1 柱状图 ==========
plt.figure(figsize=(10, 6))
bars = plt.bar([f'Fold {f}' for f in folds], macro_f1s, color='darkred', alpha=0.8)
plt.axhline(y=mean_macro_f1, color='blue', linestyle='--', linewidth=2,
            label=f'Mean Macro F1 = {mean_macro_f1:.3f} ± {std_macro_f1:.3f}')
plt.xlabel('Fold')
plt.ylabel('Macro F1 Score')
plt.title('7-fold Cross Validation Macro F1 Score')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('cv_macro_f1.png', dpi=150)
plt.show()

# ========== 图3：按事件的 Acc 和 F1 ==========
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(events))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, macro_f1s, width, label='Macro F1', color='darkred', alpha=0.8)

ax.set_xlabel('Event ID (Val Event)')
ax.set_ylabel('Score')
ax.set_title('Performance by Validation Event')
ax.set_xticks(x)
ax.set_xticklabels(events)
ax.legend()
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# 在柱子上标注数值
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('performance_by_event.png', dpi=150)
plt.show()

# ========== 图4：Boxplot 展示分布 ==========
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy 箱线图
ax[0].boxplot(accuracies, vert=True)
ax[0].set_title('Accuracy Distribution (7 folds)')
ax[0].set_ylabel('Accuracy')
ax[0].set_xticklabels(['All Folds'])

# Macro F1 箱线图
ax[1].boxplot(macro_f1s, vert=True)
ax[1].set_title('Macro F1 Distribution (7 folds)')
ax[1].set_ylabel('Macro F1')
ax[1].set_xticklabels(['All Folds'])

plt.tight_layout()
plt.savefig('cv_boxplot.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("图表已保存:")
print("  - cv_accuracy.png")
print("  - cv_macro_f1.png")
print("  - performance_by_event.png")
print("  - cv_boxplot.png")
print("=" * 50)