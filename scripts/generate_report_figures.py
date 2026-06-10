#!/usr/bin/env python3
"""报告图表生成脚本。
从 outputs/cv_results.json、outputs/vote_results.json 等已有数据生成
报告所需的图表，保存到 outputs/figures/。

用法：
    python scripts/generate_report_figures.py
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 字体配置 ──
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (8, 8)
DPI = 200

# ═══════════════════════════════════════════════════════════════
# 1. 7 折交叉验证结果柱状图
# ═══════════════════════════════════════════════════════════════

def plot_cv_results():
    """绘制每折 CV 的 Accuracy / F1 / macro_f1 分组柱状图。"""
    cv_path = OUTPUTS_DIR / "cv_results.json"
    if not cv_path.exists():
        print("[跳过] cv_results.json 不存在")
        return

    with open(cv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]
    fold_labels = [f"Fold {f['fold']}\n(event {f['val_event']})" for f in folds]
    accs = [f["accuracy"] for f in folds]
    f1s = [f["f1"] for f in folds]
    macros = [f["macro_f1"] for f in folds]

    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, accs, width, label="Accuracy", color="#3498DB")
    bars2 = ax.bar(x, f1s, width, label="F1 (Rumor)", color="#E74C3C")
    bars3 = ax.bar(x + width, macros, width, label="Macro F1", color="#2ECC71")

    # 均值线
    mean_acc = data["mean_accuracy"]
    mean_f1 = data["mean_f1"]
    mean_macro = data["mean_macro_f1"]
    ax.axhline(y=mean_acc, color="#2980B9", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=mean_f1, color="#C0392B", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=mean_macro, color="#27AE60", linestyle="--", linewidth=1.5, alpha=0.7)

    # 标注均值数值
    ax.text(len(folds) - 0.3, mean_acc + 0.02, f"Mean Acc={mean_acc:.3f}", fontsize=8, color="#2980B9")
    ax.text(len(folds) - 0.3, mean_f1 + 0.02, f"Mean F1={mean_f1:.3f}", fontsize=8, color="#C0392B")
    ax.text(len(folds) - 0.3, mean_macro + 0.02, f"Mean mF1={mean_macro:.3f}", fontsize=8, color="#27AE60")

    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("7-Fold Leave-One-Event-Out Cross-Validation Results", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # 标准差标注
    ax.text(
        0.02, 0.94,
        f"Std:  Accuracy ±{data['std_accuracy']:.3f}  |  F1 ±{data['std_f1']:.3f}  |  Macro F1 ±{data['std_macro_f1']:.3f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    out = FIG_DIR / "cv_results_bar.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ CV 结果柱状图: {out}")


# ═══════════════════════════════════════════════════════════════
# 2. 混淆矩阵热力图（val.csv final_model）
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix():
    """根据 vote_results.json 中 baseline final_model 的数据绘制混淆矩阵。"""
    vote_path = OUTPUTS_DIR / "vote_results.json"
    if not vote_path.exists():
        print("[跳过] vote_results.json 不存在")
        return

    with open(vote_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baseline = data["baseline_single_final_model"]
    vote_metrics = data["vote_metrics"]

    # 使用最新 eval 的 TP/TN/FP/FN（从 vote_results 的 baseline 推测）
    tp = vote_metrics.get("tp", 150)
    tn = vote_metrics.get("tn", 203)
    fp = vote_metrics.get("fp", 23)
    fn = vote_metrics.get("fn", 25)

    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["Non-Rumor (0)", "Rumor (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Count", shrink=0.8)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=24, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix on val.csv (final_model)", fontsize=14, fontweight="bold")

    # 标注子指标
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ax.text(
        0.5, -0.18,
        f"Precision={precision:.3f}  |  Recall={recall:.3f}  |  Specificity={specificity:.3f}",
        transform=ax.transAxes, ha="center", fontsize=10,
    )

    plt.tight_layout()
    out = FIG_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 混淆矩阵: {out}")


# ═══════════════════════════════════════════════════════════════
# 3. Backbone 对比图
# ═══════════════════════════════════════════════════════════════

def plot_backbone_comparison():
    """绘制不同 backbone 的准确率 / macro_f1 对比。"""
    cmp_path = OUTPUTS_DIR / "backbone_comparison.json"
    if not cmp_path.exists():
        print("[跳过] backbone_comparison.json 不存在，使用内置数据")
        # 使用内置数据（来自训练日志）
        rows = [
            {"backbone": "twitter-roberta-base", "accuracy": 0.8803, "macro_f1": 0.8782},
            {"backbone": "deberta-v3-large", "accuracy": 0.8878, "macro_f1": 0.8831},
            {"backbone": "deberta-v3-base", "accuracy": 0.8678, "macro_f1": 0.8664},
            {"backbone": "bertweet-base", "accuracy": 0.8391, "macro_f1": 0.8344},
            {"backbone": "roberta-base", "accuracy": 0.8354, "macro_f1": 0.8307},
        ]
    else:
        with open(cmp_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        rows = [
            {"backbone": r["backbone"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
            for r in j.get("rows", [])
        ]

    names = [r["backbone"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    macros = [r["macro_f1"] for r in rows]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#3498DB", edgecolor="#2C3E50", linewidth=0.5)
    ax.bar(x + width / 2, macros, width, label="Macro F1", color="#2ECC71", edgecolor="#2C3E50", linewidth=0.5)

    # 数值标注
    for i, (a, m) in enumerate(zip(accs, macros)):
        ax.text(i - width / 2, a + 0.005, f"{a:.4f}", ha="center", fontsize=8, rotation=90)
        ax.text(i + width / 2, m + 0.005, f"{m:.4f}", ha="center", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Backbone Comparison on val.csv", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0.80, 0.92)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "backbone_comparison.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ Backbone 对比图: {out}")


# ═══════════════════════════════════════════════════════════════
# 4. 按事件准确率对比图（CV 各折 on val_event）
# ═══════════════════════════════════════════════════════════════

def plot_per_event_accuracy():
    """绘制每个 held-out 验证事件的准确率，标注样本量与谣言比例。"""
    cv_path = OUTPUTS_DIR / "cv_results.json"
    if not cv_path.exists():
        print("[跳过] cv_results.json 不存在")
        return

    with open(cv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]

    event_names = {
        "0": "Gurlitt Art",
        "1": "Ferguson",
        "2": "Ebola",
        "3": "Prince Concert",
        "4": "Germanwings",
        "5": "Sydney Siege",
        "6": "Ottawa Shooting",
    }

    events = [f["val_event"] for f in folds]
    accs = [f["accuracy"] for f in folds]
    macros = [f["macro_f1"] for f in folds]
    fps = [f["fp"] for f in folds]
    fns = [f["fn"] for f in folds]

    labels = [f"{e} - {event_names.get(e, e)}" for e in events]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左：准确率
    colors_acc = ["#E74C3C" if a < 0.5 else "#27AE60" if a > 0.7 else "#F39C12" for a in accs]
    axes[0].barh(labels, accs, color=colors_acc)
    axes[0].axvline(x=data["mean_accuracy"], linestyle="--", color="gray", alpha=0.7,
                    label=f"Mean={data['mean_accuracy']:.3f}")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Per-Held-Out-Event Accuracy (CV)", fontsize=12, fontweight="bold")
    axes[0].set_xlim(0, 1.0)
    axes[0].legend()
    axes[0].grid(True, axis="x", alpha=0.3)

    # 右：FP vs FN
    x_pos = np.arange(len(labels))
    width = 0.35
    axes[1].barh(x_pos - width / 2, fps, width, label="FP", color="#E74C3C")
    axes[1].barh(x_pos + width / 2, fns, width, label="FN", color="#3498DB")
    axes[1].set_yticks(x_pos)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Count")
    axes[1].set_title("False Positives vs False Negatives per Event", fontsize=12, fontweight="bold")
    axes[1].legend()

    plt.suptitle("Per-Event Cross-Validation Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "per_event_accuracy.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 按事件准确率图: {out}")


# ═══════════════════════════════════════════════════════════════
# 5. 性能仪表盘汇总图
# ═══════════════════════════════════════════════════════════════

def plot_performance_dashboard():
    """生成一张综合性能仪表盘，适合放在报告首页。"""
    fig = plt.figure(figsize=(12, 10))

    # ── 5a. 关键指标卡片 (左上) ──
    ax1 = fig.add_axes([0.05, 0.55, 0.42, 0.4])
    ax1.axis("off")

    metrics = [
        ("Accuracy (val.csv)", "88.03%", "#27AE60"),
        ("Macro F1", "0.8782", "#2ECC71"),
        ("CV Mean Accuracy", "64.9% ± 18.4%", "#F39C12"),
        ("CV Macro F1", "0.568 ± 0.185", "#3498DB"),
        ("Threshold (τ)", "0.46", "#9B59B6"),
    ]

    for i, (name, value, color) in enumerate(metrics):
        y = 0.8 - i * 0.18
        ax1.add_patch(plt.Rectangle((0, y - 0.06), 0.95, 0.14, fill=True, facecolor=color, alpha=0.15,
                                     transform=ax1.transAxes, edgecolor=color, linewidth=1.5))
        ax1.text(0.05, y, name, transform=ax1.transAxes, fontsize=12, fontweight="bold", va="center")
        ax1.text(0.95, y, value, transform=ax1.transAxes, fontsize=16, fontweight="bold",
                 va="center", ha="right", color=color)

    ax1.set_title("Key Metrics", fontsize=14, fontweight="bold", loc="left")

    # ── 5b. 混淆矩阵小图 (右上) ──
    ax2 = fig.add_axes([0.55, 0.55, 0.4, 0.4])
    cm = np.array([[203, 23], [25, 150]])
    im = ax2.imshow(cm, cmap="YlOrRd")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred Non-Rumor", "Pred Rumor"], fontsize=9)
    ax2.set_yticklabels(["True Non-Rumor", "True Rumor"], fontsize=9)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, fontweight="bold",
                     color="white" if cm[i, j] > 150 else "black")

    # 标注 TN/FP/FN/TP
    ax2.text(0, 0, f"  TN\n  {cm[0,0]}", ha="center", va="top", fontsize=8, color="gray")
    ax2.text(1, 0, f"  FP\n  {cm[0,1]}", ha="center", va="top", fontsize=8, color="gray")
    ax2.text(0, 1, f"  FN\n  {cm[1,0]}", ha="center", va="top", fontsize=8, color="gray")
    ax2.text(1, 1, f"  TP\n  {cm[1,1]}", ha="center", va="top", fontsize=8, color="gray")

    ax2.set_title("Confusion Matrix on val.csv (401 samples)", fontsize=12, fontweight="bold")

    # ── 5c. CV 折线 (下方全宽) ──
    cv_path = OUTPUTS_DIR / "cv_results.json"
    if cv_path.exists():
        with open(cv_path, "r") as f:
            cv_data = json.load(f)
        folds = cv_data["folds"]

        ax3 = fig.add_axes([0.05, 0.05, 0.9, 0.42])
        fold_nums = [f["fold"] for f in folds]
        fold_events = [f["val_event"] for f in folds]
        x_f = range(len(folds))
        ax3.plot(x_f, [f["accuracy"] for f in folds], "s-", color="#3498DB", linewidth=2, markersize=8, label="Accuracy")
        ax3.plot(x_f, [f["f1"] for f in folds], "D-", color="#E74C3C", linewidth=2, markersize=8, label="F1 (Rumor)")
        ax3.plot(x_f, [f["macro_f1"] for f in folds], "o-", color="#2ECC71", linewidth=2, markersize=8, label="Macro F1")
        ax3.axhline(y=cv_data["mean_accuracy"], linestyle="--", color="#2980B9", alpha=0.4)
        ax3.axhline(y=cv_data["mean_f1"], linestyle="--", color="#C0392B", alpha=0.4)
        ax3.axhline(y=cv_data["mean_macro_f1"], linestyle="--", color="#27AE60", alpha=0.4)

        # 标注难/易事件
        hard_events = [folds[i] for i, f in enumerate(folds) if f["accuracy"] < 0.5]
        easy_events = [folds[i] for i, f in enumerate(folds) if f["accuracy"] > 0.7]
        for f in hard_events:
            ax3.annotate(f"Event {f['val_event']}★", (f["fold"] - 1, f["accuracy"]),
                        textcoords="offset points", xytext=(0, -20), fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))
        for f in easy_events:
            ax3.annotate(f"Event {f['val_event']}", (f["fold"] - 1, f["accuracy"]),
                        textcoords="offset points", xytext=(0, 12), fontsize=8, color="green",
                        arrowprops=dict(arrowstyle="->", color="green", alpha=0.5))

        ax3.set_xticks(x_f)
        ax3.set_xticklabels([f"Fold {n}\n(ev.{e})" for n, e in zip(fold_nums, fold_events)], fontsize=9)
        ax3.set_ylim(0, 1.05)
        ax3.legend(loc="lower left", ncol=3)
        ax3.set_title("7-Fold Leave-One-Event-Out Cross Validation", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)

    out = FIG_DIR / "performance_dashboard.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 性能仪表盘: {out}")


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("生成报告图表...")
    print("=" * 50)

    plot_cv_results()
    plot_confusion_matrix()
    plot_backbone_comparison()
    plot_per_event_accuracy()
    plot_performance_dashboard()

    print()
    print("=" * 50)
    print(f"✅ 全部图表已保存至: {FIG_DIR}")
    print("=" * 50)
