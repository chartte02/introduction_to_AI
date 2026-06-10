#!/usr/bin/env python3
"""可解释性分析图表生成脚本。

基于现有数据（关键词 CSV / 特殊字符 CSV / CV 结果 / val 评估日志）
生成报告 §2.4「判断依据分析」所需的分析图。

不需要加载模型，仅依赖 matplotlib + numpy。

用法：
    python scripts/generate_explainability_figures.py
"""

import csv
import json
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 200

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
COLORS_RUMOR = "#E74C3C"
COLORS_NONRUMOR = "#3498DB"


# ──────────────────────────────────────────────
# 1. 关键词重要性对比图（谣言 vs 非谣言 Top-15）
# ──────────────────────────────────────────────

def plot_keyword_importance():
    """基于 CS2112 关键词统计绘制对比柱状图。"""
    rumor_kw = _load_keywords(OUTPUTS_DIR / "tables" / "rumor_keywords.csv")
    non_kw   = _load_keywords(OUTPUTS_DIR / "tables" / "non_rumor_keywords.csv")

    if not rumor_kw or not non_kw:
        print("[跳过] 关键词 CSV 缺失")
        return

    # 取 Top-10 并合并为统一关键词集（交集 + 差异）
    rumor_set = {w: c for w, c in rumor_kw[:10]}
    non_set   = {w: c for w, c in non_kw[:10]}
    all_words = list(dict.fromkeys([w for w, _ in rumor_kw[:10]] + [w for w, _ in non_kw[:10]]))

    rumor_vals = [rumor_set.get(w, 0) for w in all_words]
    non_vals   = [non_set.get(w, 0) for w in all_words]

    x = np.arange(len(all_words))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, rumor_vals, width, label="Rumor", color=COLORS_RUMOR, alpha=0.85)
    ax.bar(x + width / 2, non_vals, width, label="Non-Rumor", color=COLORS_NONRUMOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_words, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Frequency in Training Set", fontsize=12)
    ax.set_title("Top Keywords: Rumor vs Non-Rumor", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "explain_keyword_importance.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 关键词重要性对比: {out}")


# ──────────────────────────────────────────────
# 2. 语言特征分析图（URL / @ / # / ! / ?）
# ──────────────────────────────────────────────

def plot_language_features():
    """基于 special_characters_analysis.csv 绘制谣言 vs 非谣言语言特征对比。"""
    csv_path = OUTPUTS_DIR / "tables" / "special_characters_analysis.csv"
    if not csv_path.exists():
        print("[跳过] 语言特征 CSV 缺失")
        return

    features, rumor_pct, non_pct = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            features.append(row["Feature"])
            rumor_pct.append(float(row["Rumor (%)"]))
            non_pct.append(float(row["Non-Rumor (%)"]))

    # 转英文显示名
    label_map = {"URL": "Contains URL", "@mention": "@mention", "hashtag": "# Hashtag",
                 "exclamation": "! Exclamation", "question": "? Question"}

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, rumor_pct, width, label="Rumor", color=COLORS_RUMOR, alpha=0.85)
    bars2 = ax.bar(x + width / 2, non_pct, width, label="Non-Rumor", color=COLORS_NONRUMOR, alpha=0.85)

    # 数值标注
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8, color=COLORS_RUMOR)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8, color=COLORS_NONRUMOR)

    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(f, f) for f in features], fontsize=11)
    ax.set_ylabel("Occurrence Rate (%)", fontsize=12)
    ax.set_title("Language Feature Comparison: Rumor vs Non-Rumor", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(rumor_pct + non_pct) * 1.2)

    # 关键发现标注
    diff_hashtag = rumor_pct[2] - non_pct[2]
    ax.annotate(
        f"#Hashtag gap:\nNon-Rumor +{abs(diff_hashtag):.1f}%",
        xy=(2, non_pct[2]), xytext=(3.5, 70),
        fontsize=9, color="#8E44AD",
        arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", alpha=0.8),
    )

    plt.tight_layout()
    out = FIG_DIR / "explain_language_features.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 语言特征分析: {out}")


# ──────────────────────────────────────────────
# 3. 可解释性系统架构图
# ──────────────────────────────────────────────

def plot_explainability_architecture():
    """绘制可解释性双通道架构图。"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 配色
    C_INPUT = "#F39C12"
    C_MODEL = "#3498DB"
    C_ATTN = "#2ECC71"
    C_LLM = "#9B59B6"
    C_OUTPUT = "#E74C3C"
    C_ARROW = "#7F8C8D"

    def box(x, y, w, h, text, color, fontsize=11, bold=False):
        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor="white", alpha=0.9, linewidth=2)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=weight, color="white")

    def arrow(x1, y1, x2, y2, label="", color=C_ARROW):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my + 0.15, label, fontsize=8, color=color, style="italic")

    # 标题
    ax.text(7, 7.6, "Explainability System Architecture", ha="center", fontsize=16, fontweight="bold")

    # Row 1: 输入
    box(2, 6.5, 3, 1.0, "Input Tweet\n(Text)", C_INPUT, bold=True)

    # Row 2: 预处理 + BERT
    box(2, 4.8, 3, 1.0, "Preprocessing +\nBERT Classifier", C_MODEL, bold=True)
    arrow(2, 6.0, 2, 5.3)

    # Row 2.5: 分类输出
    box(2, 3.2, 3, 0.8, "Label + Confidence", C_MODEL)

    # Row 3 左: 注意力通道
    box(5.5, 4.8, 3.2, 1.2, "Attention Channel\n- Extract 12-layer weights\n- Aggregate cross-head\n- Token importance", C_ATTN)

    # Row 3 右: LLM 通道
    box(9.5, 4.8, 3.2, 1.2, "LLM Channel\n- SJTU DeepSeek API\n- 5-dimension analysis\n- Natural language output", C_LLM)

    # Row 4 左: 注意力输出
    box(5.5, 3.2, 3.2, 0.9, "Attention Outputs\n- Top-K keywords\n- Highlighted text\n- Heatmap", C_ATTN)

    # Row 4 右: LLM 输出
    box(9.5, 3.2, 3.2, 0.9, "LLM Outputs\n- Reasoning text\n- Multi-model comparison\n- Faithfulness eval", C_LLM)

    # Row 5: 融合
    box(7.5, 1.5, 5, 1.0, "Fused Explanation\n(Keywords + Natural Language)", C_OUTPUT, bold=True, fontsize=12)

    # 箭头
    arrow(2, 4.3, 5.5, 4.8, "BERT\nattentions")
    arrow(2, 4.3, 9.5, 4.8, "label +\nconfidence")
    arrow(5.5, 4.2, 5.5, 3.65)
    arrow(9.5, 4.2, 9.5, 3.65)
    arrow(5.5, 2.75, 7.5, 2.0, "")
    arrow(9.5, 2.75, 7.5, 2.0, "")

    # 标签
    ax.text(2, 1.0, "Classification\n(what?)", ha="center", fontsize=10, color=C_MODEL, fontweight="bold")
    ax.text(5.5, 1.0, "Attention\n(what did model see?)", ha="center", fontsize=10, color=C_ATTN, fontweight="bold")
    ax.text(9.5, 1.0, "LLM\n(why?)", ha="center", fontsize=10, color=C_LLM, fontweight="bold")

    # 图例
    ax.text(0.3, 0.3, "Dual-channel: Attention (offline, zero-cost) + LLM (online, rich semantics)",
            fontsize=9, color=C_ARROW, style="italic")

    plt.tight_layout()
    out = FIG_DIR / "explain_architecture.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 可解释性架构: {out}")


# ──────────────────────────────────────────────
# 4. 解释方式对比示意图（三种模式）
# ──────────────────────────────────────────────

def plot_explain_mode_comparison():
    """绘制三种解释方式的说服力对比示意图（展示管道）。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Three Explanation Modes", fontsize=15, fontweight="bold", y=1.01)

    modes = [
        ("Mode 1: Attention Only\n(Keywords)", COLORS_RUMOR,
         ["Input → BERT", "Extract\nattentions", "Aggregate\nlayers+heads", "Top-K\nkeywords", "Highlighted\ntext"]),
        ("Mode 2: LLM Only\n(Free Analysis)", "#9B59B6",
         ["Input → BERT", "Get label +\nconfidence", "Build LLM\nprompt", "SJTU API\ncall", "Reasoning\ntext"]),
        ("Mode 3: LLM + Attention\n(Guided Analysis)", "#E67E22",
         ["Input → BERT", "Get label +\nconfidence", "Extract\nTop-K words", "Build prompt\nwith keywords", "Guided\nreasoning"]),
    ]

    for ax, (title, color, steps) in zip(axes, modes):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color)

        for i, step in enumerate(steps):
            y = 10.5 - i * 2.2
            rect = FancyBboxPatch((1, y - 0.7), 8, 1.4, boxstyle="round,pad=0.1",
                                   facecolor=color, alpha=0.15 + i * 0.15,
                                   edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(5, y, step, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

            if i < len(steps) - 1:
                ax.annotate("", xy=(5, y - 0.75), xytext=(5, y - 1.5),
                           arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.8))

        # 特征标签
        features = {
            "Mode 1": "Offline\nZero API cost\nModel-intrinsic",
            "Mode 2": "Online\nRequires API\nRich semantics",
            "Mode 3": "Online\nRequires API\nBest of both",
        }
        key = title.split("\n")[0]
        ax.text(5, 0.3, features.get(key, ""), ha="center", fontsize=8, color="#7F8C8D", style="italic")

    plt.tight_layout()
    out = FIG_DIR / "explain_mode_comparison.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 解释方式对比: {out}")


# ──────────────────────────────────────────────
# 5. CV 每事件「难易度 vs 可解释性」散点图
# ──────────────────────────────────────────────

def plot_event_explainability():
    """基于 cv_results.json 绘制每事件准确率 + 样本量气泡图。"""
    cv_path = OUTPUTS_DIR / "cv_results.json"
    if not cv_path.exists():
        print("[跳过] cv_results.json 缺失")
        return

    with open(cv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    event_names = {"0": "Gurlitt", "1": "Ferguson", "2": "Ebola", "3": "Prince",
                   "4": "Germanwings", "5": "Sydney", "6": "Ottawa"}
    sizes = {"0": 66, "1": 799, "2": 9, "3": 162, "4": 327, "5": 854, "6": 623}

    folds = data["folds"]
    events = [f["val_event"] for f in folds]
    accs = [f["accuracy"] for f in folds]
    macros = [f["macro_f1"] for f in folds]
    fps = [f["fp"] for f in folds]
    fns = [f["fn"] for f in folds]
    bubble_sizes = [sizes.get(e, 50) * 0.5 for e in events]

    fig, ax = plt.subplots(figsize=(10, 7))

    # 颜色按准确率映射
    colors = plt.cm.RdYlGn([(a - 0.15) / 0.65 for a in accs])
    scatter = ax.scatter(macros, accs, s=bubble_sizes, c=accs, cmap="RdYlGn",
                          edgecolors="black", linewidth=1.2, alpha=0.85, zorder=5)

    for e, a, m, n in zip(events, accs, macros, [event_names.get(e, e) for e in events]):
        offset = 15 if e in ("3", "0") else -12
        ax.annotate(f"{n}\n(Acc={a:.2f})", (m, a), textcoords="offset points",
                    xytext=(0, offset), fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_xlabel("Macro F1", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Event Generalization Difficulty\n(Bubble size = training samples)", fontsize=14, fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Accuracy", fontsize=10)

    # 难易分区
    ax.axhline(y=0.65, linestyle="--", color="gray", alpha=0.4)
    ax.axvline(x=0.6, linestyle="--", color="gray", alpha=0.4)
    ax.text(0.7, 0.8, "Easy to\ngeneralize", fontsize=9, color="green", alpha=0.7, ha="center")
    ax.text(0.3, 0.25, "Hard to\ngeneralize", fontsize=9, color="red", alpha=0.7, ha="center")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "explain_event_difficulty.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 事件泛化难度: {out}")


# ──────────────────────────────────────────────
# helper
# ──────────────────────────────────────────────

def _load_keywords(path: Path) -> list:
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append((row["Keyword"], int(row["Frequency"])))
    return items


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("生成可解释性分析图表 ...")
    print(f"输出目录: {FIG_DIR}")
    print("=" * 50)

    plot_keyword_importance()
    plot_language_features()
    plot_explainability_architecture()
    plot_explain_mode_comparison()
    plot_event_explainability()

    print()
    print("=" * 50)
    print(f"完成: {len(list(FIG_DIR.glob('explain_*.png')))} 张图表已保存")
    print("=" * 50)
