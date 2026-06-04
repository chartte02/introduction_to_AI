#!/usr/bin/env python3
"""阈值校准入口脚本。

在 final_model 训练时使用的同一 dev 集上扫描 τ ∈ [0.05, 0.95]，
找到使 accuracy / macro_f1 最大的阈值，写入 outputs/threshold.json。
后续 evaluate() 与 predict_single() 会自动读取该文件，所有下游推理
（eval.py / predict.py / vote.py）默认走最优阈值。

用法：
    python scripts/calibrate.py
    python scripts/calibrate.py --model outputs/final_model.pt --metric accuracy
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台 UTF-8 输出
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 国内 HuggingFace 镜像（须在 import transformers 之前）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("no_proxy", "*")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.data import RumorDataset, load_data, stratified_split
from model.model import RumorClassifier
from model.trainer import _compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def collect_probs(
    model: RumorClassifier,
    dataloader: DataLoader,
    device: str,
):
    """前向收集每条样本的 P(谣言) 与对应 label。"""
    model.eval()
    probs_pos, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            probs_pos.extend(probs[:, 1].cpu().tolist())
            labels.extend(batch["label"].tolist())
    return probs_pos, labels


def sweep_threshold(probs_pos, labels, taus):
    """对每个 τ 计算 metric dict。"""
    sweep = []
    for tau in taus:
        preds = [1 if p > tau else 0 for p in probs_pos]
        m = _compute_metrics(preds, labels)
        sweep.append({
            "tau": round(tau, 4),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "f1": m["f1"],
            "balanced_accuracy": m["balanced_accuracy"],
        })
    return sweep


def main():
    parser = argparse.ArgumentParser(description="阈值校准：在 dev 集上扫最优 τ")
    parser.add_argument(
        "--model",
        type=str,
        default=str(OUTPUTS_DIR / "final_model.pt"),
        help="模型文件路径（默认 outputs/final_model.pt）",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "macro_f1"],
        help="写入 threshold.json 的 'tau' 字段以哪个指标为目标（默认 accuracy）",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dev_ratio", type=float, default=0.1,
                        help="必须与训练时一致（默认 0.1），保证重建相同的 dev 集")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1. 加载模型 + tokenizer
    model = RumorClassifier.load(args.model, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    print(f"模型: {model.model_name}  ←  {args.model}")

    # 2. 重建 final_model 训练时用的 dev 集（同 seed=42 同 dev_ratio=0.1）
    texts, labels, _events = load_data()
    _train_texts, _train_labels, dev_texts, dev_labels = stratified_split(
        texts, labels, dev_ratio=args.dev_ratio
    )
    print(f"dev 集大小: {len(dev_texts)}（谣言 {sum(dev_labels)} / 非谣言 {len(dev_labels) - sum(dev_labels)}）")

    dev_ds = RumorDataset(dev_texts, dev_labels, tokenizer, args.max_length)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    # 3. 收集 probs，扫 τ
    probs_pos, dev_labels_collected = collect_probs(model, dev_loader, device)
    assert dev_labels_collected == dev_labels, "dev 顺序错乱"

    taus = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05 … 0.95
    sweep = sweep_threshold(probs_pos, dev_labels, taus)

    # 4. 找各指标最优 τ
    best_acc = max(sweep, key=lambda r: (r["accuracy"], -abs(r["tau"] - 0.5)))
    best_mf1 = max(sweep, key=lambda r: (r["macro_f1"], -abs(r["tau"] - 0.5)))

    # baseline @ τ=0.5
    base = next(r for r in sweep if r["tau"] == 0.5)

    # 选写入 threshold.json 的 τ
    chosen = best_acc if args.metric == "accuracy" else best_mf1

    out = {
        "tau": chosen["tau"],
        "metric": args.metric,
        "tau_acc": best_acc["tau"],
        "tau_macro_f1": best_mf1["tau"],
        "dev_acc_at_tau": chosen["accuracy"],
        "dev_macro_f1_at_tau": chosen["macro_f1"],
        "baseline_at_0.5": {"accuracy": base["accuracy"], "macro_f1": base["macro_f1"]},
        "sweep": sweep,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "threshold.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"  dev @ τ=0.5    :  acc={base['accuracy']:.4f}  macro_f1={base['macro_f1']:.4f}")
    print(f"  dev @ τ={best_acc['tau']:.2f} (best acc)     :  acc={best_acc['accuracy']:.4f}  macro_f1={best_acc['macro_f1']:.4f}")
    print(f"  dev @ τ={best_mf1['tau']:.2f} (best macro_f1):  acc={best_mf1['accuracy']:.4f}  macro_f1={best_mf1['macro_f1']:.4f}")
    print("=" * 60)
    print(f"  已写入 {out_path}（生效阈值：τ={chosen['tau']:.2f}，目标={args.metric}）")


if __name__ == "__main__":
    main()
