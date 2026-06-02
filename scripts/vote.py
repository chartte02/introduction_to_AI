#!/usr/bin/env python3
"""8 模型 soft voting 评估脚本。

把 outputs/ 下的 7 个 CV fold 模型与 final_model 全部加载，对 val.csv
逐模型推理收集 softmax probs，按位平均后用 outputs/threshold.json 中
校准过的 τ（若不存在则用 0.5）做最终判定，并与 single final_model 基线对比。

用法：
    python scripts/vote.py
    python scripts/vote.py --data data/val.csv --models outputs/fold_1_best.pt outputs/final_model.pt
"""

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("no_proxy", "*")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.data import RumorDataset
from model.model import RumorClassifier
from model.preprocessing import clean_text
from model.trainer import _compute_metrics, load_threshold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODELS = [
    str(OUTPUTS_DIR / "final_model.pt"),
    *[str(OUTPUTS_DIR / f"fold_{i}_best.pt") for i in range(1, 8)],
]


def load_val(csv_path: Path):
    texts, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            texts.append(clean_text(row[1]))
            labels.append(int(row[2]))
    return texts, labels


def model_probs(model_path: str, texts, labels, device: str, batch_size: int, max_length: int):
    """加载单个模型，对 val 推理返回 P(谣言)[N] 列表。"""
    model = RumorClassifier.load(model_path, device=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    ds = RumorDataset(texts, labels, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs_pos = []
    with torch.no_grad():
        for b in loader:
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1)
            probs_pos.extend(probs[:, 1].cpu().tolist())

    # 显存释放
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return probs_pos


def main():
    parser = argparse.ArgumentParser(description="8 模型 soft voting 评估")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="参与投票的模型路径列表（默认 final_model + 7 fold）")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "val.csv"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    threshold = load_threshold()
    print(f"设备: {device}")
    print(f"阈值: τ={threshold:.4f}  (来自 outputs/threshold.json 或回退 0.5)")
    print(f"数据: {args.data}")
    print(f"参与投票的模型 ({len(args.models)} 个):")
    for p in args.models:
        print(f"  - {p}")

    # 1. 加载 val
    texts, labels = load_val(Path(args.data))
    print(f"样本数: {len(texts)}")

    # 2. 逐模型推理并收集 probs
    all_probs = []
    per_model = []
    for path in args.models:
        print(f"\n  >> 推理 {Path(path).name} ...")
        probs_pos = model_probs(path, texts, labels, device, args.batch_size, args.max_length)
        preds_single = [1 if p > threshold else 0 for p in probs_pos]
        m_single = _compute_metrics(preds_single, labels)
        per_model.append({
            "model_path": path,
            "accuracy": m_single["accuracy"],
            "macro_f1": m_single["macro_f1"],
            "f1": m_single["f1"],
        })
        print(f"     单模型 @ τ={threshold:.2f}: acc={m_single['accuracy']:.4f}  macro_f1={m_single['macro_f1']:.4f}")
        all_probs.append(probs_pos)

    # 3. soft vote: probs 按位平均
    n = len(texts)
    avg = [sum(all_probs[k][i] for k in range(len(all_probs))) / len(all_probs) for i in range(n)]
    preds_vote = [1 if p > threshold else 0 for p in avg]
    m_vote = _compute_metrics(preds_vote, labels)

    # 4. 基线：single final_model @ τ
    baseline = per_model[0]  # final_model 是 DEFAULT_MODELS 第一个

    # 5. 汇总打印
    print()
    print("=" * 64)
    print(f"  baseline (single final_model @ τ={threshold:.2f}):")
    print(f"     acc={baseline['accuracy']:.4f}   macro_f1={baseline['macro_f1']:.4f}")
    print(f"  vote-{len(args.models)} (probs 平均 @ τ={threshold:.2f}):")
    print(f"     acc={m_vote['accuracy']:.4f}   macro_f1={m_vote['macro_f1']:.4f}")
    delta_acc = m_vote["accuracy"] - baseline["accuracy"]
    delta_mf1 = m_vote["macro_f1"] - baseline["macro_f1"]
    print(f"     Δ acc      = {delta_acc:+.4f}")
    print(f"     Δ macro_f1 = {delta_mf1:+.4f}")
    print(f"  混淆矩阵 (vote): TP={m_vote['tp']}, TN={m_vote['tn']}, FP={m_vote['fp']}, FN={m_vote['fn']}")
    print("=" * 64)

    # 6. 写 vote_results.json
    out = {
        "variant": f"vote-{len(args.models)} (probs 平均)",
        "threshold": threshold,
        "n_samples": len(texts),
        "models": args.models,
        "per_model_at_threshold": per_model,
        "vote_metrics": m_vote,
        "baseline_single_final_model": baseline,
        "delta_vs_baseline": {
            "accuracy": delta_acc,
            "macro_f1": delta_mf1,
        },
    }
    out_path = OUTPUTS_DIR / "vote_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
