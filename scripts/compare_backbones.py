#!/usr/bin/env python3
"""Backbone 对比评估脚本。

对给定的若干 .pt 模型分别加载并在 val.csv 上评估，
汇总成 outputs/backbone_comparison.json 与一张控制台表格。

用法（默认对比 twitter-roberta-base 基线 + 4 个新 backbone）：
    python scripts/compare_backbones.py

    # 自定义模型列表
    python scripts/compare_backbones.py --models outputs/final_model.pt outputs/final_debertav3base.pt
"""

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

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
    str(OUTPUTS_DIR / "final_model.pt"),           # twitter-roberta-base 基线
    str(OUTPUTS_DIR / "final_debertav3base.pt"),
    str(OUTPUTS_DIR / "final_bertweetbase.pt"),
    str(OUTPUTS_DIR / "final_robertabase.pt"),
    str(OUTPUTS_DIR / "final_debertav3large.pt"),
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


def eval_one(model_path: str, texts, labels, device: str, batch_size: int, max_length: int, threshold: float):
    """加载并评估单个模型，返回 metrics dict + 元数据。"""
    model = RumorClassifier.load(model_path, device=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    backbone = model.model_name

    ds = RumorDataset(texts, labels, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for b in loader:
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1)
            p = (probs[:, 1] > threshold).long().cpu().tolist()
            preds.extend(p)

    metrics = _compute_metrics(preds, labels)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return backbone, metrics


def main():
    parser = argparse.ArgumentParser(description="Backbone 对比评估")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "val.csv"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    threshold = load_threshold()
    print(f"设备: {device}")
    print(f"阈值: τ={threshold:.4f}")
    print(f"数据: {args.data}")
    print()

    texts, labels = load_val(Path(args.data))
    print(f"样本数: {len(texts)}")

    rows = []
    for path in args.models:
        if not Path(path).exists():
            print(f"  [跳过] {path} 不存在")
            continue
        print(f"\n  >> {Path(path).name}")
        backbone, m = eval_one(path, texts, labels, device, args.batch_size, args.max_length, threshold)
        print(f"     backbone={backbone}")
        print(f"     acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
              f"f1(谣言)={m['f1']:.4f}  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
        rows.append({
            "checkpoint": path,
            "backbone": backbone,
            **m,
        })

    # 控制台对比表
    print()
    print("=" * 92)
    print(f"{'checkpoint':<38} | {'backbone':<32} | {'acc':>7} | {'mF1':>7}")
    print("-" * 92)
    base_acc = rows[0]["accuracy"] if rows else 0
    base_mf1 = rows[0]["macro_f1"] if rows else 0
    for r in rows:
        ckpt = Path(r["checkpoint"]).name
        da = r["accuracy"] - base_acc
        dm = r["macro_f1"] - base_mf1
        print(f"{ckpt:<38} | {r['backbone']:<32} | {r['accuracy']:>7.4f} | {r['macro_f1']:>7.4f}  "
              f"(Δacc={da:+.4f}, Δmf1={dm:+.4f})")
    print("=" * 92)

    # 输出 JSON
    out = {
        "threshold": threshold,
        "n_samples": len(texts),
        "baseline_checkpoint": rows[0]["checkpoint"] if rows else None,
        "rows": rows,
    }
    out_path = OUTPUTS_DIR / "backbone_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
