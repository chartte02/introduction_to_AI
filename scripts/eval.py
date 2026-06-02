#!/usr/bin/env python3
"""评估入口脚本。

用法：
    # 在 val.csv 上评估最终模型
    python scripts/eval.py

    # 评估指定模型和数据集
    python scripts/eval.py --model outputs/final_model.pt --data data/val.csv
"""

import argparse
import csv
import io
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 上（支持直接运行 python ai_scripts/eval.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 国内环境 HuggingFace 镜像配置（必须在 import transformers 之前）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("no_proxy", "*")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.preprocessing import clean_text
from model.model import RumorClassifier
from model.data import RumorDataset
from model.trainer import evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="谣言检测模型评估")
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "final_model.pt"),
        help="模型文件路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "val.csv"),
        help="评估数据集 CSV",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--max_length", type=int, default=128, help="最大 token 长度")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu），默认自动检测",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    model = RumorClassifier.load(args.model, device=device)
    model.eval()

    # 加载数据
    texts, labels = [], []
    with open(args.data, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            texts.append(clean_text(row[1]))
            labels.append(int(row[2]))

    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    dataset = RumorDataset(texts, labels, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 评估
    metrics = evaluate(model, dataloader, device)

    print(f"\n{'='*40}")
    print(f"评估结果 — {args.data}")
    print(f"  样本数:        {len(texts)}")
    print(f"  准确率:        {metrics['accuracy']:.4f}")
    print(f"  平衡准确率:    {metrics['balanced_accuracy']:.4f}")
    print(f"  macro_f1:      {metrics['macro_f1']:.4f}")
    print(f"  [谣言]   P/R/F1: {metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}")
    print(f"  [非谣言] P/R/F1: {metrics['precision_neg']:.4f} / {metrics['recall_neg']:.4f} / {metrics['f1_neg']:.4f}")
    print(f"  混淆矩阵: TP={metrics['tp']}, TN={metrics['tn']}, "
          f"FP={metrics['fp']}, FN={metrics['fn']}")


if __name__ == "__main__":
    main()
