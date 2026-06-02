#!/usr/bin/env python3
"""推理入口脚本。

用法：
    # 交互式推理
    python scripts/predict.py

    # 单条推理
    python scripts/predict.py --text "BREAKING: Ferguson police chief says..."
"""

import argparse
import io
import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 国内环境 HuggingFace 镜像配置（必须在 import transformers 之前）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("no_proxy", "*")

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.preprocessing import clean_text
from model.model import RumorClassifier
from model.trainer import load_threshold

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def predict_single(text: str, model: RumorClassifier, tokenizer, device: str, max_length: int = 128) -> dict:
    """对单条推文进行推理。

    判定规则：P(谣言) > τ，其中 τ 由 outputs/threshold.json 提供；
    未校准时回退到 0.5，等价于原 argmax 行为。

    Args:
        text: 原始推文文本。
        model: 已加载的分类模型。
        tokenizer: HuggingFace tokenizer。
        device: 设备。
        max_length: token 最大长度。

    Returns:
        {"label": 0/1, "label_name": "谣言"/"非谣言", "confidence": 0.XX, "threshold": τ}
    """
    clean = clean_text(text)
    encoded = tokenizer(
        clean,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    threshold = load_threshold()
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = 1 if probs[0, 1].item() > threshold else 0
        confidence = probs[0, pred].item()

    return {
        "label": pred,
        "label_name": "谣言" if pred == 1 else "非谣言",
        "confidence": confidence,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="谣言检测单条推理")
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "final_model.pt"),
        help="模型文件路径",
    )
    parser.add_argument("--text", type=str, default=None, help="待检测的推文文本")
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
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    print(f"模型: {model.model_name}")

    if args.text:
        # 单条推理模式
        result = predict_single(args.text, model, tokenizer, device, args.max_length)
        print(f"\n输入文本: {args.text}")
        print(f"检测结果: {result['label_name']}")
        print(f"置信度:   {result['confidence']:.4f}")
    else:
        # 交互式模式
        print("\n交互式推理模式（输入 quit 退出）")
        print("-" * 50)
        while True:
            try:
                text = input("\n请输入推文文本: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() == "quit":
                break
            if not text:
                continue

            result = predict_single(text, model, tokenizer, device, args.max_length)
            print(f"→ {result['label_name']}（置信度: {result['confidence']:.4f}）")


if __name__ == "__main__":
    main()
