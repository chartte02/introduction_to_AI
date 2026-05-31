#!/usr/bin/env python3
"""训练入口脚本。

用法：
    # 运行留一事件交叉验证
    python ai_scripts/train.py --cv

    # 在全部训练集上训练最终模型
    python ai_scripts/train.py

    # 指定模型和其他超参
    python ai_scripts/train.py --model bert-base-uncased --epochs 8 --lr 3e-5
"""

import argparse

from ai_model.trainer import run_cross_validation, train_final_model


def main():
    parser = argparse.ArgumentParser(description="谣言检测模型训练")
    parser.add_argument(
        "--model",
        type=str,
        default="cardiffnlp/twitter-roberta-base",
        help="HuggingFace 预训练模型名",
    )
    parser.add_argument("--max_length", type=int, default=128, help="最大 token 长度")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--warmup", type=float, default=0.1, help="warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 率")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="运行留一事件交叉验证（7折）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu），默认自动检测",
    )

    args = parser.parse_args()

    if args.cv:
        print("=" * 60)
        print("留一事件交叉验证模式（Leave-One-Event-Out CV）")
        print("=" * 60)
        run_cross_validation(
            model_name=args.model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            warmup_ratio=args.warmup,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            device=args.device,
        )
    else:
        print("=" * 60)
        print("最终模型训练模式（全部训练集）")
        print("=" * 60)
        train_final_model(
            model_name=args.model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            warmup_ratio=args.warmup,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            device=args.device,
        )


if __name__ == "__main__":
    main()
