#!/usr/bin/env python3
"""训练入口脚本。

用法：
    # 留一事件交叉验证（评估泛化能力）
    python ai_scripts/train.py --cv

    # 训练最终模型（默认：分层 dev 集 + 早停 + 保存最佳 epoch）
    python ai_scripts/train.py

    # 指定模型和其他超参
    python ai_scripts/train.py --model bert-base-uncased --epochs 8 --lr 3e-5
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 上（支持直接运行 python ai_scripts/train.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_model.trainer import run_cross_validation, train_final_model


def main():
    parser = argparse.ArgumentParser(description="谣言检测模型训练")
    parser.add_argument("--model", type=str, default="cardiffnlp/twitter-roberta-base",
                        help="HuggingFace 预训练模型名")
    parser.add_argument("--max_length", type=int, default=128, help="最大 token 长度")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--warmup", type=float, default=0.1, help="warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 率")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")

    # 类别不平衡：默认开启类别加权损失
    parser.add_argument("--class_weights", dest="class_weights", action="store_true",
                        default=True, help="启用类别加权损失（默认开启）")
    parser.add_argument("--no_class_weights", dest="class_weights", action="store_false",
                        help="关闭类别加权损失")

    # 最佳 epoch / 折选择指标
    parser.add_argument("--select_metric", type=str, default="macro_f1",
                        choices=["macro_f1", "f1", "accuracy", "balanced_accuracy"],
                        help="选择最佳 epoch / 折的指标")

    # 最终模型训练：验证集 + 早停
    parser.add_argument("--dev_ratio", type=float, default=0.1,
                        help="从训练集分层切出的 dev 集比例（<=0 则全量训练并保存最后一轮）")
    parser.add_argument("--patience", type=int, default=2, help="早停容忍轮数")

    parser.add_argument("--cv", action="store_true", help="运行留一事件交叉验证（7折）")
    parser.add_argument("--device", type=str, default=None, help="设备（cuda/cpu），默认自动检测")

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
            max_grad_norm=args.max_grad_norm,
            use_class_weights=args.class_weights,
            select_metric=args.select_metric,
            device=args.device,
        )
    else:
        print("=" * 60)
        print("最终模型训练模式")
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
            max_grad_norm=args.max_grad_norm,
            use_class_weights=args.class_weights,
            select_metric=args.select_metric,
            dev_ratio=args.dev_ratio,
            patience=args.patience,
            device=args.device,
        )


if __name__ == "__main__":
    main()
