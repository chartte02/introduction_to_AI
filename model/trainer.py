"""训练模块。

提供：
  - train_epoch()：单轮训练
  - evaluate()：验证集评估
  - run_cross_validation()：留一事件交叉验证主循环
  - train_final_model()：全量训练最终模型

网络说明：
  HuggingFace 在国内可能无法直连，支持通过环境变量配置镜像：
    export HF_ENDPOINT=https://hf-mirror.com
    export no_proxy="*"
  或在运行前设置：
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
"""

import json
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Windows 控制台 UTF-8 输出（仅当 stdout 未被上游替换时）
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 自动配置 HuggingFace 镜像（国内环境）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("no_proxy", "*")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model.data import (
    load_data,
    get_event_folds,
    build_fold_datasets,
    RumorDataset,
    SEED,
)
from model.model import RumorClassifier

# 路径常量
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 固定随机种子
torch.manual_seed(SEED)

# 默认超参
DEFAULT_HP = {
    "model_name": "cardiffnlp/twitter-roberta-base",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def create_dataloader(dataset: RumorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """创建 DataLoader。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> float:
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """在验证集上评估，返回准确率、精确率、召回率、F1。"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 计算指标
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    accuracy = (tp + tn) / max(len(all_preds), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_cross_validation(
    model_name: str = DEFAULT_HP["model_name"],
    max_length: int = DEFAULT_HP["max_length"],
    batch_size: int = DEFAULT_HP["batch_size"],
    learning_rate: float = DEFAULT_HP["learning_rate"],
    num_epochs: int = DEFAULT_HP["num_epochs"],
    warmup_ratio: float = DEFAULT_HP["warmup_ratio"],
    weight_decay: float = DEFAULT_HP["weight_decay"],
    dropout: float = DEFAULT_HP["dropout"],
    device: str = None,
) -> Dict:
    """运行留一事件交叉验证。

    对 7 个事件各做一次 held-out 验证，记录每个 fold 的评估结果。

    Returns:
        包含所有 fold 结果和汇总统计的字典。
    """
    if device is None:
        device = DEFAULT_HP["device"]

    print(f"设备: {device}")
    print(f"模型: {model_name}")
    print(f"超参: lr={learning_rate}, epochs={num_epochs}, batch={batch_size}, max_len={max_length}")

    # 加载全量数据
    texts, labels, events = load_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    folds = get_event_folds()
    all_results = []

    for i, fold in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {i+1}/7: 验证事件 = {fold['val_event']}, 训练事件 = {fold['train_events']}")
        print(f"{'='*50}")

        # 构建当前 fold 的数据集
        train_ds, val_ds = build_fold_datasets(
            texts, labels, events, fold, tokenizer, max_length
        )
        train_loader = create_dataloader(train_ds, batch_size, shuffle=True)
        val_loader = create_dataloader(val_ds, batch_size, shuffle=False)

        print(f"  训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

        # 初始化模型
        model = RumorClassifier(
            model_name=model_name,
            num_labels=2,
            dropout=dropout,
        )
        model.to(device)

        # 优化器与调度器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # 训练循环
        best_f1 = 0.0
        best_epoch = 0
        for epoch in range(num_epochs):
            start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            metrics = evaluate(model, val_loader, device)
            elapsed = time.time() - start

            print(
                f"  Epoch {epoch+1}/{num_epochs} | "
                f"loss={train_loss:.4f} | "
                f"acc={metrics['accuracy']:.4f} | "
                f"f1={metrics['f1']:.4f} | "
                f"耗时={elapsed:.1f}s"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_epoch = epoch + 1

                # 保存最佳模型
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                model.save(str(OUTPUTS_DIR / f"fold_{i+1}_best.pt"))

        fold_result = {
            "fold": i + 1,
            "val_event": fold["val_event"],
            "best_f1": best_f1,
            "best_epoch": best_epoch,
        }
        # 汇总最终 epoch 的完整指标
        final_metrics = evaluate(model, val_loader, device)
        fold_result.update(final_metrics)
        all_results.append(fold_result)

        print(f"  ✅ 最佳 F1={best_f1:.4f} (epoch {best_epoch})")

    # 汇总结果
    accuracies = [r["accuracy"] for r in all_results]
    f1s = [r["f1"] for r in all_results]
    summary = {
        "model_name": model_name,
        "hyperparams": {
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "dropout": dropout,
        },
        "folds": all_results,
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "std_accuracy": (sum((a - sum(accuracies) / len(accuracies)) ** 2 for a in accuracies) / len(accuracies)) ** 0.5,
        "mean_f1": sum(f1s) / len(f1s),
        "std_f1": (sum((f - sum(f1s) / len(f1s)) ** 2 for f in f1s) / len(f1s)) ** 0.5,
    }

    # 保存完整结果
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"交叉验证汇总")
    print(f"  平均准确率: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"  平均 F1:    {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"  结果已保存至: {OUTPUTS_DIR / 'cv_results.json'}")

    return summary


def train_final_model(
    model_name: str = DEFAULT_HP["model_name"],
    max_length: int = DEFAULT_HP["max_length"],
    batch_size: int = DEFAULT_HP["batch_size"],
    learning_rate: float = DEFAULT_HP["learning_rate"],
    num_epochs: int = DEFAULT_HP["num_epochs"],
    warmup_ratio: float = DEFAULT_HP["warmup_ratio"],
    weight_decay: float = DEFAULT_HP["weight_decay"],
    dropout: float = DEFAULT_HP["dropout"],
    device: str = None,
) -> RumorClassifier:
    """在全部 train.csv 上训练最终模型。

    Args:
        （参数同 run_cross_validation）

    Returns:
        训练好的 RumorClassifier 实例。
    """
    if device is None:
        device = DEFAULT_HP["device"]

    print(f"设备: {device}")
    print("在全部训练数据上训练最终模型...")

    texts, labels, _events = load_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = RumorDataset(texts, labels, tokenizer, max_length)
    dataloader = create_dataloader(dataset, batch_size, shuffle=True)

    model = RumorClassifier(model_name=model_name, num_labels=2, dropout=dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(num_epochs):
        start = time.time()
        loss = train_epoch(model, dataloader, optimizer, scheduler, device)
        elapsed = time.time() - start
        print(f"  Epoch {epoch+1}/{num_epochs} | loss={loss:.4f} | 耗时={elapsed:.1f}s")

    # 保存最终模型
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUTS_DIR / "final_model.pt"))
    tokenizer.save_pretrained(str(OUTPUTS_DIR / "tokenizer"))
    print(f"最终模型已保存至: {OUTPUTS_DIR / 'final_model.pt'}")

    return model
