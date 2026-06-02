"""训练模块。

提供：
  - train_epoch()：单轮训练（支持类别加权损失 + 梯度裁剪）
  - evaluate()：验证集评估（含 macro_f1 / balanced_accuracy）
  - run_cross_validation()：留一事件交叉验证主循环（按最佳 epoch 汇总）
  - train_final_model()：带验证集 + 早停 + 保存最佳 epoch 的最终模型训练

网络说明：
  HuggingFace 在国内可能无法直连，支持通过环境变量配置镜像：
    export HF_ENDPOINT=https://hf-mirror.com
    export no_proxy="*"
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
    stratified_split,
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
    "max_grad_norm": 1.0,
    "use_class_weights": True,
    "select_metric": "macro_f1",
    "dev_ratio": 0.1,
    "patience": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def create_dataloader(dataset: RumorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """创建 DataLoader。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(labels: List[int], num_classes: int = 2, device: str = "cpu") -> torch.Tensor:
    """按类别频率的逆比例计算损失权重（balanced 策略）。

    weight_c = total / (num_classes * count_c)；样本越少的类别权重越大，
    缓解类别不平衡导致的多数类偏置。
    """
    counts = [labels.count(c) for c in range(num_classes)]
    total = len(labels)
    weights = [total / (num_classes * max(cnt, 1)) for cnt in counts]
    return torch.tensor(weights, dtype=torch.float, device=device)


def _build_criterion(labels: List[int], use_class_weights: bool, device: str) -> nn.Module:
    """构建损失函数：可选按类别频率加权。"""
    if use_class_weights:
        weights = compute_class_weights(labels, device=device)
        print(f"  类别加权损失 weight={[round(w, 3) for w in weights.tolist()]}")
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """计算均值与（总体）标准差。"""
    n = max(len(values), 1)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return mean, std


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    criterion: nn.Module,
    max_grad_norm: float = 1.0,
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
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    """在验证集上评估。

    返回正类（谣言）与负类（非谣言）的 precision/recall/f1，
    以及对类别不平衡更稳健的 macro_f1 与 balanced_accuracy。
    """
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

    # 混淆矩阵（正类 = 谣言 = 1）
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    accuracy = (tp + tn) / max(len(all_preds), 1)

    # 正类（谣言）指标
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # 负类（非谣言）指标
    precision_neg = tn / max(tn + fn, 1)
    recall_neg = tn / max(tn + fp, 1)
    f1_neg = 2 * precision_neg * recall_neg / max(precision_neg + recall_neg, 1e-8)

    # 宏平均 / 平衡准确率（对类别不平衡更稳健）
    macro_f1 = (f1 + f1_neg) / 2
    balanced_accuracy = (recall + recall_neg) / 2

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
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
    max_grad_norm: float = DEFAULT_HP["max_grad_norm"],
    use_class_weights: bool = DEFAULT_HP["use_class_weights"],
    select_metric: str = DEFAULT_HP["select_metric"],
    device: str = None,
) -> Dict:
    """运行留一事件交叉验证。

    对 7 个事件各做一次 held-out 验证。每折按 select_metric 跟踪并保存最佳 epoch，
    汇总时统一使用各折「最佳 epoch」的完整指标（避免最佳轮与最后一轮口径不一致）。

    Returns:
        包含所有 fold 结果和汇总统计的字典。
    """
    if device is None:
        device = DEFAULT_HP["device"]

    print(f"设备: {device}")
    print(f"模型: {model_name}")
    print(f"超参: lr={learning_rate}, epochs={num_epochs}, batch={batch_size}, max_len={max_length}")
    print(f"类别加权: {use_class_weights} | 选择指标: {select_metric} | 梯度裁剪: {max_grad_norm}")

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

        # 损失函数（权重由当前 fold 的训练标签计算）
        criterion = _build_criterion(train_ds.labels, use_class_weights, device)

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

        # 训练循环：按 select_metric 跟踪最佳 epoch
        best_score = -1.0
        best_epoch = 0
        best_metrics = None
        for epoch in range(num_epochs):
            start = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device, criterion, max_grad_norm
            )
            metrics = evaluate(model, val_loader, device)
            elapsed = time.time() - start

            print(
                f"  Epoch {epoch+1}/{num_epochs} | "
                f"loss={train_loss:.4f} | "
                f"acc={metrics['accuracy']:.4f} | "
                f"f1={metrics['f1']:.4f} | "
                f"macro_f1={metrics['macro_f1']:.4f} | "
                f"耗时={elapsed:.1f}s"
            )

            if metrics[select_metric] > best_score:
                best_score = metrics[select_metric]
                best_epoch = epoch + 1
                best_metrics = metrics
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                model.save(str(OUTPUTS_DIR / f"fold_{i+1}_best.pt"))

        fold_result = {
            "fold": i + 1,
            "val_event": fold["val_event"],
            "best_epoch": best_epoch,
            "select_metric": select_metric,
            "best_score": best_score,
        }
        fold_result.update(best_metrics)  # 最佳 epoch 的完整指标
        all_results.append(fold_result)

        print(f"  ✅ 最佳 {select_metric}={best_score:.4f} (epoch {best_epoch})")

    # 汇总结果（基于各折「最佳 epoch」）
    mean_acc, std_acc = _mean_std([r["accuracy"] for r in all_results])
    mean_f1, std_f1 = _mean_std([r["f1"] for r in all_results])
    mean_macro_f1, std_macro_f1 = _mean_std([r["macro_f1"] for r in all_results])
    mean_bal_acc, std_bal_acc = _mean_std([r["balanced_accuracy"] for r in all_results])
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
            "max_grad_norm": max_grad_norm,
            "use_class_weights": use_class_weights,
            "select_metric": select_metric,
        },
        "folds": all_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "mean_balanced_accuracy": mean_bal_acc,
        "std_balanced_accuracy": std_bal_acc,
    }

    # 保存完整结果
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"交叉验证汇总（各折取最佳 {select_metric} 的 epoch）")
    print(f"  平均准确率:      {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  平均 F1(谣言):   {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  平均 macro_f1:   {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"  平均平衡准确率:  {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")
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
    max_grad_norm: float = DEFAULT_HP["max_grad_norm"],
    use_class_weights: bool = DEFAULT_HP["use_class_weights"],
    select_metric: str = DEFAULT_HP["select_metric"],
    dev_ratio: float = DEFAULT_HP["dev_ratio"],
    patience: int = DEFAULT_HP["patience"],
    device: str = None,
) -> RumorClassifier:
    """训练最终模型。

    dev_ratio > 0（默认）：从 train.csv 分层切出 dev 集，每轮在 dev 上评估，
    按 select_metric 保存最佳 epoch 作为最终模型，并在连续 patience 轮无提升时早停。

    dev_ratio <= 0：在全部训练数据上训练固定轮数，保存最后一轮（旧行为）。

    Returns:
        训练好的 RumorClassifier 实例（dev 模式下为最佳 epoch 的权重）。
    """
    if device is None:
        device = DEFAULT_HP["device"]

    print(f"设备: {device}")
    print(f"类别加权: {use_class_weights} | 梯度裁剪: {max_grad_norm}")

    texts, labels, _events = load_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    final_path = str(OUTPUTS_DIR / "final_model.pt")

    # ── 无 dev 集：全量训练，保存最后一轮 ──
    if dev_ratio is None or dev_ratio <= 0:
        print("模式: 全部训练数据（无验证集，保存最后一轮）")
        dataset = RumorDataset(texts, labels, tokenizer, max_length)
        dataloader = create_dataloader(dataset, batch_size, shuffle=True)
        criterion = _build_criterion(labels, use_class_weights, device)

        model = RumorClassifier(model_name=model_name, num_labels=2, dropout=dropout)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        total_steps = len(dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * warmup_ratio), total_steps
        )

        for epoch in range(num_epochs):
            start = time.time()
            loss = train_epoch(
                model, dataloader, optimizer, scheduler, device, criterion, max_grad_norm
            )
            print(f"  Epoch {epoch+1}/{num_epochs} | loss={loss:.4f} | 耗时={time.time()-start:.1f}s")

        model.save(final_path)
        tokenizer.save_pretrained(str(OUTPUTS_DIR / "tokenizer"))
        print(f"最终模型已保存至: {final_path}")
        return model

    # ── 有 dev 集：分层划分 + 早停 + 保存最佳 epoch ──
    train_texts, train_labels, dev_texts, dev_labels = stratified_split(
        texts, labels, dev_ratio=dev_ratio
    )
    print(f"模式: 分层划分 (dev_ratio={dev_ratio}) | 训练 {len(train_texts)} / dev {len(dev_texts)}")
    print(f"选择指标: {select_metric} | 早停 patience: {patience}")

    train_ds = RumorDataset(train_texts, train_labels, tokenizer, max_length)
    dev_ds = RumorDataset(dev_texts, dev_labels, tokenizer, max_length)
    train_loader = create_dataloader(train_ds, batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_ds, batch_size, shuffle=False)

    criterion = _build_criterion(train_labels, use_class_weights, device)

    model = RumorClassifier(model_name=model_name, num_labels=2, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )

    best_score = -1.0
    best_epoch = 0
    best_metrics = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion, max_grad_norm
        )
        metrics = evaluate(model, dev_loader, device)
        elapsed = time.time() - start

        print(
            f"  Epoch {epoch+1}/{num_epochs} | loss={train_loss:.4f} | "
            f"dev_acc={metrics['accuracy']:.4f} | dev_f1={metrics['f1']:.4f} | "
            f"dev_macro_f1={metrics['macro_f1']:.4f} | 耗时={elapsed:.1f}s"
        )

        if metrics[select_metric] > best_score:
            best_score = metrics[select_metric]
            best_epoch = epoch + 1
            best_metrics = metrics
            epochs_no_improve = 0
            model.save(final_path)
            print(f"    ↑ {select_metric} 提升至 {best_score:.4f}，已保存最佳模型")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    ⏹ 连续 {patience} 轮无提升，早停于 epoch {epoch+1}")
                break

    tokenizer.save_pretrained(str(OUTPUTS_DIR / "tokenizer"))

    print(f"\n最终模型 = 最佳 epoch {best_epoch}（dev {select_metric}={best_score:.4f}）")
    print(
        f"  dev 指标: acc={best_metrics['accuracy']:.4f}, "
        f"f1={best_metrics['f1']:.4f}, macro_f1={best_metrics['macro_f1']:.4f}, "
        f"balanced_acc={best_metrics['balanced_accuracy']:.4f}"
    )
    print(f"  已保存至: {final_path}")

    # 重新加载最佳权重返回（内存中的 model 是最后一轮，不一定是最佳）
    return RumorClassifier.load(final_path, device=device)
