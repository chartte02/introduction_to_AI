"""数据加载与事件感知的交叉验证划分模块。

提供：
  - load_data()：从 CSV 加载并清洗文本
  - get_event_folds()：按事件分组，生成留一事件交叉验证的 train/val 划分
  - stratified_split()：按标签分层切出 dev 集（用于最终模型早停）
"""

import csv
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from model.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 所有事件 ID 列表
ALL_EVENTS = ["0", "1", "2", "3", "4", "5", "6"]

# 固定随机种子
SEED = 42

class RumorDataset(Dataset):
    """谣言检测数据集。

    每个样本返回 (input_ids, attention_mask, label)
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data(csv_path: Path = None) -> Tuple[List[str], List[int], List[str]]:
    """
    Args:
        csv_path: CSV 文件路径，默认值=data/train.csv

    Returns:
        (texts, labels, events) 三元组。
        - texts: 清洗后的文本列表
        - labels: 标签列表 (0/1)
        - events: 事件 ID 列表 (字符串 "0"~"6")
    """
    if csv_path is None:
        csv_path = DATA_DIR / "train.csv"

    texts, labels, events = [], [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过 header
        for row in reader:
            if len(row) < 4:
                continue  # 跳过格式异常行
            texts.append(clean_text(row[1]))
            labels.append(int(row[2]))
            events.append(row[3])

    return texts, labels, events


def get_event_folds() -> List[Dict[str, List[str]]]:
    """生成留一事件交叉验证的 fold 划分。

    返回 7 个 fold，每个 fold 中：
      - "train_events": 用于训练的事件 ID 列表
      - "val_event":   用于验证的事件 ID

    例：fold 0 用事件 1,2,3,4,5,6 训练，事件 0 验证。

    Returns:
        folds 列表，每个元素为 {"train_events": [...], "val_event": "0"}。
    """
    folds = []
    for held_out in ALL_EVENTS:
        train_events = [e for e in ALL_EVENTS if e != held_out]
        folds.append({
            "train_events": train_events,
            "val_event": held_out,
        })
    return folds


def build_fold_datasets(
    texts: List[str],
    labels: List[int],
    events: List[str],
    fold: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> Tuple[RumorDataset, RumorDataset]:
    """根据 fold 配置构建训练集和验证集。

    Args:
        texts: 全部文本
        labels: 全部标签
        events: 全部事件 ID
        fold: get_event_folds() 返回的单个 fold 配置

    Returns:
        (train_dataset, val_dataset)
    """
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []

    for text, label, event in zip(texts, labels, events):
        if event in fold["train_events"]:
            train_texts.append(text)
            train_labels.append(label)
        elif event == fold["val_event"]:
            val_texts.append(text)
            val_labels.append(label)

    train_dataset = RumorDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = RumorDataset(val_texts, val_labels, tokenizer, max_length)

    return train_dataset, val_dataset


def stratified_split(
    texts: List[str],
    labels: List[int],
    dev_ratio: float = 0.1,
    seed: int = SEED,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """按标签分层，从训练数据中切出验证集。

    用于最终模型训练时的早停 / 选最佳 epoch。分层（stratify）保证 train/dev
    的谣言比例一致，避免少数类在验证集中缺失。

    dev 集来自 train.csv，与 val.csv（测试集）无关，避免在测试集上
    调参导致信息泄漏。

    Args:
        texts: 全部文本
        labels: 全部标签 (0/1)
        dev_ratio: dev 集占比
        seed: 随机种子（保证可复现）

    Returns:
        (train_texts, train_labels, dev_texts, dev_labels)
    """
    from sklearn.model_selection import train_test_split

    indices = list(range(len(texts)))
    train_idx, dev_idx = train_test_split(
        indices,
        test_size=dev_ratio,
        stratify=labels,
        random_state=seed,
    )
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    dev_texts = [texts[i] for i in dev_idx]
    dev_labels = [labels[i] for i in dev_idx]
    return train_texts, train_labels, dev_texts, dev_labels
