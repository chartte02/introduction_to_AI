"""快速诊断：检查 PyTorch 和 DataLoader 是否正常工作。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("no_proxy", "*")

import torch
from transformers import AutoTokenizer
from model.data import load_data, RumorDataset, stratified_split
from model.model import RumorClassifier
from torch.utils.data import DataLoader

print("1. PyTorch 版本:", torch.__version__)
print("2. CPU 核心数:", os.cpu_count())

# 加载数据
texts, labels, events = load_data()
print(f"3. 数据加载完成: {len(texts)} 条")

# 分层划分
train_texts, train_labels, dev_texts, dev_labels = stratified_split(texts, labels)
print(f"4. 分层划分: train={len(train_texts)}, dev={len(dev_texts)}")

# tokenizer + dataset
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
print("5. Tokenizer 加载完成")

train_ds = RumorDataset(train_texts, train_labels, tokenizer, max_length=128)
print(f"6. Dataset 创建完成: {len(train_ds)} 条")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
print(f"7. DataLoader 创建完成: {len(train_loader)} 个 batch")

# 取一个 batch 试试
print("8. 开始加载第一个 batch...")
try:
    batch = next(iter(train_loader))
    print(f"   ✅ batch 加载成功! input_ids shape: {batch['input_ids'].shape}")
except Exception as e:
    print(f"   ❌ batch 加载失败: {e}")
    sys.exit(1)

# 加载模型并跑一次前向传播
print("9. 开始创建模型...")
model = RumorClassifier(model_name="cardiffnlp/twitter-roberta-base")
print("10. 模型创建完成")

model.eval()
print("11. 开始前向传播...")
with torch.no_grad():
    logits = model(batch["input_ids"], batch["attention_mask"])
print(f"12. ✅ 前向传播成功! logits shape: {logits.shape}")

print("\n🎉 全部通过！模型和数据都正常。现在可以训练了。")
