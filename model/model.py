"""BERT 分类器模型模块。

提供 RumorClassifier：基于预训练 Transformer 的二分类模型。
分类头设计：取 [CLS] token → Linear(768, 2)。
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class RumorClassifier(nn.Module):
    """基于预训练 BERT/RoBERTa 的谣言二分类器。

    架构：
        text → encoder → [CLS] vector → Linear(768, 2) → logits

    Args:
        model_name: HuggingFace 模型名，默认 twitter-roberta-base。
        num_labels: 分类数，固定为 2。
        dropout: [CLS] 向量后的 dropout 率。
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base",
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 通常为 768

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: token ID 张量，形状 (batch, seq_len)。
            attention_mask: 注意力掩码，形状 (batch, seq_len)。
            return_attentions: 是否同时返回注意力权重。

        Returns:
            如果 return_attentions=False，返回 logits 张量，形状 (batch, 2)。
            如果 return_attentions=True，返回 (logits, attentions)，
            其中 attentions 是各层注意力权重的元组。
        """
        # encoder 输出（需要时开启 attention 输出）
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions,
        )

        # 取 [CLS] 向量（序列第一个 token）
        cls_vec = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        # 分类头
        cls_vec = self.dropout(cls_vec)
        cls_vec = cls_vec.float()
        logits = self.classifier(cls_vec)  # (batch, 2)

        if return_attentions:
            # attentions: 元组，每层一个 (batch, num_heads, seq_len, seq_len)
            return logits, outputs.attentions

        return logits

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_name": self.model_name,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RumorClassifier":
        """从文件加载模型。

        Args:
            path: 模型文件路径。
            device: 加载到的设备。

        Returns:
            RumorClassifier 实例。
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(model_name=checkpoint["model_name"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
