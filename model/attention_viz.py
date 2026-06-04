"""注意力可视化模块。

从 BERT 模型中提取注意力权重，实现：
  - 提取各层的 attention weights
  - 聚合跨层/跨头的注意力分数
  - 生成关键词重要性排序
  - 文本高亮可视化
"""

from typing import List, Tuple, Optional

import torch
import numpy as np

from model.model import RumorClassifier


def extract_attention(
    model: RumorClassifier,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple:
    """执行前向传播并提取注意力权重。

    利用 model.forward() 的 return_attentions 参数获取各层注意力。

    Args:
        model: RumorClassifier 实例。
        input_ids: token ID 张量，形状 (batch, seq_len)。
        attention_mask: 注意力掩码，形状 (batch, seq_len)。

    Returns:
        (logits, attentions) 其中 attentions 是各层注意力权重的元组，
        每层形状 (batch, num_heads, seq_len, seq_len)。
    """
    model.eval()
    with torch.no_grad():
        logits, attentions = model(input_ids, attention_mask, return_attentions=True)
    return logits, attentions


def aggregate_attention(
    attentions: tuple,
    method: str = "mean",
) -> torch.Tensor:
    """聚合各层的注意力权重。

    Args:
        attentions: 每层 attention 的元组，每个形状 (batch, num_heads, seq_len, seq_len)。
        method: 聚合方式——"mean" 取均值，"last" 只用最后一层，"first" 只用第一层。

    Returns:
        聚合后的注意力矩阵，形状 (batch, seq_len, seq_len)。
    """
    if not attentions:
        raise ValueError("注意力列表为空")

    # stack: (num_layers, batch, num_heads, seq_len, seq_len)
    stacked = torch.stack(list(attentions))

    if method == "last":
        attn = stacked[-1]
    elif method == "first":
        attn = stacked[0]
    else:  # "mean"
        attn = stacked.mean(dim=0)

    # 跨头平均: (batch, seq_len, seq_len)
    attn = attn.mean(dim=1)

    return attn


def extract_token_importance(
    attentions: tuple,
    method: str = "mean",
) -> torch.Tensor:
    """提取每个 token 的重要性分数。

    基于 [CLS] token 对其他 token 的注意力权重作为重要性指标。

    Args:
        attentions: 各层注意力元组。
        method: 聚合方式。

    Returns:
        每个 token 的重要性分数，形状 (batch, seq_len)。
    """
    agg_attn = aggregate_attention(attentions, method)
    # [CLS] token（索引 0）对所有 token 的注意力
    cls_attention = agg_attn[:, 0, :]
    return cls_attention


def get_top_keywords(
    tokenizer,
    input_ids: torch.Tensor,
    importance_scores: torch.Tensor,
    top_k: int = 10,
    exclude_special: bool = True,
) -> List[Tuple[str, float]]:
    """获取重要性最高的 top-k 关键词。

    Args:
        tokenizer: HuggingFace tokenizer。
        input_ids: token ID 张量，形状 (1, seq_len)。
        importance_scores: 重要性分数，形状 (1, seq_len)。
        top_k: 返回前 k 个。
        exclude_special: 是否排除 [CLS]/[SEP]/[PAD]。

    Returns:
        (词, 分数) 列表，按分数降序排列。
    """
    scores = importance_scores[0]
    ids = input_ids[0]

    token_scores = []
    for i in range(len(ids)):
        token_id = ids[i].item()
        if exclude_special and token_id in [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ]:
            continue
        word = tokenizer.decode([token_id])
        score = scores[i].item()
        token_scores.append((word, score))

    token_scores.sort(key=lambda x: x[1], reverse=True)
    return token_scores[:top_k]


def generate_attention_highlight(
    tokenizer,
    input_ids: torch.Tensor,
    importance_scores: torch.Tensor,
) -> str:
    """生成带注意力高亮的文本（Markdown 格式）。

    分数 >= 0.7: **加粗**
    分数 >= 0.4: *斜体*
    分数 < 0.4:  普通

    Args:
        tokenizer: HuggingFace tokenizer。
        input_ids: token ID 张量，形状 (1, seq_len)。
        importance_scores: 重要性分数，形状 (1, seq_len)。

    Returns:
        带高亮标记的文本。
    """
    scores = importance_scores[0]
    ids = input_ids[0]

    # 归一化分数到 0~1
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    parts = []
    buffer = ""

    for i in range(len(ids)):
        token_id = ids[i].item()
        if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue

        word = tokenizer.decode([token_id])
        score = scores[i].item()

        # 处理 subword（如 "##ing"）
        if word.startswith("##"):
            buffer += word[2:]
            continue

        if buffer:
            parts.append(buffer)
            buffer = ""

        if score >= 0.7:
            parts.append(f"**{word}**")
        elif score >= 0.4:
            parts.append(f"*{word}*")
        else:
            parts.append(word)

    if buffer:
        parts.append(buffer)

    return " ".join(parts)


def visualize_attention_heatmap(
    attentions: tuple,
    tokenizer,
    input_ids: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "注意力热力图",
):
    """绘制注意力热力图。

    Args:
        attentions: 各层注意力元组。
        tokenizer: HuggingFace tokenizer。
        input_ids: token ID 张量，形状 (1, seq_len)。
        save_path: 图片保存路径（如 "outputs/attention_heatmap.png"）。
        title: 图表标题。
    """
    import matplotlib.pyplot as plt

    agg_attn = aggregate_attention(attentions, method="mean")
    attn_matrix = agg_attn[0].cpu().numpy()

    # 解码 token，过滤特殊 token
    tokens = []
    for i in range(input_ids.shape[1]):
        tid = input_ids[0, i].item()
        if tid in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        tokens.append(tokenizer.decode([tid]).replace("##", ""))

    # 截取到有效长度
    valid_len = len(tokens)
    valid_attn = attn_matrix[:valid_len, :valid_len]

    # 如果 token 太多，只显示头尾各 20 个
    if valid_len > 40:
        keep = 20
        head_tokens = tokens[:keep]
        tail_tokens = tokens[-keep:]
        display_tokens = head_tokens + ["..."] + tail_tokens
        head_attn = valid_attn[:keep, :keep]
        tail_attn = valid_attn[-keep:, -keep:]
        top = np.concatenate([head_attn, np.zeros((keep, 1)), np.zeros((keep, keep))], axis=1)
        bottom = np.concatenate([np.zeros((keep, keep)), np.zeros((keep, 1)), tail_attn], axis=1)
        middle = np.zeros((1, 2 * keep + 1))
        valid_attn = np.concatenate([top, middle, bottom], axis=0)
    else:
        display_tokens = tokens

    fig_width = max(8, len(display_tokens) * 0.35)
    fig_height = max(6, len(display_tokens) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(valid_attn, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(display_tokens)))
    ax.set_yticks(range(len(display_tokens)))
    ax.set_xticklabels(display_tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(display_tokens, fontsize=8)
    ax.set_xlabel("Attended to →")
    ax.set_ylabel("Attending from →")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Attention weight", shrink=0.8)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"注意力热力图已保存: {save_path}")

    plt.close(fig)

