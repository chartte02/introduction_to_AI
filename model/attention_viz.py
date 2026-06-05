"""注意力可视化模块。

从 BERT 模型中提取注意力权重，实现：
  - 提取各层的 attention weights
  - 聚合跨层/跨头的注意力分数
  - 生成关键词重要性排序
  - 文本高亮可视化
"""

import re
from typing import List, Tuple, Optional

import torch
import numpy as np

from model.model import RumorClassifier


# 仅标点符号的 token 模式（含 Ġ 前缀和纯标点）
_PUNCT_PATTERN = re.compile(r"^[Ġ]?[^a-zA-Z0-9\u4e00-\u9fff]+$")


def _is_punct(token: str) -> bool:
    """判断一个 token 是否纯标点/符号（不含字母数字和中文）。"""
    return bool(_PUNCT_PATTERN.match(token))


def _is_continuation(token: str) -> bool:
    """判断是否为 subword continuation（不以 Ġ 开头且非特殊 token 且非标点）。"""
    clean = token.replace("##", "").replace("Ġ", "")
    if _is_punct(clean):
        return False
    return not token.startswith("Ġ") and token not in ("<s>", "</s>", "<pad>", "<mask>")


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
    """获取重要性最高的 top-k 关键词（自动合并 subword 并过滤标点）。

    Args:
        tokenizer: HuggingFace tokenizer。
        input_ids: token ID 张量，形状 (1, seq_len)。
        importance_scores: 重要性分数，形状 (1, seq_len)。
        top_k: 返回前 k 个。
        exclude_special: 是否排除特殊 token。

    Returns:
        (词, 分数) 列表，按分数降序排列。
    """
    scores = importance_scores[0]
    ids = input_ids[0]

    # 将 token ID 转为带 subword 标记的字符串
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())

    # 合并 subword 并过滤标点
    word_scores = []
    buffer = ""
    buffer_score = 0.0
    buffer_count = 0

    for i, (tid, tok, score) in enumerate(zip(ids.tolist(), tokens, scores.tolist())):
        if exclude_special and tid in [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ]:
            continue

        if _is_continuation(tok):
            # subword continuation：拼接到前一个词
            clean = tok.replace("##", "").replace("Ġ", "")
            buffer += clean
            buffer_score += score
            buffer_count += 1
        else:
            # 新词开始，先保存前一个词
            if buffer:
                avg_score = buffer_score / max(buffer_count, 1)
                word_scores.append((buffer, avg_score))

            # 去除 Ġ 前缀
            clean = tok.replace("Ġ", "")
            if _is_punct(clean):
                buffer = ""
                buffer_score = 0.0
                buffer_count = 0
                continue
            buffer = clean
            buffer_score = score
            buffer_count = 1

    # 最后一个词
    if buffer and not _is_punct(buffer):
        avg_score = buffer_score / max(buffer_count, 1)
        word_scores.append((buffer, avg_score))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:top_k]


def generate_attention_highlight(
    tokenizer,
    input_ids: torch.Tensor,
    importance_scores: torch.Tensor,
) -> str:
    """生成带注意力高亮的文本（Markdown 格式，自动合并 subword）。

    Args:
        tokenizer: HuggingFace tokenizer。
        input_ids: token ID 张量，形状 (1, seq_len)。
        importance_scores: 重要性分数，形状 (1, seq_len)。

    Returns:
        带高亮标记的文本（**加粗**=高关注，*斜体*=中等关注）。
    """
    scores = importance_scores[0]
    ids = input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())

    # 归一化分数
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    parts = []
    buffer = ""
    buffer_scores = []

    for i, (tid, tok, score) in enumerate(zip(ids.tolist(), tokens, scores.tolist())):
        if tid in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue

        if _is_continuation(tok):
            clean = tok.replace("##", "").replace("Ġ", "")
            buffer += clean
            buffer_scores.append(score)
        else:
            # 保存前一个完整词
            if buffer:
                avg = sum(buffer_scores) / max(len(buffer_scores), 1)
                _append_highlighted(parts, buffer, avg)
                buffer = ""
                buffer_scores = []

            clean = tok.replace("Ġ", "")
            buffer = clean
            buffer_scores = [score]

    # 最后一个词
    if buffer:
        avg = sum(buffer_scores) / max(len(buffer_scores), 1)
        _append_highlighted(parts, buffer, avg)

    return " ".join(parts)


def _append_highlighted(parts: list, word: str, score: float):
    """根据分数将词加上高亮标记，追加到 parts 列表。"""
    if _is_punct(word):
        parts.append(word)
    elif score >= 0.7:
        parts.append(f"**{word}**")
    elif score >= 0.4:
        parts.append(f"*{word}*")
    else:
        parts.append(word)


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

