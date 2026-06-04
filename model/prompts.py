"""Prompt 模板模块。

构造用于可解释性生成的 prompt，让 LLM 基于推文原文 + 分类结果
输出判断依据文本。
"""

from typing import List, Dict


# ── 系统提示词 ──

SYSTEM_PROMPT = """你是一个专业的谣言检测分析师。你的任务是根据一条推文和分类模型的检测结果，分析并解释该分类结果的依据。

请从以下几个角度进行分析（选择适用的）：
1. **信息来源与引用**：推文中是否引用可信/不可信来源？是否包含 URL 链接？
2. **措辞与情绪**：推文措辞是否客观冷静，还是带有煽动性、情绪化表达？
3. **事实可验证性**：推文中的声称是否有具体的时间、地点、人物等可验证细节？
4. **逻辑一致性**：推文内容是否自洽？是否存在逻辑矛盾？
5. **语言特征**：是否使用全大写、感叹号、夸张修辞等谣言常见手法？

要求：
- 用 2-4 句话简要分析，言之有物
- 基于推文原文的具体内容进行分析，不要泛泛而谈
- 使用中文回答
- 不要说"作为AI模型"之类的自我介绍
- 直接给出分析"""


# ── 不同场景的 prompt 构建函数 ──


def build_explanation_prompt(
    text: str,
    label: int,
    label_name: str,
    confidence: float,
) -> List[Dict[str, str]]:
    """构建可解释性生成的 prompt。

    Args:
        text: 原始推文文本。
        label: 分类标签（0 或 1）。
        label_name: 标签名称（"谣言" 或 "非谣言"）。
        confidence: 模型置信度（0~1）。

    Returns:
        符合 OpenAI 格式的消息列表。
    """
    user_prompt = f"""以下是一条社交媒体推文，已被谣言检测模型判定为「{label_name}」（置信度: {confidence:.2%}）。

请分析这条推文，解释为什么它被判定为{label_name}。

推文内容：
---
{text}
---"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_batch_explanation_prompt(
    samples: List[dict],
) -> List[List[Dict[str, str]]]:
    """批量构建多个样本的解释 prompt。

    Args:
        samples: 样本列表，每个元素包含 text, label, label_name, confidence。

    Returns:
        prompt 消息列表的列表。
    """
    prompts = []
    for s in samples:
        prompts.append(
            build_explanation_prompt(
                text=s["text"],
                label=s["label"],
                label_name=s["label_name"],
                confidence=s["confidence"],
            )
        )
    return prompts


def build_attention_analysis_prompt(
    text: str,
    label: int,
    label_name: str,
    confidence: float,
    top_words: list,
) -> List[Dict[str, str]]:
    """基于注意力权重构建 prompt，让 LLM 结合关键词进行分析。

    Args:
        text: 原始推文文本。
        label: 分类标签。
        label_name: 标签名称。
        confidence: 模型置信度。
        top_words: 注意力权重最高的关键词列表 [(word, weight), ...]。

    Returns:
        消息列表。
    """
    keywords_str = ", ".join([f"{w}({s:.3f})" for w, s in top_words])

    user_prompt = f"""以下是一条社交媒体推文，已被谣言检测模型判定为「{label_name}」（置信度: {confidence:.2%}）。

模型注意力主要集中在以下关键词上（按重要性排序）：
{keywords_str}

请结合这些关键词，分析推文被判定为{label_name}的原因。

推文内容：
---
{text}
---"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
