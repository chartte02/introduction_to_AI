"""文本预处理模块。

按照 AGENTS.md §8 的定义实现预处理 pipeline：
  原始推文 → HTML解码 → URL替换 → @mention替换 → #去除 → 空白合并 → BERT Tokenizer
"""

import re
import html


# URL 匹配模式：http:// 或 https:// 开头的非空白字符序列
URL_PATTERN = re.compile(r"https?://\S+")

# @mention 匹配模式：@ 后跟非空白字符
MENTION_PATTERN = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    """对单条推文执行完整预处理（不含 tokenization）。

    Args:
        text: 原始推文文本。

    Returns:
        清洗后的文本。
    """
    # 1. HTML 实体解码：&amp; → & 等
    text = html.unescape(text)

    # 2. URL → [URL]
    text = URL_PATTERN.sub("[URL]", text)

    # 3. @mention → @USER
    text = MENTION_PATTERN.sub("@USER", text)

    # 4. 去除 # 符号，保留标签文字
    text = text.replace("#", "")

    # 5. 合并多余空白
    text = re.sub(r"\s+", " ", text).strip()

    return text
